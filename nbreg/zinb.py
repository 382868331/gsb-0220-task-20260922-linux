"""Zero-inflated NB2 regression (ZINB) with a constant logit parameter.

Model
-----
Each observation is drawn from a two-component *mixture* (the component
label is never observed and is never assigned by hard thresholding):

    with probability pi            : Y = 0        (structural zero)
    with probability (1 - pi)      : Y ~ NB2(mu_i, alpha)
    mu_i  = exp(x_i . beta + offset_i)
    logit(pi) = z                  (one scalar, no covariates)

The observation density is

    P(Y=0)   = pi + (1 - pi) * p0_i
    P(Y=k>0) = (1 - pi) * f_NB(k; mu_i, alpha)

where p0_i = P_NB(Y=0) = (1 + alpha*mu_i)^(-1/alpha).  The log-likelihood
is the *mixture* log-likelihood; zeros contribute through

    tau_i = (1-pi) p0_i / (pi + (1-pi) p0_i)

which is the posterior probability that a zero came from the count
component.  This soft weight -- not a hard split -- multiplies the
count-component score for zero observations.  Positive observations are
count-component with posterior weight 1.  Only Python's standard library
is used; NB2 primitives are reused from :mod:`nbreg.nb2`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .nb2 import (
    NBFitError,
    LOG_ALPHA_MIN,
    LOG_ALPHA_MAX,
    _ETA_LIMIT,
    _NEAR_POISSON_ALPHA,
    _STEP_CAP,
    _SUM_EXACT_MAX_Y,
    _cholesky_solve,
    _check_full_rank,
    _digamma,
    _trigamma,
    _is_real,
    _validate_positive_alpha,
    _validate_xy,
    nb2_pmf,
)

# pi is parameterized by z = logit(pi); the fitter keeps z in this box.
# z = -30  <=>  pi ~= 9.4e-14, numerically the ordinary NB2 model, and the
# no-inflation degeneracy is flagged well before hitting the bound.
LOGIT_PI_MIN = -30.0
LOGIT_PI_MAX = 30.0

_NEAR_NO_INFLATION_PI = 1e-6  # fitted pi below this is flagged NB2-like
_BOUND_TOL = 1e-9


class ZINBFitError(NBFitError):
    """Invalid input for a ZINB fit. Same hierarchy as NBFitError, so
    existing ``except NBFitError`` call sites keep working."""


@dataclass(frozen=True)
class ZINBResult:
    """Outcome of :func:`fit_zinb`.

    ``converged`` is True only when ``status == "converged"``.  Statuses:

    * ``"converged"``           -- projected gradient below tolerance
                                   (KKT at the box boundaries counts).
    * ``"max_iter"``            -- iteration budget exhausted; the best
                                   point found is returned, not certified.
    * ``"line_search_failed"``  -- no improving step found (numerical
                                   budget exhausted; not certified).
    * ``"no_finite_mle"``       -- certified structural boundary: all
                                   counts are zero.  The log-likelihood is
                                   strictly below 0 for every finite
                                   parameter and approaches its supremum 0
                                   only as mu -> 0 or pi -> 1, neither of
                                   which is attained at a finite point.

    ``gradient`` is d(loglik)/d(beta, log_alpha, logit_pi), length k+2,
    at the returned point.  ``pi_boundary`` reports parameter-boundary
    degeneracy: ``"lower"`` (pi collapsed to 0; the model reduces to NB2),
    ``"upper"`` (pi at 1; only possible with no positive counts), or
    ``"none"``.
    """

    converged: bool
    status: str
    message: str
    beta: list
    log_alpha: float
    alpha: float
    logit_pi: float
    pi: float
    loglik: float
    gradient: list
    n_iter: int
    n_rejected: int
    near_poisson: bool
    near_no_inflation: bool
    pi_boundary: str
    n_obs: int
    n_zeros: int
    n_params: int


# ---------------------------------------------------------------------------
# Small stable helpers
# ---------------------------------------------------------------------------

def _softplus(u):
    if u > 35.0:
        return u + math.log1p(math.exp(-u))
    return math.log1p(math.exp(u))


def _logistic(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logaddexp(a, b):
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


# ---------------------------------------------------------------------------
# Mixture log-likelihood, gradient and observed Hessian
# ---------------------------------------------------------------------------

def _zinb_ll_grad_hess(yv, Xv, off, beta, theta, z):
    """Return (loglik, grad[k+2], Hessian[k+2][k+2], etas).

    Parameter order: beta[0..k-1], theta=log alpha at index k,
    z=logit(pi) at index k+1.

    Everything is evaluated in overflow-safe log space; the only
    exponentiated mixture quantity is the posterior weight tau, computed
    as 1/(1+exp(z - log p0)) so it underflows straight to 0 when
    appropriate (a zero that is certainly structural at the parameters).
    """
    n = len(yv)
    k = len(beta)
    it = k          # theta index
    iz = k + 1      # z index
    alpha = math.exp(theta)
    r = 1.0 / alpha
    pi = _logistic(z)
    s = 1.0 - pi
    spz = _softplus(z)  # log s = -spz, log pi = z - spz

    ll = 0.0
    g = [0.0] * (k + 2)
    H = [[0.0] * (k + 2) for _ in range(k + 2)]
    etas = [0.0] * n

    for i in range(n):
        xi = Xv[i]
        yi = yv[i]
        eta = off[i]
        for j in range(k):
            eta += xi[j] * beta[j]
        etas[i] = eta
        t = theta + eta
        if t > 35.0:
            L = t + math.log1p(math.exp(-t))
        else:
            L = math.log1p(math.exp(t))
        q = math.exp(-L)            # 1/(1+alpha*mu)
        w = math.exp(eta - L)       # mu/(1+alpha*mu), 0 < w <= r
        h = 1.0 - q                 # alpha*mu/(1+alpha*mu)

        if yi == 0:
            # ---- structural/count ambiguity for a zero ------------------
            # log p0 = -r L;  m(0) = pi + s*p0.
            l0 = -r * L
            # tau = s*p0/(pi+s*p0) = 1/(1+exp(z - l0)); pi/D = 1-tau.
            m_arg = z - l0
            tau = 0.0 if m_arg > 50.0 else 1.0 / (1.0 + math.exp(m_arg))
            one_minus_tau = 1.0 - tau

            # log m(0) = log s + logaddexp(z, l0)
            ll += -spz + _logaddexp(z, l0)

            # d log p0 / d eta = -w ; d log p0 / d theta = r*(L - h)
            if t < -2.0:
                v = math.exp(t)
                l_minus_h = math.log1p(v) - v / (1.0 + v)
            else:
                l_minus_h = L - h
            u_theta = r * l_minus_h

            # d log D/dz = (1-pi) - tau  (D = pi + (1-pi)*p0)
            g[iz] += s - tau
            ge = -tau * w
            for j in range(k):
                g[j] += ge * xi[j]
            g[it] += tau * u_theta

            # second derivatives (mixture curvature is NOT just the
            # weighted NB Hessian: tau itself depends on the parameters)
            taut = tau * one_minus_tau            # tau*(1-tau) in (0, 1/4]
            H[iz][iz] += taut - pi * s
            hz_e = taut * w                       # d2 l / dz deta
            hz_t = -taut * u_theta                # d2 l / dz dtheta
            for j in range(k):
                xj = xi[j]
                H[iz][j] += hz_e * xj
                H[j][iz] += hz_e * xj
                H[it][j] += (-taut * w * u_theta + tau * w * h) * xj
                for jj in range(j + 1):
                    H[j][jj] += (taut * w * w - tau * w * q) * xj * xi[jj]
            H[it][iz] += hz_t
            H[iz][it] += hz_t
            H[it][it] += taut * u_theta * u_theta - tau * u_theta + tau * r * h * h
            continue

        # ---- positive count: certainly from the count component ---------
        if yi <= _SUM_EXACT_MAX_Y:
            lg_ratio = 0.0   # lgamma(y+r) - lgamma(r) = sum log(r+m)
            dpsi = 0.0       # psi(y+r) - psi(r)      = sum 1/(r+m)
            dtrig = 0.0      # psi1(r) - psi1(y+r)    = sum 1/(r+m)^2
            for m in range(yi):
                sm = r + m
                lg_ratio += math.log(sm)
                dpsi += 1.0 / sm
                dtrig += 1.0 / (sm * sm)
        else:
            lg_ratio = math.lgamma(yi + r) - math.lgamma(r)
            dpsi = _digamma(yi + r) - _digamma(r)
            dtrig = _trigamma(r) - _trigamma(yi + r)

        # log P(Y=yi) = log s + log f_NB(yi)
        ll += (lg_ratio - math.lgamma(yi + 1)
               + yi * (theta + eta) - (yi + r) * L - spz)

        yr = yi + r
        gb = yi - yr * h
        for j in range(k):
            g[j] += gb * xi[j]
        g[it] += yi + r * L - r * dpsi - yr * h
        g[iz] += -pi

        hw = yr * h * q                # -d2 l_NB/d eta^2 (>= 0)
        cf = h * (r - yr * q)          # d2 l_NB/d eta dtheta
        for j in range(k):
            xj = xi[j]
            H[it][j] += cf * xj
            H[j][it] += cf * xj
            for jj in range(j + 1):
                H[j][jj] -= hw * xj * xi[jj]
        H[it][it] += (-r * L + r * dpsi - r * r * dtrig
                      + 2.0 * r * h - yr * h * q)
        H[iz][iz] += -pi * s

    # mirror the symmetric lower triangles written by the beta-beta loop
    # and by H[it][j] / H[iz][j] cross-block accumulation
    for j in range(k):
        H[j][it] = H[it][j]
        H[j][iz] = H[iz][j]
        for jj in range(j):
            H[jj][j] = H[j][jj]
    return ll, g, H, etas


# ---------------------------------------------------------------------------
# Public pure functions
# ---------------------------------------------------------------------------

def _validate_params(beta, k, log_alpha, logit_pi):
    if isinstance(beta, (str, bytes)):
        raise ZINBFitError("beta must be a sequence of k real numbers")
    try:
        bv = list(beta)
    except TypeError:
        raise ZINBFitError("beta must be a sequence of k real numbers") from None
    if len(bv) != k:
        raise ZINBFitError(f"beta must have length k={k}, got {len(bv)}")
    for j, v in enumerate(bv):
        if not _is_real(v) or not math.isfinite(v):
            raise ZINBFitError(f"beta[{j}] must be a finite real number, got {v!r}")
    if not _is_real(log_alpha) or not math.isfinite(log_alpha):
        raise ZINBFitError(
            f"log_alpha must be a finite real number, got {log_alpha!r}")
    if not _is_real(logit_pi) or not math.isfinite(logit_pi):
        raise ZINBFitError(
            f"logit_pi must be a finite real number, got {logit_pi!r}")
    return [float(v) for v in bv], float(log_alpha), float(logit_pi)


def zinb_loglik(y, X, beta, log_alpha, logit_pi, offset=None):
    """Mixture log-likelihood of ZINB at (beta, log_alpha, logit_pi).

    Pure function; accepts any finite ``logit_pi`` (the fit-time box
    applies only to :func:`fit_zinb`).  Raises NBFitError (specifically
    ZINBFitError) on invalid input.
    """
    yv, Xv, off = _validate_xy(y, X, offset)
    bv, theta, z = _validate_params(beta, len(Xv[0]), log_alpha, logit_pi)
    ll, _, _, _ = _zinb_ll_grad_hess(yv, Xv, off, bv, theta, z)
    return ll


def zinb_objective(y, X, beta, log_alpha, logit_pi, offset=None):
    """Return ``(loglik, gradient)`` of the ZINB mixture likelihood.

    Gradient length k+2: d/d(beta_j), then d/d log_alpha, then
    d/d logit_pi.  Provided for independent derivative checks and used
    unchanged inside :func:`fit_zinb`.
    """
    yv, Xv, off = _validate_xy(y, X, offset)
    bv, theta, z = _validate_params(beta, len(Xv[0]), log_alpha, logit_pi)
    ll, g, _, _ = _zinb_ll_grad_hess(yv, Xv, off, bv, theta, z)
    return ll, g


def zinb_pmf(y_count, mu, alpha, pi):
    """Mixture probability mass P(Y = y_count) for ZINB.

    ``y_count`` non-negative int (bool/float rejected), ``mu``/``alpha``
    positive finite, ``pi`` a finite real in [0, 1] (the endpoints are
    allowed for this pure evaluation: pi=0 is NB2, pi=1 is point mass at
    zero).
    """
    if type(y_count) is not int:
        raise ZINBFitError(
            "y_count must be an int (bool rejected), got "
            f"{type(y_count).__name__} {y_count!r}")
    if y_count < 0:
        raise ZINBFitError(f"y_count must be non-negative, got {y_count}")
    if not _is_real(mu) or not math.isfinite(mu) or mu <= 0.0:
        raise ZINBFitError(f"mu must be a positive finite number, got {mu!r}")
    if not _is_real(alpha) or not math.isfinite(alpha) or alpha <= 0.0:
        raise ZINBFitError(f"alpha must be a positive finite number, got {alpha!r}")
    if not _is_real(pi) or not math.isfinite(pi) or not 0.0 <= pi <= 1.0:
        raise ZINBFitError(f"pi must be a finite number in [0, 1], got {pi!r}")
    if y_count == 0:
        return pi + (1.0 - pi) * nb2_pmf(0, mu, alpha)
    return (1.0 - pi) * nb2_pmf(y_count, mu, alpha)


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

def _validate_pi_init(value):
    if not _is_real(value):
        raise ZINBFitError(
            f"pi_init must be a real number in (0, 1) (bool rejected), "
            f"got {type(value).__name__} {value!r}")
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ZINBFitError(
            f"pi_init must be strictly inside (0, 1) (use the likelihood "
            f"box for the boundary; 0 and 1 give infinite logit), "
            f"got {value!r}")
    return float(value)


def fit_zinb(y, X, offset=None, pi_init=None, alpha_init=None,
             max_iter=200, gtol=1e-8):
    """Fit zero-inflated NB2 by maximizing the mixture log-likelihood.

    Parameters
    ----------
    y, X, offset : as for :func:`nbreg.fit_nb2` (1..100 non-negative int
        counts, 1..5 finite full-rank regressor columns, optional offset).
    pi_init : optional structural-zero probability, strictly in (0, 1).
        Default estimates it from zero excess over the NB2 starting fit.
        (Only the *constant* logit z = logit(pi) is estimated; the zero
        component has no covariates.)
    alpha_init : optional positive finite initial alpha (same range as in
        :func:`nbreg.fit_nb2`).
    max_iter, gtol : same meaning and validation as in :func:`nbreg.fit_nb2`.

    Returns a :class:`ZINBResult` for every fitting outcome; raises
    NBFitError only for invalid input or a rank-deficient design (before
    any state is produced -- the fitter is a stateless batch function).
    """
    yv, Xv, off = _validate_xy(y, X, offset)
    n = len(yv)
    k = len(Xv[0])

    if type(max_iter) is not int or not 1 <= max_iter <= 1_000_000:
        raise ZINBFitError(
            f"max_iter must be an int in [1, 1000000] (bool rejected), "
            f"got {max_iter!r}")
    if not _is_real(gtol) or not math.isfinite(gtol) or not 0.0 < gtol <= 1.0:
        raise ZINBFitError(f"gtol must be a finite real in (0, 1], got {gtol!r}")
    if alpha_init is not None:
        alpha_init = _validate_positive_alpha(alpha_init, "alpha_init")
    if pi_init is not None:
        pi_init = _validate_pi_init(pi_init)

    _check_full_rank(Xv, n, k)
    dim = k + 2
    it = k
    iz = k + 1

    # ---- starting values -------------------------------------------------
    mean_y = sum(yv) / n
    mean_off = sum(off) / n
    beta0 = [0.0] * k
    for j in range(k):  # a constant column, if any, absorbs the mean level
        c0 = Xv[0][j]
        if c0 != 0.0 and all(Xv[i][j] == c0 for i in range(n)):
            beta0[j] = (math.log(mean_y + 0.5) - mean_off) / c0
            break
    if alpha_init is not None:
        theta0 = math.log(alpha_init)
    else:  # method of moments, clamped inside the supported box
        var_y = sum((v - mean_y) ** 2 for v in yv) / n
        a0 = (var_y - mean_y) / max(mean_y, 0.5) ** 2
        a0 = min(max(a0, math.exp(LOG_ALPHA_MIN + 2.0)), math.exp(6.0))
        theta0 = math.log(a0)

    if pi_init is not None:
        z0 = math.log(pi_init / (1.0 - pi_init))
    else:
        # zero excess: pi = (obs. zero fraction - mean NB2 p0) / (1 - p0)
        r0 = math.exp(-theta0)
        p0_sum = 0.0
        for i in range(n):
            eta0 = off[i] + sum(Xv[i][j] * beta0[j] for j in range(k))
            tt = theta0 + eta0
            L0 = tt + math.log1p(math.exp(-tt)) if tt > 35.0 else math.log1p(math.exp(tt))
            p0_sum += math.exp(-r0 * L0)
        frac0 = sum(1 for v in yv if v == 0) / n
        pi0 = (frac0 - p0_sum / n) / max(1e-12, 1.0 - p0_sum / n)
        pi0 = min(max(pi0, 1e-4), 0.9)
        z0 = math.log(pi0 / (1.0 - pi0))

    p = list(beta0) + [theta0, z0]

    def evaluate(pv):
        theta, zc = pv[it], pv[iz]
        if not LOG_ALPHA_MIN <= theta <= LOG_ALPHA_MAX:
            return None
        if not LOGIT_PI_MIN <= zc <= LOGIT_PI_MAX:
            return None
        ll, g, H, etas = _zinb_ll_grad_hess(yv, Xv, off, pv[:k], theta, zc)
        if not math.isfinite(ll):
            return None
        for e in etas:
            if e > _ETA_LIMIT or e < -_ETA_LIMIT:
                return None
        return ll, g, H, etas

    ev = evaluate(p)
    if ev is None:
        raise ZINBFitError(
            "initial point is outside the computable region: need "
            f"|x_i.beta + offset_i| <= {_ETA_LIMIT:.0f} for every row, "
            f"log_alpha in [{LOG_ALPHA_MIN}, {LOG_ALPHA_MAX}] and "
            f"logit_pi in [{LOGIT_PI_MIN}, {LOGIT_PI_MAX}]; rescale X or "
            f"offset (values are never clipped silently)")

    n_zeros = sum(1 for v in yv if v == 0)
    all_zero = n_zeros == n

    if all_zero:
        # Proved structural boundary: with pi < 1 and finite mu every
        # factor pi + (1-pi) p0 is strictly < 1, so ll < 0. The supremum
        # 0 is approached both as mu -> 0 (p0 -> 1) and as pi -> 1,
        # neither attained at a finite parameter.
        ll, g, _, _ = ev
        return ZINBResult(
            converged=False,
            status="no_finite_mle",
            message=("all counts are zero: the ZINB log-likelihood is < 0 "
                     "for every finite (mu, pi) and reaches its supremum 0 "
                     "only as mu -> 0 or pi -> 1; no finite MLE exists"),
            beta=list(p[:k]),
            log_alpha=p[it],
            alpha=math.exp(p[it]),
            logit_pi=p[iz],
            pi=_logistic(p[iz]),
            loglik=ll,
            gradient=list(g),
            n_iter=0,
            n_rejected=0,
            near_poisson=False,
            near_no_inflation=False,
            pi_boundary="none",
            n_obs=n,
            n_zeros=n_zeros,
            n_params=dim,
        )

    n_rejected = 0
    it_count = 0
    status = None
    message = ""

    # box bounds per coordinate: beta unbounded, theta/z boxed
    lo = [-math.inf] * k + [LOG_ALPHA_MIN, LOGIT_PI_MIN]
    hi = [math.inf] * k + [LOG_ALPHA_MAX, LOGIT_PI_MAX]

    while True:
        ll, g, H, _ = ev
        # --- projected-gradient (KKT) convergence at the box boundaries --
        gp = list(g)
        frozen = []
        for c in (it, iz):
            if p[c] <= lo[c] + _BOUND_TOL and g[c] < 0.0:
                gp[c] = 0.0
                frozen.append(c)
            elif p[c] >= hi[c] - _BOUND_TOL and g[c] > 0.0:
                gp[c] = 0.0
                frozen.append(c)
        gnorm = max(abs(v) for v in gp)
        if gnorm <= gtol * max(1.0, abs(ll)):
            status = "converged"
            message = ("projected gradient infinity-norm below "
                       "gtol * max(1, |loglik|) (boundary KKT included)")
            break
        if it_count >= max_iter:
            status = "max_iter"
            message = (f"iteration budget exhausted (max_iter={max_iter}); "
                       "returned point is the best found, not a certified "
                       "optimum")
            break

        # --- damped Newton on the observed Hessian -----------------------
        # The mixture Hessian can have positive curvature in z (a log-sum
        # of mixture weights is convex), so the Cholesky may need damping;
        # outward box-bound coordinates are frozen (KKT active set).
        negH = [[-H[i][j] for j in range(dim)] for i in range(dim)]
        diag = max(1.0, max(negH[i][i] for i in range(dim)))
        lam = 1e-10 * diag
        rhs = list(g)
        d = None
        gtd = 0.0
        for _ in range(14):
            M = [[negH[i][j] + (lam if i == j else 0.0)
                  for j in range(dim)] for i in range(dim)]
            for ac in frozen:
                for i in range(dim):
                    M[i][ac] = 0.0
                    M[ac][i] = 0.0
                M[ac][ac] = 1.0
                rhs[ac] = 0.0
            d = _cholesky_solve(M, rhs)
            if d is not None:
                gtd = sum(g[i] * d[i] for i in range(dim))
                if gtd > 0.0:
                    break
            lam *= 100.0
            d = None
        if d is None:  # steepest-ascent fallback (frozen coords zeroed)
            d = list(g)
            for ac in frozen:
                d[ac] = 0.0
            gtd = sum(v * v for v in d)
        dn = max(abs(v) for v in d)
        if dn > _STEP_CAP:
            d = [v * (_STEP_CAP / dn) for v in d]
            gtd = sum(g[i] * d[i] for i in range(dim))

        # --- Armijo backtracking; out-of-box/overflow steps rejected -----
        step = 1.0
        accepted = None
        for _ in range(60):
            cand = [p[i] + step * d[i] for i in range(dim)]
            evc = evaluate(cand)
            if evc is not None and evc[0] >= ll + 1e-4 * step * gtd:
                accepted = (cand, evc)
                break
            step *= 0.5
            n_rejected += 1
        if accepted is None:
            status = "line_search_failed"
            message = ("backtracking line search found no improving step "
                       "in 60 halvings; returned point is not a certified "
                       "optimum")
            break
        p, ev = accepted
        it_count += 1

    ll, g, _, _ = ev
    theta_f, z_f = p[it], p[iz]
    alpha_f = math.exp(theta_f)
    pi_f = _logistic(z_f)
    if z_f <= LOGIT_PI_MIN + _BOUND_TOL and g[iz] <= 0.0:
        pi_boundary = "lower"
    elif z_f >= LOGIT_PI_MAX - _BOUND_TOL and g[iz] >= 0.0:
        pi_boundary = "upper"
    else:
        pi_boundary = "none"
    return ZINBResult(
        converged=(status == "converged"),
        status=status,
        message=message,
        beta=list(p[:k]),
        log_alpha=theta_f,
        alpha=alpha_f,
        logit_pi=z_f,
        pi=pi_f,
        loglik=ll,
        gradient=list(g),
        n_iter=it_count,
        n_rejected=n_rejected,
        near_poisson=(alpha_f < _NEAR_POISSON_ALPHA),
        near_no_inflation=(pi_f < _NEAR_NO_INFLATION_PI),
        pi_boundary=pi_boundary,
        n_obs=n,
        n_zeros=n_zeros,
        n_params=dim,
    )
