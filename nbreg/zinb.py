"""Zero-inflated NB2 negative-binomial regression (stdlib only).

Model
-----
A constant zero-inflation component on top of the existing NB2 count
model (the count mean keeps the original covariates; the zero component
has *no* covariates -- a single constant logit parameter):

    Y = 0                 with probability pi
    Y ~ NB2(mu_i, alpha)  with probability 1 - pi
    mu_i = exp(x_i . beta + offset_i)

with gamma = logit(pi), theta = log(alpha).  The observation pmf is the
*mixture* likelihood

    P(Y=0)   = pi + (1 - pi) * f_NB(0 | mu, alpha)
    P(Y=k>0) = (1 - pi) * f_NB(k | mu, alpha)

Zeros are NEVER partitioned into a hard group: estimation keeps the soft
posterior responsibility (E-step weight)

    s_i = (1-pi) f_NB(0) / [pi + (1-pi) f_NB(0)]     for y_i = 0
    s_i = 1                                           for y_i > 0

and the score/information are the responsibility-weighted count scores
plus the posterior-variance outer product (the usual EM identity).

Boundary behavior
-----------------
* pi -> 0 (gamma -> -inf): the model collapses to plain NB2 ("no
  structural zeros"); reported as a converged boundary fit, flagged.
* pi -> 1 (gamma -> +inf): only possible for an all-zero sample; then
  every likelihood factor is < 1 for finite parameters and tends to 1
  only as pi -> 1, so the supremum log-likelihood 0 is not attained and
  beta/alpha are unidentifiable.  Certified as status "no_finite_mle".

Only the Python standard library is used; NB2 terms reuse nb2.py.
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
    _is_real,
    _nb_terms,
    _validate_xy,
    fit_nb2,
)

# logit(pi) is optimized inside this symmetric box.  pi spans about
# [2e-9, 1 - 2e-9]; both ends are degeneracy boundaries that are flagged
# rather than reported as interior optima.
LOGIT_PI_MIN = -20.0
LOGIT_PI_MAX = 20.0

# A fitted pi below this (or above 1 - this) is reported as a boundary
# degeneracy (collapses to NB2 / to a point mass at zero), even though
# Newton stopped just inside the box: at pi ~ 1e-6 the mixture and plain
# NB2 differ by less than 1e-6 in every mass and the MLE is, practically,
# the boundary value.
_PI_BOUNDARY_EPS = 1e-6


@dataclass(frozen=True)
class ZINBResult:
    """Outcome of :func:`fit_zinb`.

    ``converged`` is True only when ``status == "converged"``.  Statuses:

    * ``"converged"``           -- projected gradient below tolerance
                                   (possibly at a flagged box boundary).
    * ``"max_iter"``            -- iteration budget exhausted; the best
                                   point found is returned, not certified.
    * ``"line_search_failed"``  -- no improving step found (numerical
                                   budget exhausted; not certified).
    * ``"no_finite_mle"``       -- certified structural boundary: all
                                   counts are zero, the supremum is
                                   approached only as pi -> 1 and the
                                   count parameters are unidentifiable.

    Fields:

    * ``beta``/``log_alpha``/``alpha`` -- count-component parameters.
    * ``logit_pi``/``pi``              -- constant zero-inflation parameter.
    * ``gradient``  -- d(loglik)/d(beta, log_alpha, logit_pi), length k+2.
    * ``pi_at_lower_boundary`` -- True when the fit sits on gamma_min:
      no evidence of structural zeros, model degenerates to plain NB2.
    * ``pi_at_upper_boundary`` -- True when the fit sits on gamma_max
      (all-zero structural verdict).
    * ``near_poisson`` -- alpha < 1e-6 (count component near Poisson).
    * ``structural_zero_degenerate`` -- all-zero sample: no finite MLE.
    * ``responsibilities`` -- per-observation soft posterior probability
      that the observation comes from the *structural* zero state
      (0 for positive counts; values strictly in (0, 1) for zeros at a
      finite interior pi).  Nothing is hard-classified.
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
    pi_at_lower_boundary: bool
    pi_at_upper_boundary: bool
    structural_zero_degenerate: bool
    n_obs: int
    n_zero: int
    n_params: int
    responsibilities: list


# ---------------------------------------------------------------------------
# Stable scalar helpers
# ---------------------------------------------------------------------------

def _softplus(x):
    """log1p(exp(x)), stable for large |x|."""
    if x > 35.0:
        return x + math.log1p(math.exp(-x))
    if x < -745.0:
        return 0.0
    return math.log1p(math.exp(x))


def _sigmoid(x):
    """1/(1+exp(-x)), stable."""
    if x >= 0.0:
        if x > 700.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))
    if x < -700.0:
        return 0.0
    e = math.exp(x)
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# Mixture objective: log-likelihood, score, observed information
# ---------------------------------------------------------------------------

def _zinb_core(yv, Xv, off, beta, theta, gamma):
    """Return (loglik, grad[k+2], Hessian[k+2][k+2], etas, responsibilities).

    Index conventions: coordinates 0..k-1 are beta, coordinate k is
    theta = log(alpha), coordinate k+1 is gamma = logit(pi).

    For a zero observation let u = log f_NB(0), s = sigmoid(u - gamma)
    the count-state responsibility and t = 1 - s the structural-zero
    posterior; v = s*t is the posterior variance of the latent state.
    Writing a = d u / d(eta,theta) (which equals the ordinary NB2 zero
    score components), the per-observation information is

        s * I_NB  +  v * a a'

    and the gamma couplings are -v * a, with v - pi(1-pi) on gamma.
    Positive observations contribute the plain NB2 terms (s = 1) plus
    -pi(1-pi) on the gamma diagonal.
    """
    n = len(yv)
    k = len(beta)
    ti = k          # theta coordinate
    zi = k + 1      # gamma coordinate
    terms, etas = _nb_terms(yv, Xv, off, beta, theta)
    pi = _sigmoid(gamma)
    sp_g = _softplus(gamma)

    ll = 0.0
    g = [0.0] * (k + 2)
    H = [[0.0] * (k + 2) for _ in range(k + 2)]
    resp = [0.0] * n

    for i, (ll_i, gb, gth, hw, cf, hth, u) in enumerate(terms):
        xi = Xv[i]
        if yv[i] == 0:
            # log P(0) = u + softplus(gamma - u) - softplus(gamma)
            zug = gamma - u
            sp_zu = _softplus(zug)
            s = math.exp(-sp_zu)          # count-state responsibility
            tpost = 1.0 - s               # structural-zero posterior
            v = s * tpost                 # posterior variance
            ll += u + sp_zu - sp_g
            g[zi] += tpost - pi
            hgg = v
            resp[i] = tpost
        else:
            # log P(y>0) = log(1-pi) + log f_NB(y)
            s = 1.0
            v = 0.0
            ll += ll_i - sp_g
            g[zi] += -pi
            hgg = 0.0

        for j in range(k):
            g[j] += s * gb * xi[j]
        g[ti] += s * gth

        for j in range(k):
            xj = xi[j]
            H[j][j] -= s * hw * xj * xj
            for jj in range(j):
                H[j][jj] -= s * hw * xj * xi[jj]
            # EM outer-product term (zeros only; v = 0 otherwise)
            if v:
                duj = gb * xj
                for jj in range(j + 1):
                    H[j][jj] += v * duj * (gb * xi[jj])
                H[zi][j] += -v * gb * xj
            H[ti][j] += s * cf * xj + v * gth * gb * xj
        if v:
            H[zi][ti] += -v * gth
        H[ti][ti] += s * hth + v * gth * gth
        H[zi][zi] += hgg - pi * (1.0 - pi)

    for a in range(k + 2):
        for b in range(a):
            H[b][a] = H[a][b]
    return ll, g, H, etas, resp


def _validate_logit(value, name):
    if not _is_real(value):
        raise NBFitError(
            f"{name} must be a real number (bool rejected), "
            f"got {type(value).__name__} {value!r}")
    if not math.isfinite(value):
        raise NBFitError(f"{name} must be finite, got {value!r}")
    return float(value)


def _validate_point(beta, log_alpha, logit_pi, k):
    if isinstance(beta, (str, bytes)):
        raise NBFitError("beta must be a sequence of k real numbers")
    try:
        bv = list(beta)
    except TypeError:
        raise NBFitError("beta must be a sequence of k real numbers") from None
    if len(bv) != k:
        raise NBFitError(f"beta must have length k={k}, got {len(bv)}")
    for j, v in enumerate(bv):
        if not _is_real(v) or not math.isfinite(v):
            raise NBFitError(f"beta[{j}] must be a finite real number, got {v!r}")
    la = _validate_logit(log_alpha, "log_alpha")
    gp = _validate_logit(logit_pi, "logit_pi")
    return [float(v) for v in bv], la, gp


# ---------------------------------------------------------------------------
# Public pure functions
# ---------------------------------------------------------------------------

def zinb_loglik(y, X, beta, log_alpha, logit_pi, offset=None):
    """Log-likelihood of the zero-inflated NB2 mixture at a parameter point.

    Pure function; ``logit_pi`` may be any finite real (the model is
    well-defined for every finite gamma, with boundaries approached as
    gamma -> +/-inf).  Raises NBFitError on invalid input.
    """
    yv, Xv, off = _validate_xy(y, X, offset)
    k = len(Xv[0])
    bv, la, gp = _validate_point(beta, log_alpha, logit_pi, k)
    ll, _, _, _, _ = _zinb_core(yv, Xv, off, bv, la, gp)
    return ll


def zinb_objective(y, X, beta, log_alpha, logit_pi, offset=None):
    """Return ``(loglik, gradient)`` of the ZINB mixture.

    Gradient length k+2: d/d(beta), d/d log(alpha), d/d logit(pi).
    """
    yv, Xv, off = _validate_xy(y, X, offset)
    k = len(Xv[0])
    bv, la, gp = _validate_point(beta, log_alpha, logit_pi, k)
    ll, g, _, _, _ = _zinb_core(yv, Xv, off, bv, la, gp)
    return ll, g


def _zinb_pmf_parts(y_count, mu, alpha, pi):
    """Validation + raw pieces shared by the pmf/posterior helpers."""
    if type(y_count) is not int:
        raise NBFitError(
            "y_count must be an int (bool rejected), got "
            f"{type(y_count).__name__} {y_count!r}")
    if y_count < 0:
        raise NBFitError(f"y_count must be non-negative, got {y_count}")
    if not _is_real(mu) or not math.isfinite(mu) or mu <= 0.0:
        raise NBFitError(f"mu must be a positive finite number, got {mu!r}")
    if not _is_real(alpha) or not math.isfinite(alpha) or alpha <= 0.0:
        raise NBFitError(f"alpha must be a positive finite number, got {alpha!r}")
    if not _is_real(pi) or not math.isfinite(pi) or not 0.0 <= pi <= 1.0:
        raise NBFitError(
            f"pi must be a finite real in [0, 1], got {pi!r}")
    r = 1.0 / alpha
    L = _softplus(math.log(alpha) + math.log(mu))  # log1p(alpha*mu), stable
    return r, L, pi


def zinb_pmf(y_count, mu, alpha, pi):
    """Mixture probability P(Y = y_count).

    P(0) = pi + (1-pi) f_NB(0); P(k>0) = (1-pi) f_NB(k).  Boundary
    values pi = 0 (plain NB2) and pi = 1 (point mass at zero) are
    accepted.  Computed in log space.
    """
    r, L, pi = _zinb_pmf_parts(y_count, mu, alpha, pi)
    log_p0 = -r * L
    if y_count == 0:
        if pi <= 0.0:
            log_p = log_p0
        elif pi >= 1.0:
            return 1.0
        else:
            a = math.log1p(-pi) + log_p0
            b = math.log(pi)
            log_p = max(a, b) + _softplus(-abs(a - b))  # logsumexp
    else:
        if pi >= 1.0:
            return 0.0
        if y_count <= _SUM_EXACT_MAX_Y:
            lg_ratio = sum(math.log(r + m) for m in range(y_count))
        else:
            lg_ratio = math.lgamma(y_count + r) - math.lgamma(r)
        log_f = (lg_ratio - math.lgamma(y_count + 1)
                 + y_count * (math.log(alpha) + math.log(mu))
                 - (y_count + r) * L)
        log_p = log_f if pi <= 0.0 else math.log1p(-pi) + log_f
    if log_p < -745.0:
        return 0.0
    return math.exp(log_p)


def zinb_structural_probability(y_count, mu, alpha, pi):
    """Posterior probability that ``y_count`` came from the structural state.

    tau(0) = pi / [pi + (1-pi) f_NB(0)] in (0, 1) for 0 < pi < 1;
    tau(k>0) = 0.  This is the soft responsibility used by the fitter --
    observations are never assigned to a hard zero/non-zero group.
    """
    r, L, pi = _zinb_pmf_parts(y_count, mu, alpha, pi)
    if y_count > 0 or pi <= 0.0:
        return 0.0
    if pi >= 1.0:
        return 1.0
    log_p0 = -r * L
    # tau = sigmoid(logit(pi) - log_p0) = sigmoid(gamma - u)
    gamma = math.log(pi) - math.log1p(-pi)
    return _sigmoid(gamma - log_p0)


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

def _run_newton(yv, Xv, off, p0, max_iter, gtol, bounds):
    """Damped Newton/Armijo maximization of the ZINB likelihood from one
    start.  Returns ``(p, ll, g, H, resp, status, message, n_iter,
    n_rejected)``.  Coordinates on box bounds with an outward gradient are
    frozen (KKT active set); out-of-bounds/overflow candidate steps are
    rejected by backtracking, never clipped.
    """
    k = len(Xv[0])
    dim = k + 2

    def evaluate(pv):
        theta, gamma = pv[k], pv[k + 1]
        if not LOG_ALPHA_MIN <= theta <= LOG_ALPHA_MAX:
            return None
        if not LOGIT_PI_MIN <= gamma <= LOGIT_PI_MAX:
            return None
        ll, g, H, etas, resp = _zinb_core(yv, Xv, off, pv[:k], theta, gamma)
        if not math.isfinite(ll):
            return None
        for e in etas:
            if e > _ETA_LIMIT or e < -_ETA_LIMIT:
                return None
        return ll, g, H, resp

    p = list(p0)
    ev = evaluate(p)
    if ev is None:
        return None
    n_rejected = 0
    it = 0
    status = None
    message = ""
    while True:
        ll, g, H, resp = ev
        gp = list(g)
        for c, (lo, hi) in bounds.items():
            if p[c] <= lo + 1e-9 and g[c] < 0.0:
                gp[c] = 0.0
            if p[c] >= hi - 1e-9 and g[c] > 0.0:
                gp[c] = 0.0
        gnorm = max(abs(v) for v in gp)
        if gnorm <= gtol * max(1.0, abs(ll)):
            status = "converged"
            message = ("projected gradient infinity-norm below "
                       "gtol * max(1, |loglik|)")
            break
        if it >= max_iter:
            status = "max_iter"
            message = (f"iteration budget exhausted (max_iter={max_iter}); "
                       "returned point is the best found, not a certified "
                       "optimum")
            break

        active = []
        for c, (lo, hi) in bounds.items():
            if p[c] <= lo + 1e-9 and g[c] < 0.0:
                active.append(c)
            elif p[c] >= hi - 1e-9 and g[c] > 0.0:
                active.append(c)
        negH = [[-H[i][j] for j in range(dim)] for i in range(dim)]
        diag = max(1.0, max(negH[i][i] for i in range(dim)))
        lam = 1e-10 * diag
        rhs = list(g)
        d = None
        gtd = 0.0
        for _ in range(14):
            M = [[negH[i][j] + (lam if i == j else 0.0)
                  for j in range(dim)] for i in range(dim)]
            if active:
                for c in active:
                    for i in range(dim):
                        M[i][c] = 0.0
                        M[c][i] = 0.0
                    M[c][c] = 1.0
                    rhs[c] = 0.0
            d = _cholesky_solve(M, rhs)
            if d is not None:
                gtd = sum(g[i] * d[i] for i in range(dim))
                if gtd > 0.0:
                    break
            lam *= 100.0
            d = None
        if d is None:
            d = list(g)
            for c in active:
                d[c] = 0.0
            gtd = sum(v * v for v in d)
        dn = max(abs(v) for v in d)
        if dn > _STEP_CAP:
            d = [v * (_STEP_CAP / dn) for v in d]
            gtd = sum(g[i] * d[i] for i in range(dim))

        s = 1.0
        accepted = None
        for _ in range(60):
            cand = [p[i] + s * d[i] for i in range(dim)]
            evc = evaluate(cand)
            if evc is not None and evc[0] >= ll + 1e-4 * s * gtd:
                accepted = (cand, evc)
                break
            s *= 0.5
            n_rejected += 1
        if accepted is None:
            status = "line_search_failed"
            message = ("backtracking line search found no improving step "
                       "in 60 halvings; returned point is not a certified "
                       "optimum")
            break
        p, ev = accepted
        it += 1

    ll, g, H, resp = ev
    return (p, ll, g, H, resp, status, message, it, n_rejected)


def fit_zinb(y, X, offset=None, max_iter=200, gtol=1e-8):
    """Fit the constant-logit zero-inflated NB2 model by ML.

    Parameters
    ----------
    y : sequence of int, 1..100 non-negative counts (bool/float rejected).
    X : sequence of n rows, each 1..5 finite reals (no auto-intercept;
        rank deficiency rejected, never silently corrected).
    offset : None or n finite reals added to the count linear predictor.
    max_iter : int (bool rejected), in [1, 10^6]; per Newton start.
    gtol : convergence tolerance on the projected gradient infinity
        norm, relative to max(1, |loglik|).

    The mixture likelihood is not concave in general: pi = 0 (plain NB2)
    can be a local/global boundary solution while an interior MLE also
    exists, so the fitter runs Newton from several deterministic starts
    (the NB2 warm start with a small excess-zero pi, plus interior
    logit values) and keeps the point with the highest log-likelihood.
    Returns a ZINBResult for every fitting outcome; raises NBFitError
    only on invalid input or rank-deficient design (before any state is
    produced).  Inputs are never modified.
    """
    yv, Xv, off = _validate_xy(y, X, offset)
    n = len(yv)
    k = len(Xv[0])

    if type(max_iter) is not int or not 1 <= max_iter <= 1_000_000:
        raise NBFitError(
            "max_iter must be an int in [1, 1000000] (bool rejected), "
            f"got {max_iter!r}")
    if not _is_real(gtol) or not math.isfinite(gtol) or not 0.0 < gtol <= 1.0:
        raise NBFitError(f"gtol must be a finite real in (0, 1], got {gtol!r}")

    _check_full_rank(Xv, n, k)
    dim = k + 2
    n_zero = sum(1 for v in yv if v == 0)
    bounds = {k: (LOG_ALPHA_MIN, LOG_ALPHA_MAX),
              k + 1: (LOGIT_PI_MIN, LOGIT_PI_MAX)}

    # ---- all-zero sample: certified structural boundary ----------------
    if n_zero == n:
        mean_y = 0.0
        mean_off = sum(off) / n
        beta0 = [0.0] * k
        for j in range(k):
            c0 = Xv[0][j]
            if c0 != 0.0 and all(Xv[i][j] == c0 for i in range(n)):
                beta0[j] = (math.log(mean_y + 0.5) - mean_off) / c0
                break
        theta0 = LOG_ALPHA_MIN + 2.0
        gamma0 = LOGIT_PI_MAX
        ll, g, _, _, resp = _zinb_core(yv, Xv, off, beta0, theta0, gamma0)
        return ZINBResult(
            converged=False,
            status="no_finite_mle",
            message=("all counts are zero: every mixture factor is < 1 for "
                     "finite mu and pi < 1, and the log-likelihood reaches "
                     "its supremum 0 only as pi -> 1 (logit_pi -> +inf); "
                     "beta and alpha are unidentifiable at that boundary, "
                     "so no finite MLE exists"),
            beta=list(beta0),
            log_alpha=theta0,
            alpha=math.exp(theta0),
            logit_pi=gamma0,
            pi=_sigmoid(gamma0),
            loglik=ll,
            gradient=list(g),
            n_iter=0,
            n_rejected=0,
            near_poisson=False,
            pi_at_lower_boundary=False,
            pi_at_upper_boundary=True,
            structural_zero_degenerate=True,
            n_obs=n,
            n_zero=n_zero,
            n_params=dim,
            responsibilities=list(resp),
        )

    # ---- starting values: NB2 warm start for the count parameters -------
    warm = fit_nb2(yv, Xv, off)
    beta_w = list(warm.beta)
    theta_w = warm.log_alpha
    terms, _ = _nb_terms(yv, Xv, off, beta_w, theta_w)
    p0_bar = sum(math.exp(t[6]) for t in terms) / n
    zfrac = n_zero / n
    if p0_bar >= 1.0 - 1e-15:
        pi_excess = 1e-3
    else:
        pi_excess = (zfrac - p0_bar) / (1.0 - p0_bar)
    pi_excess = min(max(pi_excess, 1e-3), 0.999)

    def logit(p):
        return math.log(p) - math.log1p(-p)

    # Deterministic starts on gamma (logit pi): the NB2-side excess-zero
    # estimate plus a grid of interior values.  The surface can have a
    # boundary optimum (pi -> 0) and a separate interior optimum, so a
    # single warm start is not enough.
    gamma_starts = [logit(pi_excess), -1.5, 0.0, logit(0.8)]
    seen = set()
    starts = []
    for gm in gamma_starts:
        key = round(gm, 9)
        if key not in seen:
            seen.add(key)
            starts.append(beta_w + [theta_w, min(max(gm, LOGIT_PI_MIN + 1e-6),
                                                 LOGIT_PI_MAX - 1e-6)])

    runs = []
    for p in starts:
        out = _run_newton(yv, Xv, off, p, max_iter, gtol, bounds)
        if out is not None:
            runs.append(out)
    if not runs:
        raise NBFitError(
            "initial points are outside the computable region: need "
            f"|x_i.beta + offset_i| <= {_ETA_LIMIT:.0f} for every row, "
            f"log_alpha in [{LOG_ALPHA_MIN}, {LOG_ALPHA_MAX}] and "
            f"logit_pi in [{LOGIT_PI_MIN}, {LOGIT_PI_MAX}]; rescale X or "
            "offset (values are never clipped silently)")

    # Highest log-likelihood wins.  With the default budget every start
    # reaches its own (possibly boundary) stationary point, so this is the
    # global best found; with a tiny budget it is still honest to report
    # the best point and that start's non-converged status.
    runs.sort(key=lambda run: -run[1])
    p, ll, g, _, resp, status, message, it, n_rejected = runs[0]
    if len(runs) > 1:
        message += f" (best of {len(runs)} deterministic starts)"

    theta, gamma = p[k], p[k + 1]
    pi = _sigmoid(gamma)
    pi_lo = pi <= _PI_BOUNDARY_EPS or gamma <= LOGIT_PI_MIN + 1e-7
    pi_hi = pi >= 1.0 - _PI_BOUNDARY_EPS or gamma >= LOGIT_PI_MAX - 1e-7
    if pi_lo and status == "converged":
        message += ("; logit_pi at its lower bound: no structural-zero "
                    "component is supported, model degenerates to NB2")
    if pi_hi and status == "converged":
        message += "; logit_pi at its upper bound (all mass in zero state)"
    return ZINBResult(
        converged=(status == "converged"),
        status=status,
        message=message,
        beta=list(p[:k]),
        log_alpha=theta,
        alpha=math.exp(theta),
        logit_pi=gamma,
        pi=pi,
        loglik=ll,
        gradient=list(g),
        n_iter=it,
        n_rejected=n_rejected,
        near_poisson=(math.exp(theta) < _NEAR_POISSON_ALPHA),
        pi_at_lower_boundary=pi_lo,
        pi_at_upper_boundary=pi_hi,
        structural_zero_degenerate=False,
        n_obs=n,
        n_zero=n_zero,
        n_params=dim,
        responsibilities=list(resp),
    )
