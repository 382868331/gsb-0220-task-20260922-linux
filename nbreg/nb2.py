"""NB2 negative-binomial regression for count measurements (stdlib only).

Model
-----
    y_i ~ NB2(mu_i, alpha)
    mu_i  = exp(x_i . beta + offset_i)
    Var(y_i) = mu_i + alpha * mu_i**2,   alpha > 0

Log-likelihood of one observation, with r = 1/alpha, q = 1/(1+alpha*mu),
L = log1p(alpha*mu), eta = log(mu):

    ll_i = lgamma(y+r) - lgamma(r) - lgamma(y+1)
           + y*log(alpha) + y*eta - (y+r)*L

Parameters (beta, log alpha) are optimized jointly with a damped Newton
method on the *observed* Hessian plus an Armijo backtracking line search.
Steps that would overflow (|eta| > 700, or log alpha outside its bounds)
are rejected by backtracking; data are never clipped or altered.

Only Python's standard library is used.  digamma/trigamma are implemented
here by recurrence plus asymptotic expansion (math has neither).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MAX_SAMPLES = 100
MAX_COLUMNS = 5

# Bounds on log(alpha) used during optimization.  alpha = exp(-20) ~= 2.1e-9
# is indistinguishable from Poisson for any practical count data; the
# near-Poisson flag is raised well before that boundary.
LOG_ALPHA_MIN = -20.0
LOG_ALPHA_MAX = 25.0

_ETA_LIMIT = 700.0          # exp(eta) overflows float64 beyond ~709
_NEAR_POISSON_ALPHA = 1e-6  # fitted alpha below this is flagged near-Poisson
_SUM_EXACT_MAX_Y = 4096     # exact finite sums for gamma ratios up to this y
_STEP_CAP = 100.0           # trust-region-style cap on a single Newton step


class NBFitError(ValueError):
    """Invalid input or structural problem (e.g. rank-deficient design).

    The message always names the offending parameter and, where applicable,
    the exact row/column index, so errors are locatable.  Nothing is
    silently corrected.
    """


_BOUND_TOL = 1e-10        # a bounded coefficient at <= this is "at the bound"


@dataclass(frozen=True)
class NB2BoundedResult:
    """Outcome of :func:`fit_nb2_bounded`.

    Same status vocabulary as :class:`NB2Result`.  ``converged`` is True
    only when BOTH the free-subspace gradient and the boundary KKT
    conditions are below tolerance -- never because a parameter step
    happened to be small.

    Extra fields:

    * ``nonneg``            -- sorted tuple of the constrained column
                               indices (beta_j >= 0).
    * ``active_bounds``     -- constrained indices sitting at 0 in the
                               returned point.
    * ``free_grad_norm``    -- infinity norm of the gradient over the
                               free coordinates (unconstrained betas and
                               bounded betas not at their bound, plus
                               log alpha in joint mode).
    * ``max_kkt_violation`` -- max over bounded coefficients at 0 of
                               max(0, d ll / d beta_j); a positive value
                               means the bound wants to release.  Zero
                               when no bound is active.
    """

    converged: bool
    status: str
    message: str
    beta: list
    log_alpha: float
    alpha: float
    loglik: float
    gradient: list
    n_iter: int
    n_rejected: int
    near_poisson: bool
    fixed_alpha: bool
    n_obs: int
    n_params: int
    nonneg: tuple
    active_bounds: tuple
    free_grad_norm: float
    max_kkt_violation: float


@dataclass(frozen=True)
class NB2Result:
    """Outcome of :func:`fit_nb2`.

    ``converged`` is True only when ``status == "converged"``.  Statuses:

    * ``"converged"``           -- projected gradient below tolerance.
    * ``"max_iter"``            -- iteration budget exhausted (NOT a proof
                                   of anything; best point found is returned).
    * ``"line_search_failed"``  -- no improving step found (numerical
                                   budget exhausted; not a certified optimum).
    * ``"no_finite_mle"``       -- certified boundary case: the likelihood
                                   has no finite maximizer (e.g. all-zero
                                   counts, where the supremum is approached
                                   as mu -> 0).  This is a proved structural
                                   verdict, distinct from budget exhaustion.

    ``gradient`` is d(loglik)/d(beta, log_alpha) at the returned point
    (d/d beta only when ``fixed_alpha`` mode was used).
    ``n_rejected`` counts backtracking rejections, including
    overflow-rejected steps.
    """

    converged: bool
    status: str
    message: str
    beta: list
    log_alpha: float
    alpha: float
    loglik: float
    gradient: list
    n_iter: int
    n_rejected: int
    near_poisson: bool
    fixed_alpha: bool
    n_obs: int
    n_params: int


# ---------------------------------------------------------------------------
# Special functions: digamma / trigamma by recurrence + asymptotic expansion.
# ---------------------------------------------------------------------------

def _digamma(x):
    """psi(x) = d/dx log Gamma(x), for x > 0.

    Recurrence psi(x) = psi(x+1) - 1/x up to x >= 10, then the asymptotic
    expansion psi(x) ~ log x - 1/(2x) - sum B_{2k}/(2k x^{2k}).
    Absolute error ~1e-14 or better.
    """
    r = 0.0
    while x < 10.0:
        r -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    u = inv * inv
    r += math.log(x) - 0.5 * inv + u * (
        -1.0 / 12.0 + u * (
            1.0 / 120.0 + u * (
                -1.0 / 252.0 + u * (
                    1.0 / 240.0 + u * (
                        -1.0 / 132.0 + u * (
                            691.0 / 32760.0))))))
    return r


def _trigamma(x):
    """psi1(x) = d^2/dx^2 log Gamma(x), for x > 0.

    Recurrence psi1(x) = psi1(x+1) + 1/x^2 up to x >= 12, then the
    asymptotic expansion in powers of 1/x.  Absolute error ~1e-14.
    """
    r = 0.0
    while x < 12.0:
        r += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    u = inv * inv
    r += inv * (1.0 + inv * (0.5 + inv * (1.0 / 6.0 + u * (
        -1.0 / 30.0 + u * (
            1.0 / 42.0 + u * (
                -1.0 / 30.0 + u * (
                    5.0 / 66.0 + u * (
                        -691.0 / 2730.0))))))))
    return r


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _is_real(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_xy(y, X, offset):
    if isinstance(y, (str, bytes)):
        raise NBFitError("y must be a sequence of non-negative integers, not a string")
    try:
        yv = list(y)
    except TypeError:
        raise NBFitError("y must be a sequence of non-negative integers") from None
    if not 1 <= len(yv) <= MAX_SAMPLES:
        raise NBFitError(
            f"y must contain 1..{MAX_SAMPLES} observations, got {len(yv)}")
    for i, v in enumerate(yv):
        if type(v) is not int:
            raise NBFitError(
                f"y[{i}] must be an int count (bool and float are rejected), "
                f"got {type(v).__name__} {v!r}")
        if v < 0:
            raise NBFitError(f"y[{i}] must be non-negative, got {v}")
    n = len(yv)

    if isinstance(X, (str, bytes)):
        raise NBFitError("X must be a sequence of n rows, not a string")
    try:
        Xv = [list(row) for row in X]
    except TypeError:
        raise NBFitError("X must be a sequence of n rows of real numbers") from None
    if len(Xv) != n:
        raise NBFitError(f"X must have exactly n={n} rows, got {len(Xv)}")
    k = len(Xv[0]) if Xv else 0
    if not 1 <= k <= MAX_COLUMNS:
        raise NBFitError(
            f"X rows must have 1..{MAX_COLUMNS} columns, got {k}")
    for i, row in enumerate(Xv):
        if len(row) != k:
            raise NBFitError(
                f"X row {i} has {len(row)} columns, expected {k}")
        for j, v in enumerate(row):
            if not _is_real(v):
                raise NBFitError(
                    f"X[{i}][{j}] must be a real number (bool rejected), "
                    f"got {type(v).__name__} {v!r}")
            if not math.isfinite(v):
                raise NBFitError(f"X[{i}][{j}] must be finite, got {v!r}")

    if offset is None:
        off = [0.0] * n
    else:
        try:
            off = list(offset)
        except TypeError:
            raise NBFitError("offset must be a sequence of n real numbers") from None
        if len(off) != n:
            raise NBFitError(f"offset must have length n={n}, got {len(off)}")
        for i, v in enumerate(off):
            if not _is_real(v):
                raise NBFitError(
                    f"offset[{i}] must be a real number (bool rejected), "
                    f"got {type(v).__name__} {v!r}")
            if not math.isfinite(v):
                raise NBFitError(f"offset[{i}] must be finite, got {v!r}")
    return yv, [[float(v) for v in row] for row in Xv], [float(v) for v in off]


def _validate_positive_alpha(value, name):
    if not _is_real(value):
        raise NBFitError(
            f"{name} must be a real number (bool rejected), "
            f"got {type(value).__name__} {value!r}")
    if not math.isfinite(value) or value <= 0.0:
        raise NBFitError(f"{name} must be a positive finite number, got {value!r}")
    lo, hi = math.exp(LOG_ALPHA_MIN), math.exp(LOG_ALPHA_MAX)
    if not lo <= value <= hi:
        raise NBFitError(
            f"{name}={value!r} outside supported range [{lo:.3e}, {hi:.3e}] "
            f"(log_alpha in [{LOG_ALPHA_MIN}, {LOG_ALPHA_MAX}])")
    return float(value)


def _check_full_rank(Xv, n, k):
    """Modified Gram-Schmidt; raises if a column is (numerically) dependent."""
    qcols = []
    for j in range(k):
        v = [Xv[i][j] for i in range(n)]
        nrm0 = math.sqrt(sum(c * c for c in v))
        for q in qcols:
            d = sum(a * b for a, b in zip(q, v))
            v = [a - d * b for a, b in zip(v, q)]
        nrm = math.sqrt(sum(c * c for c in v))
        if nrm <= 1e-10 * max(1.0, nrm0):
            raise NBFitError(
                f"X is rank deficient: column {j} is (numerically) a linear "
                f"combination of columns 0..{j - 1} (e.g. duplicate or "
                f"all-zero column); coefficients are not identifiable")
        qcols.append([c / nrm for c in v])


# ---------------------------------------------------------------------------
# Log-likelihood, gradient, observed Hessian (stable formulations)
# ---------------------------------------------------------------------------

def _loglik_grad_hess(yv, Xv, off, beta, theta):
    """Return (loglik, grad[k+1], Hessian[k+1][k+1], etas).

    All quantities are computed with overflow-safe rearrangements:
      L = log1p(exp(theta+eta)) evaluated via its large-argument form,
      w = mu/(1+alpha*mu) = exp(eta - L)  in (0, 1/alpha),
      q = 1/(1+alpha*mu) = exp(-L).
    Gamma ratios and (tri)gamma differences for integer y are exact finite
    sums (y <= _SUM_EXACT_MAX_Y), avoiding catastrophic cancellation.
    """
    n = len(yv)
    k = len(beta)
    alpha = math.exp(theta)
    r = 1.0 / alpha
    ll = 0.0
    g = [0.0] * (k + 1)
    H = [[0.0] * (k + 1) for _ in range(k + 1)]
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
        if yi <= _SUM_EXACT_MAX_Y:
            lg_ratio = 0.0   # lgamma(y+r) - lgamma(r) = sum log(r+m)
            dpsi = 0.0       # psi(y+r) - psi(r)      = sum 1/(r+m)
            dtrig = 0.0      # psi1(r) - psi1(y+r)    = sum 1/(r+m)^2
            for m in range(yi):
                s = r + m
                lg_ratio += math.log(s)
                dpsi += 1.0 / s
                dtrig += 1.0 / (s * s)
        else:
            lg_ratio = math.lgamma(yi + r) - math.lgamma(r)
            dpsi = _digamma(yi + r) - _digamma(r)
            dtrig = _trigamma(r) - _trigamma(yi + r)
        ll += lg_ratio - math.lgamma(yi + 1) + yi * (theta + eta) - (yi + r) * L

        w = math.exp(eta - L)          # mu/(1+alpha*mu) <= 1/alpha
        q = math.exp(-L)
        yr = yi + r
        aw = alpha * w                 # alpha*mu/(1+alpha*mu) < 1

        gb = yi - yr * aw
        for j in range(k):
            g[j] += gb * xi[j]
        g[k] += yi + r * L - r * dpsi - yr * aw

        hw = yr * aw * q               # -d2l/deta^2 weight (>= 0)
        cf = aw * (r - yr * q)         # d2l/deta dtheta
        for j in range(k):
            xj = xi[j]
            H[k][j] += cf * xj
            for jj in range(j + 1):
                H[j][jj] -= hw * xj * xi[jj]
        H[k][k] += (-r * L + r * dpsi - r * r * dtrig
                    + 2.0 * r * aw - yr * aw * q)
    for j in range(k + 1):
        for jj in range(j):
            H[jj][j] = H[j][jj]
    return ll, g, H, etas


def nb2_loglik(y, X, beta, log_alpha, offset=None):
    """Log-likelihood of the NB2 model at (beta, log_alpha).

    Pure function; stable for any finite inputs (no overflow guard is
    needed thanks to the log-space formulation).  Raises NBFitError on
    invalid input.  Intended for independent verification of the fitter.
    """
    yv, Xv, off = _validate_xy(y, X, offset)
    k = len(Xv[0])
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
    if not _is_real(log_alpha) or not math.isfinite(log_alpha):
        raise NBFitError(f"log_alpha must be a finite real number, got {log_alpha!r}")
    ll, _, _, _ = _loglik_grad_hess(yv, Xv, off, [float(v) for v in bv],
                                    float(log_alpha))
    return ll


def nb2_objective(y, X, beta, log_alpha, offset=None):
    """Return ``(loglik, gradient)`` at (beta, log_alpha).

    The gradient has length k+1: d(loglik)/d(beta_j), then
    d(loglik)/d(log_alpha).  This pure function is exposed so that the
    fitter's derivatives can be checked at arbitrary points (it is the
    same objective used inside :func:`fit_nb2`).
    """
    yv, Xv, off = _validate_xy(y, X, offset)
    k = len(Xv[0])
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
    if not _is_real(log_alpha) or not math.isfinite(log_alpha):
        raise NBFitError(f"log_alpha must be a finite real number, got {log_alpha!r}")
    ll, g, _, _ = _loglik_grad_hess(yv, Xv, off, [float(v) for v in bv],
                                    float(log_alpha))
    return ll, g


def nb2_pmf(y_count, mu, alpha):
    """Probability mass P(Y = y_count) for NB2(mu, alpha).

    ``y_count`` must be a non-negative Python int (bool/float rejected),
    ``mu`` and ``alpha`` positive finite reals.  Computed entirely in log
    space, so it is finite for arbitrarily large counts (it underflows
    to 0.0 only when the true probability is below the float floor).
    """
    if type(y_count) is not int:
        raise NBFitError(
            "y_count must be an int (bool rejected), got "
            f"{type(y_count).__name__} {y_count!r}")
    if y_count < 0:
        raise NBFitError(f"y_count must be non-negative, got {y_count}")
    if not _is_real(mu) or not math.isfinite(mu) or mu <= 0.0:
        raise NBFitError(f"mu must be a positive finite number, got {mu!r}")
    if not _is_real(alpha) or not math.isfinite(alpha) or alpha <= 0.0:
        raise NBFitError(
            f"alpha must be a positive finite number, got {alpha!r}")
    r = 1.0 / alpha
    # lgamma(k+r) - lgamma(r) as an exact finite sum of log(r+m):
    # subtracting the two huge lgamma values directly loses all digits
    # when alpha -> 0 (r -> inf), which is precisely the Poisson limit.
    if y_count <= _SUM_EXACT_MAX_Y:
        lg_ratio = sum(math.log(r + m) for m in range(y_count))
    else:
        lg_ratio = math.lgamma(y_count + r) - math.lgamma(r)
    log_p = (lg_ratio
             - math.lgamma(y_count + 1)
             + y_count * math.log(alpha * mu)
             - (y_count + r) * math.log1p(alpha * mu))
    if log_p < -745.0:
        return 0.0
    return math.exp(log_p)


# ---------------------------------------------------------------------------
# Small dense linear algebra
# ---------------------------------------------------------------------------

def _cholesky_solve(M, b):
    """Solve M d = b for symmetric positive definite M; None if not PD."""
    n = len(b)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = M[i][j]
            for p in range(j):
                s -= L[i][p] * L[j][p]
            if i == j:
                if not s > 0.0 or not math.isfinite(s):
                    return None
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    yv = [0.0] * n
    for i in range(n):
        s = b[i]
        for p in range(i):
            s -= L[i][p] * yv[p]
        yv[i] = s / L[i][i]
    d = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = yv[i]
        for p in range(i + 1, n):
            s -= L[p][i] * d[p]
        d[i] = s / L[i][i]
    return d


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

def fit_nb2(y, X, offset=None, fixed_alpha=None, alpha_init=None,
            max_iter=200, gtol=1e-8):
    """Fit the NB2 regression by maximizing the log-likelihood.

    Parameters
    ----------
    y : sequence of int, 1..100 non-negative counts (bool/float rejected).
    X : sequence of n rows, each 1..5 finite real columns.  No intercept
        is added automatically; include a column of ones if wanted.
    offset : None or sequence of n finite reals, added to the linear
        predictor.
    fixed_alpha : None -> joint optimization of (beta, log alpha);
        a positive finite float -> beta-only optimization at that alpha
        (for independent verification / profiling).
    alpha_init : optional positive finite initial alpha (joint mode).
    max_iter : int (bool rejected), iteration budget.
    gtol : convergence tolerance on the projected gradient, relative to
        max(1, |loglik|).

    Returns an NB2Result in all fitting outcomes (never raises for
    non-convergence); raises NBFitError only for invalid input or a
    rank-deficient design.
    """
    yv, Xv, off = _validate_xy(y, X, offset)
    n = len(yv)
    k = len(Xv[0])

    if type(max_iter) is not int or not 1 <= max_iter <= 1_000_000:
        raise NBFitError(
            f"max_iter must be an int in [1, 1000000] (bool rejected), "
            f"got {max_iter!r}")
    if not _is_real(gtol) or not math.isfinite(gtol) or not 0.0 < gtol <= 1.0:
        raise NBFitError(f"gtol must be a finite real in (0, 1], got {gtol!r}")
    if fixed_alpha is not None:
        fixed_alpha = _validate_positive_alpha(fixed_alpha, "fixed_alpha")
    if alpha_init is not None:
        alpha_init = _validate_positive_alpha(alpha_init, "alpha_init")

    _check_full_rank(Xv, n, k)
    joint = fixed_alpha is None
    dim = k + 1 if joint else k
    theta_fixed = math.log(fixed_alpha) if fixed_alpha is not None else None

    # ---- starting values -------------------------------------------------
    mean_y = sum(yv) / n
    mean_off = sum(off) / n
    beta0 = [0.0] * k
    for j in range(k):  # a constant column, if any, absorbs the mean level
        c0 = Xv[0][j]
        if c0 != 0.0 and all(Xv[i][j] == c0 for i in range(n)):
            beta0[j] = (math.log(mean_y + 0.5) - mean_off) / c0
            break
    if joint:
        if alpha_init is not None:
            theta0 = math.log(alpha_init)
        else:  # method of moments, clamped inside the supported box
            var_y = sum((v - mean_y) ** 2 for v in yv) / n
            a0 = (var_y - mean_y) / max(mean_y, 0.5) ** 2
            a0 = min(max(a0, math.exp(LOG_ALPHA_MIN + 2.0)), math.exp(6.0))
            theta0 = math.log(a0)
        p = beta0 + [theta0]
    else:
        p = list(beta0)

    def evaluate(pv):
        theta = pv[k] if joint else theta_fixed
        if not LOG_ALPHA_MIN <= theta <= LOG_ALPHA_MAX:
            return None  # out-of-bounds step: rejected by backtracking
        ll, g, H, etas = _loglik_grad_hess(yv, Xv, off, pv[:k], theta)
        if not math.isfinite(ll):
            return None
        for e in etas:
            if e > _ETA_LIMIT or e < -_ETA_LIMIT:
                return None  # overflow territory: rejected by backtracking
        if not joint:
            g = g[:k]
            H = [row[:k] for row in H[:k]]
        return ll, g, H, etas

    ev = evaluate(p)
    if ev is None:
        raise NBFitError(
            "initial point is outside the computable region: need "
            f"|x_i.beta + offset_i| <= {_ETA_LIMIT:.0f} for every row and "
            f"log_alpha in [{LOG_ALPHA_MIN}, {LOG_ALPHA_MAX}]; rescale X or "
            f"offset (values are never clipped silently)")

    all_zero = all(v == 0 for v in yv)
    n_rejected = 0
    it = 0
    status = None
    message = ""

    if all_zero:
        # Structural verdict (proved, not a budget outcome): every factor
        # of the likelihood is exp(-r log(1+alpha*mu)) < 1 and tends to 1
        # only as mu -> 0 (eta -> -inf), so the supremum 0 is not attained
        # at any finite parameter in either mode.
        ll, g, _, _ = ev
        if not joint:
            g = g[:k]
        return NB2Result(
            converged=False,
            status="no_finite_mle",
            message=("all counts are zero: the log-likelihood is < 0 for "
                     "every finite mu and reaches its supremum 0 only as "
                     "mu -> 0 (eta -> -inf); no finite MLE exists, in "
                     "joint or fixed-alpha mode"),
            beta=list(p[:k]),
            log_alpha=(p[k] if joint else theta_fixed),
            alpha=(math.exp(p[k]) if joint else fixed_alpha),
            loglik=ll,
            gradient=list(g),
            n_iter=0,
            n_rejected=0,
            near_poisson=False,
            fixed_alpha=not joint,
            n_obs=n,
            n_params=dim,
        )

    while True:
        ll, g, H, etas = ev
        # --- convergence: projected gradient (KKT at the alpha bounds) ---
        gp = list(g)
        if joint:
            th = p[k]
            if th <= LOG_ALPHA_MIN + 1e-9 and g[k] < 0.0:
                gp[k] = 0.0
            if th >= LOG_ALPHA_MAX - 1e-9 and g[k] > 0.0:
                gp[k] = 0.0
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

        # --- damped Newton direction on the observed Hessian -------------
        # Active-set handling: if log alpha sits at a box bound with the
        # gradient pointing outward, freeze that coordinate (KKT) and take
        # the Newton step in beta only.  Without this the near-singular
        # Poisson-bound Hessian (H_theta theta -> 0 as alpha -> 0) would
        # couple tiny theta moves into the beta block and stall.
        active = -1
        if joint:
            if p[k] <= LOG_ALPHA_MIN + 1e-9 and g[k] < 0.0:
                active = k
            elif p[k] >= LOG_ALPHA_MAX - 1e-9 and g[k] > 0.0:
                active = k
        negH = [[-H[i][j] for j in range(dim)] for i in range(dim)]
        diag = max(1.0, max(negH[i][i] for i in range(dim)))
        lam = 1e-10 * diag
        rhs = list(g)
        d = None
        gtd = 0.0
        for _ in range(14):
            M = [[negH[i][j] + (lam if i == j else 0.0)
                  for j in range(dim)] for i in range(dim)]
            if active >= 0:
                for i in range(dim):
                    M[i][active] = 0.0
                    M[active][i] = 0.0
                M[active][active] = 1.0
                rhs[active] = 0.0
            d = _cholesky_solve(M, rhs)
            if d is not None:
                gtd = sum(g[i] * d[i] for i in range(dim))
                if gtd > 0.0:
                    break
            lam *= 100.0
            d = None
        if d is None:  # fall back to steepest ascent (active coord zero)
            d = list(g)
            if active >= 0:
                d[active] = 0.0
            gtd = sum(v * v for v in d)
        dn = max(abs(v) for v in d)
        if dn > _STEP_CAP:
            d = [v * (_STEP_CAP / dn) for v in d]
            gtd = sum(g[i] * d[i] for i in range(dim))

        # --- Armijo backtracking; overflow/bounds => rejected step -------
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

    ll, g, _, _ = ev
    if joint:
        log_alpha = p[k]
        alpha = math.exp(log_alpha)
    else:
        log_alpha = theta_fixed
        alpha = fixed_alpha
    return NB2Result(
        converged=(status == "converged"),
        status=status,
        message=message,
        beta=list(p[:k]),
        log_alpha=log_alpha,
        alpha=alpha,
        loglik=ll,
        gradient=list(g),
        n_iter=it,
        n_rejected=n_rejected,
        near_poisson=(alpha < _NEAR_POISSON_ALPHA),
        fixed_alpha=not joint,
        n_obs=n,
        n_params=dim,
    )

# ---------------------------------------------------------------------------
# Bounded-coefficient fitter (projected Newton with an active set)
# ---------------------------------------------------------------------------

def _validate_nonneg(nonneg, k):
    """Return a sorted tuple of unique column indices, or raise."""
    if nonneg is None:
        return ()
    if isinstance(nonneg, (str, bytes)):
        raise NBFitError("nonneg must be a sequence of column indices, not a string")
    try:
        idxs = list(nonneg)
    except TypeError:
        raise NBFitError("nonneg must be a sequence of column indices") from None
    seen = set()
    for j in idxs:
        if type(j) is not int:
            raise NBFitError(
                "nonneg entries must be int column indices (bool rejected), "
                f"got {type(j).__name__} {j!r}")
        if not 0 <= j < k:
            raise NBFitError(
                f"nonneg index {j} out of range: X has columns 0..{k - 1}")
        if j in seen:
            raise NBFitError(f"nonneg index {j} listed twice")
        seen.add(j)
    return tuple(sorted(seen))


def fit_nb2_bounded(y, X, offset=None, nonneg=None, fixed_alpha=None,
                    alpha_init=None, max_iter=200, gtol=1e-8):
    """Fit the NB2 regression with non-negativity bounds on some betas.

    Same model, likelihood, offset handling and alpha modes as
    :func:`fit_nb2`; the only difference is that the coefficients listed
    in `nonneg` are constrained to `beta_j >= 0`.  The constraint acts
    on the coefficients only -- y and mu are never clipped or altered.

    Parameters are identical to :func:`fit_nb2` plus:

    nonneg : None or sequence of int column indices (0-based, bool
        rejected, duplicates rejected).  These coefficients are kept
        non-negative; all other betas and log alpha remain free.

    Method: projected Newton with an active set.  A bounded coefficient
    sitting at 0 whose gradient points out of the feasible region
    (d ll/d beta_j <= 0, since we maximize) is frozen for the Newton
    solve; one whose gradient points inward (positive) is released.
    The Newton system is solved in the free subspace only; the trial
    step is projected back onto the box, and the Armijo backtracking
    accepts a step only if it is feasible AND improves the objective
    (sufficient decrease on the actual projected step).  Steps that
    would overflow (|eta| > 700, log alpha outside its box) are rejected
    by the backtracking, exactly as in the unconstrained fitter.

    Convergence requires BOTH the free-subspace gradient infinity-norm
    AND the boundary KKT violation (positive gradient at an active
    bound) to fall below `gtol * max(1, |loglik|)`; a small parameter
    move alone never declares success.

    Returns an NB2BoundedResult in all fitting outcomes; raises
    NBFitError only for invalid input or a rank-deficient design.
    """
    yv, Xv, off = _validate_xy(y, X, offset)
    n = len(yv)
    k = len(Xv[0])

    if type(max_iter) is not int or not 1 <= max_iter <= 1_000_000:
        raise NBFitError(
            f"max_iter must be an int in [1, 1000000] (bool rejected), "
            f"got {max_iter!r}")
    if not _is_real(gtol) or not math.isfinite(gtol) or not 0.0 < gtol <= 1.0:
        raise NBFitError(f"gtol must be a finite real in (0, 1], got {gtol!r}")
    if fixed_alpha is not None:
        fixed_alpha = _validate_positive_alpha(fixed_alpha, "fixed_alpha")
    if alpha_init is not None:
        alpha_init = _validate_positive_alpha(alpha_init, "alpha_init")
    bounded = _validate_nonneg(nonneg, k)
    bset = set(bounded)

    _check_full_rank(Xv, n, k)
    joint = fixed_alpha is None
    dim = k + 1 if joint else k
    theta_fixed = math.log(fixed_alpha) if fixed_alpha is not None else None

    # ---- starting values (same heuristic as fit_nb2, then projected) ---
    mean_y = sum(yv) / n
    mean_off = sum(off) / n
    beta0 = [0.0] * k
    for j in range(k):
        c0 = Xv[0][j]
        if c0 != 0.0 and all(Xv[i][j] == c0 for i in range(n)):
            beta0[j] = (math.log(mean_y + 0.5) - mean_off) / c0
            break
    for j in bset:
        beta0[j] = max(0.0, beta0[j])
    if joint:
        if alpha_init is not None:
            theta0 = math.log(alpha_init)
        else:
            var_y = sum((v - mean_y) ** 2 for v in yv) / n
            a0 = (var_y - mean_y) / max(mean_y, 0.5) ** 2
            a0 = min(max(a0, math.exp(LOG_ALPHA_MIN + 2.0)), math.exp(6.0))
            theta0 = math.log(a0)
        p = beta0 + [theta0]
    else:
        p = list(beta0)

    def evaluate(pv):
        theta = pv[k] if joint else theta_fixed
        if not LOG_ALPHA_MIN <= theta <= LOG_ALPHA_MAX:
            return None  # out-of-bounds step: rejected by backtracking
        ll, g, H, etas = _loglik_grad_hess(yv, Xv, off, pv[:k], theta)
        if not math.isfinite(ll):
            return None
        for e in etas:
            if e > _ETA_LIMIT or e < -_ETA_LIMIT:
                return None  # overflow territory: rejected by backtracking
        if not joint:
            g = g[:k]
            H = [row[:k] for row in H[:k]]
        return ll, g, H, etas

    ev = evaluate(p)
    if ev is None:
        raise NBFitError(
            "initial point is outside the computable region: need "
            f"|x_i.beta + offset_i| <= {_ETA_LIMIT:.0f} for every row and "
            f"log_alpha in [{LOG_ALPHA_MIN}, {LOG_ALPHA_MAX}]; rescale X or "
            f"offset (values are never clipped silently)")

    def kkt_report(pv, gv):
        """(free_grad_norm, max_kkt_violation, active_bounds) at a point.

        log alpha at a box bound with the gradient pointing outward is
        KKT-satisfied (projected away), exactly as in fit_nb2.
        """
        free_norm = 0.0
        viol = 0.0
        active = []
        for j in range(dim):
            if j in bset and pv[j] <= _BOUND_TOL:
                active.append(j)
                if gv[j] > viol:
                    viol = gv[j]
            elif (joint and j == k and (
                    (pv[k] <= LOG_ALPHA_MIN + 1e-9 and gv[k] < 0.0) or
                    (pv[k] >= LOG_ALPHA_MAX - 1e-9 and gv[k] > 0.0))):
                pass  # KKT at the log-alpha box bound
            else:
                if abs(gv[j]) > free_norm:
                    free_norm = abs(gv[j])
        return free_norm, viol, active

    def make_result(status, message, it, n_rejected, pv, evv):
        ll, g, _, _ = evv
        free_norm, viol, active = kkt_report(pv, g)
        if joint:
            log_alpha = pv[k]
            alpha = math.exp(log_alpha)
        else:
            log_alpha = theta_fixed
            alpha = fixed_alpha
        return NB2BoundedResult(
            converged=(status == "converged"),
            status=status,
            message=message,
            beta=list(pv[:k]),
            log_alpha=log_alpha,
            alpha=alpha,
            loglik=ll,
            gradient=list(g),
            n_iter=it,
            n_rejected=n_rejected,
            near_poisson=(alpha < _NEAR_POISSON_ALPHA),
            fixed_alpha=not joint,
            n_obs=n,
            n_params=dim,
            nonneg=tuple(bounded),
            active_bounds=tuple(active),
            free_grad_norm=free_norm,
            max_kkt_violation=viol,
        )

    if all(v == 0 for v in yv):
        # Same proved structural verdict as fit_nb2, but only when the
        # supremum direction eta -> -inf is feasible under the bounds.
        # Sufficient check: some column can drive every eta to -inf --
        # a free column of one sign, or a bounded column of all-negative
        # entries (beta_j -> +inf).  Otherwise the feasible box keeps
        # eta finite and the boundary optimum is found by the iteration.
        escapable = False
        for j in range(k):
            col = [Xv[i][j] for i in range(n)]
            if j in bset:
                if all(c < 0.0 for c in col):
                    escapable = True
            else:
                if all(c > 0.0 for c in col) or all(c < 0.0 for c in col):
                    escapable = True
        if escapable:
            return make_result(
                "no_finite_mle",
                "all counts are zero and the bounds do not block eta -> "
                "-inf: the log-likelihood is < 0 for every finite mu and "
                "reaches its supremum 0 only as mu -> 0; no finite MLE "
                "exists, in joint or fixed-alpha mode",
                0, 0, p, ev)

    n_rejected = 0
    it = 0
    status = None
    message = ""

    while True:
        ll, g, H, etas = ev
        scale = max(1.0, abs(ll))
        free_norm, viol, active_list = kkt_report(p, g)
        if free_norm <= gtol * scale and viol <= gtol * scale:
            status = "converged"
            message = ("free-subspace gradient and boundary KKT violation "
                       "both below gtol * max(1, |loglik|)")
            break
        if it >= max_iter:
            status = "max_iter"
            message = (f"iteration budget exhausted (max_iter={max_iter}); "
                       "returned point is the best found, not a certified "
                       "optimum")
            break

        # --- active set: freeze bounded coords at 0 whose gradient does
        # not point into the feasible interior; release the rest. -------
        active = set(j for j in active_list if g[j] <= 0.0)
        if joint:  # freeze log alpha at a box bound with outward gradient
            if p[k] <= LOG_ALPHA_MIN + 1e-9 and g[k] < 0.0:
                active.add(k)
            elif p[k] >= LOG_ALPHA_MAX - 1e-9 and g[k] > 0.0:
                active.add(k)
        free = [j for j in range(dim) if j not in active]
        nf = len(free)

        # --- damped Newton in the free subspace -------------------------
        negH = [[-H[free[a]][free[b]] for b in range(nf)] for a in range(nf)]
        gf = [g[j] for j in free]
        diag = max(1.0, max(negH[a][a] for a in range(nf)))
        lam = 1e-10 * diag
        d = None
        for _ in range(14):
            M = [[negH[a][b] + (lam if a == b else 0.0)
                  for b in range(nf)] for a in range(nf)]
            d = _cholesky_solve(M, gf)
            if d is not None and sum(gf[a] * d[a] for a in range(nf)) > 0.0:
                break
            lam *= 100.0
            d = None
        if d is None:  # fall back to steepest ascent in the free subspace
            d = list(gf)
        dn = max(abs(v) for v in d)
        if dn > _STEP_CAP:
            d = [v * (_STEP_CAP / dn) for v in d]

        # --- projected Armijo backtracking: feasible AND improving -----
        s = 1.0
        accepted = None
        for _ in range(60):
            cand = list(p)
            for a in range(nf):
                cand[free[a]] = p[free[a]] + s * d[a]
            for j in bset:
                if cand[j] < 0.0:
                    cand[j] = 0.0
            if joint:  # project the log-alpha box as well
                if cand[k] < LOG_ALPHA_MIN:
                    cand[k] = LOG_ALPHA_MIN
                elif cand[k] > LOG_ALPHA_MAX:
                    cand[k] = LOG_ALPHA_MAX
            step = [cand[i] - p[i] for i in range(dim)]
            gtd = sum(g[i] * step[i] for i in range(dim))
            evc = evaluate(cand)
            if evc is not None and gtd > 0.0 and evc[0] >= ll + 1e-4 * gtd:
                accepted = (cand, evc)
                break
            s *= 0.5
            n_rejected += 1
        if accepted is None:
            status = "line_search_failed"
            message = ("projected backtracking line search found no "
                       "feasible improving step in 60 halvings; returned "
                       "point is not a certified optimum")
            break
        p, ev = accepted
        it += 1

    return make_result(status, message, it, n_rejected, p, ev)
