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
_TAIL_TOL_DEFAULT = 1e-12   # relative residual-mass tolerance for tail sums
_TAIL_BUDGET_DEFAULT = 200_000  # max terms in one interval/tail summation


class NBFitError(ValueError):
    """Invalid input or structural problem (e.g. rank-deficient design).

    The message always names the offending parameter and, where applicable,
    the exact row/column index, so errors are locatable.  Nothing is
    silently corrected.
    """


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


# chunk A: observation parsing + scalar logpmf
class _TailBudgetError(Exception):
    """Internal: a tail/interval summation exhausted its term budget.

    Carries a human-readable reason; converted into a rejected objective
    evaluation inside the fitter (never silently treated as an exact
    likelihood), or into NBFitError at the public boundary.
    """


def _validate_tail_controls(tail_tol, tail_budget):
    if not _is_real(tail_tol) or not math.isfinite(tail_tol) \
            or not 0.0 < tail_tol <= 1.0:
        raise NBFitError(
            f"tail_tol must be a finite real in (0, 1], got {tail_tol!r}")
    if type(tail_budget) is not int or not 1 <= tail_budget <= 100_000_000:
        raise NBFitError(
            "tail_budget must be an int in [1, 100000000] (bool rejected), "
            f"got {tail_budget!r}")
    return float(tail_tol), tail_budget


def _validate_obs(obs):
    """Parse the observation list into (lo, hi) pairs; hi=None means tail.

    Each observation is either a non-negative int (exact count) or a
    2-element tuple/list (lo, hi) of non-negative ints with lo <= hi
    (closed integer interval), or (lo, None) for "at least lo".
    """
    if isinstance(obs, (str, bytes)):
        raise NBFitError("obs must be a sequence of observations, not a string")
    try:
        items = list(obs)
    except TypeError:
        raise NBFitError("obs must be a sequence of observations") from None
    if not 1 <= len(items) <= MAX_SAMPLES:
        raise NBFitError(
            f"obs must contain 1..{MAX_SAMPLES} observations, got {len(items)}")
    parsed = []
    for i, ob in enumerate(items):
        if type(ob) is int:
            if ob < 0:
                raise NBFitError(f"obs[{i}] must be non-negative, got {ob}")
            parsed.append((ob, ob))
            continue
        if isinstance(ob, bool) or not isinstance(ob, (tuple, list)):
            raise NBFitError(
                f"obs[{i}] must be an int count or a (lo, hi) pair, got "
                f"{type(ob).__name__} {ob!r}")
        if len(ob) != 2:
            raise NBFitError(
                f"obs[{i}] must have exactly 2 elements (lo, hi), "
                f"got {len(ob)}")
        lo, hi = ob
        if type(lo) is not int or lo < 0:
            raise NBFitError(
                f"obs[{i}][0] (lo) must be a non-negative int (bool "
                f"rejected), got {lo!r}")
        if hi is not None:
            if type(hi) is not int or hi < 0:
                raise NBFitError(
                    f"obs[{i}][1] (hi) must be a non-negative int or None "
                    f"(bool rejected), got {hi!r}")
            if hi < lo:
                raise NBFitError(
                    f"obs[{i}] has hi < lo ({hi} < {lo}); intervals are "
                    "closed [lo, hi]")
        parsed.append((lo, hi))
    return parsed


def _log_pmf_scalar(k, eta, theta):
    """log P(Y=k) at log-mu eta and log-alpha theta (stable, no overflow)."""
    alpha = math.exp(theta)
    r = 1.0 / alpha
    t = theta + eta
    if t > 35.0:
        L = t + math.log1p(math.exp(-t))
    else:
        L = math.log1p(math.exp(t))
    if k <= _SUM_EXACT_MAX_Y:
        lg_ratio = 0.0
        for m in range(k):
            lg_ratio += math.log(r + m)
    else:
        lg_ratio = math.lgamma(k + r) - math.lgamma(r)
    return lg_ratio - math.lgamma(k + 1) + k * t - (k + r) * L


def _interval_sums(lo, hi, eta, theta, tail_tol, tail_budget):
    """Sums over P(lo <= Y <= hi) at (eta, theta) = (log mu, log alpha).

    Returns (logP, dlogP/deta, d2logP/deta2).  The probability is a direct
    sum of pmf terms (never a CDF difference, never a midpoint surrogate).
    For hi=None (right tail) or hi far out, the sum stops early only when
    the *explicit* geometric residual bounds R0/R1/R2 (for the mass, the
    first derivative numerator and the second derivative numerator) are
    below tail_tol relative to the accumulated sums; the residual bound
    uses rho = w * max(1, (K+1+r)/(K+2)) with w = alpha*mu/(1+alpha*mu),
    which dominates every later pmf ratio.  If tail_budget terms do not
    suffice, _TailBudgetError is raised -- the truncated sum is NOT a
    valid likelihood and must not be used as one.
    """
    if lo == 0 and hi is None:
        # Whole support: P = 1 exactly, derivatives exactly zero.
        return 0.0, 0.0, 0.0
    alpha = math.exp(theta)
    r = 1.0 / alpha
    t = theta + eta
    if t > 35.0:
        L = t + math.log1p(math.exp(-t))
    else:
        L = math.log1p(math.exp(t))
    w = math.exp(eta - L)          # mu/(1+alpha*mu)
    q = math.exp(-L)               # 1/(1+alpha*mu)
    aw = alpha * w                 # alpha*mu/(1+alpha*mu) in (0, 1)

    s0 = 0.0   # sum p_k
    s1 = 0.0   # sum p_k * g_k,         g_k = d log p_k / d eta
    s2 = 0.0   # sum p_k * (g_k^2 + h_k), h_k = d2 log p_k / d eta^2
    terms = 0
    k = lo
    pk = 0.0
    while True:
        if hi is not None and k > hi:
            break
        logp = _log_pmf_scalar(k, eta, theta)
        pk = math.exp(logp) if logp >= -745.0 else 0.0
        gb = k - (k + r) * aw
        hw = (k + r) * aw * q
        s0 += pk
        s1 += pk * gb
        s2 += pk * (gb * gb - hw)
        terms += 1
        if terms > tail_budget:
            raise _TailBudgetError(
                f"tail summation for P(Y >= {lo}) at mu=exp({eta:.6g}), "
                f"alpha={alpha:.6g} did not converge within the term "
                f"budget ({tail_budget} terms); the truncated tail sum "
                "is not an exact likelihood, so this objective evaluation "
                "was aborted")
        # Early stop on an explicit residual bound (tail or far hi).
        if k >= lo + 2 and (hi is None or k < hi):
            rho = aw * max(1.0, (k + 1.0 + r) / (k + 2.0))
            if rho < 1.0:
                orho = 1.0 - rho
                m = k + 1.0 + r
                r0 = pk * rho / orho
                r1 = 2.0 * pk * (m * rho / orho
                                 + rho * rho / (orho * orho))
                r2 = 5.0 * pk * (m * m * rho / orho
                                 + 2.0 * m * rho * rho / (orho * orho)
                                 + 2.0 * rho ** 3 / (orho ** 3))
                if (r0 <= tail_tol * max(s0, 1e-300)
                        and r1 <= tail_tol * (abs(s1) + m * s0 + 1e-300)
                        and r2 <= tail_tol * (abs(s2) + m * m * s0 + 1e-300)):
                    break
        k += 1
    if s0 <= 0.0:
        # Every pmf term underflowed the float floor: the interval
        # probability is below ~1e-324, so the log-likelihood is
        # effectively -inf at this point.  Treated as an invalid
        # evaluation (rejected step), never as a finite number.
        raise _TailBudgetError(
            f"interval [{lo}, {hi if hi is not None else 'inf'}] has "
            f"probability below the float floor at mu=exp({eta:.6g}), "
            f"alpha={alpha:.6g}; log-likelihood is -inf here")
    d1 = s1 / s0
    d2 = s2 / s0 - d1 * d1
    return math.log(s0), d1, d2


def nb2_interval_logprob(lo, hi, mu, alpha,
                         tail_tol=_TAIL_TOL_DEFAULT,
                         tail_budget=_TAIL_BUDGET_DEFAULT):
    """log P(lo <= Y <= hi) for NB2(mu, alpha); hi=None means P(Y >= lo).

    Direct log-space pmf summation with an explicit residual-mass bound;
    raises NBFitError if the term budget is exhausted (the truncated sum
    is never returned as if it were exact).
    """
    if type(lo) is not int or lo < 0:
        raise NBFitError(
            f"lo must be a non-negative int (bool rejected), got {lo!r}")
    if hi is not None:
        if type(hi) is not int or hi < 0:
            raise NBFitError(
                f"hi must be a non-negative int or None (bool rejected), "
                f"got {hi!r}")
        if hi < lo:
            raise NBFitError(f"hi={hi} < lo={lo}; intervals are [lo, hi]")
    if not _is_real(mu) or not math.isfinite(mu) or mu <= 0.0:
        raise NBFitError(f"mu must be a positive finite number, got {mu!r}")
    if not _is_real(alpha) or not math.isfinite(alpha) or alpha <= 0.0:
        raise NBFitError(
            f"alpha must be a positive finite number, got {alpha!r}")
    tail_tol, tail_budget = _validate_tail_controls(tail_tol, tail_budget)
    try:
        logp, _, _ = _interval_sums(lo, hi, math.log(mu), math.log(alpha),
                                    tail_tol, tail_budget)
    except _TailBudgetError as exc:
        raise NBFitError(str(exc)) from None
    return logp


def _validate_beta(bv, k):
    if isinstance(bv, (str, bytes)):
        raise NBFitError("beta must be a sequence of k real numbers")
    try:
        bv = list(bv)
    except TypeError:
        raise NBFitError("beta must be a sequence of k real numbers") from None
    if len(bv) != k:
        raise NBFitError(f"beta must have length k={k}, got {len(bv)}")
    for j, v in enumerate(bv):
        if not _is_real(v) or not math.isfinite(v):
            raise NBFitError(f"beta[{j}] must be a finite real number, got {v!r}")
    return [float(v) for v in bv]


def _censored_loglik_grad_hess(parsed, Xv, off, beta, theta,
                               tail_tol, tail_budget):
    """(loglik, grad[k], Hessian[k][k]) for interval-censored observations.

    For observation i with eta_i = x_i.beta + offset_i the per-observation
    scalar derivatives d1 = d logP/d eta and d2 = d2 logP/d eta2 from
    _interval_sums give grad += d1 * x_i and H += d2 * x_i x_i^T.
    """
    n = len(parsed)
    k = len(beta)
    ll = 0.0
    g = [0.0] * k
    H = [[0.0] * k for _ in range(k)]
    etas = [0.0] * n
    for i in range(n):
        xi = Xv[i]
        eta = off[i]
        for j in range(k):
            eta += xi[j] * beta[j]
        etas[i] = eta
        lo, hi = parsed[i]
        logp, d1, d2 = _interval_sums(lo, hi, eta, theta,
                                      tail_tol, tail_budget)
        ll += logp
        for j in range(k):
            g[j] += d1 * xi[j]
            xj = xi[j]
            for jj in range(j + 1):
                H[j][jj] += d2 * xj * xi[jj]
    for j in range(k):
        for jj in range(j):
            H[jj][j] = H[j][jj]
    return ll, g, H, etas


def _validate_censored_inputs(obs, X, offset, alpha, tail_tol, tail_budget):
    parsed = _validate_obs(obs)
    n = len(parsed)
    # Reuse the X/offset validation by passing a dummy exact-count vector.
    _, Xv, off = _validate_xy([0] * n, X, offset)
    alpha = _validate_positive_alpha(alpha, "alpha")
    tail_tol, tail_budget = _validate_tail_controls(tail_tol, tail_budget)
    return parsed, Xv, off, alpha, tail_tol, tail_budget


def nb2_censored_loglik(obs, X, beta, alpha, offset=None,
                        tail_tol=_TAIL_TOL_DEFAULT,
                        tail_budget=_TAIL_BUDGET_DEFAULT):
    """Log-likelihood of interval-censored observations at (beta, alpha).

    obs entries: int k (exact), (lo, hi) closed integer interval, or
    (lo, None) for "at least lo".  alpha is fixed (positive finite).
    Pure function for independent verification.  Raises NBFitError on
    invalid input or tail-budget exhaustion.
    """
    parsed, Xv, off, alpha, tail_tol, tail_budget = \
        _validate_censored_inputs(obs, X, offset, alpha, tail_tol, tail_budget)
    bv = _validate_beta(beta, len(Xv[0]))
    try:
        ll, _, _, _ = _censored_loglik_grad_hess(
            parsed, Xv, off, bv, math.log(alpha), tail_tol, tail_budget)
    except _TailBudgetError as exc:
        raise NBFitError(str(exc)) from None
    return ll


def nb2_censored_objective(obs, X, beta, alpha, offset=None,
                           tail_tol=_TAIL_TOL_DEFAULT,
                           tail_budget=_TAIL_BUDGET_DEFAULT):
    """Return (loglik, gradient); gradient is w.r.t. beta only."""
    parsed, Xv, off, alpha, tail_tol, tail_budget = \
        _validate_censored_inputs(obs, X, offset, alpha, tail_tol, tail_budget)
    bv = _validate_beta(beta, len(Xv[0]))
    try:
        ll, g, _, _ = _censored_loglik_grad_hess(
            parsed, Xv, off, bv, math.log(alpha), tail_tol, tail_budget)
    except _TailBudgetError as exc:
        raise NBFitError(str(exc)) from None
    return ll, g


def fit_nb2_censored(obs, X, alpha, offset=None, max_iter=200, gtol=1e-8,
                     tail_tol=_TAIL_TOL_DEFAULT,
                     tail_budget=_TAIL_BUDGET_DEFAULT):
    """Fit NB2 regression coefficients to interval-censored observations.

    alpha is FIXED (positive finite); only beta is optimized.  Each
    observation contributes its exact likelihood:

    * int k        -> P(Y = k)                (pmf)
    * (lo, hi)     -> P(lo <= Y <= hi)        (direct pmf sum, log space)
    * (lo, None)   -> P(Y >= lo)              (right-tail pmf sum with an
                                               explicit residual-mass bound
                                               and a term budget)

    Interval midpoints are never used as observations.  When every
    observation is an exact count the call degenerates to the original
    fixed-alpha path (fit_nb2 with fixed_alpha=alpha).

    Returns an NB2Result (fixed_alpha=True, gradient w.r.t. beta only).
    Statuses mirror fit_nb2; additionally, if every observation is a
    right tail ("at least lo"), the likelihood is strictly increasing in
    every mu_i and no finite MLE exists -> status "no_finite_mle".
    A tail summation that exhausts tail_budget aborts that objective
    evaluation (rejected step, counted in n_rejected); if it happens at
    the initial point an NBFitError is raised with the reason.
    """
    parsed, Xv, off, alpha, tail_tol, tail_budget = \
        _validate_censored_inputs(obs, X, offset, alpha, tail_tol, tail_budget)
    n = len(parsed)
    k = len(Xv[0])
    if type(max_iter) is not int or not 1 <= max_iter <= 1_000_000:
        raise NBFitError(
            f"max_iter must be an int in [1, 1000000] (bool rejected), "
            f"got {max_iter!r}")
    if not _is_real(gtol) or not math.isfinite(gtol) or not 0.0 < gtol <= 1.0:
        raise NBFitError(f"gtol must be a finite real in (0, 1], got {gtol!r}")
    _check_full_rank(Xv, n, k)
    theta = math.log(alpha)

    # Degenerate case: all observations exact -> original fixed-alpha path.
    if all(lo == hi for lo, hi in parsed):
        return fit_nb2([lo for lo, _ in parsed], X, offset=off,
                       fixed_alpha=alpha, max_iter=max_iter, gtol=gtol)

    # Structural non-identifiability: only lower bounds.  Every factor
    # P(Y >= lo_i) is strictly increasing in mu_i and tends to 1 as
    # mu_i -> inf, so the supremum 0 of the log-likelihood is not attained
    # at any finite beta.
    if all(hi is None for _, hi in parsed):
        ll0, g0, _, _ = _censored_loglik_grad_hess(
            parsed, Xv, off, [0.0] * k, theta, tail_tol, tail_budget)
        return NB2Result(
            converged=False,
            status="no_finite_mle",
            message=("every observation is a lower bound (at-least-L): "
                     "each factor P(Y >= L) is strictly increasing in mu "
                     "and the supremum 0 of the log-likelihood is reached "
                     "only as mu -> inf (eta -> +inf); beta is not "
                     "identifiable, no finite MLE exists"),
            beta=[0.0] * k,
            log_alpha=theta,
            alpha=alpha,
            loglik=ll0,
            gradient=list(g0),
            n_iter=0,
            n_rejected=0,
            near_poisson=(alpha < _NEAR_POISSON_ALPHA),
            fixed_alpha=True,
            n_obs=n,
            n_params=k,
        )

    # Starting values: a constant column absorbs a proxy mean level.  The
    # proxy (midpoint/lower bound) is used ONLY for this initial guess,
    # never in the likelihood.
    proxy = []
    for lo, hi in parsed:
        if hi is None:
            proxy.append(float(lo))
        else:
            proxy.append(0.5 * (lo + hi))
    mean_p = sum(proxy) / n + 0.5
    mean_off = sum(off) / n
    p = [0.0] * k
    for j in range(k):
        c0 = Xv[0][j]
        if c0 != 0.0 and all(Xv[i][j] == c0 for i in range(n)):
            p[j] = (math.log(mean_p) - mean_off) / c0
            break

    abort_reason = [None]

    def evaluate(pv):
        abort_reason[0] = None
        try:
            ll, g, H, etas = _censored_loglik_grad_hess(
                parsed, Xv, off, pv, theta, tail_tol, tail_budget)
        except _TailBudgetError as exc:
            abort_reason[0] = str(exc)
            return None
        if not math.isfinite(ll):
            return None
        for e in etas:
            if e > _ETA_LIMIT or e < -_ETA_LIMIT:
                return None
        return ll, g, H, etas

    ev = evaluate(p)
    if ev is None:
        reason = abort_reason[0] or (
            f"initial |x_i.beta + offset_i| exceeds {_ETA_LIMIT:.0f}")
        raise NBFitError(
            "initial point is outside the computable region for the "
            f"censored likelihood: {reason}")

    n_rejected = 0
    it = 0
    status = None
    message = ""
    while True:
        ll, g, H, etas = ev
        gnorm = max(abs(v) for v in g)
        if gnorm <= gtol * max(1.0, abs(ll)):
            status = "converged"
            message = ("gradient infinity-norm below "
                       "gtol * max(1, |loglik|)")
            break
        if it >= max_iter:
            status = "max_iter"
            message = (f"iteration budget exhausted (max_iter={max_iter}); "
                       "returned point is the best found, not a certified "
                       "optimum")
            break

        negH = [[-H[i][j] for j in range(k)] for i in range(k)]
        diag = max(1.0, max(negH[i][i] for i in range(k)))
        lam = 1e-10 * diag
        d = None
        gtd = 0.0
        for _ in range(14):
            M = [[negH[i][j] + (lam if i == j else 0.0)
                  for j in range(k)] for i in range(k)]
            d = _cholesky_solve(M, g)
            if d is not None:
                gtd = sum(g[i] * d[i] for i in range(k))
                if gtd > 0.0:
                    break
            lam *= 100.0
            d = None
        if d is None:
            d = list(g)
            gtd = sum(v * v for v in d)
        dn = max(abs(v) for v in d)
        if dn > _STEP_CAP:
            d = [v * (_STEP_CAP / dn) for v in d]
            gtd = sum(g[i] * d[i] for i in range(k))

        s = 1.0
        accepted = None
        for _ in range(60):
            cand = [p[i] + s * d[i] for i in range(k)]
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
            if abort_reason[0] is not None:
                message += "; last aborted evaluation: " + abort_reason[0]
            break
        p, ev = accepted
        it += 1

    ll, g, _, _ = ev
    return NB2Result(
        converged=(status == "converged"),
        status=status,
        message=message,
        beta=list(p),
        log_alpha=theta,
        alpha=alpha,
        loglik=ll,
        gradient=list(g),
        n_iter=it,
        n_rejected=n_rejected,
        near_poisson=(alpha < _NEAR_POISSON_ALPHA),
        fixed_alpha=True,
        n_obs=n,
        n_params=k,
    )


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
