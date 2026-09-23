"""NB2 negative-binomial regression core.

Model
-----
For observation ``i``::

    mu_i = exp(X_i @ beta + offset_i)
    y_i | mu_i, alpha ~ NB2,  Var = mu + alpha * mu**2,  alpha > 0

theta = 1 / alpha is the target-number ("size") parameter.  The PMF is

    f(y) = Gamma(y+theta)/(Gamma(theta) Gamma(y+1))
           * theta**theta * mu**y / (mu + theta)**(y+theta)

Joint maximum-likelihood estimation over ``(beta, log theta)`` uses
Newton steps on the observed Hessian with a Levenberg fallback and an
Armijo backtracking line search.  Candidate steps that overflow
(non-finite objective or linear predictor beyond the representable
range) are *rejected and shortened*; data values are never clipped.

Error semantics are part of the public contract (see README); in
particular a provably non-existent finite estimator (all-zero counts
with an intercept) raises `NoFinibleMLEError`, which is distinct from
`ConvergenceBudgetError` (search budget exhausted).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .linalg import design_rank, solve_pd
from .special import (
    digamma_diff,
    gamma_ratio_log,
    trigamma_diff,
)

__all__ = [
    "NB2Result",
    "NB2InputError",
    "RankDeficientDesignError",
    "NoFiniteMLEError",
    "ConvergenceBudgetError",
    "StepFailureError",
    "fit_nb2",
    "nb2_loglik",
    "nb2_gradient",
    "log_pmf",
    "pmf",
    "range_probability",
    "conditional_pmf",
    "LIMITS",
]

LIMITS = {
    "max_n": 100,
    "max_p": 5,
    "eta_max": 700.0,
    "theta_max": 1.0e8,   # alpha_min = 1e-8
    "theta_min": 1.0e-8,  # alpha_max = 1e8
    "near_poisson_theta": 1.0e6,  # alpha <= 1e-6 flagged near-Poisson
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NB2InputError(ValueError):
    """An input violates the documented contract."""


class RankDeficientDesignError(NB2InputError):
    """The design matrix is not full column rank.

    ``dependent_columns`` are reproducible locators (columns that lie in
    the span of earlier pivoted columns), not recommendations.
    """

    def __init__(self, dependent_columns: list[int], rank: int, n_cols: int):
        self.dependent_columns = list(dependent_columns)
        self.rank = rank
        self.n_cols = n_cols
        super().__init__(
            "design matrix is rank deficient: rank "
            f"{rank} < {n_cols} columns; dependent column index(es) "
            f"{self.dependent_columns}"
        )


class NoFiniteMLEError(Exception):
    """A finite MLE provably does not exist for this data/design."""


class ConvergenceBudgetError(Exception):
    """The iteration budget ran out; no impossibility is claimed."""


class StepFailureError(Exception):
    """No acceptable finite step could be found (overflow rejection)."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class NB2Result:
    beta: list[float]
    alpha: float
    theta: float
    loglik: float
    gradient: list[float]
    """Gradient at the terminal point.

    Joint mode: ``[d/db_0, ..., d/db_{p-1}, d/d log theta]``.
    Fixed-alpha mode: beta gradient only.
    """
    converged: bool
    iterations: int
    status: str
    """One of ``converged``, ``converged_boundary`` (gradient tolerance
    met while sitting on the near-Poisson theta box), ``max_iter``
    (budget exhausted), ``step_failure`` (overflow/no finite step),
    ``boundary_theta_min`` (ran into the large-alpha box)."""
    message: str
    near_poisson: bool
    alpha_at_boundary: bool
    fixed_alpha: bool
    mu: list[float] = field(default_factory=list)
    hessian: list[list[float]] | None = None
    vcov: list[list[float]] | None = None
    rejected_steps: int = 0

    def raise_if_not_converged(self) -> None:
        """Convenience: raise on budget/step failure, return on success."""
        if self.status == "max_iter":
            raise ConvergenceBudgetError(self.message)
        if self.status == "step_failure":
            raise StepFailureError(self.message)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _finite_float(v, what: str, index=None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        where = "" if index is None else f" at {index}"
        raise NB2InputError(f"{what}{where} must be a real number, got {v!r}")
    f = float(v)
    if not math.isfinite(f):
        where = "" if index is None else f" at {index}"
        raise NB2InputError(f"{what}{where} must be finite (no NaN/Infinity)")
    return f


def _validate(y, X, offset, alpha) -> tuple[list[float], list[list[float]],
                                            list[float], int, int]:
    if not isinstance(y, (list, tuple)):
        raise NB2InputError("y must be a list or tuple of non-negative integers")
    n = len(y)
    if n < 1:
        raise NB2InputError("y must contain at least one observation")
    if n > LIMITS["max_n"]:
        raise NB2InputError(
            f"n={n} exceeds the supported maximum of {LIMITS['max_n']}"
        )

    yf: list[float] = []
    for i, v in enumerate(y):
        if isinstance(v, bool):
            raise NB2InputError(
                f"y[{i}] must be a non-negative integer, got bool {v!r}"
            )
        if not isinstance(v, (int, float)):
            raise NB2InputError(f"y[{i}] must be a non-negative integer, got {v!r}")
        fv = float(v)
        if not math.isfinite(fv):
            raise NB2InputError(f"y[{i}] must be finite (no NaN/Infinity)")
        if fv < 0.0:
            raise NB2InputError(f"y[{i}]={fv} is negative; counts are >= 0")
        if not (isinstance(v, int) or fv.is_integer()):
            raise NB2InputError(
                f"y[{i}]={fv!r} is not an integer; fractional counts are rejected"
            )
        yf.append(fv)

    if X is None:
        Xm = [[1.0] for _ in range(n)]
    else:
        if not isinstance(X, (list, tuple)):
            raise NB2InputError("X must be a list of rows (or None)")
        if len(X) != n:
            raise NB2InputError(f"X has {len(X)} rows but y has {n}")
        p = None
        Xm = []
        for i, row in enumerate(X):
            if not isinstance(row, (list, tuple)):
                raise NB2InputError(f"X[{i}] must be a list row")
            if p is None:
                p = len(row)
                if p < 1 or p > LIMITS["max_p"]:
                    raise NB2InputError(
                        f"X has {p} columns; supported range is 1..{LIMITS['max_p']}"
                    )
            elif len(row) != p:
                raise NB2InputError(
                    f"X[{i}] has {len(row)} entries, expected {p} (rectangular)"
                )
            Xm.append([
                _finite_float(v, "X", f"[{i}][{j}]") for j, v in enumerate(row)
            ])
        assert p is not None

    n, p = len(Xm), len(Xm[0])

    if offset is None:
        off = [0.0] * n
    else:
        if not isinstance(offset, (list, tuple)):
            raise NB2InputError(
                f"offset must be a length-{n} list/tuple, got "
                f"{type(offset).__name__}"
            )
        if len(offset) != n:
            raise NB2InputError(
                f"offset has length {len(offset)}, expected {n}"
            )
        off = [_finite_float(v, "offset", i) for i, v in enumerate(offset)]

    if alpha is not None:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise NB2InputError(f"alpha must be a positive number, got {alpha!r}")
        a = float(alpha)
        if not math.isfinite(a) or a <= 0.0:
            raise NB2InputError(f"alpha must be finite and > 0, got {alpha!r}")

    rank, deps = design_rank(Xm)
    if rank < p:
        raise RankDeficientDesignError(deps, rank, p)

    return yf, Xm, off, n, p


# ---------------------------------------------------------------------------
# Likelihood primitives
# ---------------------------------------------------------------------------


def _stable_ratio_logs(mu: float, theta: float) -> tuple[float, float, float]:
    """Return (r, log r, log(1-r)) for r = mu/(mu+theta), all finite."""
    if mu <= theta:
        u = mu / theta
        logr = math.log(u) - math.log1p(u) if u > 0.0 else -math.inf
        log1mr = -math.log1p(u)
        r = u / (1.0 + u)
    else:
        v = theta / mu
        logr = -math.log1p(v)
        log1mr = math.log(v) - math.log1p(v)
        r = 1.0 / (1.0 + v)
    return r, logr, log1mr


def _objective(
    beta: list[float],
    log_theta: float,
    y: list[float],
    X: list[list[float]],
    off: list[float],
    need_derivs: bool = True,
):
    """Log-likelihood and, when requested, gradient and observed Hessian.

    Parameter order: (beta[0..p-1], log theta).  Returns
    (f, grad[p+1], H[(p+1)x(p+1)]).  Overflow signals via f == -inf.
    """
    theta = math.exp(log_theta)
    n = len(y)
    p = len(beta)
    mu = [0.0] * n
    for i in range(n):
        e = off[i]
        Xi = X[i]
        for j in range(p):
            e += Xi[j] * beta[j]
        if e > LIMITS["eta_max"] or not math.isfinite(e):
            return -math.inf, None, None
        mu[i] = math.exp(e)

    f = 0.0
    # Per-observation quantities
    d = [0.0] * n          # dL/deta
    h_eta = [0.0] * n      # d2L/deta2
    e_th = [0.0] * n       # d2L/deta dtheta
    g_th = 0.0             # dL/dtheta
    h_th = 0.0             # d2L/dtheta2
    for i in range(n):
        yi = y[i]
        mui = mu[i]
        r, logr, log1mr = _stable_ratio_logs(mui, theta)
        s = yi + theta
        mt = mui + theta
        a = 1.0 / mt
        # Guard 0 * (-inf) when both y and mu underflow to zero.
        y_logr = yi * logr if yi > 0.0 else 0.0
        # Stable gamma/digamma/trigamma differences in theta (avoids
        # catastrophic cancellation when theta >> y, the near-Poisson
        # regime).
        yint = int(yi)
        f += (
            gamma_ratio_log(theta, yint)
            - math.lgamma(yi + 1.0)
            + y_logr
            + theta * log1mr
        )
        if not need_derivs:
            continue
        ya = yi * a          # y/(mu+theta), bounded
        yr = ya - r          # (y-mu)/(mu+theta), bounded in [-r, ya]
        # dL/deta = theta (y - mu)/(mu + theta)
        di = theta * yr
        d[i] = di
        # d2L/deta2 = - r(1-r)(y+theta)  [= -mu theta (y+theta)/(mu+theta)^2]
        h_eta[i] = -r * (1.0 - r) * s
        # d2L/deta dtheta = r (y-mu)/(mu+theta), no overflowing products
        e_th[i] = r * yr
        # dL/dtheta
        g_th += (
            log1mr
            + r
            - ya
            + digamma_diff(theta, yint)
        )
        # d2L/dtheta2
        h_th += (
            r / theta
            + yr * a
            + trigamma_diff(theta, yint)
        )

    if not need_derivs:
        return f, None, None

    q = p + 1
    grad = [0.0] * q
    H = [[0.0] * q for _ in range(q)]
    for i in range(n):
        Xi = X[i]
        di = d[i]
        hi = h_eta[i]
        ei = e_th[i]
        for j in range(p):
            xij = Xi[j]
            grad[j] += di * xij
            H[j][p] += theta * ei * xij
            row = H[j]
            for k in range(j, p):
                row[k] += hi * xij * Xi[k]
    for j in range(p):
        for k in range(j + 1, p):
            H[k][j] = H[j][k]
        H[p][j] = H[j][p]
    # chain rule z = log theta
    grad[p] = theta * g_th
    H[p][p] = theta * theta * h_th + theta * g_th
    return f, grad, H


# ---------------------------------------------------------------------------
# Public likelihood/gradient entry points (fixed or free alpha)
# ---------------------------------------------------------------------------


def _resolve_dispersion(alpha, theta, log_theta, allow_free_point: bool):
    """Validate the dispersion specification; return (alpha_or_None, log_theta)."""
    n_given = sum(v is not None for v in (alpha, theta, log_theta))
    if n_given == 0:
        if allow_free_point:
            return None, 0.0
        raise NB2InputError(
            "a dispersion point is required: give exactly one of "
            "alpha, theta or log_theta"
        )
    if n_given > 1:
        raise NB2InputError(
            "give exactly one of alpha, theta or log_theta"
        )
    name, v = (
        ("alpha", alpha) if alpha is not None
        else ("theta", theta) if theta is not None
        else ("log_theta", log_theta)
    )
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise NB2InputError(f"{name} must be a real number, got {v!r}")
    f = float(v)
    if not math.isfinite(f):
        raise NB2InputError(f"{name} must be finite (no NaN/Infinity)")
    if name == "alpha":
        if f <= 0.0:
            raise NB2InputError(f"alpha must be > 0, got {v!r}")
        return f, math.log(1.0 / f)
    if name == "theta":
        if f <= 0.0:
            raise NB2InputError(f"theta must be > 0, got {v!r}")
        return 1.0 / f, math.log(f)
    return 1.0 / math.exp(f), f


def nb2_loglik(y, X=None, *, offset=None, beta=None,
               alpha=None, theta=None, log_theta=None) -> float:
    """Evaluate the NB2 log-likelihood at a finite parameter point.

    Give dispersion as exactly one of ``alpha`` (Var = mu + alpha mu^2),
    ``theta`` (size, alpha = 1/theta) or ``log_theta``.
    """
    a, lt = _resolve_dispersion(alpha, theta, log_theta, False)
    yf, Xm, off, n, p = _validate(y, X, offset, a)
    b = [0.0] * p if beta is None else list(beta)
    if len(b) != p:
        raise NB2InputError(f"beta length {len(b)} does not match {p} columns")
    for j, v in enumerate(b):
        b[j] = _finite_float(v, "beta", j)
    f, _, _ = _objective(b, lt, yf, Xm, off, need_derivs=False)
    if not math.isfinite(f):
        raise NB2InputError("log-likelihood is not finite at this parameter point")
    return f


def nb2_gradient(y, X=None, *, offset=None, beta=None,
                 alpha=None, theta=None, log_theta=None) -> list[float]:
    """Gradient of the NB2 log-likelihood.

    Fixed mode (``alpha`` or ``theta`` given): the p beta derivatives.
    Joint mode (``log_theta`` given): ``[d/d beta..., d/d log theta]``,
    i.e. the exact gradient used by the joint optimiser.
    """
    joint = log_theta is not None
    if joint and (alpha is not None or theta is not None):
        raise NB2InputError(
            "give exactly one of alpha, theta or log_theta"
        )
    if joint:
        if isinstance(log_theta, bool) or not isinstance(
            log_theta, (int, float)
        ) or not math.isfinite(float(log_theta)):
            raise NB2InputError("log_theta must be a finite real number")
        lt = float(log_theta)
        a = 1.0 / math.exp(lt)
    else:
        a, lt = _resolve_dispersion(alpha, theta, None, False)
    yf, Xm, off, n, p = _validate(y, X, offset, a)
    b = [0.0] * p if beta is None else list(beta)
    if len(b) != p:
        raise NB2InputError(f"beta length {len(b)} does not match {p} columns")
    for j, v in enumerate(b):
        b[j] = _finite_float(v, "beta", j)
    _, grad, _ = _objective(b, lt, yf, Xm, off, need_derivs=True)
    return grad if joint else grad[:p]


# ---------------------------------------------------------------------------
# PMF
# ---------------------------------------------------------------------------


def log_pmf(y_value, mu, alpha) -> float:
    """Log NB2 PMF: mean ``mu`` > 0, dispersion ``alpha`` > 0, y >= 0."""
    for name, v in (("y", y_value), ("mu", mu), ("alpha", alpha)):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise NB2InputError(f"{name} must be a real number, got {v!r}")
        if not math.isfinite(float(v)):
            raise NB2InputError(f"{name} must be finite")
    if float(y_value) < 0 or not float(y_value).is_integer():
        raise NB2InputError(f"y must be a non-negative integer, got {y_value!r}")
    if mu <= 0.0 or alpha <= 0.0:
        raise NB2InputError("mu and alpha must be > 0")
    yv = float(y_value)
    th = 1.0 / alpha
    r, logr, log1mr = _stable_ratio_logs(float(mu), th)
    y_logr = yv * logr if yv > 0.0 else 0.0
    return (
        gamma_ratio_log(th, int(yv))
        - math.lgamma(yv + 1.0)
        + y_logr
        + th * log1mr
    )


def pmf(y_value, mu, alpha) -> float:
    """NB2 probability mass at ``y_value``."""
    return math.exp(log_pmf(y_value, mu, alpha))


def range_probability(lo, hi, mu, alpha) -> float:
    """Sum P(Y = k) for integer k in [lo, hi]; ``hi=None`` is unbounded.

    The unbounded case uses the complement P(Y < lo) for numerical
    reasons.  Finite truncation sets use a direct positive-term sum.
    """
    lo = int(lo)
    if lo < 0:
        raise NB2InputError("lo must be >= 0")
    if hi is not None:
        hi = int(hi)
        if hi < lo:
            raise NB2InputError("hi must be >= lo")
        return sum(math.exp(log_pmf(k, mu, alpha)) for k in range(lo, hi + 1))
    return max(0.0, 1.0 - sum(
        math.exp(log_pmf(k, mu, alpha)) for k in range(lo)
    ))


def conditional_pmf(y_value, lo, hi, mu, alpha) -> float:
    """P(Y = y | lo <= Y <= hi) under NB2 truncation to [lo, hi]."""
    y_value = int(y_value)
    if not (lo <= y_value <= hi):
        raise NB2InputError("y_value must lie within [lo, hi]")
    denom = range_probability(lo, hi, mu, alpha)
    return math.exp(log_pmf(y_value, mu, alpha)) / denom


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _positive_int_param(v, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise NB2InputError(f"{name} must be an integer, got {v!r}")
    if v <= 0:
        raise NB2InputError(f"{name} must be > 0, got {v}")
    return v


def _newton_step(H, g, p_plus_1: int):
    """Solve -H d = g with Cholesky + Levenberg damping. Returns d or None."""
    scale = [max(abs(H[i][i]), 1e-12) for i in range(p_plus_1)]
    lam = 0.0
    for _ in range(40):
        M = [row[:] for row in H]
        for i in range(p_plus_1):
            M[i][i] = -M[i][i] + lam * scale[i]
            for j in range(p_plus_1):
                if i != j:
                    M[i][j] = -M[i][j]
        try:
            d = solve_pd(M, g)
            return d
        except Exception:
            lam = max(lam * 10.0, 1e-10)
    return None


def fit_nb2(
    y,
    X=None,
    *,
    offset=None,
    alpha: float | None = None,
    max_iter: int = 100,
    gtol: float = 1.0e-7,
    step_tol: float = 1.0e-6,
) -> NB2Result:
    """Fit NB2 by maximum likelihood.

    Parameters
    ----------
    y:
        Length-n list of non-negative integer counts, n <= 100.
    X:
        n x p design matrix, 1 <= p <= 5, full column rank; ``None``
        means an intercept-only model.
    offset:
        Optional length-n finite real vector added to the linear
        predictor.
    alpha:
        ``None`` (default) jointly estimates ``(beta, log theta)``.
        A positive finite value fixes alpha and optimises beta only.
    max_iter:
        Positive integer (bool rejected). Budget exhaustion yields
        ``status='max_iter'`` / `ConvergenceBudgetError` on request; it
        is never reported as convergence.
    """
    max_iter = _positive_int_param(max_iter, "max_iter")
    if isinstance(gtol, bool) or not isinstance(gtol, (int, float)) or not (
        math.isfinite(float(gtol)) and gtol > 0
    ):
        raise NB2InputError("gtol must be a finite positive number")
    if isinstance(step_tol, bool) or not isinstance(step_tol, (int, float)) or not (
        math.isfinite(float(step_tol)) and step_tol > 0
    ):
        raise NB2InputError("step_tol must be a finite positive number")

    fixed = alpha is not None
    yf, Xm, off, n, p = _validate(y, X, offset, alpha)

    # Provable non-existence of a finite *beta* MLE: every count is zero
    # and the design contains an intercept column.  For y=0 the PMF
    # (theta/(mu+theta))**theta is strictly decreasing in mu.  Along
    # beta_0 -> -infinity (the intercept direction) every mu_i -> 0, so
    # the likelihood is strictly increasing toward a supremum that no
    # finite parameter point attains.  Designs without an intercept do
    # not admit this argument and are optimised normally.
    if max(yf) == 0.0 and any(
        all(Xm[i][j] == 1.0 for i in range(n)) for j in range(p)
    ):
        raise NoFiniteMLEError(
            "all counts are zero and the design contains an intercept: "
            "the y=0 mass is strictly decreasing in mu, and along the "
            "intercept direction mu -> 0 for every row, so the likelihood "
            "supremum is not attained at any finite parameter point"
        )

    theta_fixed = 1.0 / alpha if fixed else 1.0
    beta = [0.0] * p
    log_theta = math.log(theta_fixed)

    lt_min = math.log(LIMITS["theta_min"])
    lt_max = math.log(LIMITS["theta_max"])

    def obj(b, lt):
        if fixed:
            return _objective(b, log_theta, yf, Xm, off, need_derivs=True)
        return _objective(b, lt, yf, Xm, off, need_derivs=True)

    f, grad, H = obj(beta, log_theta)
    rejected = 0
    status = "max_iter"
    message = f"iteration budget ({max_iter}) exhausted; MLE not certified"
    iterations = 0
    box_tol = 1e-9

    for it in range(1, max_iter + 1):
        iterations = it
        q = p if fixed else p + 1

        # Active set: a coordinate pinned at a box face whose gradient
        # points out of the box is held fixed (its step is zero), and
        # KKT convergence is judged on the free coordinates only.
        active = [False] * q
        at_upper = at_lower = False
        if not fixed:
            at_upper = log_theta >= lt_max - box_tol and grad[p] > 0.0
            at_lower = log_theta <= lt_min + box_tol and grad[p] < 0.0
            active[p] = at_upper or at_lower
        free_idx = [k for k in range(q) if not active[k]]

        gf = [grad[k] for k in free_idx]
        Hf = [[H[i][j] for j in free_idx] for i in free_idx]
        m = len(free_idx)

        # Newton (damped) direction solving -H_f d = g_f
        df = _newton_step(Hf, gf, m)
        if df is None:
            status = "step_failure"
            message = "could not factorise a damped Hessian; step rejected"
            break

        # Convergence test before stepping: once the free gradient is
        # within tolerance and the predicted Newton step is within the
        # step tolerance, no line search could certify a larger gain than
        # function-evaluation noise anyway, so declare convergence
        # immediately (essential at the theta box).
        gf_norm = max((abs(v) for v in gf), default=0.0)
        pred_step = math.sqrt(sum(v * v for v in df))
        if gf_norm <= gtol and pred_step <= step_tol:
            if at_lower:
                status = "boundary_theta_min"
                message = (
                    "beta converged but log theta is pinned at the lower "
                    f"computational box (theta={LIMITS['theta_min']:g}); "
                    "constrained optimum at the box, unconstrained MLE "
                    "uncertified"
                )
            else:
                status = "converged_boundary" if at_upper else "converged"
                if at_upper:
                    message = (
                        "beta converged and the log-theta estimate is "
                        "pinned at the upper computational box "
                        f"theta={LIMITS['theta_max']:g} with outward "
                        "gradient: constrained optimum at the near-Poisson "
                        "boundary"
                    )
                else:
                    message = "gradient and predicted step within tolerances"
            break

        slope = sum(gf[k] * df[k] for k in range(m))
        if slope <= 0.0:
            # Damped Newton failed to give an ascent direction; fall back
            # to steepest ascent on the free coordinates.
            df = list(gf)
            slope = sum(gf[k] * df[k] for k in range(m))
        d = [0.0] * q
        for k, idx in enumerate(free_idx):
            d[idx] = df[k]

        # Line search with overflow rejection.  t=1 (full Newton) is
        # tried first; while it satisfies Armijo the step is repeatedly
        # doubled, so a long run toward a distant optimum is not taken
        # one unit at a time.  The first candidate that overflows or
        # breaks Armijo is rejected, and the largest *acceptable*
        # candidate is taken.  If t=1 itself is unacceptable, classical
        # halving backtracking runs.  The parameter point is untouched
        # until an acceptance.
        f_new = -math.inf
        accepted = False
        cand_b = beta
        cand_lt = log_theta
        g_new = H_new = None

        def candidate(tt: float):
            cb = [beta[j] + tt * d[j] for j in range(p)]
            clt = log_theta if fixed else min(
                lt_max, max(lt_min, log_theta + tt * d[p])
            )
            f_try, g_try, H_try = _objective(
                cb, log_theta if fixed else clt, yf, Xm, off, need_derivs=True
            )
            return cb, clt, f_try, g_try, H_try

        # Growth phase: only when the current point is far from the
        # optimum (large free-gradient norm) do we double acceptable
        # steps to cover a long march.  Near the optimum the full Newton
        # step is taken at t=1 exactly, preserving quadratic
        # convergence and avoiding overshoot oscillation.
        grow = max((abs(v) for v in gf), default=0.0) > 50.0

        acceptable: list[tuple] = []
        t = 1.0
        for _grow in range(13 if grow else 1):
            cb, clt, f_try, g_try, H_try = candidate(t)
            if math.isfinite(f_try) and f_try >= f + 1.0e-4 * t * slope:
                acceptable.append((t, f_try, cb, clt, g_try, H_try))
                if not grow:
                    break
                # Stop growing once the directional derivative at the
                # candidate changes sign: the doubled step would cross
                # the optimum.
                gx = list(g_try[:p]) + (
                    [] if fixed else [g_try[p]]
                )
                dd = [d[j] for j in range(p)] + (
                    [] if fixed else [d[p]]
                )
                slope_try = sum(gx[k] * dd[k] for k in range(len(gx)))
                if slope_try <= 0.0:
                    break
                t *= 2.0
                continue
            # An overflow here is a genuinely rejected step; an Armijo
            # failure just marks the end of the growth phase.
            if not math.isfinite(f_try):
                rejected += 1
            break

        if acceptable:
            t, f_new, cand_b, cand_lt, g_new, H_new = acceptable[-1]
            accepted = True
        else:
            t = 0.5
            for _ls in range(80):
                cb, clt, f_try, g_try, H_try = candidate(t)
                if math.isfinite(f_try) and f_try >= f + 1.0e-4 * t * slope:
                    f_new = f_try
                    cand_b, cand_lt = cb, clt
                    g_new, H_new = g_try, H_try
                    accepted = True
                    break
                rejected += 1
                t *= 0.5
                if t < 1.0e-12:
                    break
        if not accepted or not math.isfinite(f_new):
            status = "step_failure"
            message = (
                "line search rejected every candidate step (overflow or "
                "no finite Armijo increase); parameter point unchanged"
            )
            break

        moved = math.sqrt(
            sum((cand_b[j] - beta[j]) ** 2 for j in range(p))
            + (0.0 if fixed else (cand_lt - log_theta) ** 2)
        )
        beta, log_theta = cand_b, cand_lt
        f, grad, H = f_new, g_new, H_new

        # Recompute boundary status at the new point.
        at_upper = at_lower = False
        if not fixed:
            at_upper = log_theta >= lt_max - box_tol and grad[p] >= -gtol
            at_lower = log_theta <= lt_min + box_tol and grad[p] < 0.0
        free_norm = max(abs(grad[j]) for j in range(p))
        if not fixed and not at_upper and not at_lower:
            free_norm = max(free_norm, abs(grad[p]))
        if free_norm <= gtol and moved <= step_tol:
            status = "converged_boundary" if at_upper else "converged"
            if at_upper:
                message = (
                    "beta converged and the log-theta estimate is pinned "
                    f"at the upper computational box theta={LIMITS['theta_max']:g} "
                    "with outward gradient: constrained optimum at the "
                    "near-Poisson boundary"
                )
            else:
                message = "gradient and step size within tolerances"
            break

    theta = math.exp(log_theta)
    alpha_hat = 1.0 / theta
    near_poisson = theta >= LIMITS["near_poisson_theta"]
    at_boundary = (not fixed and theta >= LIMITS["theta_max"] * (1 - 1e-9))

    # Variance-covariance from the observed Hessian at the terminal point
    vcov = None
    try:
        q = p if fixed else p + 1
        M = [[-H[i][j] for j in range(q)] for i in range(q)]
        cols = []
        for k in range(q):
            e = [1.0 if i == k else 0.0 for i in range(q)]
            cols.append(solve_pd(M, e))
        vcov = [[cols[k][i] for k in range(q)] for i in range(q)]
    except Exception:
        vcov = None

    mu_hat = [
        math.exp(min(LIMITS["eta_max"],
                     off[i] + sum(Xm[i][j] * beta[j] for j in range(p))))
        for i in range(n)
    ]

    hessian_out = [row[:q] for row in H[:q]] if fixed else H

    res = NB2Result(
        beta=beta,
        alpha=alpha_hat,
        theta=theta,
        loglik=f,
        gradient=list(grad[:p] if fixed else grad),
        converged=status in ("converged", "converged_boundary"),
        iterations=iterations,
        status=status,
        message=message,
        near_poisson=bool(near_poisson or (fixed and alpha_hat <= 1.0e-6)),
        alpha_at_boundary=bool(at_boundary),
        fixed_alpha=fixed,
        mu=mu_hat,
        hessian=hessian_out,
        vcov=vcov,
        rejected_steps=rejected,
    )
    return res
