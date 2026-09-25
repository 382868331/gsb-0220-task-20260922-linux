"""Independent reference computations for NB2 tests.

This module deliberately does NOT import the package under test.  Everything
here is re-derived from scratch with the standard library so it can supply
expected values:

  * NB2 pmf / log-likelihood directly from the gamma/Poisson mixture form;
  * pmf recursion p(k+1)/p(k), a different numerical route, for truncated
    probability sums;
  * exact finite-difference gradients of the reference log-likelihood;
  * an independent fixed-alpha beta optimizer (numerical-gradient ascent);
  * deterministic NB sampling by CDF inversion with a private LCG.
"""

import math
from itertools import combinations


def nb_log_pmf(k, mu, alpha):
    """log P(Y=k), mu>0, alpha>0, derived from the NB2 pmf directly."""
    r = 1.0 / alpha
    return (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
            + k * (math.log(mu) - math.log(r))
            - (k + r) * math.log1p(mu / r))


def nb_pmf_recursion(mu, alpha, kmax):
    """List [P(Y=0), ..., P(Y=kmax)] via the pmf ratio recurrence.

    Different route from the lgamma formula:
        p(0) = (1 + alpha mu)^(-1/alpha)
        p(k+1)/p(k) = (k + 1/alpha)/(k+1) * (alpha mu)/(1 + alpha mu)
    """
    r = 1.0 / alpha
    p = [(1.0 + alpha * mu) ** (-r)]
    ratio = alpha * mu / (1.0 + alpha * mu)
    for k in range(kmax):
        p.append(p[-1] * (k + r) / (k + 1) * ratio)
    return p


def poisson_log_pmf(k, mu):
    return -mu + k * math.log(mu) - math.lgamma(k + 1)


def ref_loglik(y, X, beta, log_alpha, offset=None):
    alpha = math.exp(log_alpha)
    n = len(y)
    if offset is None:
        offset = [0.0] * n
    total = 0.0
    for i in range(n):
        eta = offset[i] + sum(X[i][j] * beta[j] for j in range(len(beta)))
        total += nb_log_pmf(y[i], math.exp(eta), alpha)
    return total


def ref_fd_gradient(y, X, beta, log_alpha, offset=None, h=1e-6):
    """Central finite differences of the reference log-likelihood."""
    b = list(beta)
    g = []
    for j in range(len(b)):
        b[j] += h
        up = ref_loglik(y, X, b, log_alpha, offset)
        b[j] -= 2 * h
        dn = ref_loglik(y, X, b, log_alpha, offset)
        b[j] = beta[j]
        g.append((up - dn) / (2 * h))
    la = log_alpha + h
    up = ref_loglik(y, X, beta, la, offset)
    la = log_alpha - h
    dn = ref_loglik(y, X, beta, la, offset)
    g.append((up - dn) / (2 * h))
    return g


def ref_fit_beta_fixed_alpha(y, X, alpha, offset=None, free_mask=None,
                             max_iter=20000, tol=1e-10):
    """Independent beta-only MLE at fixed alpha.

    Numerical-gradient ascent (central differences on the reference
    likelihood) with per-coordinate step halving.  Slow but completely
    independent of the analytic machinery under test.

    ``free_mask`` (optional) restricts movement to the coordinates
    flagged True; the others stay at 0.  Used by the bounded-fit
    enumeration reference below.
    """
    n = len(y)
    k = len(X[0])
    if free_mask is None:
        free_mask = [True] * k
    if offset is None:
        offset = [0.0] * n
    beta = [0.0] * k
    base_step = 1e-3
    cur = ref_loglik(y, X, beta, math.log(alpha), offset)
    for _ in range(max_iter):
        moved = False
        for j in range(k):
            if not free_mask[j]:
                continue
            h = 1e-5
            bp = list(beta)
            bp[j] += h
            up = ref_loglik(y, X, bp, math.log(alpha), offset)
            bm = list(beta)
            bm[j] -= h
            dn = ref_loglik(y, X, bm, math.log(alpha), offset)
            deriv = (up - dn) / (2 * h)
            if abs(deriv) < tol:
                continue
            step = base_step
            for _halves in range(40):
                cand = list(beta)
                cand[j] += step * (1.0 if deriv > 0 else -1.0)
                val = ref_loglik(y, X, cand, math.log(alpha), offset)
                if val > cur + 1e-12:
                    beta, cur, moved = cand, val, True
                    break
                step *= 0.5
        if not moved:
            break
    return beta, cur


def ref_beta_gradient(y, X, beta, alpha, offset=None, h=1e-6):
    """Central finite differences of the reference loglik, beta only."""
    return ref_fd_gradient(y, X, beta, math.log(alpha), offset, h=h)[:len(beta)]


def ref_fit_bounded_enum(y, X, nonneg, alpha, offset=None):
    """Independent bounded-fit reference by exhaustive active-set enum.

    Fixed alpha.  Every subset S of the non-negativity-constrained
    coefficients is pinned at 0 in turn; the remaining coefficients are
    optimized in the free subspace by the independent numerical-gradient
    optimizer above.  A candidate is accepted only if it is feasible
    (bounded free coordinates >= 0) and satisfies the boundary KKT
    condition (gradient <= tolerance at every pinned coordinate, for a
    maximization with beta_j >= 0).  The best feasible KKT point wins.

    Returns (beta, loglik, pinned_set) of the winner.
    """
    k = len(X[0])
    bounded = [j for j in range(k) if nonneg[j]]
    best = None
    for r in range(len(bounded) + 1):
        for pinned in combinations(bounded, r):
            free_mask = [j not in pinned for j in range(k)]
            beta, ll = ref_fit_beta_fixed_alpha(
                y, X, alpha, offset, free_mask=free_mask)
            if any(nonneg[j] and beta[j] < -1e-9 for j in range(k)):
                continue  # optimizer left the feasible box: discard
            beta = [max(0.0, b) if nonneg[j] else b
                    for j, b in enumerate(beta)]
            g = ref_beta_gradient(y, X, beta, alpha, offset)
            if any(g[j] > 1e-4 for j in pinned):
                continue  # boundary KKT violated: not a candidate
            if best is None or ll > best[1]:
                best = (beta, ll, pinned)
    if best is None:
        raise AssertionError("enumeration found no feasible KKT point")
    return best


# Deterministic generator ---------------------------------------------------

def _lcg(seed):
    state = seed & 0xFFFFFFFF
    while True:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def sample_nb(mu, alpha, u):
    """One NB2 draw by CDF inversion; u a uniform float in [0, 1)."""
    r = 1.0 / alpha
    p0 = (1.0 + alpha * mu) ** (-r)
    cdf = p0
    k = 0
    ratio = alpha * mu / (1.0 + alpha * mu)
    pk = p0
    while u > cdf and k < 100000:
        pk *= (k + r) / (k + 1) * ratio
        cdf += pk
        k += 1
    return k


def make_dataset(n, beta_true, alpha_true, offset=None, seed=12345):
    rng = _lcg(seed)
    k = len(beta_true)
    X = [[float((i * (j + 3)) % 7 - 3) / 2.0 for j in range(k)]
         for i in range(n)]
    if offset is None:
        offset = [0.0] * n
    y = []
    for i in range(n):
        eta = offset[i] + sum(X[i][j] * beta_true[j] for j in range(k))
        y.append(sample_nb(math.exp(eta), alpha_true, next(rng)))
    return y, X, offset
