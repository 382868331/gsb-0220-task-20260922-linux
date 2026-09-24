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


def ref_fit_beta_fixed_alpha(y, X, alpha, offset=None,
                             max_iter=20000, tol=1e-10):
    """Independent beta-only MLE at fixed alpha.

    Numerical-gradient ascent (central differences on the reference
    likelihood) with per-coordinate step halving.  Slow but completely
    independent of the analytic machinery under test.
    """
    n = len(y)
    k = len(X[0])
    if offset is None:
        offset = [0.0] * n
    beta = [0.0] * k
    base_step = 1e-3
    cur = ref_loglik(y, X, beta, math.log(alpha), offset)
    for _ in range(max_iter):
        moved = False
        for j in range(k):
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


# ---------------------------------------------------------------------------
# Independent ZINB reference (constant logit zero-inflation).
# Derived directly from the two-component mixture definition; never imports
# the package under test.
# ---------------------------------------------------------------------------

def zinb_log_pmf(k, mu, alpha, pi):
    """log P(Y=k) for the ZINB mixture, from the definition."""
    if k == 0:
        p0 = math.exp(nb_log_pmf(0, mu, alpha))
        return math.log(pi + (1.0 - pi) * p0)
    return math.log(1.0 - pi) + nb_log_pmf(k, mu, alpha)


def zinb_mass(k, mu, alpha, pi):
    """P(Y=k), direct mixture formula (a second numerical route)."""
    p0 = (1.0 + alpha * mu) ** (-1.0 / alpha)
    if k == 0:
        return pi + (1.0 - pi) * p0
    return (1.0 - pi) * math.exp(nb_log_pmf(k, mu, alpha))


def ref_zinb_loglik(y, X, beta, log_alpha, logit_pi, offset=None):
    alpha = math.exp(log_alpha)
    pi = 1.0 / (1.0 + math.exp(-logit_pi))
    n = len(y)
    if offset is None:
        offset = [0.0] * n
    total = 0.0
    for i in range(n):
        eta = offset[i] + sum(X[i][j] * beta[j] for j in range(len(beta)))
        total += zinb_log_pmf(y[i], math.exp(eta), alpha, pi)
    return total


def ref_zinb_fd_gradient(y, X, beta, log_alpha, logit_pi, offset=None,
                         h=1e-6):
    """Central finite differences of the reference ZINB log-likelihood."""
    g = []
    b = list(beta)
    for j in range(len(b)):
        b[j] += h
        up = ref_zinb_loglik(y, X, b, log_alpha, logit_pi, offset)
        b[j] -= 2 * h
        dn = ref_zinb_loglik(y, X, b, log_alpha, logit_pi, offset)
        b[j] = beta[j]
        g.append((up - dn) / (2 * h))
    up = ref_zinb_loglik(y, X, beta, log_alpha + h, logit_pi, offset)
    dn = ref_zinb_loglik(y, X, beta, log_alpha - h, logit_pi, offset)
    g.append((up - dn) / (2 * h))
    up = ref_zinb_loglik(y, X, beta, log_alpha, logit_pi + h, offset)
    dn = ref_zinb_loglik(y, X, beta, log_alpha, logit_pi - h, offset)
    g.append((up - dn) / (2 * h))
    return g


def ref_fit_zinb(y, X, starts, offset=None, max_iter=20000, tol=1e-10):
    """Independent ZINB MLE: numerical-gradient coordinate ascent over
    (beta, log_alpha, logit_pi), multi-start.  Slow but shares no code
    with the analytic fitter under test.  Returns (params, loglik).
    """
    n = len(y)
    if offset is None:
        offset = [0.0] * n

    def ll(p):
        k = len(X[0])
        return ref_zinb_loglik(y, X, p[:k], p[k], p[k + 1], offset)

    best = None
    for start in starts:
        p = list(start)
        # keep the optimizer inside the same finite box the fitter uses
        lo = [-20.0] * (len(p) - 2) + [-20.0, -30.0]
        hi = [20.0] * (len(p) - 2) + [25.0, 30.0]
        cur = ll(p)
        base_step = 2e-3
        for _ in range(max_iter):
            moved = False
            for j in range(len(p)):
                h = 1e-5
                bp = list(p); bp[j] += h
                bm = list(p); bm[j] -= h
                deriv = (ll(bp) - ll(bm)) / (2 * h)
                if abs(deriv) < tol:
                    continue
                step = base_step
                for _halves in range(40):
                    cand = list(p)
                    cand[j] += step * (1.0 if deriv > 0 else -1.0)
                    if cand[j] <= lo[j] or cand[j] >= hi[j]:
                        step *= 0.5
                        continue
                    val = ll(cand)
                    if val > cur + 1e-12:
                        p, cur, moved = cand, val, True
                        break
                    step *= 0.5
            if not moved:
                break
        if best is None or cur > best[1]:
            best = (p, cur)
    return best


def sample_zinb(mu, alpha, pi, u_struct, u_count):
    """One ZINB draw: u_struct < pi -> structural 0, else an NB draw."""
    if u_struct < pi:
        return 0
    return sample_nb(mu, alpha, u_count)


def make_zinb_dataset(n, beta_true, alpha_true, pi_true, seed=12345):
    """Deterministic ZINB sample with an explicit two-stage draw: the
    structural/count coin is drawn separately, so the generative process
    really is a mixture (the analyzer is never told which zeros were
    structural)."""
    rng = _lcg(seed)
    k = len(beta_true)
    X = [[float((i * (j + 3)) % 7 - 3) / 2.0 for j in range(k)]
         for i in range(n)]
    y = []
    labels = []
    for i in range(n):
        eta = sum(X[i][j] * beta_true[j] for j in range(k))
        u_struct = next(rng)
        if u_struct < pi_true:
            y.append(0)
            labels.append(1)
        else:
            y.append(sample_nb(math.exp(eta), alpha_true, next(rng)))
            labels.append(0)
    return y, X, labels
