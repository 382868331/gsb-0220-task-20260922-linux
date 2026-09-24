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
# Zero-inflated NB2 reference (derived independently of nbreg.zinb)
# ---------------------------------------------------------------------------

def zinb_mixture_pmf(k, mu, alpha, pi):
    """Mixture pmf straight from its definition, using nb pmf above."""
    p_nb = math.exp(nb_log_pmf(k, mu, alpha))
    if k == 0:
        return pi + (1.0 - pi) * p_nb
    return (1.0 - pi) * p_nb


def zinb_structural_tau(k, mu, alpha, pi):
    """Posterior state probability tau = P(structural | y=k)."""
    if k > 0:
        return 0.0
    return pi / zinb_mixture_pmf(0, mu, alpha, pi)


def ref_zinb_loglik(y, X, beta, log_alpha, logit_pi, offset=None):
    """ZINB mixture log-likelihood computed directly from the pmf."""
    alpha = math.exp(log_alpha)
    pi = 1.0 / (1.0 + math.exp(-logit_pi))
    n = len(y)
    if offset is None:
        offset = [0.0] * n
    total = 0.0
    for i in range(n):
        eta = offset[i] + sum(X[i][j] * beta[j] for j in range(len(beta)))
        mu = math.exp(eta)
        if y[i] == 0:
            p0_nb = (1.0 + alpha * mu) ** (-1.0 / alpha)
            total += math.log(pi + (1.0 - pi) * p0_nb)
        else:
            total += math.log1p(-pi) + nb_log_pmf(y[i], mu, alpha)
    return total


def ref_zinb_fd_gradient(y, X, beta, log_alpha, logit_pi, offset=None,
                         h=1e-6):
    """Central finite differences of the reference ZINB likelihood."""
    def ll(bb, la, gp):
        return ref_zinb_loglik(y, X, bb, la, gp, offset)

    g = []
    b = list(beta)
    for j in range(len(b)):
        b[j] += h
        up = ll(b, log_alpha, logit_pi)
        b[j] -= 2 * h
        dn = ll(b, log_alpha, logit_pi)
        b[j] = beta[j]
        g.append((up - dn) / (2 * h))
    g.append((ll(beta, log_alpha + h, logit_pi)
              - ll(beta, log_alpha - h, logit_pi)) / (2 * h))
    g.append((ll(beta, log_alpha, logit_pi + h)
              - ll(beta, log_alpha, logit_pi - h)) / (2 * h))
    return g


def ref_zinb_fit(y, X, offset=None, max_sweeps=4000, tol=1e-11):
    """Independent ZINB MLE: numerical-gradient coordinate ascent.

    Completely independent of the analytic score/Hessian under test:
    central-difference derivatives and per-coordinate step halving on
    the direct mixture likelihood.  Started from several deterministic
    points (the surface is not concave); the best log-likelihood found
    is returned.  Slow but straightforward -- used only on small data.
    """
    n = len(y)
    k = len(X[0])
    if offset is None:
        offset = [0.0] * n
    starts = [[0.2] * k + [0.0, gp] for gp in (-2.0, -0.5, 1.0)]
    starts += [[-0.5] * k + [1.0, -1.5]]
    best = None
    box = 18.0
    for beta0 in starts:
        p = list(beta0)
        cur = ref_zinb_loglik(y, X, p[:k], p[k], p[k + 1], offset)
        step = 2e-3
        for _ in range(max_sweeps):
            moved = False
            for j in range(k + 2):
                h = 1e-5
                pp, pm = list(p), list(p)
                pp[j] += h
                pm[j] -= h
                up = ref_zinb_loglik(y, X, pp[:k], pp[k], pp[k + 1], offset)
                dn = ref_zinb_loglik(y, X, pm[:k], pm[k], pm[k + 1], offset)
                deriv = (up - dn) / (2 * h)
                if abs(deriv) < tol:
                    continue
                s = step
                for _halves in range(45):
                    cand = list(p)
                    cand[j] += s * (1.0 if deriv > 0 else -1.0)
                    if abs(cand[j]) > box and j >= k:
                        s *= 0.5
                        continue
                    val = ref_zinb_loglik(y, X, cand[:k], cand[k],
                                          cand[k + 1], offset)
                    if val > cur + 1e-12:
                        p, cur, moved = cand, val, True
                        step = min(step * 1.6, 0.2)
                        break
                    s *= 0.5
            if not moved:
                break
        if best is None or cur > best[1]:
            best = (p, cur)
    return best[0], best[1]


def make_zinb_dataset(n, beta_true, alpha_true, pi_true, offset=None,
                      seed=12345):
    """Deterministic ZINB draws: structural zero with probability pi_true,
    otherwise an NB2 count (which may also be zero)."""
    rng = _lcg(seed)
    k = len(beta_true)
    X = [[float((i * (j + 3)) % 7 - 3) / 2.0 for j in range(k)]
         for i in range(n)]
    if offset is None:
        offset = [0.0] * n
    y = []
    structural = []
    for i in range(n):
        if next(rng) < pi_true:
            y.append(0)
            structural.append(True)
        else:
            eta = offset[i] + sum(X[i][j] * beta_true[j] for j in range(k))
            y.append(sample_nb(math.exp(eta), alpha_true, next(rng)))
            structural.append(False)
    return y, X, offset, structural
