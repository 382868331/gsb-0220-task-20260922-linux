"""Independent reference calculations for cross-checking the fitter.

Nothing in this module imports or calls the package under test
(``nb2reg``).  It is a deliberately simple, separately written
implementation used only by the tests:

* the NB2 log-PMF straight from Gamma/binomial coefficients;
* the log-likelihood by summing those PMFs;
* Poisson-gamma mixture RNG (deterministic seed supplied by callers);
* central finite differences for the gradient.

Parameter regimes are kept moderate here (theta O(0.1..10)) so the
naive lgamma differences in the reference do not themselves cancel;
cancellation-resistant behaviour near theta -> infinity is covered
separately by the stable-difference unit tests.
"""

from __future__ import annotations

import math
import random


def ref_log_pmf(k: int, mu: float, alpha: float) -> float:
    theta = 1.0 / alpha
    return (
        math.lgamma(k + theta)
        - math.lgamma(theta)
        - math.lgamma(k + 1.0)
        + theta * math.log(theta)
        + k * math.log(mu)
        - (k + theta) * math.log(theta + mu)
    )


def ref_loglik(y, X, beta, alpha, offset=None):
    theta = 1.0 / alpha
    total = 0.0
    for i, yi in enumerate(y):
        eta = 0.0 if offset is None else offset[i]
        for j, b in enumerate(beta):
            eta += X[i][j] * b
        total += ref_log_pmf(int(yi), math.exp(eta), alpha)
    return total


def ref_finite_diff_grad(y, X, beta, log_theta, offset=None, eps=1.0e-6):
    p = len(beta)
    point = list(beta) + [log_theta]

    def ll(q):
        return ref_loglik(y, X, q[:p], 1.0 / math.exp(q[p]), offset)

    out = []
    for k in range(p + 1):
        up = point[:]
        dn = point[:]
        up[k] += eps
        dn[k] -= eps
        out.append((ll(up) - ll(dn)) / (2.0 * eps))
    return out


def ref_range_sum(lo, hi, mu, alpha):
    """Independent truncated-range sum (hi may be None)."""
    if hi is None:
        tail_ge_lo = 1.0 - sum(
            math.exp(ref_log_pmf(k, mu, alpha)) for k in range(lo)
        )
        return max(0.0, tail_ge_lo)
    return sum(math.exp(ref_log_pmf(k, mu, alpha)) for k in range(lo, hi + 1))


def sample_poisson(rng: random.Random, lam: float) -> int:
    """Knuth's Poisson sampler (lam is not huge in the test data)."""
    border = math.exp(-lam)
    k = 0
    p = 1.0
    while p > border:
        k += 1
        p *= rng.random()
    return k - 1


def sample_nb2(rng: random.Random, mu: float, alpha: float) -> int:
    """Gamma-Poisson mixture with the NB2 shape theta = 1/alpha."""
    theta = 1.0 / alpha
    gamma_mean_one = rng.gammavariate(theta, 1.0 / theta)
    return sample_poisson(rng, mu * gamma_mean_one)


def make_dataset(seed, n, beta, alpha, x_sampler=None, with_intercept=True):
    rng = random.Random(seed)
    y = []
    X = []
    offset = []
    for _ in range(n):
        row = []
        if with_intercept:
            row.append(1.0)
        if x_sampler is not None:
            row.append(float(x_sampler(rng)))
        eta = sum(row[j] * beta[j] for j in range(len(beta)))
        off = rng.uniform(-0.2, 0.2)
        offset.append(off)
        mu = math.exp(eta + off)
        y.append(sample_nb2(rng, mu, alpha))
        X.append(row)
    return y, X, offset
