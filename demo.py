"""Fixed-input demonstration for the count-regression modules.

Runs entirely offline, deterministically, standard library only.  It
exercises both models with genuinely computed data:

  NB2 (existing):
    1. a normal joint NB2 fit on overdispersed counts, with diagnostics
       (gradient vs central finite differences, fixed-alpha cross-check);
  ZINB (new constant-logit zero-inflated NB2):
    2. a normal interior fit on data with structural zeros -- the fitted
       zero-inflation pi is strictly between 0 and 1, zeros carry soft
       posterior responsibilities (never a hard group assignment), and
       the mixture likelihood beats plain NB2;
    3. a boundary fit on NB2 data with many NATURAL zeros: pi -> 0,
       flagged as a collapse to plain NB2;
    4. an actual triggered *exception*: pi outside [0, 1] is rejected
       with a locatable NBFitError;
    5. a genuine structural *boundary*: an all-zero sample returns
       status "no_finite_mle" (supremum reached only as pi -> 1).

Run:  python demo.py     (expected runtime well under 90 seconds)
"""

import math
import time

from nbreg import (
    fit_nb2,
    fit_zinb,
    nb2_objective,
    zinb_objective,
    zinb_pmf,
    NBFitError,
)


def lcg(seed):
    state = seed & 0xFFFFFFFF
    while True:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def sample_nb(mu, alpha, u):
    r = 1.0 / alpha
    p0 = (1.0 + alpha * mu) ** (-r)
    pk, cdf, k = p0, p0, 0
    ratio = alpha * mu / (1.0 + alpha * mu)
    while u > cdf and k < 100000:
        pk *= (k + r) / (k + 1) * ratio
        cdf += pk
        k += 1
    return k


def make_nb_data():
    """Deterministic overdispersed NB2 sample: 60 obs, 2 regressors."""
    beta_true, alpha_true = [0.45, -0.3], 3.0
    rng = lcg(158)
    n = 60
    X, y, offset = [], [], []
    for i in range(n):
        row = [1.0, ((i * 5) % 11 - 5) / 5.0]
        X.append(row)
        off = 0.15 if i % 4 == 0 else 0.0
        offset.append(off)
        eta = off + row[0] * beta_true[0] + row[1] * beta_true[1]
        y.append(sample_nb(math.exp(eta), alpha_true, next(rng)))
    return y, X, offset, beta_true, alpha_true


def make_zinb_data(pi_true, beta_true, alpha_true, n, seed, spread=0.5):
    """Deterministic ZINB sample: structural zero with prob pi_true."""
    rng = lcg(seed)
    X, y = [], []
    n_struct = 0
    for i in range(n):
        row = [1.0, spread * ((i % 5) - 2)]
        X.append(row)
        if next(rng) < pi_true:
            y.append(0)
            n_struct += 1
        else:
            eta = sum(row[j] * beta_true[j] for j in range(2))
            y.append(sample_nb(math.exp(eta), alpha_true, next(rng)))
    return y, X, n_struct


def make_natural_zero_nb_data():
    """Plain NB2 with alpha = 6 and small mean -> many natural zeros."""
    rng = lcg(3)
    n = 40
    X = [[1.0, 0.4 * ((i % 5) - 2)] for i in range(n)]
    y = []
    for i, row in enumerate(X):
        eta = 0.2 - 0.1 * row[1]
        y.append(sample_nb(math.exp(eta), 6.0, next(rng)))
    return y, X


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("Count-regression demo: NB2 and constant-logit ZINB (offline)")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Normal joint NB2 fit
    # ------------------------------------------------------------------
    y, X, offset, beta_true, alpha_true = make_nb_data()
    mean_y = sum(y) / len(y)
    var_y = sum((v - mean_y) ** 2 for v in y) / len(y)
    print(f"\n[NB2 data] n={len(y)}, mean={mean_y:.3f}, "
          f"variance={var_y:.3f} (var/mean={var_y / mean_y:.2f} >> 1)")

    res = fit_nb2(y, X, offset=offset)
    print("[1] NB2 JOINT FIT")
    print(f"    status     : {res.status} (converged={res.converged})")
    print(f"    beta       : {[round(b, 6) for b in res.beta]}")
    print(f"    alpha      : {res.alpha:.6f}   loglik: {res.loglik:.6f}")

    h = 1e-6
    fd = []
    for j in range(len(res.beta)):
        bp, bm = list(res.beta), list(res.beta)
        bp[j] += h
        bm[j] -= h
        up, _ = nb2_objective(y, X, bp, res.log_alpha, offset)
        dn, _ = nb2_objective(y, X, bm, res.log_alpha, offset)
        fd.append((up - dn) / (2 * h))
    up, _ = nb2_objective(y, X, res.beta, res.log_alpha + h, offset)
    dn, _ = nb2_objective(y, X, res.beta, res.log_alpha - h, offset)
    fd.append((up - dn) / (2 * h))
    worst = max(abs(a - b) for a, b in zip(res.gradient, fd))
    print(f"    grad vs FD : worst |analytic - central diff| = {worst:.2e}")
    rb = fit_nb2(y, X, offset=offset, max_iter=1)
    print(f"    budget case: max_iter=1 -> status={rb.status} "
          f"(converged={rb.converged})")

    # ------------------------------------------------------------------
    # 2. Normal ZINB fit on data with structural zeros
    # ------------------------------------------------------------------
    pi_true, b_true, a_true = 0.40, [0.9, -0.25], 0.7
    yz, Xz, n_struct = make_zinb_data(pi_true, b_true, a_true, 60, seed=71)
    n_zero = sum(1 for v in yz if v == 0)
    print(f"\n[ZINB data] n={len(yz)}, zeros={n_zero} "
          f"(structural draws={n_struct}, truth pi={pi_true})")
    rz = fit_zinb(yz, Xz)
    rn = fit_nb2(yz, Xz)
    print("[2] ZINB INTERIOR FIT (constant logit, mixture likelihood)")
    print(f"    status     : {rz.status} (converged={rz.converged})")
    print(f"    message    : {rz.message}")
    print(f"    beta       : {[round(b, 6) for b in rz.beta]}")
    print(f"    alpha      : {rz.alpha:.6f}")
    print(f"    logit(pi)  : {rz.logit_pi:.6f}   pi = {rz.pi:.6f}")
    print(f"    loglik     : {rz.loglik:.6f}  (plain NB2: {rn.loglik:.6f}; "
          f"gain {rz.loglik - rn.loglik:+.4f})")
    print(f"    gradient   : {[f'{g:+.2e}' for g in rz.gradient]}")
    # soft (not hard) zero attribution
    zero_resp = [rz.responsibilities[i] for i in range(len(yz)) if yz[i] == 0]
    print(f"    zero responsibilities: min={min(zero_resp):.4f}, "
          f"max={max(zero_resp):.4f} (strictly inside (0,1): "
          f"{all(0.0 < t < 1.0 for t in zero_resp)}); "
          f"positive counts all 0: "
          f"{all(rz.responsibilities[i] == 0.0 for i in range(len(yz)) if yz[i] > 0)}")
    # independent FD gradient of the same mixture objective
    fd = []
    for j in range(len(rz.beta)):
        bp, bm = list(rz.beta), list(rz.beta)
        bp[j] += h
        bm[j] -= h
        up, _ = zinb_objective(yz, Xz, bp, rz.log_alpha, rz.logit_pi)
        dn, _ = zinb_objective(yz, Xz, bm, rz.log_alpha, rz.logit_pi)
        fd.append((up - dn) / (2 * h))
    up, _ = zinb_objective(yz, Xz, rz.beta, rz.log_alpha + h, rz.logit_pi)
    dn, _ = zinb_objective(yz, Xz, rz.beta, rz.log_alpha - h, rz.logit_pi)
    fd.append((up - dn) / (2 * h))
    up, _ = zinb_objective(yz, Xz, rz.beta, rz.log_alpha, rz.logit_pi + h)
    dn, _ = zinb_objective(yz, Xz, rz.beta, rz.log_alpha, rz.logit_pi - h)
    fd.append((up - dn) / (2 * h))
    worst = max(abs(a - b) for a, b in zip(rz.gradient, fd))
    print(f"    grad vs FD : worst |analytic - central diff| = {worst:.2e}")

    # ------------------------------------------------------------------
    # 3. Boundary: no structural zeros -> pi collapses to 0 (= NB2)
    # ------------------------------------------------------------------
    yn, Xn = make_natural_zero_nb_data()
    print(f"\n[NB2-only data] n={len(yn)}, zeros={sum(1 for v in yn if v == 0)} "
          f"(all NATURAL zeros from a high-alpha NB)")
    rnn = fit_zinb(yn, Xn)
    print("[3] ZINB LOWER-BOUNDARY FIT")
    print(f"    status     : {rnn.status}")
    print(f"    pi         : {rnn.pi:.3e}  pi_at_lower_boundary="
          f"{rnn.pi_at_lower_boundary}")
    print(f"    alpha      : {rnn.alpha:.6f}   loglik: {rnn.loglik:.6f}")
    print(f"    message    : {rnn.message}")

    # ------------------------------------------------------------------
    # 4. Illegal input boundary (exception, not silently corrected)
    # ------------------------------------------------------------------
    print("\n[4] ILLEGAL INPUT (exception, located)")
    try:
        zinb_pmf(0, mu=2.0, alpha=0.5, pi=1.3)
    except NBFitError as exc:
        print(f"    zinb_pmf(pi=1.3) raised NBFitError:\n      -> {exc}")
    ybad = list(yz)
    ybad[0] = 1.5  # float counts are rejected
    try:
        fit_zinb(ybad, Xz)
    except NBFitError as exc:
        print(f"    fit_zinb(float count) raised NBFitError:\n      -> {exc}")

    # ------------------------------------------------------------------
    # 5. Structural boundary: all-zero sample
    # ------------------------------------------------------------------
    print("\n[5] ALL-ZERO SAMPLE (proved structural boundary)")
    r0 = fit_zinb([0] * 8, [[1.0, 0.0]] + [[1.0, 0.25]] * 3
                  + [[1.0, -0.25]] * 4)
    print(f"    status     : {r0.status} (converged={r0.converged})")
    print(f"    pi         : {r0.pi:.6f}  pi_at_upper_boundary="
          f"{r0.pi_at_upper_boundary}")
    print(f"    loglik     : {r0.loglik:.3e} (supremum 0, unreached at "
          f"finite pi)")
    print(f"    verdict    : {r0.message}")

    print(f"\n(elapsed {time.perf_counter() - t0:.2f} s; every value "
          f"computed live, none pre-recorded, no sleeps)")


if __name__ == "__main__":
    main()
