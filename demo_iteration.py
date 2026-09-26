"""Fixed-input demo for interval-censored NB2 fitting (offline, stdlib only).

Observations may be exact counts, closed integer intervals [lo, hi], or
right tails "at least L".  alpha is fixed; only the regression
coefficients beta are fitted.  Everything below is computed live:

  1. a normal fit on synthetic mixed observations (exact / interval /
     at-least-L), with an independent finite-difference gradient check
     and a likelihood comparison against the true coefficients;
  2. a genuinely triggered structural boundary: only lower bounds ->
     beta is not identifiable, status "no_finite_mle";
  3. a genuinely triggered budget boundary: a tail summation whose term
     budget is too small aborts that objective evaluation with a reason
     (the truncated tail sum is never used as an exact likelihood).

Run:  python demo_iteration.py     (expected runtime well under 90 s)
"""

import math
import time

from nbreg import (fit_nb2_censored, nb2_censored_objective,
                   nb2_interval_logprob, NBFitError)


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


def make_data():
    """Deterministic synthetic sample, then deterministic censoring."""
    beta_true, alpha_true = [0.5, -0.35], 0.8
    rng = lcg(20260926)
    n = 60
    X, offset, obs = [], [], []
    n_exact = n_interval = n_tail = 0
    for i in range(n):
        row = [1.0, ((i * 7) % 13 - 6) / 6.0]
        X.append(row)
        off = 0.1 if i % 5 == 0 else 0.0
        offset.append(off)
        eta = off + row[0] * beta_true[0] + row[1] * beta_true[1]
        v = sample_nb(math.exp(eta), alpha_true, next(rng))
        m = i % 3
        if m == 0:
            obs.append(v)
            n_exact += 1
        elif m == 1:
            width = 1 + (i % 3)
            obs.append((max(0, v - width), v + width))
            n_interval += 1
        else:
            obs.append((max(0, v - (i % 2)), None))
            n_tail += 1
    return obs, X, offset, beta_true, alpha_true, (n_exact, n_interval, n_tail)


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("Interval-censored NB2 regression demo (fixed alpha, beta only)")
    print("=" * 72)

    obs, X, offset, beta_true, alpha_true, counts = make_data()
    print(f"\n[data] n={len(obs)} observations: {counts[0]} exact, "
          f"{counts[1]} intervals [lo, hi], {counts[2]} at-least-L")
    print(f"[data] true beta={beta_true}, fixed alpha={alpha_true}")
    print(f"[data] first 6 obs: {obs[:6]}")

    # ------------------------------------------------------------------
    # 1. Normal fit on mixed censored observations
    # ------------------------------------------------------------------
    res = fit_nb2_censored(obs, X, alpha_true, offset=offset)
    print("\n[1] MIXED CENSORED FIT (alpha fixed, beta fitted)")
    print(f"    status       : {res.status} (converged={res.converged})")
    print(f"    message      : {res.message}")
    print(f"    beta         : {[round(b, 6) for b in res.beta]} "
          f"(true {beta_true})")
    print(f"    alpha (fixed): {res.alpha}")
    print(f"    loglik       : {res.loglik:.8f}")
    print(f"    gradient     : {[f'{g:+.2e}' for g in res.gradient]}")
    print(f"    iterations   : {res.n_iter}, rejected steps: {res.n_rejected}")

    # Independent check: central finite differences of the same objective.
    h = 1e-6
    fd = []
    for j in range(len(res.beta)):
        bp, bm = list(res.beta), list(res.beta)
        bp[j] += h
        bm[j] -= h
        up, _ = nb2_censored_objective(obs, X, bp, alpha_true, offset=offset)
        dn, _ = nb2_censored_objective(obs, X, bm, alpha_true, offset=offset)
        fd.append((up - dn) / (2 * h))
    worst = max(abs(a - b) for a, b in zip(res.gradient, fd))
    print(f"    grad vs FD   : worst |analytic - central diff| = {worst:.2e}")

    ll_true, _ = nb2_censored_objective(obs, X, beta_true, alpha_true,
                                        offset=offset)
    print(f"    loglik at true beta: {ll_true:.6f} "
          f"(fitted >= true: {res.loglik >= ll_true - 1e-8})")

    # Interval probabilities used by the likelihood, computed live.
    p_exact = nb2_interval_logprob(3, 3, 2.0, alpha_true)
    p_inter = nb2_interval_logprob(2, 5, 2.0, alpha_true)
    p_tail = nb2_interval_logprob(4, None, 2.0, alpha_true)
    print(f"    live probs   : log P(Y=3)={p_exact:.6f}, "
          f"log P(2..5)={p_inter:.6f}, log P(Y>=4)={p_tail:.6f} "
          f"(mu=2, alpha={alpha_true})")

    # ------------------------------------------------------------------
    # 2. Structural boundary: only lower bounds -> not identifiable
    # ------------------------------------------------------------------
    print("\n[2] STRUCTURAL BOUNDARY (only lower bounds, proved no finite MLE)")
    only_lo = [(2, None), (3, None), (1, None), (4, None), (2, None)]
    rz = fit_nb2_censored(only_lo, [[1.0]] * 5, alpha_true)
    print(f"    status       : {rz.status} (converged={rz.converged})")
    print(f"    loglik       : {rz.loglik:.6f} (supremum 0 only as mu -> inf)")
    print(f"    verdict      : {rz.message}")

    # ------------------------------------------------------------------
    # 3. Budget boundary: tail summation aborts the evaluation
    # ------------------------------------------------------------------
    print("\n[3] TAIL-BUDGET BOUNDARY (truncated tail is never exact)")
    try:
        nb2_interval_logprob(1, None, 50.0, 0.01, tail_budget=10)
    except NBFitError as exc:
        print(f"    nb2_interval_logprob raised NBFitError:\n      -> {exc}")
    # Same mechanism inside a fit: the initial evaluation aborts.
    try:
        fit_nb2_censored([(1, None), (2, None), 3], [[1.0]] * 3, 0.01,
                         offset=[4.0, 4.0, 4.0], tail_budget=5)
    except NBFitError as exc:
        print(f"    fit_nb2_censored raised NBFitError:\n      -> {exc}")

    print(f"\n(elapsed {time.perf_counter() - t0:.2f} s; all computations "
          f"real, none pre-recorded)")


if __name__ == "__main__":
    main()
