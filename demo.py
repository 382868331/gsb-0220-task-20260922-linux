"""Fixed-input demonstration for the NB2 count-regression module.

Runs entirely offline, deterministically, standard library only.  It prints:

  1. a normal joint NB2 fit on overdispersed synthetic counts, with
     diagnostics (gradient vs. independent central finite differences,
     fixed-alpha cross-check, rejected-step counts);
  2. a genuinely triggered *exception* boundary: NaN in X is rejected
     with a locatable NBFitError;
  3. a genuinely triggered *structural* boundary: all-zero counts return
     status "no_finite_mle" -- a proved boundary, distinct from a budget
     exhaustion ("max_iter"), which is also shown.

Run:  python demo.py     (expected runtime well under 8 seconds)
"""

import math
import time

from nbreg import fit_nb2, nb2_objective, NBFitError


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


def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("NB2 negative-binomial regression demo (offline, stdlib only)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Normal joint fit
    # ------------------------------------------------------------------
    y, X, offset, beta_true, alpha_true = make_data()
    mean_y = sum(y) / len(y)
    var_y = sum((v - mean_y) ** 2 for v in y) / len(y)
    print(f"\n[data] n={len(y)}, columns={len(X[0])}, mean={mean_y:.3f}, "
          f"variance={var_y:.3f} (variance/mean={var_y / mean_y:.2f} >> 1)")
    print(f"[data] true beta={beta_true}, true alpha={alpha_true}")

    res = fit_nb2(y, X, offset=offset)
    print("\n[1] JOINT FIT")
    print(f"    status       : {res.status} (converged={res.converged})")
    print(f"    message      : {res.message}")
    print(f"    beta         : {[round(b, 6) for b in res.beta]}")
    print(f"    log(alpha)   : {res.log_alpha:.6f}  alpha={res.alpha:.6f}")
    print(f"    loglik       : {res.loglik:.8f}")
    print(f"    gradient     : {[f'{g:+.2e}' for g in res.gradient]}")
    print(f"    iterations   : {res.n_iter}, backtracking rejections: "
          f"{res.n_rejected}")
    print(f"    near poisson : {res.near_poisson}")

    # Independent check: central finite differences of the SAME objective
    # (the numerical route is different from the analytic gradient code).
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
    print(f"    grad vs FD   : worst |analytic - central diff| = {worst:.2e}")

    # Fixed-alpha mode cross-check: profile likelihood must be <= joint LL.
    rf = fit_nb2(y, X, offset=offset, fixed_alpha=alpha_true)
    print(f"[1b] fixed alpha={alpha_true}: status={rf.status}, "
          f"loglik={rf.loglik:.6f} (<= joint {res.loglik:.6f}: "
          f"{rf.loglik <= res.loglik + 1e-9}), grad length={len(rf.gradient)}")

    # Budget exhaustion is reported honestly, not as convergence.
    rb = fit_nb2(y, X, offset=offset, max_iter=1)
    print(f"[1c] max_iter=1 : status={rb.status}, converged={rb.converged}, "
          f"message='{rb.message[:48]}...'")

    # ------------------------------------------------------------------
    # 2. Exception boundary actually triggered: NaN rejected, located
    # ------------------------------------------------------------------
    print("\n[2] ILLEGAL INPUT BOUNDARY (exception, not silently corrected)")
    X_bad = [list(row) for row in X]
    X_bad[3][1] = float("nan")
    try:
        fit_nb2(y, X_bad, offset=offset)
    except NBFitError as exc:
        print(f"    fit_nb2 raised NBFitError:\n      -> {exc}")

    # ------------------------------------------------------------------
    # 3. Structural boundary actually triggered: all-zero counts
    # ------------------------------------------------------------------
    print("\n[3] STRUCTURAL BOUNDARY (proved, distinct from budget exhaustion)")
    rz = fit_nb2([0] * 8, [[1.0, 0.0]] + [[1.0, 0.25]] * 3
                 + [[1.0, -0.25]] * 4)
    print(f"    all-zero y   : status={rz.status}, converged={rz.converged}")
    print(f"    loglik       : {rz.loglik:.6f} (supremum 0 unreached at "
          f"finite mu)")
    print(f"    verdict      : {rz.message}")

    print(f"\n(elapsed {time.perf_counter() - t0:.2f} s; all computations "
          f"real, none pre-recorded)")


if __name__ == "__main__":
    main()
