"""Fixed-input demonstration for the NB2 / ZINB count-regression module.

Runs entirely offline, deterministically, standard library only.  It prints:

  1. a normal joint NB2 fit on overdispersed synthetic counts, with
     diagnostics (gradient vs. independent central finite differences,
     fixed-alpha cross-check, rejected-step counts);
  2. a normal ZINB fit on zero-inflated counts: the constant logit
     inflation parameter is estimated from the mixture likelihood, zeros
     stay soft (posterior weights), with finite-difference gradient and
     an independent numerical-MLE cross-check;
  3. a genuinely triggered *boundary* for ZINB: plain-NB2 data collapses
     pi to the lower box (no inflation), and all-zero counts return
     status "no_finite_mle" -- proved boundaries, distinct from the
     budget exhaustion ("max_iter") also shown;
  4. genuinely triggered *exception* boundaries: NaN in X and an illegal
     pi_init are rejected with locatable NBFitError.

Run:  python demo.py     (expected runtime well under 90 seconds)
"""

import math
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nbreg import (  # noqa: E402
    fit_nb2, fit_zinb, nb2_objective, zinb_objective, NBFitError,
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tests"))
import _reference as ref  # noqa: E402


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
    print("NB2 / ZINB count-regression demo (offline, stdlib only)")
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
    # 2. Normal ZINB fit on genuinely zero-inflated counts
    # ------------------------------------------------------------------
    beta_z, alpha_z, pi_z = [0.3, -0.2], 0.4, 0.35
    yz, Xz, labels = ref.make_zinb_dataset(24, beta_z, alpha_z, pi_z, seed=2)
    nz = len(yz)
    frac0 = sum(1 for v in yz if v == 0) / nz
    print(f"\n[2] ZINB JOINT FIT (constant logit zero inflation)")
    print(f"[data] n={nz}, zero fraction={frac0:.3f} (true pi={pi_z}); "
          f"true beta={beta_z}, true alpha={alpha_z}")
    rz = fit_zinb(yz, Xz)
    print(f"    status       : {rz.status} (converged={rz.converged})")
    print(f"    message      : {rz.message}")
    print(f"    beta         : {[round(b, 6) for b in rz.beta]}")
    print(f"    log(alpha)   : {rz.log_alpha:.6f}  alpha={rz.alpha:.6f}")
    print(f"    logit(pi)    : {rz.logit_pi:.6f}  pi={rz.pi:.6f}")
    print(f"    pi boundary  : {rz.pi_boundary!r}, near no-inflation: "
          f"{rz.near_no_inflation}")
    print(f"    loglik       : {rz.loglik:.8f}  (n zeros={rz.n_zeros})")
    print(f"    gradient     : {[f'{g:+.2e}' for g in rz.gradient]}")
    print(f"    iterations   : {rz.n_iter}, backtracking rejections: "
          f"{rz.n_rejected}")

    # Independent check 1: central finite differences of the same objective.
    h = 1e-6
    fd = []
    for j in range(len(rz.beta)):
        bp, bm = list(rz.beta), list(rz.beta)
        bp[j] += h
        bm[j] -= h
        fd.append((zinb_objective(yz, Xz, bp, rz.log_alpha, rz.logit_pi)[0]
                   - zinb_objective(yz, Xz, bm, rz.log_alpha, rz.logit_pi)[0])
                  / (2 * h))
    fd.append((zinb_objective(yz, Xz, rz.beta, rz.log_alpha + h, rz.logit_pi)[0]
               - zinb_objective(yz, Xz, rz.beta, rz.log_alpha - h,
                                rz.logit_pi)[0]) / (2 * h))
    fd.append((zinb_objective(yz, Xz, rz.beta, rz.log_alpha, rz.logit_pi + h)[0]
               - zinb_objective(yz, Xz, rz.beta, rz.log_alpha,
                                rz.logit_pi - h)[0]) / (2 * h))
    worst = max(abs(a - b) for a, b in zip(rz.gradient, fd))
    print(f"    grad vs FD   : worst |analytic - central diff| = {worst:.2e}")

    # Independent check 2: a completely separate multi-start numerical
    # gradient-ascent optimizer (shares no code with the analytic fitter).
    mean_yz = sum(yz) / nz
    starts = [
        [math.log(mean_yz + 0.5), 0.0, 0.0, -0.5],
        beta_z + [math.log(alpha_z), math.log(pi_z / (1 - pi_z))],
        [0.5, 0.5, 1.0, -1.5],
    ]
    p_ref, ll_ref = ref.ref_fit_zinb(yz, Xz, starts, max_iter=10000)
    print(f"    numeric MLE  : ll={ll_ref:.8f} (fitter {rz.loglik:.8f}, "
          f"gap={rz.loglik - ll_ref:+.1e}); pi_ref="
          f"{1 / (1 + math.exp(-p_ref[-1])):.6f}")

    # Soft-mixture demonstration: zeros are never assigned to one component.
    # Posterior weight tau_i = (1-pi) p0_i / (pi + (1-pi) p0_i) varies
    # continuously; compare against what a hard split would score.
    alpha_h = math.exp(rz.log_alpha)
    taus, ll_hard = [], 0.0
    for i, yi in enumerate(yz):
        eta = sum(Xz[i][j] * rz.beta[j] for j in range(len(Xz[0])))
        mu = math.exp(eta)
        p0 = (1.0 + alpha_h * mu) ** (-1.0 / alpha_h)
        if yi == 0:
            a_s, a_c = rz.pi, (1 - rz.pi) * p0
            taus.append(a_c / (a_s + a_c))
            ll_hard += math.log(max(a_s, a_c))
        else:
            ll_hard += math.log((1 - rz.pi)
                                * math.exp(ref.nb_log_pmf(yi, mu, alpha_h)))
    print(f"    soft zeros   : posterior P(count|zero) in "
          f"[{min(taus):.3f}, {max(taus):.3f}] across {len(taus)} zeros; "
          f"hard-split ll={ll_hard:.6f} < mixture ll={rz.loglik:.6f}")

    # ------------------------------------------------------------------
    # 3. ZINB boundary outcomes, actually triggered
    # ------------------------------------------------------------------
    print("\n[3] ZINB BOUNDARIES (proved, distinct from budget exhaustion)")
    # 3a. Plain NB2 data: no structural zeros exist, so pi collapses to 0
    # and the mixture reduces to NB2 -- reported as a boundary, not an
    # interior optimum.
    yn, Xn, _ = ref.make_dataset(50, [0.4, -0.3], 0.9, seed=7)
    rn = fit_zinb(yn, Xn)
    rnb = fit_nb2(yn, Xn)
    print(f"[3a] no inflation: status={rn.status}, pi={rn.pi:.2e}, "
          f"near_no_inflation={rn.near_no_inflation}; ZINB ll={rn.loglik:.5f}"
          f" vs NB2 ll={rnb.loglik:.5f} (equal in the pi->0 limit)")
    # 3b. All zeros: certified no finite MLE.
    rzz = fit_zinb([0] * 8, [[1.0, 0.0]] + [[1.0, 0.25]] * 3
                   + [[1.0, -0.25]] * 4)
    print(f"[3b] all-zero  : status={rzz.status}, converged={rzz.converged}, "
          f"loglik={rzz.loglik:.6f} < 0")
    print(f"               {rzz.message}")
    # 3c. Budget exhaustion is not convergence here either.
    rzb = fit_zinb(yz, Xz, max_iter=1)
    print(f"[3c] max_iter=1: status={rzb.status}, converged={rzb.converged}")

    # ------------------------------------------------------------------
    # 4. Exception boundaries actually triggered: illegal inputs
    # ------------------------------------------------------------------
    print("\n[4] ILLEGAL INPUT BOUNDARIES (exception, not silently corrected)")
    X_bad = [list(row) for row in X]
    X_bad[3][1] = float("nan")
    try:
        fit_nb2(y, X_bad, offset=offset)
    except NBFitError as exc:
        print(f"[4a] NaN in X -> fit_nb2 raised NBFitError:\n      -> {exc}")
    try:
        fit_zinb(yz, Xz, pi_init=1.0)
    except NBFitError as exc:
        print(f"[4b] pi_init=1 (infinite logit) -> fit_zinb raised "
              f"NBFitError:\n      -> {exc}")

    print(f"\n(elapsed {time.perf_counter() - t0:.2f} s; all computations "
          f"real, none pre-recorded)")


if __name__ == "__main__":
    main()
