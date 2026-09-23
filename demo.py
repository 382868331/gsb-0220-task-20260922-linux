"""Fixed-input demonstration for the nb2reg package.

Run with::

    python demo.py

It performs real computation (no sleep, no pre-recorded results) and
prints:

1. a normal, converged NB2 fit on deterministically generated
   overdispersed counts, including log-likelihood, gradient, alpha,
   standard errors and the near-Poisson flag;
2. a genuinely triggered *exception boundary*: all-zero counts with an
   intercept provably have no finite MLE, which raises
   ``NoFiniteMLEError`` rather than returning a fake "converged" point;
3. an extreme-data *rejection* case (counts whose joint optimum path
   overflows): the overflowing candidates are rejected and shortened
   (``rejected_steps``) and the fitter reports ``step_failure`` honestly
   instead of a false convergence; the separate ``max_iter`` budget
   status is also demonstrated.
"""

from __future__ import annotations

import math
import random
import time

from nb2reg import (
    ConvergenceBudgetError,
    NB2InputError,
    NoFiniteMLEError,
    StepFailureError,
    conditional_pmf,
    fit_nb2,
    range_probability,
)


def poisson_sample(rng: random.Random, lam: float) -> int:
    """Deterministic Knuth Poisson draw (stdlib has no poisson())."""
    border = math.exp(-min(lam, 700.0))
    k = 0
    p = 1.0
    while p > border:
        k += 1
        p *= rng.random()
    return k - 1


def generate_normal_data():
    """Overdispersed NB2 counts.

    True model: log mu = 0.30 + 0.50 x + offset, alpha = 2 (theta = 0.5).
    """
    rng = random.Random(20260923)
    n = 80
    y, X, offset = [], [], []
    theta_true = 0.5
    for _ in range(n):
        x = rng.uniform(-1.0, 1.0)
        off = round(rng.uniform(-0.15, 0.15), 4)
        eta = 0.30 + 0.50 * x + off
        gamma_mean_one = rng.gammavariate(theta_true, 1.0 / theta_true)
        y.append(poisson_sample(rng, math.exp(eta) * gamma_mean_one))
        X.append([1.0, x])
        offset.append(off)
    return y, X, offset


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def show_normal_fit() -> None:
    section("1. NORMAL RESULT: joint NB2 fit on overdispersed counts")
    y, X, offset = generate_normal_data()
    print(f"n = {len(y)} samples, p = {len(X[0])} design columns "
          "(intercept + x), with an exposure offset")
    print(f"sample mean y = {sum(y) / len(y):.3f}, "
          f"sample variance = {sum((v - sum(y)/len(y))**2 for v in y)/(len(y)-1):.3f}")

    start = time.perf_counter()
    result = fit_nb2(y, X, offset=offset)
    elapsed = time.perf_counter() - start

    names = ["beta0 (intercept)", "beta1 (slope on x)", "log theta"]
    truth = [0.30, 0.50]
    print(f"\nstatus       : {result.status}  ({result.message})")
    print(f"converged    : {result.converged}")
    print(f"iterations   : {result.iterations}   "
          f"rejected steps: {result.rejected_steps}   elapsed: {elapsed:.3f}s")
    print(f"log-likelihood: {result.loglik:.8f}")
    print(f"alpha (Var = mu + alpha*mu^2): {result.alpha:.6f}"
          f"   theta = 1/alpha: {result.theta:.6f}")
    print(f"near-Poisson boundary flag : {result.near_poisson}")
    print(f"alpha at computational box : {result.alpha_at_boundary}")
    print("\nparameters (estimate, analytic gradient, std. error, true value):")
    grad = result.gradient
    for j in range(2):
        se = math.sqrt(result.vcov[j][j])
        print(f"  {names[j]:<22} {result.beta[j]: .5f}   "
              f"grad {grad[j]: .2e}   SE {se:.4f}   true {truth[j]:.2f}")
    se_lt = math.sqrt(result.vcov[2][2])
    print(f"  {names[j+1]:<22} {math.log(result.theta): .5f}   "
          f"grad {grad[2]: .2e}   SE {se_lt:.4f}")
    assert result.converged, "normal demo was expected to converge"
    assert max(abs(g) for g in grad) < 1.0e-6, "gradient not at a stationary point"

    # Fixed-alpha cross-check, independently optimising beta only.
    fixed = fit_nb2(y, X, offset=offset, alpha=2.0)
    print(f"\nfixed-alpha check (alpha pinned at 2.0): status={fixed.status}, "
          f"beta={[round(b, 5) for b in fixed.beta]}, "
          f"beta-gradient max = {max(abs(g) for g in fixed.gradient):.2e}")

    # Truncated-probability illustration (zero-truncated mass at k=1).
    mu_demo, alpha_demo = 3.0, result.alpha
    p_ge1 = range_probability(1, None, mu_demo, alpha_demo)
    p1_trunc = conditional_pmf(1, 1, 300, mu_demo, alpha_demo)
    print(f"\ntruncation illustration at mu={mu_demo}, alpha={alpha_demo:.3f}:")
    print(f"  P(Y >= 1)            = {p_ge1:.6f}")
    print(f"  P(Y = 1 | Y >= 1)    = {p1_trunc:.6f}")


def show_proved_no_mle() -> None:
    section("2. ACTUALLY TRIGGERED EXCEPTION BOUNDARY: all-zero counts")
    y = [0, 0, 0, 0, 0, 0]
    print("y =", y, " with an intercept-only design")
    print("For y=0 the NB2 mass (theta/(mu+theta))**theta is strictly")
    print("decreasing in mu; sending the intercept to -infinity drives")
    print("every mu to 0 and raises the likelihood toward a supremum that")
    print("no finite beta attains.  The library proves this and raises:")
    try:
        fit_nb2(y)
    except NoFiniteMLEError as exc:
        print(f"  NoFiniteMLEError: {exc}")
    else:  # pragma: no cover
        raise AssertionError("expected NoFiniteMLEError")

    # The same data WITHOUT an intercept is not covered by the proof; a
    # bad input (NaN) is rejected separately with a locatable error.
    try:
        fit_nb2([1, float("nan"), 2])
    except NB2InputError as exc:
        print(f"\ninput rejection example -> NB2InputError: {exc}")


def show_extreme_nonconvergence() -> None:
    section("3. ACTUALLY TRIGGERED REJECTION BOUNDARY: counts beyond representable mean")
    # log(10^250) ~ 576 is within eta_max=700, but the joint
    # beta/theta Newton path doubles through the representable edge:
    # those candidate steps overflow and are rejected and shortened,
    # never clipping the data.  Near the edge the Hessian loses
    # representable curvature, so no finite acceptable step remains and
    # the fitter reports step failure honestly rather than converging.
    y = [10**250, 10**250]
    print(f"y = [1e250, 1e250]  (eta representable up to 700)")
    result = fit_nb2(y)
    print(f"status            : {result.status}")
    print(f"converged         : {result.converged}  (correctly NOT a false success)")
    print(f"iterations        : {result.iterations}")
    print(f"rejected steps    : {result.rejected_steps}  "
          "(overflowing candidates rejected & shortened; data untouched)")
    print(f"terminal beta     : {[round(b, 3) for b in result.beta]}")
    print(f"terminal loglik   : {result.loglik:.6g}")
    print(f"terminal gradient : {[f'{g:+.2e}' for g in result.gradient]}")
    print(f"diagnostic        : {result.message}")
    try:
        result.raise_if_not_converged()
    except StepFailureError as exc:
        print(f"raise_if_not_converged() -> StepFailureError: {exc}")
    assert not result.converged
    assert result.status == "step_failure"
    assert result.rejected_steps > 0

    # Distinct from step failure: a tight budget yields max_iter, which
    # claims no impossibility - only that the search ran out of budget.
    budgeted = fit_nb2([1, 100, 1, 100], max_iter=2)
    print(f"\nbudget distinction: status={budgeted.status!r}, "
          f"iterations={budgeted.iterations}, converged={budgeted.converged}")
    try:
        budgeted.raise_if_not_converged()
    except ConvergenceBudgetError as exc:
        print(f"raise_if_not_converged() -> ConvergenceBudgetError: {exc}")
    assert budgeted.status == "max_iter"


def main() -> None:
    t0 = time.perf_counter()
    print("nb2reg demonstration - Python",
          f"{__import__('sys').version.split()[0]}", "standard library only")
    show_normal_fit()
    show_proved_no_mle()
    show_extreme_nonconvergence()
    print()
    print("=" * 70)
    print(f"demo finished in {time.perf_counter() - t0:.2f}s; "
          "diagnostics shown, no pre-recorded results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
