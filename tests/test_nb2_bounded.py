"""Tests for fit_nb2_bounded: non-negativity bounds on selected betas.

References are independent of the optimizer under test:
  * tests/_reference.py re-derives the likelihood from the gamma/Poisson
    mixture and computes central finite-difference gradients;
  * the active-set enumeration below solves each free subspace with a
    private coordinate-ascent on the reference likelihood (no package
    code), then picks the KKT-compatible candidate.

Run:  python -m unittest discover -s tests -v
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nbreg import (  # noqa: E402
    NB2BoundedResult,
    NBFitError,
    fit_nb2,
    fit_nb2_bounded,
    nb2_objective,
)
import _reference as ref  # noqa: E402


def lcg(seed):
    state = seed & 0xFFFFFFFF
    while True:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def make_negative_coef_data():
    """60 obs, true beta = [0.5, -0.8], alpha = 2 (deterministic)."""
    rng = lcg(42)
    X, y = [], []
    for i in range(60):
        row = [1.0, ((i * 5) % 11 - 5) / 5.0]
        X.append(row)
        eta = 0.5 - 0.8 * row[1]
        y.append(ref.sample_nb(math.exp(eta), 2.0, next(rng)))
    return y, X


def ref_fit_bounded_enumerate(y, X, nonneg, offset=None, tol=1e-9):
    """Independent bounded fit by enumerating every active-boundary set.

    For each subset A of the bounded columns: fix those betas to 0 and
    maximize the reference log-likelihood over the remaining betas and
    log alpha by cyclic coordinate ascent with step halving (purely
    numerical, uses only _reference.ref_loglik).  Among all candidates
    the feasible one with the highest likelihood that also satisfies the
    KKT sign condition (gradient of every fixed coordinate <= tol) is
    returned as (beta, log_alpha, loglik, active_set).
    """
    n = len(y)
    k = len(X[0])
    if offset is None:
        offset = [0.0] * n
    bounded = sorted(nonneg)
    best = None
    for mask in range(1 << len(bounded)):
        active = {bounded[b] for b in range(len(bounded)) if mask >> b & 1}
        free_b = [j for j in range(k) if j not in active]
        beta = [0.0] * k
        la = 0.0
        cur = ref.ref_loglik(y, X, beta, la, offset)
        for _ in range(4000):
            moved = False
            for j in free_b:
                h = 1e-5
                bp = list(beta)
                bp[j] += h
                up = ref.ref_loglik(y, X, bp, la, offset)
                bp[j] -= 2 * h
                dn = ref.ref_loglik(y, X, bp, la, offset)
                deriv = (up - dn) / (2 * h)
                step = 1e-2
                for _h in range(50):
                    cand = list(beta)
                    cand[j] += step if deriv > 0 else -step
                    # candidate must stay feasible
                    if all(cand[b] >= 0.0 for b in bounded):
                        val = ref.ref_loglik(y, X, cand, la, offset)
                        if val > cur + 1e-12:
                            beta, cur, moved = cand, val, True
                            break
                    step *= 0.5
            h = 1e-5
            up = ref.ref_loglik(y, X, beta, la + h, offset)
            dn = ref.ref_loglik(y, X, beta, la - h, offset)
            deriv = (up - dn) / (2 * h)
            step = 1e-2
            for _h in range(50):
                cand_la = la + (step if deriv > 0 else -step)
                val = ref.ref_loglik(y, X, beta, cand_la, offset)
                if val > cur + 1e-12:
                    la, cur, moved = cand_la, val, True
                    break
                step *= 0.5
            if not moved:
                break
        # KKT sign check on every bounded coordinate sitting at 0
        # (reference finite-difference gradient)
        g = ref.ref_fd_gradient(y, X, beta, la, offset)
        kkt_ok = all(g[b] <= 1e-4 for b in bounded if beta[b] <= 1e-9)
        if kkt_ok and (best is None or cur > best[2] + 1e-9):
            best = (beta, la, cur)
    beta, la, cur = best
    active = {b for b in bounded if beta[b] <= 1e-9}
    return beta, la, cur, active


class BoundedValidationTests(unittest.TestCase):

    def test_bad_nonneg_index_rejected(self):
        y = [1, 2, 3]
        X = [[1.0, 0.5], [1.0, -0.5], [1.0, 0.25]]
        for bad in ([2], [-1], [0, 0], [True], [0.5], "1"):
            with self.assertRaises(NBFitError, msg=f"nonneg={bad!r}"):
                fit_nb2_bounded(y, X, nonneg=bad)

    def test_rank_deficiency_still_rejected(self):
        # collinear columns: column 2 is a copy of column 1
        y = [1, 0, 3, 2]
        X = [[1.0, 0.5, 0.5], [1.0, -0.5, -0.5],
             [1.0, 0.25, 0.25], [1.0, -1.0, -1.0]]
        with self.assertRaises(NBFitError) as cm:
            fit_nb2_bounded(y, X, nonneg=[1])
        self.assertIn("rank deficient", str(cm.exception))

    def test_result_type_and_fields(self):
        y, X = make_negative_coef_data()
        res = fit_nb2_bounded(y, X, nonneg=[1])
        self.assertIsInstance(res, NB2BoundedResult)
        self.assertEqual(res.nonneg, (1,))
        self.assertEqual(res.n_obs, 60)
        self.assertEqual(res.n_params, 3)


class BoundedFitTests(unittest.TestCase):

    def test_negative_coefficient_blocked_at_zero(self):
        y, X = make_negative_coef_data()
        unb = fit_nb2(y, X)
        self.assertLess(unb.beta[1], 0.0)  # unconstrained MLE is negative
        res = fit_nb2_bounded(y, X, nonneg=[1])
        self.assertEqual(res.status, "converged")
        self.assertEqual(res.beta[1], 0.0)
        self.assertEqual(res.active_bounds, (1,))
        # bound is a genuine restriction: lower likelihood than the MLE
        self.assertLess(res.loglik, unb.loglik)
        # KKT: gradient at the active bound must not want to release
        self.assertLessEqual(res.gradient[1], 1e-8 * max(1.0, abs(res.loglik)))
        self.assertEqual(res.max_kkt_violation, 0.0)
        self.assertLessEqual(res.free_grad_norm,
                             1e-8 * max(1.0, abs(res.loglik)))

    def test_unconstrained_positive_coefficient_unaffected(self):
        rng = lcg(7)
        X, y = [], []
        for i in range(50):
            row = [1.0, ((i * 3) % 7 - 3) / 3.0]
            X.append(row)
            y.append(ref.sample_nb(math.exp(0.4 + 0.6 * row[1]), 1.5,
                                   next(rng)))
        unb = fit_nb2(y, X)
        res = fit_nb2_bounded(y, X, nonneg=[1])
        self.assertEqual(res.status, "converged")
        self.assertEqual(res.active_bounds, ())
        self.assertAlmostEqual(res.beta[1], unb.beta[1], places=6)
        self.assertAlmostEqual(res.loglik, unb.loglik, places=8)

    def test_optimum_exactly_zero(self):
        # symmetric design: the unconstrained coefficient of column 1 is
        # exactly 0, so the bounded fit must sit exactly on the bound with
        # a vanishing gradient there (KKT holds with equality).
        y = [4, 4, 4, 4]
        X = [[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, -1.0]]
        res = fit_nb2_bounded(y, X, nonneg=[1])
        self.assertEqual(res.status, "converged")
        self.assertEqual(res.beta[1], 0.0)
        self.assertEqual(res.active_bounds, (1,))
        self.assertAlmostEqual(res.gradient[1], 0.0, places=8)
        self.assertAlmostEqual(res.beta[0], math.log(4.0), places=6)

    def test_fixed_alpha_mode(self):
        y, X = make_negative_coef_data()
        res = fit_nb2_bounded(y, X, nonneg=[1], fixed_alpha=2.0)
        self.assertEqual(res.status, "converged")
        self.assertTrue(res.fixed_alpha)
        self.assertEqual(res.alpha, 2.0)
        self.assertEqual(res.beta[1], 0.0)
        self.assertEqual(len(res.gradient), 2)  # beta-only gradient
        # independent check: reference likelihood at the returned point
        # beats every nearby feasible point
        ll0 = ref.ref_loglik(y, X, res.beta, math.log(2.0))
        for db0 in (-0.05, 0.05):
            cand = [res.beta[0] + db0, 0.0]
            self.assertLessEqual(ref.ref_loglik(y, X, cand, math.log(2.0)),
                                 ll0 + 1e-9)

    def test_joint_mode_keeps_near_poisson_diagnostic(self):
        # equidispersed counts drive alpha to the Poisson boundary
        rng = lcg(99)
        X, y = [], []
        for i in range(40):
            row = [1.0, ((i * 3) % 5 - 2) / 4.0]
            X.append(row)
            mu = math.exp(0.7 + 0.3 * row[1])
            # Poisson sample by Knuth's method (alpha -> 0 limit)
            L = math.exp(-mu)
            kk, pp = 0, 1.0
            while pp > L:
                kk += 1
                pp *= next(rng)
            y.append(kk - 1)
        res = fit_nb2_bounded(y, X, nonneg=[1])
        self.assertEqual(res.status, "converged")
        self.assertTrue(res.near_poisson)
        self.assertLess(res.alpha, 1e-6)
        self.assertFalse(res.fixed_alpha)

    def test_gradient_matches_independent_finite_differences(self):
        y, X = make_negative_coef_data()
        res = fit_nb2_bounded(y, X, nonneg=[1])
        fd = ref.ref_fd_gradient(y, X, res.beta, res.log_alpha)
        for j in range(3):
            self.assertAlmostEqual(res.gradient[j], fd[j], places=4)


class ActiveSetEnumerationTests(unittest.TestCase):

    def test_matches_enumerated_active_set_reference(self):
        y, X = make_negative_coef_data()
        res = fit_nb2_bounded(y, X, nonneg=[1])
        self.assertEqual(res.status, "converged")
        beta_ref, la_ref, ll_ref, active_ref = (
            ref_fit_bounded_enumerate(y, X, [1]))
        self.assertAlmostEqual(res.loglik, ll_ref, places=4)
        self.assertEqual(set(res.active_bounds), active_ref)
        for j in range(2):
            self.assertAlmostEqual(res.beta[j], beta_ref[j], places=3)
        self.assertAlmostEqual(res.log_alpha, la_ref, places=3)

    def test_two_bounded_columns_enumeration(self):
        rng = lcg(2024)
        X, y = [], []
        for i in range(40):
            row = [1.0, ((i * 3) % 7 - 3) / 3.0, ((i * 5) % 9 - 4) / 4.0]
            X.append(row)
            eta = 0.3 - 0.5 * row[1] + 0.4 * row[2]
            y.append(ref.sample_nb(math.exp(eta), 1.0, next(rng)))
        res = fit_nb2_bounded(y, X, nonneg=[1, 2])
        self.assertEqual(res.status, "converged")
        beta_ref, la_ref, ll_ref, active_ref = (
            ref_fit_bounded_enumerate(y, X, [1, 2]))
        self.assertAlmostEqual(res.loglik, ll_ref, places=4)
        self.assertEqual(set(res.active_bounds), active_ref)


class AllZeroBoundedTests(unittest.TestCase):

    def test_all_zero_escapable_is_no_finite_mle(self):
        # free intercept: eta -> -inf is feasible, supremum not attained
        y = [0, 0, 0, 0]
        X = [[1.0, 0.5], [1.0, -0.5], [1.0, 0.25], [1.0, -1.0]]
        res = fit_nb2_bounded(y, X, nonneg=[1])
        self.assertEqual(res.status, "no_finite_mle")
        self.assertFalse(res.converged)

    def test_all_zero_blocked_is_boundary_optimum(self):
        # intercept itself bounded and X > 0: eta cannot go to -inf, the
        # optimum is the finite boundary point beta = 0
        y = [0, 0, 0]
        X = [[1.0], [1.0], [1.0]]
        res = fit_nb2_bounded(y, X, nonneg=[0])
        self.assertEqual(res.status, "converged")
        self.assertEqual(res.beta[0], 0.0)
        self.assertEqual(res.active_bounds, (0,))
        # at beta=0 with all y=0 the score is strictly negative (pushing
        # further down is infeasible): KKT holds
        self.assertLess(res.gradient[0], 0.0)
        self.assertEqual(res.max_kkt_violation, 0.0)


class OverflowAndStatusTests(unittest.TestCase):

    def test_backtracking_overflow_rejected_not_clipped(self):
        # huge regressor scale: naive Newton steps overflow exp(eta);
        # the line search must reject them and still finish feasibly
        rng = lcg(5)
        X, y = [], []
        for i in range(30):
            row = [1.0, ((i * 7) % 13 - 6) * 20.0]
            X.append(row)
            y.append(ref.sample_nb(math.exp(0.3 + 0.01 * row[1]), 1.0,
                                   next(rng)))
        res = fit_nb2_bounded(y, X, nonneg=[1], alpha_init=1e6)
        self.assertIn(res.status, ("converged", "max_iter",
                                   "line_search_failed"))
        self.assertGreater(res.n_rejected, 0)
        self.assertTrue(math.isfinite(res.loglik))
        self.assertGreaterEqual(res.beta[1], 0.0)

    def test_max_iter_is_budget_not_success(self):
        y, X = make_negative_coef_data()
        res = fit_nb2_bounded(y, X, nonneg=[1], max_iter=1)
        self.assertEqual(res.status, "max_iter")
        self.assertFalse(res.converged)


class UnboundedRegressionKeptTests(unittest.TestCase):

    def test_old_fit_entry_unchanged(self):
        # the original unconstrained entry point still works as before
        y, X = make_negative_coef_data()
        res = fit_nb2(y, X)
        self.assertEqual(res.status, "converged")
        self.assertLess(res.beta[1], 0.0)
        self.assertFalse(hasattr(res, "active_bounds"))

    def test_bounded_without_nonneg_matches_unbounded(self):
        y, X = make_negative_coef_data()
        unb = fit_nb2(y, X)
        res = fit_nb2_bounded(y, X)
        self.assertEqual(res.status, "converged")
        self.assertEqual(res.nonneg, ())
        for j in range(2):
            self.assertAlmostEqual(res.beta[j], unb.beta[j], places=8)
        self.assertAlmostEqual(res.loglik, unb.loglik, places=8)


if __name__ == "__main__":
    unittest.main()
