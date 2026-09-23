"""Tests for the NB2 fitter: recovery, boundaries, flags and error semantics."""

import math
import unittest

from nb2reg import (
    NB2InputError,
    ConvergenceBudgetError,
    NoFiniteMLEError,
    RankDeficientDesignError,
    fit_nb2,
    nb2_loglik,
)
from nb2reg.linalg import cholesky

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from reference import make_dataset, ref_loglik  # noqa: E402


def moderate_data(seed=7, n=100):
    # True beta = [0.3, 0.5], alpha = 2.0 (theta = 0.5), no offset.
    return make_dataset(
        seed, n, beta=[0.3, 0.5], alpha=2.0,
        x_sampler=lambda r: r.uniform(-1.0, 1.0),
    )


class TestFitRecovery(unittest.TestCase):
    def test_recovers_known_parameters(self):
        y, X, off = moderate_data()
        r = fit_nb2(y, X, offset=off)
        self.assertTrue(r.converged, msg=r.message)
        # n=100 MLEs have sampling error; check loose but real recovery.
        self.assertAlmostEqual(r.beta[0], 0.3, delta=0.35)
        self.assertAlmostEqual(r.beta[1], 0.5, delta=0.45)
        self.assertAlmostEqual(math.log(r.alpha), math.log(2.0), delta=0.7)
        # Stationarity of the joint gradient.
        self.assertLess(max(abs(g) for g in r.gradient), 1.0e-5)

    def test_mle_dominates_neighbouring_points(self):
        y, X, off = moderate_data()
        r = fit_nb2(y, X, offset=off)
        # The fitted point must beat random perturbations (MLE property).
        for db0, db1, dla in [
            (0.1, 0.0, 0.0), (-0.1, 0.0, 0.0), (0.0, 0.1, 0.0),
            (0.0, 0.0, 0.2), (0.0, 0.0, -0.2),
        ]:
            alt = nb2_loglik(
                y, X, offset=off,
                beta=[r.beta[0] + db0, r.beta[1] + db1],
                log_theta=math.log(r.theta) + dla,
            )
            self.assertGreaterEqual(r.loglik + 1.0e-8, alt)

    def test_fixed_alpha_mode(self):
        y, X, off = moderate_data()
        # Fixed at the truth: beta gradient zero, and it matches a
        # profile optimum within sampling noise.
        r = fit_nb2(y, X, offset=off, alpha=2.0)
        self.assertTrue(r.converged, msg=r.message)
        self.assertTrue(r.fixed_alpha)
        self.assertAlmostEqual(r.alpha, 2.0, places=10)
        self.assertEqual(len(r.gradient), 2)
        self.assertLess(max(abs(g) for g in r.gradient), 1.0e-6)

    def test_fixed_alpha_can_be_cross_checked_by_grid(self):
        # Independent profile check: at fixed alpha, analytic beta
        # optimum beats a coarse 2-D grid around it.
        y, X, off = moderate_data()
        r = fit_nb2(y, X, offset=off, alpha=2.0)
        for i in range(-2, 3):
            for j in range(-2, 3):
                b = [r.beta[0] + 0.05 * i, r.beta[1] + 0.05 * j]
                alt = ref_loglik(y, X, b, 2.0, off)
                self.assertGreaterEqual(r.loglik + 1.0e-7, alt)

    def test_vcov_positive_definite_at_convergence(self):
        y, X, off = moderate_data()
        r = fit_nb2(y, X, offset=off)
        self.assertIsNotNone(r.vcov)
        L = cholesky(r.vcov)  # raises if not PD
        self.assertEqual(len(L), 3)


class TestOffsetAndDesign(unittest.TestCase):
    def test_constant_offset_shifts_intercept(self):
        y, X, _ = moderate_data()
        c = 0.37
        r0 = fit_nb2(y, X)
        rc = fit_nb2(y, X, offset=[c] * len(y))
        self.assertTrue(rc.converged)
        # exp(X beta + c) = exp(X [beta0+c, beta1]); alpha identical.
        # The two fits terminate independently at gtol, so alpha is only
        # expected to agree to that terminal precision, not to machine
        # precision.
        self.assertAlmostEqual(rc.beta[0] + c, r0.beta[0], places=5)
        self.assertAlmostEqual(rc.beta[1], r0.beta[1], places=5)
        self.assertAlmostEqual(rc.alpha, r0.alpha, places=5)
        self.assertAlmostEqual(rc.loglik, r0.loglik, places=5)

    def test_duplicate_columns_rejected_with_locator(self):
        y = [1, 2, 3, 4]
        X = [[1.0, 1.0]] * 4
        with self.assertRaises(RankDeficientDesignError) as cm:
            fit_nb2(y, X)
        self.assertIn(1, cm.exception.dependent_columns)
        self.assertEqual(cm.exception.rank, 1)

    def test_zero_column_rejected(self):
        y = [1, 2, 3]
        X = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        with self.assertRaises(RankDeficientDesignError):
            fit_nb2(y, X)

    def test_design_without_intercept_runs(self):
        y, X, _ = moderate_data()
        X1 = [[row[1]] for row in X]
        r = fit_nb2(y, X1)
        self.assertTrue(r.converged, msg=r.message)


class TestBoundaries(unittest.TestCase):
    def test_all_zero_with_intercept_is_provably_no_mle(self):
        with self.assertRaises(NoFiniteMLEError):
            fit_nb2([0, 0, 0, 0])
        with self.assertRaises(NoFiniteMLEError):
            fit_nb2([0, 0], X=[[1.0, 0.5], [1.0, -0.5]])

    def test_all_zero_without_intercept_is_not_falsely_declared(self):
        # Full-rank design with no intercept and rows of opposite sign:
        # at beta=0 all mu=1, score beta: sum (y-mu)*theta/(mu+theta)*x =
        # -theta/2*(1-1)=0, so beta=0 is a genuine finite interior
        # stationary point.  No impossibility proof applies; the fitter
        # runs and reports an honest status.
        r = fit_nb2([0, 0], X=[[1.0], [-1.0]], max_iter=20)
        self.assertIn(r.status,
                      ("converged", "converged_boundary", "max_iter",
                       "step_failure", "boundary_theta_min"))

    def test_near_poisson_data_flags_boundary(self):
        # Effectively Poisson counts: alpha estimate driven to its box.
        rng = __import__("random").Random(11)

        def pois(lam):
            b = math.exp(-lam); k = 0; p = 1.0
            while p > b:
                k += 1; p *= rng.random()
            return k - 1

        y = [pois(2.0) for _ in range(40)]
        r = fit_nb2(y)
        self.assertTrue(r.near_poisson)
        self.assertTrue(r.converged, msg=r.message)
        self.assertEqual(r.status, "converged_boundary")
        self.assertTrue(r.alpha_at_boundary)

    def test_high_overdispersion_converges(self):
        # Bimodal log-rates => very high dispersion but finite MLE.
        rng = __import__("random").Random(3)
        def pois(lam):
            b = math.exp(-min(lam, 700)); k = 0; p = 1.0
            while p > b:
                k += 1; p *= rng.random()
            return k - 1
        y = [pois(math.exp(rng.choice([-4.0, 4.0]))) for _ in range(80)]
        r = fit_nb2(y)
        self.assertTrue(r.converged, msg=r.message)
        self.assertGreater(r.alpha, 3.0)
        self.assertLess(max(abs(g) for g in r.gradient), 1.0e-6)

    def test_budget_exhaustion_is_distinct_from_impossibility(self):
        r = fit_nb2([1, 100, 1, 100], max_iter=1)
        self.assertFalse(r.converged)
        self.assertEqual(r.status, "max_iter")
        self.assertEqual(r.iterations, 1)
        with self.assertRaises(ConvergenceBudgetError):
            r.raise_if_not_converged()

    def test_extreme_counts_report_honestly_not_fake_convergence(self):
        # Means beyond the representable eta edge: no false "converged".
        r = fit_nb2([10**220, 10**220], max_iter=15)
        self.assertFalse(r.converged)
        self.assertIn(r.status, ("max_iter", "step_failure"))


class TestInputValidation(unittest.TestCase):
    def test_nan_infinity_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(NB2InputError):
                fit_nb2([1, bad, 2])
            with self.assertRaises(NB2InputError):
                fit_nb2([1, 2], X=[[1.0, bad], [1.0, 0.0]])
            with self.assertRaises(NB2InputError):
                fit_nb2([1, 2], offset=[bad, 0.0])

    def test_bool_rejected_for_integer_control_parameter(self):
        with self.assertRaises(NB2InputError):
            fit_nb2([1, 2], max_iter=True)
        with self.assertRaises(NB2InputError):
            fit_nb2([1, 2], max_iter=False)

    def test_bool_counts_rejected(self):
        with self.assertRaises(NB2InputError):
            fit_nb2([True, 2])

    def test_negative_and_fractional_counts_rejected(self):
        with self.assertRaises(NB2InputError):
            fit_nb2([1, -2])
        with self.assertRaises(NB2InputError):
            fit_nb2([1, 2.5])

    def test_size_and_shape_limits(self):
        with self.assertRaises(NB2InputError):
            fit_nb2([1] * 101)
        with self.assertRaises(NB2InputError):
            fit_nb2([])
        X = [[1.0] * 6 for _ in range(3)]
        with self.assertRaises(NB2InputError):
            fit_nb2([1, 2, 3], X=X)
        with self.assertRaises(NB2InputError):
            fit_nb2([1, 2], X=[[1.0], [1.0, 2.0]])
        with self.assertRaises(NB2InputError):
            fit_nb2([1, 2], offset=[0.0])

    def test_alpha_validation(self):
        for bad in (0.0, -1.0, float("inf"), float("nan"), True):
            with self.assertRaises(NB2InputError):
                fit_nb2([1, 2], alpha=bad)

    def test_tolerance_validation(self):
        with self.assertRaises(NB2InputError):
            fit_nb2([1, 2], gtol=0)
        with self.assertRaises(NB2InputError):
            fit_nb2([1, 2], gtol=True)
        with self.assertRaises(NB2InputError):
            fit_nb2([1, 2], step_tol=-1.0)

    def test_error_is_value_error_and_localisable(self):
        try:
            fit_nb2([1, float("nan")])
        except NB2InputError as exc:
            self.assertIsInstance(exc, ValueError)
            self.assertIn("y[1]", str(exc))
        else:  # pragma: no cover
            self.fail("expected NB2InputError")


if __name__ == "__main__":
    unittest.main()
