"""Tests for the NB2 negative-binomial regression module.

Expected values come from tests/_reference.py, which never imports the
package under test: lgamma-formula likelihood, pmf-ratio recurrences,
central finite-difference gradients and an independent numerical-gradient
fixed-alpha optimizer.  Round-trip properties are only supplementary.

Run:  python -m unittest discover -s tests -v
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nbreg import (  # noqa: E402
    fit_nb2, nb2_loglik, nb2_objective, nb2_pmf, NBFitError,
    LOG_ALPHA_MIN,
)
from nbreg.nb2 import _digamma, _trigamma  # noqa: E402
import _reference as ref  # noqa: E402


GAMMA = 0.5772156649015329


def ones(n):
    return [[1.0] for _ in range(n)]


class SpecialFunctionsTests(unittest.TestCase):

    def test_digamma_known_values(self):
        self.assertAlmostEqual(_digamma(1.0), -GAMMA, places=12)
        self.assertAlmostEqual(_digamma(0.5), -GAMMA - 2.0 * math.log(2.0),
                               places=12)
        self.assertAlmostEqual(_digamma(10.0),
                               -GAMMA + sum(1.0 / m for m in range(1, 10)),
                               places=12)

    def test_trigamma_known_values(self):
        self.assertAlmostEqual(_trigamma(1.0), math.pi ** 2 / 6.0, places=12)
        self.assertAlmostEqual(_trigamma(0.5), math.pi ** 2 / 2.0, places=12)
        self.assertAlmostEqual(_trigamma(3.0),
                               math.pi ** 2 / 6.0 - 1.0 - 0.25, places=12)

    def test_trigamma_is_derivative_of_digamma(self):
        # Trigamma must equal the numeric derivative of digamma.
        for x in (0.07, 0.6, 1.5, 8.3):
            h = 1e-6
            fd = (_digamma(x + h) - _digamma(x - h)) / (2 * h)
            self.assertAlmostEqual(_trigamma(x), fd, delta=2e-7)


class InputValidationTests(unittest.TestCase):

    def test_nan_infinity_rejected_with_location(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            X = [[1.0, 0.0], [1.0, bad]]
            with self.assertRaisesRegex(NBFitError, r"X\[1\]\[1\]"):
                fit_nb2([1, 2], X)
        with self.assertRaisesRegex(NBFitError, r"offset\[0\]"):
            fit_nb2([1, 2], ones(2), offset=[float("nan"), 0.0])

    def test_counts_must_be_int_and_nonneg(self):
        with self.assertRaisesRegex(NBFitError, r"y\[1\]"):
            fit_nb2([1, 2.0], ones(2))
        with self.assertRaisesRegex(NBFitError, r"y\[0\].*bool"):
            fit_nb2([True, 2], ones(2))
        with self.assertRaisesRegex(NBFitError, r"y\[0\].*non-negative"):
            fit_nb2([-1, 2], ones(2))

    def test_integer_arguments_reject_bool(self):
        with self.assertRaisesRegex(NBFitError, r"max_iter.*bool"):
            fit_nb2([1, 2, 3], ones(3), max_iter=True)

    def test_shape_errors(self):
        with self.assertRaisesRegex(NBFitError, r"n=2"):
            fit_nb2([1, 2], ones(3))
        with self.assertRaisesRegex(NBFitError, r"1\.\.100"):
            fit_nb2([], [])
        X = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3
        with self.assertRaisesRegex(NBFitError, r"1\.\.5"):
            fit_nb2([1, 2, 3], X)
        with self.assertRaisesRegex(NBFitError, r"row 1"):
            fit_nb2([1, 2, 3], [[1.0, 0.0], [1.0], [1.0, 0.0]])

    def test_duplicate_and_zero_columns_rejected_not_corrected(self):
        y = [1, 2, 3, 4]
        Xdup = [[1.0, 1.0] for _ in y]
        with self.assertRaisesRegex(NBFitError, r"column 1"):
            fit_nb2(y, Xdup)
        Xzero = [[1.0, 0.0] for _ in y]
        with self.assertRaisesRegex(NBFitError, r"column 1"):
            fit_nb2(y, Xzero)

    def test_bad_fixed_alpha(self):
        with self.assertRaisesRegex(NBFitError, r"fixed_alpha.*positive"):
            fit_nb2([1, 2], ones(2), fixed_alpha=0.0)
        with self.assertRaisesRegex(NBFitError, r"fixed_alpha.*bool"):
            fit_nb2([1, 2], ones(2), fixed_alpha=True)

    def test_inputs_are_not_mutated(self):
        y, X, off = [0, 3, 7], [[1.0, 0.5], [1.0, -0.5], [1.0, 0.2]], None
        ycopy, Xcopy = list(y), [list(r) for r in X]
        fit_nb2(y, X)
        self.assertEqual(y, ycopy)
        self.assertEqual(X, Xcopy)

    def test_overflow_initial_point_raises_and_names_guard(self):
        # Valid finite numbers, but no intercept column can absorb the
        # offsets: starting eta is +/-1000, outside the computable band.
        # Data are not clipped -- a locatable error is raised instead.
        with self.assertRaisesRegex(NBFitError, r"computable region"):
            fit_nb2([1, 2], [[1.0, 0.5], [1.0, -0.5]],
                    offset=[1000.0, 3000.0])


class GradientTests(unittest.TestCase):
    """Analytic gradient vs central finite differences of an independently
    written reference likelihood, at arbitrary non-optimal points."""

    CASES = [
        # (y, X, offset, beta, log_alpha, label)
        ([0, 1, 0, 5, 2],
         [[1.0, 0.0], [1.0, 1.0], [1.0, -1.0], [1.0, 0.5], [1.0, -0.5]],
         None, [0.3, -0.2], math.log(0.7), "mixed zeros"),
        ([12, 0, 40, 3, 0, 9],
         [[1.0, -0.5, 0.25], [1.0, 0.5, -0.25], [1.0, 0.0, 0.1],
          [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.25, -0.5]],
         [0.1, -0.2, 0.0, 0.3, -0.1, 0.0],
         [-0.4, 0.25, 0.1], math.log(3.0), "offset, high alpha"),
        ([0, 0, 1], [[1.0], [1.0], [1.0]], [0.5, -0.5, 0.0],
         [0.1], math.log(0.05), "near poisson, all-but-one zero"),
    ]

    def test_gradient_matches_finite_differences(self):
        for y, X, off, beta, la, label in self.CASES:
            with self.subTest(label):
                ll, g = nb2_objective(y, X, beta, la, off)
                fd = ref.ref_fd_gradient(y, X, beta, la, off)
                self.assertAlmostEqual(
                    ll, ref.ref_loglik(y, X, beta, la, off),
                    delta=1e-10 * max(1.0, abs(ll)))
                for a, b in zip(g, fd):
                    self.assertAlmostEqual(a, b, delta=3e-5)

    def test_gradient_at_fit_is_zero_joint(self):
        y, X, _ = ref.make_dataset(50, [0.4, -0.3], 0.9, seed=7)
        res = fit_nb2(y, X)
        self.assertEqual(res.status, "converged")
        for v in res.gradient:
            self.assertLess(abs(v), 1e-6)


class PMFAndTruncationTests(unittest.TestCase):

    def test_pmf_matches_independent_recurrence(self):
        for mu, alpha in ((2.0, 0.5), (0.3, 2.0), (7.0, 0.01), (20.0, 4.0)):
            rec = ref.nb_pmf_recursion(mu, alpha, 60)
            for k, p_rec in enumerate(rec):
                p_core = nb2_pmf(k, mu, alpha)
                self.assertAlmostEqual(p_core, p_rec,
                                       delta=1e-12 * max(1.0, p_rec))

    def test_pmf_normalizes(self):
        for mu, alpha in ((2.0, 0.5), (0.3, 2.0), (7.0, 0.01)):
            total = sum(nb2_pmf(k, mu, alpha) for k in range(2000))
            self.assertAlmostEqual(total, 1.0, delta=1e-9)

    def test_truncated_probability_sums(self):
        # Two independent routes (core lgamma pmf and reference recurrence)
        # must agree on the truncated mass P(Y <= K) and on the conditional
        # truncated pmf renormalized over 0..K.
        mu, alpha, K = 3.0, 1.5, 8
        p_core = [nb2_pmf(k, mu, alpha) for k in range(K + 1)]
        p_rec = ref.nb_pmf_recursion(mu, alpha, K)
        c_core, c_rec = sum(p_core), sum(p_rec)
        self.assertAlmostEqual(c_core, c_rec, delta=1e-12)
        self.assertLess(c_core, 1.0)  # genuinely truncated
        q_core = [p / c_core for p in p_core]
        q_rec = [p / c_rec for p in p_rec]
        for a, b in zip(q_core, q_rec):
            self.assertAlmostEqual(a, b, delta=1e-12)
        self.assertAlmostEqual(sum(q_core), 1.0, delta=1e-12)
        # Truncated mean against the two routes:
        m_core = sum(k * q for k, q in enumerate(q_core))
        m_rec = sum(k * q for k, q in enumerate(q_rec))
        self.assertAlmostEqual(m_core, m_rec, delta=1e-12)

    def test_pmf_approaches_poisson(self):
        mu = 2.5
        for k in range(8):
            self.assertAlmostEqual(nb2_pmf(k, mu, 1e-9),
                                   math.exp(ref.poisson_log_pmf(k, mu)),
                                   delta=1e-7)

    def test_pmf_input_validation(self):
        with self.assertRaisesRegex(NBFitError, r"int"):
            nb2_pmf(1.0, 2.0, 1.0)
        with self.assertRaisesRegex(NBFitError, r"non-negative"):
            nb2_pmf(-1, 2.0, 1.0)
        with self.assertRaisesRegex(NBFitError, r"mu.*positive"):
            nb2_pmf(0, 0.0, 1.0)


class FixedAlphaTests(unittest.TestCase):

    def test_fixed_alpha_matches_independent_optimizer(self):
        y, X, _ = ref.make_dataset(12, [0.3, -0.4], 1.2, seed=11)
        alpha = 0.8
        res = fit_nb2(y, X, fixed_alpha=alpha)
        self.assertTrue(res.converged)
        b_ref, ll_ref = ref.ref_fit_beta_fixed_alpha(y, X, alpha)
        self.assertEqual(len(res.gradient), len(X[0]))
        for a, b in zip(res.beta, b_ref):
            self.assertAlmostEqual(a, b, delta=2e-4)
        self.assertAlmostEqual(res.loglik, ll_ref, delta=1e-6)
        # Fixed alpha must be reported verbatim.
        self.assertEqual(res.alpha, alpha)
        self.assertTrue(res.fixed_alpha)

    def test_fixed_alpha_offset_absorption(self):
        # A constant offset c must shift the fitted intercept by -c.
        y = [2, 5, 1, 4, 3]
        c = 0.7
        r1 = fit_nb2(y, ones(len(y)), fixed_alpha=0.6)
        r2 = fit_nb2(y, ones(len(y)), fixed_alpha=0.6,
                     offset=[c] * len(y))
        self.assertAlmostEqual(r1.beta[0] - c, r2.beta[0], delta=1e-8)
        self.assertAlmostEqual(r1.loglik, r2.loglik, delta=1e-10)


class AllZeroTests(unittest.TestCase):

    def test_all_zero_is_certified_no_finite_mle(self):
        res = fit_nb2([0, 0, 0, 0], [[1.0, 0.2], [1.0, -0.2],
                                     [1.0, 0.5], [1.0, -0.5]])
        self.assertFalse(res.converged)
        self.assertEqual(res.status, "no_finite_mle")
        self.assertLess(res.loglik, 0.0)  # strictly below supremum 0
        # same structural verdict in fixed-alpha mode
        res2 = fit_nb2([0, 0], ones(2), fixed_alpha=1.0)
        self.assertEqual(res2.status, "no_finite_mle")
        self.assertFalse(res2.converged)

    def test_zero_counts_partial_still_fits(self):
        res = fit_nb2([0, 0, 3, 0, 5], ones(5))
        self.assertEqual(res.status, "converged")


class FitQualityTests(unittest.TestCase):

    def test_recovers_truth_and_beats_true_params(self):
        beta_true, alpha_true = [0.5, -0.35], 0.8
        y, X, _ = ref.make_dataset(80, beta_true, alpha_true, seed=23)
        res = fit_nb2(y, X)
        self.assertEqual(res.status, "converged")
        self.assertTrue(all(math.isfinite(v) for v in res.beta))
        self.assertTrue(math.isfinite(res.log_alpha))
        # Fitted likelihood must not be below the likelihood at truth.
        ll_true = ref.ref_loglik(y, X, beta_true, math.log(alpha_true))
        self.assertGreaterEqual(res.loglik + 1e-8, ll_true)
        # Recovery on this deterministic sample.
        self.assertAlmostEqual(res.beta[0], beta_true[0], delta=0.35)
        self.assertAlmostEqual(res.beta[1], beta_true[1], delta=0.35)
        self.assertAlmostEqual(math.log(res.alpha),
                               math.log(alpha_true), delta=0.9)

    def test_high_dispersion(self):
        beta_true, alpha_true = [0.2, 0.4], 6.0
        y, X, _ = ref.make_dataset(80, beta_true, alpha_true, seed=16)
        self.assertGreater(max(y), 30)  # genuinely overdispersed sample
        res = fit_nb2(y, X)
        self.assertEqual(res.status, "converged")
        self.assertGreater(res.alpha, 1.0)
        for a, b in zip(res.beta, beta_true):
            self.assertAlmostEqual(a, b, delta=0.7)

    def test_saturated_poisson_boundary_flagged(self):
        # Identical counts, intercept only: Poisson saturates and the NB
        # likelihood increases as alpha -> 0, so the fit ends at the lower
        # alpha bound and must be flagged near-Poisson, not reported as an
        # unconstrained interior optimum.
        res = fit_nb2([3] * 20, ones(20))
        self.assertTrue(res.near_poisson)
        self.assertLessEqual(res.log_alpha, LOG_ALPHA_MIN + 1e-7)
        self.assertAlmostEqual(res.beta[0], math.log(3.0), delta=1e-6)


class StatusSeparationTests(unittest.TestCase):

    def test_budget_exhaustion_distinct_from_no_solution(self):
        y, X, _ = ref.make_dataset(40, [0.4, -0.3], 1.0, seed=7)
        res = fit_nb2(y, X, max_iter=1)
        self.assertFalse(res.converged)
        self.assertEqual(res.status, "max_iter")
        self.assertIn("budget", res.message)

    def test_loglik_pure_function_matches_fit(self):
        y = [1, 4, 0, 7, 2, 3]
        X = [[1.0, 0.5 * (i % 3 - 1)] for i in range(6)]
        res = fit_nb2(y, X)
        self.assertAlmostEqual(
            nb2_loglik(y, X, res.beta, res.log_alpha), res.loglik,
            delta=1e-10)

    def test_rejected_steps_counted(self):
        # A deliberately awkward start (large alpha init) forces rejected
        # or damped steps; the counter must be non-negative and reported.
        y, X, _ = ref.make_dataset(30, [0.3, 0.2], 0.4, seed=3)
        res = fit_nb2(y, X, alpha_init=10.0)
        self.assertGreaterEqual(res.n_rejected, 0)
        self.assertEqual(res.status, "converged")


if __name__ == "__main__":
    unittest.main(verbosity=2)
