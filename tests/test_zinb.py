"""Tests for the ZINB (zero-inflated NB2) module.

Expected values come from tests/_reference.py, which never imports the
package under test: the ZINB mixture pmf/log-likelihood is re-derived from
the two-component mixture definition, gradients by central finite
differences, and an independent multi-start numerical-gradient optimizer
gives a fully separate MLE.  One likelihood is also checked against a
hand-computed two-observation example.

Run:  python -m unittest discover -s tests -v
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nbreg import (  # noqa: E402
    fit_nb2,
    fit_zinb,
    nb2_pmf,
    zinb_loglik,
    zinb_objective,
    zinb_pmf,
    NBFitError,
    ZINBFitError,
    LOGIT_PI_MIN,
)
import _reference as ref  # noqa: E402


def ones(n):
    return [[1.0] for _ in range(n)]


class ZINBHandComputedTests(unittest.TestCase):

    def test_loglik_hand_computed_two_obs(self):
        # mu = 1, alpha = 1 (geometric): p0 = 1/2, f(2) = 1/8.
        # pi = 0.3:  m(0) = 0.3 + 0.7*0.5 = 0.65
        #             m(2) = 0.7*0.125    = 0.0875
        z = math.log(0.3 / 0.7)
        ll = zinb_loglik([0, 2], [[1.0], [1.0]], [0.0], 0.0, z)
        expected = math.log(0.65) + math.log(0.0875)
        self.assertAlmostEqual(ll, expected, delta=1e-12)

    def test_pmf_endpoints_hand_values(self):
        self.assertAlmostEqual(zinb_pmf(0, 1.0, 1.0, 0.3), 0.65, delta=1e-12)
        self.assertAlmostEqual(zinb_pmf(2, 1.0, 1.0, 0.3), 0.0875, delta=1e-12)
        # pi = 0 is exactly the NB2 mass; pi = 1 is point mass at zero.
        self.assertEqual(zinb_pmf(3, 2.0, 0.5, 0.0), nb2_pmf(3, 2.0, 0.5))
        self.assertEqual(zinb_pmf(0, 2.0, 0.5, 1.0), 1.0)
        self.assertEqual(zinb_pmf(1, 2.0, 0.5, 1.0), 0.0)


class ZINBGradientTests(unittest.TestCase):
    """Analytic gradient vs central finite differences of the independently
    written reference mixture likelihood, at arbitrary non-optimal points,
    including all-zero and high-inflation points."""

    CASES = [
        ([0, 1, 0, 5, 2],
         [[1.0, 0.0], [1.0, 1.0], [1.0, -1.0], [1.0, 0.5], [1.0, -0.5]],
         None, [0.3, -0.2], math.log(0.7), 0.5),
        ([12, 0, 40, 3, 0, 9],
         [[1.0, -0.5, 0.25], [1.0, 0.5, -0.25], [1.0, 0.0, 0.1],
          [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.25, -0.5]],
         [0.1, -0.2, 0.0, 0.3, -0.1, 0.0],
         [-0.4, 0.25, 0.1], math.log(3.0), -1.5),
        ([0, 0, 0, 1], ones(4), None, [0.2], math.log(0.05), 2.0),
    ]

    def test_gradient_matches_reference_finite_differences(self):
        for y, X, off, beta, la, z in self.CASES:
            with self.subTest(z=z):
                ll, g = zinb_objective(y, X, beta, la, z, off)
                fd = ref.ref_zinb_fd_gradient(y, X, beta, la, z, off)
                self.assertAlmostEqual(
                    ll, ref.ref_zinb_loglik(y, X, beta, la, z, off),
                    delta=1e-10 * max(1.0, abs(ll)))
                self.assertEqual(len(g), len(beta) + 2)
                for a, b in zip(g, fd):
                    self.assertAlmostEqual(a, b, delta=3e-5)

    def test_gradient_at_fit_is_small(self):
        y, X, _ = ref.make_zinb_dataset(40, [0.3, -0.2], 0.3, 0.4, seed=5)
        res = fit_zinb(y, X)
        self.assertEqual(res.status, "converged")
        for v in res.gradient:
            self.assertLess(abs(v), 1e-6)


class ZINBPMFTests(unittest.TestCase):

    def test_pmf_matches_independent_mass_route(self):
        for mu, alpha, pi in ((2.0, 0.5, 0.3), (0.3, 2.0, 0.6),
                              (7.0, 0.01, 0.2), (20.0, 4.0, 0.45)):
            for k in range(30):
                self.assertAlmostEqual(
                    zinb_pmf(k, mu, alpha, pi),
                    ref.zinb_mass(k, mu, alpha, pi),
                    delta=1e-12 * max(1.0, ref.zinb_mass(k, mu, alpha, pi)))

    def test_pmf_normalizes(self):
        for mu, alpha, pi in ((2.0, 0.5, 0.3), (0.3, 2.0, 0.6),
                              (7.0, 0.01, 0.2)):
            total = sum(zinb_pmf(k, mu, alpha, pi) for k in range(2000))
            self.assertAlmostEqual(total, 1.0, delta=1e-9)

    def test_pmf_input_validation(self):
        with self.assertRaisesRegex(NBFitError, r"int"):
            zinb_pmf(1.0, 2.0, 1.0, 0.3)
        with self.assertRaisesRegex(NBFitError, r"non-negative"):
            zinb_pmf(-1, 2.0, 1.0, 0.3)
        with self.assertRaisesRegex(NBFitError, r"mu.*positive"):
            zinb_pmf(0, 0.0, 1.0, 0.3)
        with self.assertRaisesRegex(NBFitError, r"alpha.*positive"):
            zinb_pmf(0, 2.0, -1.0, 0.3)
        for bad in (-0.01, 1.01, float("nan"), True):
            with self.assertRaisesRegex(NBFitError, r"pi"):
                zinb_pmf(0, 2.0, 1.0, bad)


class ZINBNoStructuralZeroTests(unittest.TestCase):
    """Data drawn from plain NB2 (no inflation component): the ZINB MLE
    must collapse pi toward the lower boundary, and the limiting likelihood
    must coincide with the NB2 likelihood -- the model nests NB2."""

    def test_pi_collapses_and_likelihood_equals_nb2(self):
        y, X, _ = ref.make_dataset(50, [0.4, -0.3], 0.9, seed=7)
        res = fit_zinb(y, X)
        self.assertEqual(res.status, "converged")
        self.assertLess(res.pi, 1e-5)
        self.assertTrue(res.near_no_inflation)
        # At pi = 0 the mixture likelihood IS the NB2 likelihood, checked
        # through the independent reference formula.
        rb = fit_nb2(y, X)
        ll_nb2 = ref.ref_loglik(y, X, rb.beta, rb.log_alpha)
        ll_zinb_at_zero = ref.ref_zinb_loglik(
            y, X, res.beta, res.log_alpha, LOGIT_PI_MIN)
        self.assertAlmostEqual(res.loglik, ll_zinb_at_zero, delta=1e-6)
        self.assertGreaterEqual(res.loglik + 1e-6, ll_nb2)

    def test_pi_init_strictly_inside_unit_interval(self):
        for bad in (0.0, 1.0, -0.1, 1.1, True, float("nan")):
            with self.assertRaisesRegex(NBFitError, r"pi_init"):
                fit_zinb([1, 2, 0], ones(3), pi_init=bad)


class ZINBExtraZeroTests(unittest.TestCase):
    """Genuinely inflated samples: an interior pi is estimated from the
    mixture likelihood, beats pi = 0 (NB2), beats the true-parameter
    likelihood, and agrees with an independent numerical MLE.  Zeros are
    never partitioned (see the soft-mixture identity test)."""

    def test_recovers_inflation_and_beats_nb2(self):
        beta_t, alpha_t, pi_t = [0.2, -0.3], 0.3, 0.4
        y, X, labels = ref.make_zinb_dataset(60, beta_t, alpha_t, pi_t,
                                             seed=5)
        frac0 = sum(1 for v in y if v == 0) / len(y)
        self.assertGreater(frac0, 0.5)  # sample really is zero-heavy
        res = fit_zinb(y, X)
        self.assertEqual(res.status, "converged")
        self.assertEqual(res.pi_boundary, "none")
        self.assertGreater(res.pi, 0.1)
        self.assertLess(res.pi, 0.8)
        # MLE likelihood >= likelihood at the generating parameters
        ll_true = ref.ref_zinb_loglik(
            y, X, beta_t, math.log(alpha_t), math.log(pi_t / (1 - pi_t)))
        self.assertGreaterEqual(res.loglik + 1e-8, ll_true)
        # and strictly better than the best pi = 0 (plain NB2) explanation,
        # whose likelihood comes from the independent NB2 reference at the
        # NB2 fitted point (supplementary core cross-check below that).
        rb = fit_nb2(y, X)
        ll_nb2 = ref.ref_loglik(y, X, rb.beta, rb.log_alpha)
        self.assertGreater(res.loglik, ll_nb2 + 0.05)
        self.assertAlmostEqual(res.beta[0], beta_t[0], delta=0.5)
        self.assertAlmostEqual(res.beta[1], beta_t[1], delta=0.5)
        # alpha is weakly identified relative to pi in small samples (they
        # trade off through p0); the independent MLE below certifies the
        # point itself, recovery here is correspondingly loose.
        self.assertAlmostEqual(math.log(res.alpha), math.log(alpha_t),
                               delta=1.1)

    def test_matches_independent_numerical_mle(self):
        beta_t, alpha_t, pi_t = [0.3, -0.2], 0.4, 0.35
        y, X, _ = ref.make_zinb_dataset(24, beta_t, alpha_t, pi_t, seed=2)
        res = fit_zinb(y, X)
        self.assertEqual(res.status, "converged")
        mean_y = sum(y) / len(y)
        starts = [
            [math.log(mean_y + 0.5), 0.0, 0.0, -0.5],
            beta_t + [math.log(alpha_t), math.log(pi_t / (1 - pi_t))],
            [0.5, 0.5, 1.0, -1.5],
        ]
        p_ref, ll_ref = ref.ref_fit_zinb(y, X, starts, max_iter=10000)
        self.assertAlmostEqual(res.loglik, ll_ref, delta=1e-8)
        self.assertAlmostEqual(res.logit_pi, p_ref[-1], delta=1e-5)
        self.assertAlmostEqual(res.log_alpha, p_ref[-2], delta=1e-5)
        for a, b in zip(res.beta, p_ref[:-2]):
            self.assertAlmostEqual(a, b, delta=1e-5)

    def test_zeros_are_soft_not_hard_split(self):
        # The fitted value must be the *mixture* log-likelihood summed over
        # every observation with ambiguous zeros -- not a hard partition.
        beta_t, alpha_t, pi_t = [0.2, -0.3], 0.3, 0.4
        y, X, _ = ref.make_zinb_dataset(60, beta_t, alpha_t, pi_t, seed=5)
        res = fit_zinb(y, X)
        # Independent soft mixture value at the fitted parameters.
        ll_soft = ref.ref_zinb_loglik(
            y, X, res.beta, res.log_alpha, res.logit_pi)
        self.assertAlmostEqual(res.loglik, ll_soft, delta=1e-9)

        # Hard classification of every zero into exactly one component
        # (structural iff pi > (1-pi) p0) gives a strictly smaller value:
        # log(a + b) > max(log a, log b) whenever both terms are positive.
        # A hard-split fitter could not reach the reported likelihood.
        alpha = math.exp(res.log_alpha)
        pi = res.pi
        ll_hard = 0.0
        n_ambiguous = 0
        for i, yi in enumerate(y):
            eta = sum(X[i][j] * res.beta[j] for j in range(len(X[0])))
            mu = math.exp(eta)
            p0 = (1.0 + alpha * mu) ** (-1.0 / alpha)
            if yi == 0:
                a_struct, a_count = pi, (1.0 - pi) * p0
                if a_struct > 0.0 and a_count > 0.0:
                    n_ambiguous += 1
                ll_hard += math.log(max(a_struct, a_count))
            else:
                ll_hard += math.log((1.0 - pi)
                                    * math.exp(ref.nb_log_pmf(yi, mu, alpha)))
        self.assertGreater(n_ambiguous, 0)
        self.assertGreater(res.loglik, ll_hard + 1e-9)


class ZINBAllZeroTests(unittest.TestCase):

    def test_all_zero_is_certified_no_finite_mle(self):
        res = fit_zinb([0, 0, 0, 0], [[1.0, 0.2], [1.0, -0.2],
                                      [1.0, 0.5], [1.0, -0.5]])
        self.assertFalse(res.converged)
        self.assertEqual(res.status, "no_finite_mle")
        self.assertLess(res.loglik, 0.0)  # strictly below supremum 0
        self.assertEqual(res.n_zeros, 4)
        # and the independent mixture formula confirms ll < 0 at the
        # returned finite point with a strictly improving pi direction.
        ll_ref = ref.ref_zinb_loglik(
            [0, 0, 0, 0], [[1.0, 0.2], [1.0, -0.2], [1.0, 0.5],
                           [1.0, -0.5]],
            res.beta, res.log_alpha, res.logit_pi)
        self.assertAlmostEqual(res.loglik, ll_ref, delta=1e-9)
        gz = ref.ref_zinb_fd_gradient(
            [0, 0, 0, 0], [[1.0, 0.2], [1.0, -0.2], [1.0, 0.5],
                           [1.0, -0.5]],
            res.beta, res.log_alpha, res.logit_pi)[-1]
        self.assertGreater(gz, 0.0)  # increasing pi still raises ll

    def test_partial_zeros_still_fit(self):
        res = fit_zinb([0, 0, 3, 0, 5], ones(5))
        self.assertEqual(res.status, "converged")


class ZINBValidationTests(unittest.TestCase):

    def setUp(self):
        self.y, self.X = [0, 3, 7], [[1.0, 0.5], [1.0, -0.5], [1.0, 0.2]]

    def test_nan_infinity_rejected_with_location(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            Xb = [list(r) for r in self.X]
            Xb[1][1] = bad
            with self.assertRaisesRegex(NBFitError, r"X\[1\]\[1\]"):
                fit_zinb(self.y, Xb)

    def test_counts_must_be_int_and_nonneg(self):
        with self.assertRaisesRegex(NBFitError, r"y\[1\]"):
            fit_zinb([1, 2.0], ones(2))
        with self.assertRaisesRegex(NBFitError, r"y\[0\].*bool"):
            fit_zinb([True, 2], ones(2))
        with self.assertRaisesRegex(NBFitError, r"y\[0\].*non-negative"):
            fit_zinb([-1, 2], ones(2))

    def test_shape_and_rank_errors(self):
        with self.assertRaisesRegex(NBFitError, r"n=2"):
            fit_zinb([1, 2], ones(3))
        with self.assertRaisesRegex(NBFitError, r"column 1"):
            fit_zinb([1, 2, 3, 4], [[1.0, 1.0] for _ in range(4)])

    def test_bad_optional_args(self):
        with self.assertRaisesRegex(NBFitError, r"alpha_init.*positive"):
            fit_zinb([1, 2], ones(2), alpha_init=0.0)
        with self.assertRaisesRegex(NBFitError, r"max_iter.*bool"):
            fit_zinb([1, 2, 3], ones(3), max_iter=True)
        with self.assertRaisesRegex(NBFitError, r"gtol"):
            fit_zinb([1, 2, 3], ones(3), gtol=0.0)

    def test_pure_function_validation(self):
        with self.assertRaisesRegex(NBFitError, r"beta must have length"):
            zinb_loglik([1, 2], ones(2), [0.0, 0.0], 0.0, 0.0)
        with self.assertRaisesRegex(NBFitError, r"logit_pi.*finite"):
            zinb_loglik([1, 2], ones(2), [0.0], 0.0, float("inf"))
        with self.assertRaisesRegex(NBFitError, r"log_alpha.*finite"):
            zinb_objective([1, 2], ones(2), [0.0], float("nan"), 0.0)

    def test_error_type_is_nbfit_error(self):
        # ZINBFitError is a subclass so existing except NBFitError works.
        with self.assertRaises(NBFitError):
            fit_zinb([1, 2], ones(2), pi_init=2.0)
        self.assertIsInstance(ZINBFitError("x"), ValueError)

    def test_inputs_are_not_mutated(self):
        y, X = list(self.y), [list(r) for r in self.X]
        yc, Xc = list(y), [list(r) for r in X]
        fit_zinb(y, X, pi_init=0.3, alpha_init=0.8)
        self.assertEqual(y, yc)
        self.assertEqual(X, Xc)

    def test_overflow_initial_point_raises(self):
        with self.assertRaisesRegex(NBFitError, r"computable region"):
            fit_zinb([1, 2], [[1.0, 0.5], [1.0, -0.5]],
                     offset=[1000.0, 3000.0])


class ZINBStatusTests(unittest.TestCase):

    def test_budget_exhaustion_distinct_from_convergence(self):
        y, X, _ = ref.make_zinb_dataset(40, [0.3, -0.2], 0.4, 0.4, seed=9)
        res = fit_zinb(y, X, max_iter=1)
        self.assertFalse(res.converged)
        self.assertEqual(res.status, "max_iter")
        self.assertIn("budget", res.message)

    def test_offset_shifts_intercept(self):
        y = [0, 5, 1, 0, 3, 6]
        c = 0.7
        r1 = fit_zinb(y, ones(len(y)))
        r2 = fit_zinb(y, ones(len(y)), offset=[c] * len(y))
        self.assertAlmostEqual(r1.beta[0] - c, r2.beta[0], delta=1e-7)
        self.assertAlmostEqual(r1.loglik, r2.loglik, delta=1e-9)
        self.assertAlmostEqual(r1.logit_pi, r2.logit_pi, delta=1e-9)

    def test_loglik_pure_function_matches_fit(self):
        y, X, _ = ref.make_zinb_dataset(30, [0.3, 0.1], 0.5, 0.3, seed=3)
        res = fit_zinb(y, X)
        self.assertAlmostEqual(
            zinb_loglik(y, X, res.beta, res.log_alpha, res.logit_pi),
            res.loglik, delta=1e-10)

    def test_reported_dimensions_and_zero_count(self):
        y, X, _ = ref.make_zinb_dataset(20, [0.3, -0.1], 0.4, 0.3, seed=1)
        res = fit_zinb(y, X)
        self.assertEqual(res.n_obs, 20)
        self.assertEqual(res.n_params, 4)
        self.assertEqual(res.n_zeros, sum(1 for v in y if v == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
