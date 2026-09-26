"""Tests for interval-censored NB2 observations (fixed alpha, beta only).

Expected values come from tests/_reference.py, which never imports the
package under test: interval/tail probabilities by brute-force pmf
recurrence summation, censored log-likelihood from those, and central
finite-difference gradients.

Run:  python -m unittest discover -s tests -v
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nbreg import (  # noqa: E402
    fit_nb2, fit_nb2_censored, nb2_censored_loglik, nb2_censored_objective,
    nb2_interval_logprob, nb2_pmf, NBFitError,
)
import _reference as ref  # noqa: E402


def ones(n):
    return [[1.0] for _ in range(n)]


class IntervalProbabilityTests(unittest.TestCase):
    """Small-count ranges checked against independent brute-force sums."""

    def test_interval_matches_bruteforce(self):
        for mu, alpha in ((2.0, 0.5), (0.3, 2.0), (7.0, 0.01), (5.0, 3.0)):
            for lo, hi in ((0, 0), (2, 4), (0, 12), (5, 5)):
                got = nb2_interval_logprob(lo, hi, mu, alpha)
                want = ref.ref_interval_logprob(lo, hi, mu, alpha)
                self.assertAlmostEqual(got, want, delta=1e-10)

    def test_tail_matches_bruteforce(self):
        for mu, alpha in ((2.0, 0.5), (0.3, 2.0), (7.0, 0.01), (5.0, 3.0)):
            for lo in (0, 1, 4, 9):
                got = nb2_interval_logprob(lo, None, mu, alpha)
                want = ref.ref_interval_logprob(lo, None, mu, alpha)
                self.assertAlmostEqual(got, want, delta=1e-10)

    def test_full_support_interval_is_zero_logprob(self):
        # [0, None] covers the whole support: probability exactly 1.
        self.assertEqual(nb2_interval_logprob(0, None, 3.0, 1.5), 0.0)
        # A very wide finite interval matches the brute-force sum and is
        # within the residual bound of 1.
        got = nb2_interval_logprob(0, 400, 3.0, 1.5)
        self.assertAlmostEqual(got, 0.0, delta=1e-9)

    def test_narrow_tail_interval(self):
        # Extremely narrow interval deep in the tail: P(Y in [40, 41])
        # with mu=2 -- tiny but representable probabilities.
        mu, alpha = 2.0, 0.5
        got = nb2_interval_logprob(40, 41, mu, alpha)
        want = ref.ref_interval_logprob(40, 41, mu, alpha)
        self.assertAlmostEqual(got, want, delta=1e-10)
        self.assertLess(got, -20.0)  # genuinely a rare tail event
        # consistency with the single-point pmf route
        direct = math.log(nb2_pmf(40, mu, alpha) + nb2_pmf(41, mu, alpha))
        self.assertAlmostEqual(got, direct, delta=1e-12)

    def test_exact_interval_equals_pmf(self):
        got = nb2_interval_logprob(7, 7, 3.0, 1.5)
        self.assertAlmostEqual(got, math.log(nb2_pmf(7, 3.0, 1.5)),
                               delta=1e-14)

    def test_interval_input_validation(self):
        with self.assertRaisesRegex(NBFitError, "lo"):
            nb2_interval_logprob(-1, 3, 2.0, 1.0)
        with self.assertRaisesRegex(NBFitError, "hi"):
            nb2_interval_logprob(3, 2, 2.0, 1.0)
        with self.assertRaisesRegex(NBFitError, "bool"):
            nb2_interval_logprob(True, 3, 2.0, 1.0)
        with self.assertRaisesRegex(NBFitError, "mu"):
            nb2_interval_logprob(0, 3, float("nan"), 1.0)


class TailBudgetTests(unittest.TestCase):

    def test_budget_exhaustion_aborts_with_reason(self):
        # mu=50, alpha=0.01: the tail needs far more than 10 terms.
        with self.assertRaisesRegex(NBFitError, "budget"):
            nb2_interval_logprob(1, None, 50.0, 0.01, tail_budget=10)

    def test_budget_exhaustion_not_treated_as_exact(self):
        # The truncated tail sum must never surface as a likelihood:
        # the error message must say the evaluation was aborted.
        try:
            nb2_interval_logprob(1, None, 50.0, 0.01, tail_budget=10)
        except NBFitError as exc:
            self.assertIn("not an exact likelihood", str(exc))
        else:
            self.fail("expected NBFitError")

    def test_fit_aborts_when_initial_evaluation_exceeds_budget(self):
        obs = [(0, None), (1, None), 3]
        X = ones(3)
        with self.assertRaisesRegex(NBFitError, "budget"):
            fit_nb2_censored(obs, X, 0.01, offset=[4.0, 4.0, 4.0],
                             tail_budget=5)

    def test_tail_budget_validation(self):
        with self.assertRaisesRegex(NBFitError, "tail_budget"):
            nb2_interval_logprob(0, 3, 2.0, 1.0, tail_budget=True)
        with self.assertRaisesRegex(NBFitError, "tail_tol"):
            nb2_interval_logprob(0, 3, 2.0, 1.0, tail_tol=0.0)


class CensoredObjectiveTests(unittest.TestCase):
    """Likelihood and gradient vs independent high-precision summation."""

    OBS = [2, (0, 3), (4, None), 1, (2, 5), 0, (1, None), 3]
    X = [[1.0, 0.4 * ((i % 3) - 1)] for i in range(8)]
    OFF = [0.1 * (i % 2) for i in range(8)]
    ALPHA = 1.3
    BETA = [0.4, -0.25]

    def test_loglik_matches_reference(self):
        got = nb2_censored_loglik(self.OBS, self.X, self.BETA, self.ALPHA,
                                  offset=self.OFF)
        want = ref.ref_censored_loglik(self.OBS, self.X, self.BETA,
                                       self.ALPHA, self.OFF)
        self.assertAlmostEqual(got, want, delta=1e-9)

    def test_gradient_matches_reference_fd(self):
        ll, g = nb2_censored_objective(self.OBS, self.X, self.BETA,
                                       self.ALPHA, offset=self.OFF)
        fd = ref.ref_censored_fd_gradient(self.OBS, self.X, self.BETA,
                                          self.ALPHA, self.OFF)
        self.assertAlmostEqual(
            ll, ref.ref_censored_loglik(self.OBS, self.X, self.BETA,
                                        self.ALPHA, self.OFF),
            delta=1e-9)
        for a, b in zip(g, fd):
            self.assertAlmostEqual(a, b, delta=3e-5)

    def test_gradient_at_fit_is_tiny(self):
        obs, X, off, _ = ref.make_censored_dataset(30, [0.4, -0.3], 0.9,
                                                   seed=5)
        res = fit_nb2_censored(obs, X, 0.9, offset=off)
        self.assertEqual(res.status, "converged")
        for v in res.gradient:
            self.assertLess(abs(v), 1e-6)


class DegeneratePathTests(unittest.TestCase):

    def test_all_exact_degenerates_to_fixed_alpha_path(self):
        y = [1, 4, 0, 7, 2, 3]
        X = [[1.0, 0.5 * (i % 3 - 1)] for i in range(6)]
        r_new = fit_nb2_censored(y, X, 0.8)
        r_old = fit_nb2(y, X, fixed_alpha=0.8)
        self.assertEqual(r_new.status, r_old.status)
        self.assertAlmostEqual(r_new.loglik, r_old.loglik, delta=1e-12)
        for a, b in zip(r_new.beta, r_old.beta):
            self.assertAlmostEqual(a, b, delta=1e-12)
        self.assertTrue(r_new.fixed_alpha)

    def test_exact_int_and_singleton_interval_agree(self):
        y = [2, 0, 5, 1]
        obs = [(2, 2), (0, 0), (5, 5), (1, 1)]
        X = ones(4)
        r1 = fit_nb2_censored(y, X, 1.0)
        r2 = fit_nb2_censored(obs, X, 1.0)
        self.assertAlmostEqual(r1.loglik, r2.loglik, delta=1e-10)


class IdentifiabilityTests(unittest.TestCase):

    def test_all_lower_bounds_no_finite_mle(self):
        obs = [(2, None), (3, None), (1, None), (0, None)]
        res = fit_nb2_censored(obs, ones(4), 1.0)
        self.assertFalse(res.converged)
        self.assertEqual(res.status, "no_finite_mle")
        self.assertIn("not identifiable", res.message)
        self.assertLess(res.loglik, 0.0)  # strictly below the supremum 0

    def test_mixed_lower_bounds_still_fit(self):
        obs = [(2, None), 3, (0, 4), 1, (1, None), 2]
        res = fit_nb2_censored(obs, ones(6), 1.0)
        self.assertEqual(res.status, "converged")


class CensoredFitTests(unittest.TestCase):

    def test_recovers_beta_on_synthetic_censored_data(self):
        beta_true, alpha_true = [0.5, -0.35], 0.8
        obs, X, off, _ = ref.make_censored_dataset(60, beta_true,
                                                   alpha_true, seed=17)
        res = fit_nb2_censored(obs, X, alpha_true, offset=off)
        self.assertEqual(res.status, "converged")
        # Fitted likelihood must beat the likelihood at the true beta.
        ll_true = ref.ref_censored_loglik(obs, X, beta_true, alpha_true, off)
        self.assertGreaterEqual(res.loglik + 1e-8, ll_true)
        for a, b in zip(res.beta, beta_true):
            self.assertAlmostEqual(a, b, delta=0.35)
        self.assertEqual(res.alpha, alpha_true)
        self.assertTrue(res.fixed_alpha)
        self.assertEqual(len(res.gradient), 2)

    def test_alpha_is_fixed_not_estimated(self):
        obs, X, off, _ = ref.make_censored_dataset(20, [0.3], 1.5, seed=8)
        res = fit_nb2_censored(obs, X, 2.5, offset=off)
        self.assertEqual(res.alpha, 2.5)
        self.assertEqual(res.log_alpha, math.log(2.5))
        self.assertEqual(res.n_params, 1)

    def test_midpoint_would_give_different_answer(self):
        # Guard against the forbidden shortcut: fitting interval midpoints
        # as exact counts must NOT reproduce the censored fit.
        obs = [(0, 8), (1, 9), (0, 6), (2, 10), (1, 7), (0, 9)]
        X = ones(6)
        res_c = fit_nb2_censored(obs, X, 1.0)
        mid = [int(round(0.5 * (lo + hi))) for lo, hi in obs]
        res_m = fit_nb2(mid, X, fixed_alpha=1.0)
        self.assertGreater(abs(res_c.loglik - res_m.loglik), 1e-6)

    def test_budget_exhaustion_distinct_from_no_solution(self):
        obs, X, off, _ = ref.make_censored_dataset(20, [0.4, -0.3], 1.0,
                                                   seed=7)
        res = fit_nb2_censored(obs, X, 1.0, offset=off, max_iter=1)
        self.assertFalse(res.converged)
        self.assertEqual(res.status, "max_iter")
        self.assertIn("budget", res.message)

    def test_obs_validation(self):
        X = ones(3)
        with self.assertRaisesRegex(NBFitError, r"obs\[1\]"):
            fit_nb2_censored([1, 2.5, 3], X, 1.0)
        with self.assertRaisesRegex(NBFitError, r"obs\[0\]"):
            fit_nb2_censored([(3, 1), 2, 3], X, 1.0)
        with self.assertRaisesRegex(NBFitError, r"obs\[2\]"):
            fit_nb2_censored([1, 2, (True, None)], X, 1.0)
        with self.assertRaisesRegex(NBFitError, "alpha"):
            fit_nb2_censored([1, 2, 3], X, -1.0)
        with self.assertRaisesRegex(NBFitError, "alpha"):
            fit_nb2_censored([1, 2, 3], X, "1.0")
        with self.assertRaisesRegex(NBFitError, "max_iter"):
            fit_nb2_censored([1, 2, 3], X, 1.0, max_iter=True)

    def test_inputs_are_not_mutated(self):
        obs = [2, (0, 3), (4, None), 1]
        X = [[1.0, 0.5], [1.0, -0.5], [1.0, 0.25], [1.0, 0.0]]
        obs_copy = [list(o) if isinstance(o, tuple) else o for o in obs]
        obs_in = [tuple(o) if isinstance(o, list) else o for o in obs_copy]
        fit_nb2_censored(obs_in, X, 1.0)
        self.assertEqual(obs_in, [2, (0, 3), (4, None), 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
