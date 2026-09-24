"""Tests for the constant-logit zero-inflated NB2 module (nbreg.zinb).

Expected values come from tests/_reference.py, which never imports the
package under test: mixture pmf from its definition on top of the
independent NB2 pmf, central finite-difference gradients, and an
independent numerical-gradient coordinate-ascent maximizer of the
mixture likelihood.  Hand-checkable identities (pi = 0 collapses to
NB2, offset absorption, mean posterior responsibility == pi) are used
as well; the fitter never generates its own expectations.

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
    nb2_loglik,
    nb2_pmf,
    zinb_loglik,
    zinb_objective,
    zinb_pmf,
    zinb_structural_probability,
    NBFitError,
)
import _reference as ref  # noqa: E402
from nbreg.zinb import _zinb_core  # noqa: E402


def lcg(seed):
    return ref._lcg(seed)


def sample_nb(mu, alpha, u):
    return ref.sample_nb(mu, alpha, u)


def make_inflated(n, beta_true, alpha_true, pi_true, seed, spread=0.5):
    """Well-conditioned deterministic ZINB sample with a custom design."""
    rng = lcg(seed)
    X = [[1.0, spread * ((i % 5) - 2)] for i in range(n)]
    y, structural = [], []
    for i in range(n):
        if next(rng) < pi_true:
            y.append(0)
            structural.append(True)
        else:
            eta = sum(X[i][j] * beta_true[j] for j in range(len(beta_true)))
            y.append(sample_nb(math.exp(eta), alpha_true, next(rng)))
            structural.append(False)
    return y, X, structural


def make_plain_nb(n, beta_true, alpha_true, seed, spread=0.4):
    rng = lcg(seed)
    X = [[1.0, spread * ((i % 5) - 2)] for i in range(n)]
    y = []
    for i in range(n):
        eta = sum(X[i][j] * beta_true[j] for j in range(len(beta_true)))
        y.append(sample_nb(math.exp(eta), alpha_true, next(rng)))
    return y, X


class MixturePMFTests(unittest.TestCase):

    def test_pmf_matches_definition(self):
        for mu, alpha, pi in ((2.0, 0.5, 0.3), (0.4, 3.0, 0.7),
                              (5.0, 0.1, 0.05), (1.0, 8.0, 0.6)):
            for k in range(12):
                self.assertAlmostEqual(
                    zinb_pmf(k, mu, alpha, pi),
                    ref.zinb_mixture_pmf(k, mu, alpha, pi),
                    delta=1e-12)

    def test_pmf_normalizes(self):
        for mu, alpha, pi in ((2.0, 0.5, 0.3), (0.4, 3.0, 0.7),
                              (5.0, 0.1, 0.05)):
            total = sum(zinb_pmf(k, mu, alpha, pi) for k in range(3000))
            self.assertAlmostEqual(total, 1.0, delta=1e-9)

    def test_pi_zero_is_plain_nb2(self):
        mu, alpha = 1.7, 0.8
        for k in range(15):
            self.assertAlmostEqual(zinb_pmf(k, mu, alpha, 0.0),
                                   nb2_pmf(k, mu, alpha), delta=1e-15)

    def test_pi_one_is_point_mass_at_zero(self):
        self.assertEqual(zinb_pmf(0, 1.0, 1.0, 1.0), 1.0)
        for k in range(1, 6):
            self.assertEqual(zinb_pmf(k, 1.0, 1.0, 1.0), 0.0)

    def test_pmf_input_validation(self):
        with self.assertRaisesRegex(NBFitError, r"int"):
            zinb_pmf(1.0, 2.0, 1.0, 0.5)
        with self.assertRaisesRegex(NBFitError, r"non-negative"):
            zinb_pmf(-1, 2.0, 1.0, 0.5)
        with self.assertRaisesRegex(NBFitError, r"mu.*positive"):
            zinb_pmf(0, 0.0, 1.0, 0.5)
        with self.assertRaisesRegex(NBFitError, r"alpha.*positive"):
            zinb_pmf(0, 1.0, -1.0, 0.5)
        with self.assertRaisesRegex(NBFitError, r"pi.*\[0, 1\]"):
            zinb_pmf(0, 1.0, 1.0, 1.1)
        with self.assertRaisesRegex(NBFitError, r"pi.*\[0, 1\]"):
            zinb_pmf(0, 1.0, 1.0, float("nan"))

    def test_structural_posterior_matches_definition(self):
        mu, alpha, pi = 1.2, 1.5, 0.35
        # positive counts can never be structural zeros
        for k in (1, 2, 7):
            self.assertEqual(
                zinb_structural_probability(k, mu, alpha, pi), 0.0)
        tau0 = zinb_structural_probability(0, mu, alpha, pi)
        self.assertAlmostEqual(
            tau0, ref.zinb_structural_tau(0, mu, alpha, pi), delta=1e-12)
        self.assertTrue(0.0 < tau0 < 1.0)
        self.assertAlmostEqual(
            tau0, pi / zinb_pmf(0, mu, alpha, pi), delta=1e-14)
        # boundary pi
        self.assertEqual(zinb_structural_probability(0, mu, alpha, 0.0), 0.0)
        self.assertEqual(zinb_structural_probability(0, mu, alpha, 1.0), 1.0)


class MixtureLikelihoodTests(unittest.TestCase):

    CASES = [
        # (y, X, offset, beta, log_alpha, logit_pi, label)
        ([0, 2, 0, 5, 1],
         [[1.0, 0.0], [1.0, 1.0], [1.0, -1.0], [1.0, 0.5], [1.0, -0.5]],
         None, [0.3, -0.2], math.log(0.7), 0.4, "mixed zeros, interior pi"),
        ([0, 0, 3, 0],
         [[1.0, 0.2], [1.0, -0.4], [1.0, 0.1], [1.0, -0.1]],
         [0.1, -0.2, 0.0, 0.25],
         [-0.3, 0.2], math.log(2.0), -2.5, "offset, pi near 0"),
        ([4, 1, 6], [[1.0], [1.0], [1.0]], None,
         [0.6], math.log(0.2), 1.5, "positives only, high pi"),
        ([0, 0], [[1.0, 0.3], [1.0, -0.3]], None,
         [0.0, 0.1], math.log(1.0), 5.0, "all zeros, pi near 1"),
    ]

    def test_loglik_matches_reference(self):
        for y, X, off, beta, la, gp, label in self.CASES:
            with self.subTest(label):
                self.assertAlmostEqual(
                    zinb_loglik(y, X, beta, la, gp, off),
                    ref.ref_zinb_loglik(y, X, beta, la, gp, off),
                    delta=1e-10)

    def test_gradient_matches_finite_differences(self):
        for y, X, off, beta, la, gp, label in self.CASES:
            with self.subTest(label):
                ll, g = zinb_objective(y, X, beta, la, gp, off)
                fd = ref.ref_zinb_fd_gradient(y, X, beta, la, gp, off)
                self.assertEqual(len(g), len(beta) + 2)
                for a, b in zip(g, fd):
                    self.assertAlmostEqual(a, b, delta=3e-5)

    def test_observed_information_matches_gradient_differences(self):
        # Analogous to the NB2 trigamma-vs-numeric-digamma check: the
        # analytic observed Hessian (responsibility-weighted NB
        # information plus the posterior-variance outer product) must
        # equal central differences of the analytic mixture gradient.
        y = [0, 2, 0, 5, 0, 1]
        X = [[1.0, 0.0], [1.0, 1.0], [1.0, -1.0],
             [1.0, 0.5], [1.0, -0.5], [1.0, 0.2]]
        beta, la, gp = [0.25, -0.15], math.log(0.9), 0.5
        _, _, H, _, _ = _zinb_core(y, X, [0.0] * len(y), beta, la, gp)
        p = beta + [la, gp]
        h = 1e-5
        d = len(p)

        def grad_at(q):
            _, gg = zinb_objective(y, X, q[:2], q[2], q[3])
            return gg

        scale = 1.0
        for a in range(d):
            pu, pm = list(p), list(p)
            pu[a] += h
            pm[a] -= h
            gu, gm = grad_at(pu), grad_at(pm)
            for b in range(d):
                num = (gu[b] - gm[b]) / (2 * h)
                scale = max(scale, abs(num))
                self.assertAlmostEqual(H[a][b], num, delta=1e-7)
        self.assertLess(
            max(abs(H[a][b]) for a in range(d) for b in range(d)) / scale,
            10.0)  # Hessian genuinely nonzero (not a vacuous 0 == 0)

    def test_pi_zero_collapses_likelihood_to_nb2(self):
        y = [0, 1, 0, 4, 2, 0]
        X = [[1.0, 0.3 * (i % 3 - 1)] for i in range(len(y))]
        beta, la = [0.2, -0.1], math.log(0.9)
        z = zinb_loglik(y, X, beta, la, -20.0)
        plain = nb2_loglik(y, X, beta, la)
        self.assertAlmostEqual(z, plain, delta=1e-7 * len(y))

    def test_point_validation(self):
        y, X = [0, 1, 2], [[1.0]] * 3
        with self.assertRaisesRegex(NBFitError, r"beta must have length"):
            zinb_loglik(y, X, [0.0, 1.0], 0.0, 0.0)
        with self.assertRaisesRegex(NBFitError, r"beta\[0\]"):
            zinb_loglik(y, X, [float("nan")], 0.0, 0.0)
        with self.assertRaisesRegex(NBFitError, r"log_alpha.*finite"):
            zinb_loglik(y, X, [0.0], float("inf"), 0.0)
        with self.assertRaisesRegex(NBFitError, r"logit_pi.*finite"):
            zinb_loglik(y, X, [0.0], 0.0, float("inf"))


class FitInputValidationTests(unittest.TestCase):

    def test_data_checks_reused(self):
        with self.assertRaisesRegex(NBFitError, r"y\[1\]"):
            fit_zinb([1, 2.0], [[1.0], [1.0]])
        with self.assertRaisesRegex(NBFitError, r"y\[0\].*bool"):
            fit_zinb([False], [[1.0]])
        with self.assertRaisesRegex(NBFitError, r"X\[0\]\[1\]"):
            fit_zinb([0, 1], [[1.0, float("nan")], [1.0, 0.0]])
        with self.assertRaisesRegex(NBFitError, r"offset.*length n=2"):
            fit_zinb([0, 1], [[1.0], [1.0]], offset=[0.0])

    def test_rank_deficient_rejected(self):
        with self.assertRaisesRegex(NBFitError, r"column 1"):
            fit_zinb([0, 1, 2, 3], [[1.0, 1.0]] * 4)

    def test_options_rejected(self):
        y, X = [0, 1, 2], [[1.0]] * 3
        with self.assertRaisesRegex(NBFitError, r"max_iter.*bool"):
            fit_zinb(y, X, max_iter=True)
        with self.assertRaisesRegex(NBFitError, r"max_iter"):
            fit_zinb(y, X, max_iter=0)
        with self.assertRaisesRegex(NBFitError, r"gtol"):
            fit_zinb(y, X, gtol=0.0)
        with self.assertRaisesRegex(NBFitError, r"gtol"):
            fit_zinb(y, X, gtol=2.0)

    def test_inputs_are_not_mutated(self):
        y, X = [0, 3, 0, 7, 2], [[1.0, 0.5], [1.0, -0.5], [1.0, 0.2],
                                 [1.0, -0.2], [1.0, 0.4]]
        ycopy, Xcopy = list(y), [list(r) for r in X]
        fit_zinb(y, X)
        self.assertEqual(y, ycopy)
        self.assertEqual(X, Xcopy)


class ExtraZeroFitTests(unittest.TestCase):

    def test_interior_mle_matches_independent_optimizer(self):
        # Deterministic sample that genuinely needs an interior pi.
        y, X, structural = make_inflated(
            16, [0.7, -0.2], 1.2, 0.30, seed=5)
        self.assertGreater(sum(structural), 0)
        res = fit_zinb(y, X)
        self.assertEqual(res.status, "converged")
        self.assertFalse(res.pi_at_lower_boundary)
        self.assertFalse(res.pi_at_upper_boundary)
        p_ref, ll_ref = ref.ref_zinb_fit(y, X)
        self.assertAlmostEqual(res.loglik, ll_ref, delta=2e-5)
        for a, b in zip(res.beta, p_ref[:2]):
            self.assertAlmostEqual(a, b, delta=2e-3)
        self.assertAlmostEqual(res.log_alpha, p_ref[2], delta=2e-3)
        self.assertAlmostEqual(
            res.logit_pi, p_ref[3], delta=2e-3)
        # gradient at the reported point
        self.assertLess(max(abs(v) for v in res.gradient), 1e-6)

    def test_fitted_likelihood_beats_generating_parameters(self):
        beta_true, alpha_true, pi_true = [0.9, -0.25], 0.7, 0.38
        y, X, _ = make_inflated(40, beta_true, alpha_true, pi_true, seed=71)
        res = fit_zinb(y, X)
        self.assertEqual(res.status, "converged")
        ll_truth = ref.ref_zinb_loglik(
            y, X, beta_true, math.log(alpha_true),
            math.log(pi_true) - math.log1p(-pi_true))
        self.assertGreaterEqual(res.loglik + 1e-8, ll_truth)
        # useful recovery on this deterministic sample
        self.assertAlmostEqual(res.pi, pi_true, delta=0.25)
        self.assertAlmostEqual(res.beta[0], beta_true[0], delta=0.5)

    def test_mixture_nests_nb2_so_its_likelihood_is_never_lower(self):
        for seed, pi_t in ((3, 0.3), (8, 0.0)):
            y, X, _ = make_inflated(30, [0.3, 0.1], 1.0, pi_t, seed=seed)
            rz = fit_zinb(y, X)
            rn = fit_nb2(y, X)
            # At the pi -> 0 boundary the fitted point uses pi ~ 2e-9
            # (the box edge), whose likelihood differs from exact pi = 0
            # by O(pi * n) ~ 1e-7; interior fits agree much more tightly.
            tol = 1e-7 if rz.pi_at_lower_boundary else 1e-9
            self.assertGreaterEqual(rz.loglik + tol, rn.loglik)

    def test_pure_loglik_matches_fit(self):
        y, X, _ = make_inflated(25, [0.5, -0.2], 0.9, 0.3, seed=4)
        res = fit_zinb(y, X)
        self.assertAlmostEqual(
            zinb_loglik(y, X, res.beta, res.log_alpha, res.logit_pi),
            res.loglik, delta=1e-10)


class NoStructuralZeroTests(unittest.TestCase):

    def test_boundary_collapse_flagged_and_equals_nb2(self):
        # NB2 data with alpha = 6: the sampling NB itself produces many
        # natural zeros; the honest MLE drives pi to the lower boundary.
        y, X = make_plain_nb(40, [0.2, -0.1], 6.0, seed=3)
        self.assertGreater(sum(1 for v in y if v == 0), 5)
        res = fit_zinb(y, X)
        self.assertEqual(res.status, "converged")
        self.assertTrue(res.pi_at_lower_boundary)
        self.assertLessEqual(res.pi, 1e-6)
        ll_nb = fit_nb2(y, X).loglik
        self.assertAlmostEqual(res.loglik, ll_nb, delta=1e-5)

    def test_saturated_counts_hit_poisson_and_pi_lower_boundary(self):
        res = fit_zinb([3] * 20, [[1.0]] * 20)
        self.assertEqual(res.status, "converged")
        self.assertTrue(res.near_poisson)
        self.assertTrue(res.pi_at_lower_boundary)
        self.assertAlmostEqual(res.beta[0], math.log(3.0), delta=1e-6)


class AllZeroTests(unittest.TestCase):

    def test_all_zero_is_certified_no_finite_mle(self):
        res = fit_zinb([0, 0, 0, 0],
                       [[1.0, 0.2], [1.0, -0.2], [1.0, 0.5], [1.0, -0.5]])
        self.assertFalse(res.converged)
        self.assertEqual(res.status, "no_finite_mle")
        self.assertTrue(res.pi_at_upper_boundary)
        self.assertTrue(res.structural_zero_degenerate)
        self.assertLess(res.loglik, 0.0)  # strictly below supremum 0
        # every observation is structurally attributed at the boundary
        self.assertAlmostEqual(sum(res.responsibilities), 4.0)

    def test_all_zero_intercept_only(self):
        res = fit_zinb([0, 0], [[1.0], [1.0]])
        self.assertEqual(res.status, "no_finite_mle")
        self.assertFalse(res.converged)
        self.assertGreaterEqual(res.pi, 1.0 - 1e-8)

    def test_partial_zeros_still_fit(self):
        res = fit_zinb([0, 0, 3, 0, 5], [[1.0]] * 5)
        self.assertEqual(res.status, "converged")


class SoftResponsibilityTests(unittest.TestCase):

    def test_zeros_are_fractional_not_hard_classified(self):
        y, X, _ = make_inflated(30, [0.8, -0.2], 0.8, 0.35, seed=9)
        res = fit_zinb(y, X)
        self.assertEqual(res.status, "converged")
        for i, v in enumerate(y):
            if v > 0:
                self.assertEqual(res.responsibilities[i], 0.0)
            else:
                # soft mixture weight: never exactly 0 or 1
                self.assertTrue(0.0 < res.responsibilities[i] < 1.0)
                self.assertAlmostEqual(
                    res.responsibilities[i],
                    zinb_structural_probability(
                        0, math.exp(res.beta[0] + res.beta[1] * X[i][1]),
                        res.alpha, res.pi),
                    delta=1e-10)
        # EM identity for a constant pi: mean posterior == fitted pi
        self.assertAlmostEqual(
            sum(res.responsibilities) / len(y), res.pi, delta=1e-9)

    def test_offset_shifts_intercept_only(self):
        rng = lcg(5)
        n = 20
        X = [[1.0]] * n
        y = []
        for _ in range(n):
            if next(rng) < 0.3:
                y.append(0)
            else:
                y.append(sample_nb(math.exp(0.5), 0.8, next(rng)))
        c = 0.6
        r1 = fit_zinb(y, X)
        r2 = fit_zinb(y, X, offset=[c] * n)
        self.assertAlmostEqual(r1.beta[0] - c, r2.beta[0], delta=1e-8)
        self.assertAlmostEqual(r1.loglik, r2.loglik, delta=1e-9)
        self.assertAlmostEqual(r1.pi, r2.pi, delta=1e-9)


class StatusSeparationTests(unittest.TestCase):

    def test_budget_exhaustion_is_not_convergence(self):
        y, X, _ = make_inflated(30, [0.4, -0.2], 1.0, 0.3, seed=6)
        res = fit_zinb(y, X, max_iter=1)
        self.assertFalse(res.converged)
        self.assertEqual(res.status, "max_iter")
        self.assertIn("budget", res.message)

    def test_dimensions_reported(self):
        rng = lcg(6)
        n = 20
        X = [[1.0, 0.4 * ((i % 5) - 2), 0.3 * ((i % 4) - 1.5)]
             for i in range(n)]
        beta_true = [0.4, -0.2, 0.1]
        y = []
        for i in range(n):
            if next(rng) < 0.3:
                y.append(0)
            else:
                eta = sum(X[i][j] * beta_true[j] for j in range(3))
                y.append(sample_nb(math.exp(eta), 1.0, next(rng)))
        res = fit_zinb(y, X)
        self.assertEqual(res.status, "converged")
        self.assertEqual(res.n_obs, 20)
        self.assertEqual(res.n_params, 5)
        self.assertEqual(len(res.gradient), 5)
        self.assertEqual(len(res.responsibilities), 20)
        self.assertEqual(res.n_zero, sum(1 for v in y if v == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
