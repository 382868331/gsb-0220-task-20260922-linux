"""Tests for the likelihood, PMF and truncated-range probabilities.

Expected values come from the independent implementation in
``tests/reference.py`` (which never imports the package under test),
from analytic identities, and from Poisson limits.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from reference import (  # noqa: E402
    ref_finite_diff_grad,
    ref_log_pmf,
    ref_loglik,
    ref_range_sum,
)

from nb2reg import (
    NB2InputError,
    conditional_pmf,
    log_pmf,
    nb2_gradient,
    nb2_loglik,
    pmf,
    range_probability,
)


class TestPMF(unittest.TestCase):
    def test_log_pmf_matches_independent_reference(self):
        for mu in (0.3, 2.5, 17.0):
            for alpha in (0.2, 1.0, 5.0):
                for k in range(0, 12):
                    self.assertAlmostEqual(
                        log_pmf(k, mu, alpha),
                        ref_log_pmf(k, mu, alpha),
                        places=10,
                        msg=f"k={k} mu={mu} alpha={alpha}",
                    )

    def test_pmf_normalises_over_full_support(self):
        for mu, alpha in [(1.0, 1.0), (3.0, 0.5), (8.0, 2.0)]:
            total = range_probability(0, 500, mu, alpha)
            self.assertAlmostEqual(total, 1.0, places=9)

    def test_pmf_nonnegative(self):
        for mu in (0.1, 5.0, 30.0):
            for alpha in (0.1, 1.0, 10.0):
                for k in range(0, 40):
                    self.assertGreaterEqual(pmf(k, mu, alpha), 0.0)

    def test_poisson_limit_alpha_to_zero(self):
        # NB2 -> Poisson(mu) as alpha -> 0.
        mu, alpha = 4.0, 1.0e-7
        for k in range(0, 10):
            pois = math.exp(-mu) * mu**k / math.factorial(k)
            self.assertLess(abs(pmf(k, mu, alpha) - pois), 1.0e-5)

    def test_truncated_range_matches_reference(self):
        for lo, hi in [(0, 5), (2, 8), (1, 1), (0, 0)]:
            for mu, alpha in [(2.0, 1.0), (6.0, 0.7)]:
                self.assertAlmostEqual(
                    range_probability(lo, hi, mu, alpha),
                    ref_range_sum(lo, hi, mu, alpha),
                    places=11,
                )

    def test_unbounded_range_uses_complement(self):
        mu, alpha = 3.0, 0.5
        self.assertAlmostEqual(
            range_probability(0, None, mu, alpha), 1.0, places=12
        )
        self.assertAlmostEqual(
            range_probability(2, None, mu, alpha),
            ref_range_sum(2, None, mu, alpha),
            places=11,
        )

    def test_zero_truncated_conditional_normalises(self):
        mu, alpha, hi = 3.0, 0.5, 300
        denom = range_probability(1, hi, mu, alpha)
        s = sum(conditional_pmf(k, 1, hi, mu, alpha) for k in range(1, 20))
        self.assertLess(abs(s - sum(
            math.exp(log_pmf(k, mu, alpha)) / denom for k in range(1, 20)
        )), 1.0e-12)
        # single-support truncation is a point mass of one
        self.assertAlmostEqual(
            conditional_pmf(3, 3, 3, mu, alpha), 1.0, places=12
        )

    def test_pmf_input_validation(self):
        with self.assertRaises(NB2InputError):
            pmf(-1, 1.0, 1.0)
        with self.assertRaises(NB2InputError):
            pmf(1.5, 1.0, 1.0)
        with self.assertRaises(NB2InputError):
            pmf(1, 0.0, 1.0)
        with self.assertRaises(NB2InputError):
            pmf(1, 1.0, -1.0)
        with self.assertRaises(NB2InputError):
            pmf(1, float("nan"), 1.0)
        with self.assertRaises(NB2InputError):
            pmf(1, 1.0, float("inf"))


class TestLikelihoodGradient(unittest.TestCase):
    def setUp(self):
        self.y = [2, 0, 5, 1, 3, 8, 0, 2]
        self.X = [[1.0, 0.2 * (i - 3)] for i in range(8)]
        self.off = [0.1 * (i % 3) - 0.1 for i in range(8)]
        self.beta = [0.4, -0.3]
        self.alpha = 1.7

    def test_loglik_matches_reference_fixed_alpha(self):
        self.assertAlmostEqual(
            nb2_loglik(self.y, self.X, offset=self.off,
                       beta=self.beta, alpha=self.alpha),
            ref_loglik(self.y, self.X, self.beta, self.alpha, self.off),
            places=9,
        )

    def test_loglik_alpha_theta_consistency(self):
        theta = 1.0 / self.alpha
        self.assertAlmostEqual(
            nb2_loglik(self.y, self.X, beta=self.beta, theta=theta),
            nb2_loglik(self.y, self.X, beta=self.beta, alpha=self.alpha),
            places=12,
        )

    def test_gradient_fixed_alpha_vs_finite_differences(self):
        g = nb2_gradient(self.y, self.X, offset=self.off,
                         beta=self.beta, alpha=self.alpha)
        eps = 1.0e-6
        for j in range(len(self.beta)):
            up = self.beta[:]; dn = self.beta[:]
            up[j] += eps; dn[j] -= eps
            fd = (
                nb2_loglik(self.y, self.X, offset=self.off, beta=up,
                           alpha=self.alpha)
                - nb2_loglik(self.y, self.X, offset=self.off, beta=dn,
                             alpha=self.alpha)
            ) / (2 * eps)
            self.assertLess(
                abs(g[j] - fd) / max(abs(fd), 1.0), 1.0e-7,
                msg=f"beta[{j}] analytic={g[j]} fd={fd}",
            )

    def test_joint_gradient_vs_reference_finite_differences(self):
        # Independent reference (its own loglik, not the package's) for
        # the finite-difference expected gradient.
        log_theta = math.log(0.6)
        g = nb2_gradient(self.y, self.X, offset=self.off,
                         beta=self.beta, log_theta=log_theta)
        fd = ref_finite_diff_grad(
            self.y, self.X, self.beta, log_theta, self.off
        )
        for j in range(3):
            self.assertLess(
                abs(g[j] - fd[j]) / max(abs(fd[j]), 1.0), 1.0e-6,
                msg=f"joint grad[{j}] analytic={g[j]} fd={fd[j]}",
            )

    def test_joint_hessian_vs_finite_differences(self):
        # The observed Hessian (basis of the variance-covariance matrix)
        # is checked component by component against central finite
        # differences of the analytic gradient.
        from nb2reg.core import _objective, _validate
        yf, Xm, off, _n, p = _validate(
            self.y, self.X, self.off, 1.0 / math.exp(math.log(0.6))
        )
        b = self.beta
        lt = math.log(0.6)
        _f, _g, H = _objective(b, lt, yf, Xm, off)
        eps = 1.0e-5

        def grad_a(pt, a):
            _, gg, _ = _objective(pt[:p], pt[p], yf, Xm, off)
            return gg[a]

        point = b + [lt]
        for a in range(p + 1):
            for c in range(p + 1):
                up = point[:]
                dn = point[:]
                up[c] += eps
                dn[c] -= eps
                fd = (grad_a(up, a) - grad_a(dn, a)) / (2 * eps)
                self.assertLess(
                    abs(H[a][c] - fd) / max(abs(fd), 1.0), 1.0e-5,
                    msg=f"H[{a}][{c}] analytic={H[a][c]} fd={fd}",
                )
        # Symmetry
        for a in range(p + 1):
            for c in range(a):
                self.assertEqual(H[a][c], H[c][a])

    def test_dispersion_args_exclusive(self):
        with self.assertRaises(NB2InputError):
            nb2_loglik(self.y, self.X, alpha=1.0, theta=1.0)
        with self.assertRaises(NB2InputError):
            nb2_loglik(self.y, self.X)
        with self.assertRaises(NB2InputError):
            nb2_loglik(self.y, self.X, alpha=0.0)

    def test_loglik_rejects_nonfinite_point(self):
        with self.assertRaises(NB2InputError):
            nb2_loglik(self.y, self.X, beta=[float("nan"), 0.0],
                       alpha=self.alpha)


if __name__ == "__main__":
    unittest.main()
