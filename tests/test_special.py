"""Tests for the hand-written digamma/trigamma and stable differences."""

import math
import unittest

from nb2reg.special import (
    digamma,
    digamma_diff,
    gamma_ratio_log,
    trigamma,
    trigamma_diff,
)


class TestPolyGamma(unittest.TestCase):
    def test_known_values(self):
        # psi(1) = -gamma_E ; psi'(1) = pi^2/6 ; psi(1/2) = -gamma - 2log2
        euler = 0.5772156649015329
        self.assertAlmostEqual(digamma(1.0), -euler, places=12)
        self.assertAlmostEqual(digamma(0.5), -euler - 2.0 * math.log(2.0),
                               places=12)
        self.assertAlmostEqual(trigamma(1.0), math.pi**2 / 6.0, places=12)
        self.assertAlmostEqual(trigamma(0.5), math.pi**2 / 2.0, places=12)

    def test_digamma_vs_central_finite_difference_lgamma(self):
        # Central FD of lgamma: at large x the lgamma values are O(100),
        # so ulp/(2 eps) limits the reference to ~1e-8 absolute even
        # though digamma itself is much more accurate.
        eps = 1.0e-6
        for x in [0.13, 0.7, 1.5, 4.2, 11.9, 50.0, 333.0]:
            fd = (math.lgamma(x + eps) - math.lgamma(x - eps)) / (2.0 * eps)
            self.assertLessEqual(
                abs(digamma(x) - fd) / max(abs(fd), 1.0), 5.0e-8,
                msg=f"digamma mismatch at x={x}",
            )

    def test_trigamma_vs_finite_difference_digamma(self):
        eps = 1.0e-6
        for x in [0.21, 1.0, 3.7, 12.4, 88.0]:
            fd = (digamma(x + eps) - digamma(x - eps)) / (2.0 * eps)
            self.assertLessEqual(
                abs(trigamma(x) - fd) / max(abs(fd), 1.0), 1.0e-9,
                msg=f"trigamma mismatch at x={x}",
            )

    def test_recurrence_identities(self):
        for x in [0.37, 1.2, 5.4]:
            self.assertAlmostEqual(digamma(x + 1.0) - digamma(x), 1.0 / x,
                                   places=12)
            self.assertAlmostEqual(
                trigamma(x + 1.0) - trigamma(x), -1.0 / (x * x), places=12
            )

    def test_nonpositive_rejected(self):
        for fn in (digamma, trigamma):
            with self.assertRaises(ValueError):
                fn(0.0)
            with self.assertRaises(ValueError):
                fn(-1.0)


class TestStableDifferences(unittest.TestCase):
    def test_gamma_ratio_exact_product_identity_small_y(self):
        # log Gamma(x+y)/Gamma(x) = sum_{j=0}^{y-1} log(x+j), accumulated
        # independently (math.fsum of the plain product logs).
        for x, y in [(0.37, 1), (1.0, 5), (2.7, 20), (100.0, 50)]:
            direct = math.fsum(math.log(x + j) for j in range(y))
            self.assertLessEqual(
                abs(gamma_ratio_log(x, y) - direct) / max(abs(direct), 1.0),
                1.0e-12,
            )

    def test_gamma_ratio_vs_naive_when_no_cancellation(self):
        for x, y in [(2.0, 3), (5.0, 10), (50.0, 30)]:
            naive = math.lgamma(x + y) - math.lgamma(x)
            self.assertLessEqual(abs(gamma_ratio_log(x, y) - naive), 1.0e-10)

    def test_gamma_ratio_stable_in_cancellation_regime(self):
        # theta huge vs y: the value is ~ y log(theta); the naive
        # subtraction has ~1e-7 absolute round-off but our exact sum must
        # agree with the analytic leading term to high precision.
        theta, y = 1.0e7, 10
        leading = y * math.log(theta)
        self.assertLessEqual(
            abs(gamma_ratio_log(theta, y) - leading) / leading, 1.0e-6
        )
        # and the exact y=1 case equals log(theta) essentially exactly.
        self.assertAlmostEqual(gamma_ratio_log(theta, 1), math.log(theta),
                               places=12)

    def test_digamma_diff_harmonic_identity(self):
        for x, y in [(0.91, 1), (2.0, 4), (10.0, 30), (1.0e7, 5)]:
            direct = math.fsum(1.0 / (x + j) for j in range(y))
            self.assertLessEqual(abs(digamma_diff(x, y) - direct), 1.0e-12)

    def test_trigamma_diff_identity(self):
        for x, y in [(0.91, 1), (2.0, 4), (10.0, 30), (1.0e7, 5)]:
            direct = -math.fsum(1.0 / (x + j) ** 2 for j in range(y))
            self.assertLessEqual(abs(trigamma_diff(x, y) - direct), 1.0e-12)

    def test_large_y_branches_match_small_argument_exact(self):
        # y above the exact-sum cutoff with small x: branch must agree
        # with an independent lgamma/digamma evaluation there.
        x, y = 0.5, 5000
        self.assertAlmostEqual(
            gamma_ratio_log(x, y),
            math.lgamma(x + y) - math.lgamma(x),
            places=8,
        )
        self.assertAlmostEqual(
            digamma_diff(x, y), digamma(x + y) - digamma(x), places=10
        )
        self.assertAlmostEqual(
            trigamma_diff(x, y), trigamma(x + y) - trigamma(x), places=10
        )

    def test_zero_step(self):
        for x in (0.4, 12.0, 1.0e6):
            self.assertEqual(gamma_ratio_log(x, 0), 0.0)
            self.assertEqual(digamma_diff(x, 0), 0.0)
            self.assertEqual(trigamma_diff(x, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
