"""Interval tests, each with a negative control."""

import math
import unittest

from thinktrace.stats import mean_sd, paired_diff, ratio_ci, wilson


class TestWilson(unittest.TestCase):
    def test_matches_a_hand_computed_value(self):
        lo, hi = wilson(20, 40)
        # Wilson at p = 0.5, n = 40, z = 1.959964. denom = 1 + z^2/40 = 1.0960375,
        # centre = 0.5, half = (z/denom) * sqrt(0.25/40 + z^2/6400) = 0.1480047.
        self.assertAlmostEqual(lo, 0.3519953, places=6)
        self.assertAlmostEqual(hi, 0.6480047, places=6)

    def test_brackets_the_point_estimate(self):
        for k, n in [(0, 40), (1, 40), (20, 40), (39, 40), (40, 40)]:
            lo, hi = wilson(k, n)
            self.assertLessEqual(lo, k / n + 1e-12, f"{k}/{n}")
            self.assertGreaterEqual(hi, k / n - 1e-12, f"{k}/{n}")

    def test_negative_control_interval_is_not_degenerate(self):
        # A zero width interval is the shape a sabotaged interval takes, and it
        # would make every comparison look decisive. At n = 40 it must not happen,
        # not even at the boundaries where the normal approximation collapses.
        for k in (0, 40):
            lo, hi = wilson(k, 40)
            self.assertGreater(hi - lo, 0.05, f"k={k}")

    def test_negative_control_more_data_narrows_it(self):
        w_small = wilson(5, 10)
        w_big = wilson(500, 1000)
        self.assertGreater(w_small[1] - w_small[0], (w_big[1] - w_big[0]) * 5)

    def test_empty_sample_does_not_divide_by_zero(self):
        self.assertEqual(wilson(0, 0), (0.0, 0.0))


class TestMeanSd(unittest.TestCase):
    def test_known_values(self):
        m, s = mean_sd([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        self.assertAlmostEqual(m, 5.0)
        self.assertAlmostEqual(s, math.sqrt(32.0 / 7.0))

    def test_negative_control_constant_data_has_zero_spread(self):
        _, s = mean_sd([3.0] * 10)
        self.assertAlmostEqual(s, 0.0)

    def test_single_sample_is_not_an_error(self):
        self.assertEqual(mean_sd([1.5]), (1.5, 0.0))


class TestPairedDiff(unittest.TestCase):
    def test_a_uniform_improvement_is_significant(self):
        r = paired_diff([1.0] * 40)
        self.assertAlmostEqual(r["mean"], 1.0)
        self.assertTrue(r["significant"])

    def test_negative_control_a_wash_is_not_significant(self):
        r = paired_diff(([1.0] * 8) + ([-1.0] * 8) + ([0.0] * 24))
        self.assertAlmostEqual(r["mean"], 0.0)
        self.assertFalse(r["significant"])
        self.assertLess(r["lo"], 0.0)
        self.assertGreater(r["hi"], 0.0)

    def test_a_uniform_regression_is_significant_and_negative(self):
        r = paired_diff([-1.0] * 40)
        self.assertTrue(r["significant"])
        self.assertLess(r["mean"], 0.0)

    def test_interval_contains_the_mean(self):
        diffs = ([1.0] * 12) + ([0.0] * 25) + ([-1.0] * 3)
        r = paired_diff(diffs)
        self.assertLessEqual(r["lo"], r["mean"])
        self.assertGreaterEqual(r["hi"], r["mean"])

    def test_negative_control_a_small_effect_in_a_small_sample_is_not_called(self):
        # One extra win out of forty is not detectable, and the code must say so.
        r = paired_diff(([1.0] * 1) + ([0.0] * 39))
        self.assertFalse(r["significant"])

    def test_all_zero_diffs_are_not_significant(self):
        r = paired_diff([0.0] * 40)
        self.assertFalse(r["significant"])
        self.assertEqual(r["half_width"], 0.0)


class TestRatioCi(unittest.TestCase):
    def test_ratio_of_means(self):
        r = ratio_ci([10.0] * 5, [2.0] * 5)
        self.assertAlmostEqual(r["ratio"], 5.0)

    def test_negative_control_zero_denominator_is_none_not_infinity(self):
        r = ratio_ci([10.0] * 5, [0.0] * 5)
        self.assertIsNone(r["ratio"])


if __name__ == "__main__":
    unittest.main()
