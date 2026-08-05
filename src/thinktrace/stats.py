"""Interval estimates, written in closed form so a second implementation in a
different language can reproduce them without sharing code.

Two intervals are reported:

  wilson(k, n)          95 percent Wilson score interval for one accuracy.
                        Used because the normal approximation misbehaves near
                        0 and 1, which is exactly where several of these cells sit.

  paired_diff(pairs)    95 percent interval on the mean of the per-item
                        differences d_i = correct_on - correct_off, each in
                        {-1, 0, 1}. The two conditions see the identical items,
                        so the paired interval is the one that answers "did
                        thinking change anything", and it is much tighter than
                        comparing two independent Wilson intervals.

If the paired interval contains 0, the two conditions are reported as
indistinguishable at this sample size. That phrasing is deliberate: it is a
statement about the experiment, not a claim that the effect is zero.
"""

from __future__ import annotations

import math

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95 percent Wilson score interval for k successes out of n."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_sd(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(values) / n
    if n < 2:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return (mean, math.sqrt(var))


def paired_diff(diffs: list[float], z: float = Z95) -> dict:
    """Interval on the mean paired difference, plus a verdict."""
    n = len(diffs)
    mean, sd = mean_sd(diffs)
    half = z * sd / math.sqrt(n) if n > 0 else 0.0
    lo, hi = mean - half, mean + half
    return {
        "n": n,
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "half_width": half,
        "significant": not (lo <= 0.0 <= hi),
    }


def ratio_ci(values_a: list[float], values_b: list[float], z: float = Z95) -> dict:
    """Interval on the mean of A and the mean of B, reported separately.

    Used for cost columns where a ratio of means is the headline number and the
    two means each need their own uncertainty.
    """
    ma, sa = mean_sd(values_a)
    mb, sb = mean_sd(values_b)
    na, nb = len(values_a), len(values_b)
    ha = z * sa / math.sqrt(na) if na else 0.0
    hb = z * sb / math.sqrt(nb) if nb else 0.0
    return {
        "a_mean": ma, "a_lo": ma - ha, "a_hi": ma + ha, "a_n": na,
        "b_mean": mb, "b_lo": mb - hb, "b_hi": mb + hb, "b_n": nb,
        "ratio": (ma / mb) if mb else None,
    }
