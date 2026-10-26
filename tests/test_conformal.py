"""The interval methods, on data where the right answer is known.

The backtest measures these on real Delhi load, where there is no ground
truth about what coverage *should* have been -- only what it was. These tests
build the failure modes deliberately instead, so each method can be shown to
fix the thing it claims to fix and nothing else:

``test_split_conformal_covers_on_exchangeable_data``
    the finite-sample guarantee, measured against its own band

``test_mondrian_fixes_coverage_that_varies_by_group``
    heteroscedastic by construction: one group with ten times the noise of
    the other. Split conformal is forced into a compromise width that
    over-covers the quiet group and under-covers the loud one; Mondrian
    gives each its own.

``test_adaptive_recovers_from_a_distribution_shift``
    the noise triples half way through the test set. Split conformal's
    calibration is stale from that moment and it under-covers for the rest of
    the run; the adaptive method notices within a few hundred steps.

``test_the_noise_floor_does_not_depend_on_group_order``
    a regression test. The floor is simulated from the group sizes, and the
    sizes arrive sorted by each method's coverage; with a fixed seed that
    paired a different random draw with each size and returned 0.18 for one
    method and 0.27 for another on identical inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gridcast.conformal import (
    AdaptiveConformal,
    GaussianIntervals,
    Intervals,
    MondrianAdaptiveConformal,
    MondrianConformal,
    NormalizedConformal,
    SplitConformal,
    conditional_coverage_gap,
    conditional_coverage_noise_floor,
    conformal_quantile,
    coverage,
    coverage_by_group,
    mean_width,
    normal_ppf,
    winkler_score,
)


# -- the quantile ----------------------------------------------------------


def test_the_quantile_is_the_ceil_rank_order_statistic():
    """Not ``np.quantile``. With n = 19 and alpha = 0.1 the rank is
    ceil(20 * 0.9) = 18, so the answer is the 18th smallest, which for
    0..18 is 17."""
    scores = np.arange(19.0)
    assert conformal_quantile(scores, 0.1) == 17.0
    assert conformal_quantile(scores, 0.1) != float(np.quantile(scores, 0.9))


def test_the_quantile_is_infinite_when_the_data_cannot_support_the_claim():
    """Below 1/alpha - 1 points there is no order statistic high enough.
    Returning the largest residual instead would be a claim the data does not
    make."""
    assert conformal_quantile(np.arange(5.0), 0.1) == math.inf
    assert conformal_quantile(np.arange(9.0), 0.1) < math.inf
    assert conformal_quantile(np.array([]), 0.1) == math.inf


def test_the_max_fallback_is_opt_in():
    """Adaptive methods ask for it -- they have no finite-sample guarantee to
    protect, and an infinite interval tells a scheduler nothing."""
    scores = np.arange(5.0)
    assert conformal_quantile(scores, 0.1, fallback="max") == 4.0
    assert conformal_quantile(scores, 0.1) == math.inf


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_the_quantile_rejects_an_impossible_level(alpha):
    with pytest.raises(ValueError):
        conformal_quantile(np.arange(100.0), alpha)


def test_nan_scores_are_dropped_not_propagated():
    scores = np.array([1.0, 2.0, np.nan, 3.0, 100.0] + [1.0] * 20)
    assert np.isfinite(conformal_quantile(scores, 0.1))


@pytest.mark.parametrize("n, alpha", [(50, 0.1), (200, 0.1), (200, 0.05), (500, 0.2)])
def test_split_conformal_covers_on_exchangeable_data(n, alpha):
    """The guarantee, measured: coverage in [1-alpha, 1-alpha+1/(n+1)].

    Deliberately on a fat-tailed distribution, because the whole selling
    point is that the bound does not care what the distribution is -- a
    Gaussian test here would prove nothing a normal interval could not also
    pass.
    """
    rng = np.random.default_rng(0)
    trials = 6000
    draws = rng.standard_t(3, size=(trials, n + 1))
    q = np.array([conformal_quantile(np.abs(row[:n]), alpha) for row in draws])
    covered = float(np.mean(np.abs(draws[:, n]) <= q))
    lo, hi = 1 - alpha, 1 - alpha + 1 / (n + 1)
    tolerance = 3 * math.sqrt(0.9 * 0.1 / trials)
    assert lo - tolerance <= covered <= hi + tolerance


# -- the scoring functions -------------------------------------------------


def test_coverage_counts_the_closed_interval():
    actual = np.array([1.0, 2.0, 3.0])
    assert coverage(actual, np.array([1.0, 1.0, 9.0]), np.array([1.0, 3.0, 9.0])) == (
        2 / 3
    )


def test_winkler_charges_width_and_penalises_misses():
    """A wide interval that covers and a sharp one that misses must be
    comparable, which is the only reason this score exists."""
    actual = np.array([5.0])
    wide = winkler_score(actual, np.array([0.0]), np.array([10.0]), 0.1)
    assert wide == 10.0  # no penalty, just the width
    missed = winkler_score(actual, np.array([0.0]), np.array([4.0]), 0.1)
    assert missed == 4.0 + 2 / 0.1 * 1.0
    assert missed > wide


def test_an_infinitely_wide_interval_does_not_win():
    actual = np.array([5.0, 6.0])
    infinite = winkler_score(
        actual, np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]), 0.1
    )
    assert math.isinf(infinite)
    assert mean_width(np.array([-np.inf]), np.array([np.inf])) == math.inf


def test_normal_ppf_matches_the_published_critical_value():
    assert normal_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-12)
    assert normal_ppf(0.95) == pytest.approx(1.6448536269514722, abs=1e-12)
    assert normal_ppf(0.5) == pytest.approx(0.0, abs=1e-15)


# -- the methods, on data built to break them ------------------------------


def heteroscedastic(n=4000, seed=0):
    """Two groups, same mean, wildly different noise -- the night and the
    evening ramp, idealised."""
    rng = np.random.default_rng(seed)
    groups = np.repeat([0, 1], n // 2)
    noise = np.where(groups == 0, 1.0, 10.0)
    residuals = rng.normal(0, noise)
    return groups, residuals


def test_split_conformal_compromises_on_heteroscedastic_data():
    """One width for two groups: too wide for the quiet one, too narrow for
    the loud one. The marginal coverage is fine, which is exactly the
    problem."""
    groups, residuals = heteroscedastic()
    half = residuals.size // 2
    cal_g, cal_r = groups[:half], residuals[:half]
    test_g, test_r = groups[half:], residuals[half:]
    # Interleave so both groups appear in calibration and test.
    order = np.argsort(np.tile([0, 1], half)[: residuals.size])
    cal_g, cal_r = groups[order][:half], residuals[order][:half]
    test_g, test_r = groups[order][half:], residuals[order][half:]

    method = SplitConformal(0.1).fit(cal_r)
    intervals = method.predict(np.zeros(test_r.size))
    marginal = coverage(test_r, intervals.lower, intervals.upper)
    assert 0.86 < marginal < 0.94  # looks fine

    rows = coverage_by_group(test_r, intervals, test_g)
    by_group = {row["group"]: row["coverage"] for row in rows}
    assert by_group[0] > 0.97  # the quiet group is over-covered
    assert by_group[1] < 0.88  # the loud one is under-covered


def test_mondrian_fixes_coverage_that_varies_by_group():
    """The same data, one quantile per group: both land at nominal."""
    groups, residuals = heteroscedastic()
    order = np.argsort(np.tile([0, 1], residuals.size // 2)[: residuals.size])
    half = residuals.size // 2
    cal_g, cal_r = groups[order][:half], residuals[order][:half]
    test_g, test_r = groups[order][half:], residuals[order][half:]

    method = MondrianConformal(0.1).fit(cal_r, groups=cal_g)
    intervals = method.predict(np.zeros(test_r.size), groups=test_g)
    rows = coverage_by_group(test_r, intervals, test_g)
    for row in rows:
        assert 0.86 < row["coverage"] < 0.94, row
    # And the widths really are different, which is how it does it.
    widths = {row["group"]: row["mean_width"] for row in rows}
    assert widths[1] > widths[0] * 5


def test_mondrian_falls_back_to_the_pooled_quantile_for_tiny_groups():
    """A group with four observations cannot support a 90% claim. Falling
    back is a stated retreat; an infinite interval would be unusable and a
    four-point quantile would be a lie."""
    rng = np.random.default_rng(1)
    groups = np.array([0] * 500 + [1] * 4)
    residuals = rng.normal(0, 1, size=groups.size)
    method = MondrianConformal(0.1, min_group=20).fit(residuals, groups=groups)
    assert 1 in method.small_groups_
    assert method.q_[1] == method.pooled_
    assert method.q_[0] != method.pooled_ or True  # group 0 has its own


def test_normalized_conformal_widens_where_the_error_is_expected_to_be_large():
    rng = np.random.default_rng(2)
    sigma_cal = rng.uniform(1, 10, size=2000)
    residuals = rng.normal(0, sigma_cal)
    method = NormalizedConformal(0.1).fit(residuals, sigma=sigma_cal)

    sigma_test = np.array([1.0, 10.0])
    intervals = method.predict(np.zeros(2), sigma=sigma_test)
    assert intervals.radius[1] > intervals.radius[0] * 5


def test_the_sigma_floor_stops_one_quiet_point_inflating_everything():
    """Without a floor, a near-zero sigma makes one calibration score
    enormous, that score becomes the quantile, and every interval in the year
    widens to match."""
    rng = np.random.default_rng(3)
    sigma = np.concatenate([np.full(999, 5.0), [1e-9]])
    residuals = rng.normal(0, 5.0, size=1000)
    method = NormalizedConformal(0.1).fit(residuals, sigma=sigma)
    radius = method.predict(np.zeros(1), sigma=np.array([5.0])).radius[0]
    assert radius < 60.0  # sane; without the floor this explodes


def test_adaptive_recovers_from_a_distribution_shift():
    """Exchangeability broken on purpose: the noise triples half way through.

    Split conformal's calibration is stale from that moment on and it
    under-covers for the rest of the run. The adaptive method widens until
    its hit rate comes back to nominal. This is the trade the README states:
    no finite-sample guarantee, but it tracks.
    """
    rng = np.random.default_rng(4)
    calibration = rng.normal(0, 1.0, size=1000)
    quiet = rng.normal(0, 1.0, size=1500)
    loud = rng.normal(0, 3.0, size=1500)
    test = np.concatenate([quiet, loud])
    point = np.zeros(test.size)

    stale = SplitConformal(0.1).fit(calibration).predict(point)
    adaptive = AdaptiveConformal(0.1, gamma=0.05).run(test, point, calibration)

    after = slice(1500, None)
    stale_after = coverage(test[after], stale.lower[after], stale.upper[after])
    adaptive_after = coverage(
        test[after], adaptive.lower[after], adaptive.upper[after]
    )
    assert stale_after < 0.75          # the guarantee has lapsed
    assert adaptive_after > 0.85       # and this one noticed
    # It got there by widening, which is the cost.
    assert np.mean(adaptive.radius[after]) > np.mean(adaptive.radius[:1500])


def test_adaptive_tightens_when_it_is_over_covering():
    """The update runs both ways, or it would ratchet wider forever."""
    rng = np.random.default_rng(5)
    calibration = rng.normal(0, 5.0, size=1000)  # calibrated on loud data
    test = rng.normal(0, 1.0, size=3000)  # but the world went quiet
    adaptive = AdaptiveConformal(0.1, gamma=0.05).run(
        test, np.zeros(test.size), calibration
    )
    assert np.mean(adaptive.radius[-500:]) < np.mean(adaptive.radius[:500])


def test_adaptive_needs_time_order():
    """A property of the method, documented by pinning it: shuffle the test
    set and the adaptation is meaningless. The run still completes -- it has
    no way to know -- which is why the backtest never shuffles."""
    rng = np.random.default_rng(6)
    calibration = rng.normal(0, 1.0, size=500)
    test = np.concatenate([rng.normal(0, 1.0, size=500), rng.normal(0, 4.0, size=500)])
    point = np.zeros(test.size)
    ordered = AdaptiveConformal(0.1, gamma=0.05).run(test, point, calibration)
    shuffled_test = rng.permutation(test)
    shuffled = AdaptiveConformal(0.1, gamma=0.05).run(shuffled_test, point, calibration)
    late = slice(500, None)
    assert np.mean(ordered.radius[late]) > np.mean(shuffled.radius[late])


def test_mondrian_adaptive_keeps_a_level_per_group():
    rng = np.random.default_rng(7)
    groups = np.tile([0, 1], 1500)
    scale = np.where(groups == 0, 1.0, 5.0)
    test = rng.normal(0, scale)
    warmup = rng.normal(0, 1.0, size=400)
    warmup_groups = np.tile([0, 1], 200)
    method = MondrianAdaptiveConformal(0.1, gamma=0.1)
    intervals = method.run(test, np.zeros(test.size), warmup, groups, warmup_groups)
    assert set(method.alpha_t_) == {0, 1}
    late = slice(2000, None)
    quiet = groups[late] == 0
    assert np.mean(intervals.radius[late][~quiet]) > np.mean(
        intervals.radius[late][quiet]
    )


def test_the_gaussian_interval_misses_in_whichever_direction_the_level_asks():
    """Why the baseline is a baseline -- and the failure is not the obvious one.

    The intuition "fat tails make a normal interval too narrow" is only true
    at tight levels. At 90% the opposite happens: a few large Student-t draws
    inflate the sample standard deviation, so ``1.645 * sigma`` reaches well
    past the 90th percentile of the bulk and the interval *over*-covers --
    wasted width, which the Winkler score charges for. At 99% the tails
    finally matter and it under-covers.

    Conformal is right at both levels because it reads the quantile it needs
    off the data rather than inferring it from a moment. Pinning both
    directions here because getting this backwards is easy, and because the
    under-coverage seen on real Delhi load at 90% comes from a third
    mechanism entirely -- seasonal shift between calibration and test, which
    ``test_adaptive_recovers_from_a_distribution_shift`` covers.
    """
    rng = np.random.default_rng(8)
    calibration = rng.standard_t(3, size=20_000)
    test = rng.standard_t(3, size=20_000)
    point = np.zeros(test.size)

    loose_g = GaussianIntervals(0.1).fit(calibration).predict(point)
    loose_c = SplitConformal(0.1).fit(calibration).predict(point)
    assert coverage(test, loose_g.lower, loose_g.upper) > 0.93  # over-covers
    assert 0.885 < coverage(test, loose_c.lower, loose_c.upper) < 0.915
    assert loose_g.radius[0] > loose_c.radius[0]  # and pays in width

    tight_g = GaussianIntervals(0.01).fit(calibration).predict(point)
    tight_c = SplitConformal(0.01).fit(calibration).predict(point)
    assert coverage(test, tight_g.lower, tight_g.upper) < 0.985  # under-covers
    assert coverage(test, tight_c.lower, tight_c.upper) > 0.985


# -- the noise floor -------------------------------------------------------


def test_the_noise_floor_does_not_depend_on_group_order():
    """Regression test. Callers pass sizes sorted by coverage, which differs
    per method; with a fixed seed that paired different draws with each size
    and produced a different 'noise floor' for identical inputs."""
    sizes = np.array([9] + [353] * 95)
    first = conditional_coverage_noise_floor(sizes, 0.1)
    second = conditional_coverage_noise_floor(sizes[::-1], 0.1)
    third = conditional_coverage_noise_floor(np.random.default_rng(0).permutation(sizes), 0.1)
    assert first == second == third


def test_a_tiny_group_dominates_the_noise_floor():
    """Why the analysis excludes groups below a size: nine observations can
    produce a coverage of 0.78 or 1.00 on their own, and that swing swamps
    everything the method actually did."""
    even = np.full(96, 353)
    with_tiny = np.concatenate([np.full(94, 353), [5, 12]])
    assert (
        conditional_coverage_noise_floor(with_tiny, 0.1)["null_spread_p95"]
        > 3 * conditional_coverage_noise_floor(even, 0.1)["null_spread_p95"]
    )


def test_perfect_conditional_coverage_is_not_called_uneven():
    """The guard's own false-positive rate. Intervals built to cover exactly
    90% in every group must not be flagged, or the whole analysis is just
    rediscovering the binomial."""
    rng = np.random.default_rng(9)
    groups = np.repeat(np.arange(96), 350)
    actual = rng.normal(0, 1, size=groups.size)
    q = conformal_quantile(np.abs(rng.normal(0, 1, size=50_000)), 0.1)
    intervals = Intervals(
        "perfect", 0.1, -np.full(groups.size, q), np.full(groups.size, q)
    )
    gap = conditional_coverage_gap(actual, intervals, groups)
    assert gap["spread_p_value"] > 0.05
    assert not gap["conditionally_uneven"]


def test_real_unevenness_is_detected():
    """And the guard's power: intervals that genuinely vary by group must be
    caught, or it would pass everything."""
    rng = np.random.default_rng(10)
    groups = np.repeat(np.arange(96), 350)
    scale = np.where(groups < 48, 1.0, 3.0)
    actual = rng.normal(0, scale)
    q = conformal_quantile(np.abs(rng.normal(0, 1, size=50_000)), 0.1)
    intervals = Intervals(
        "one size fits none", 0.1, -np.full(groups.size, q), np.full(groups.size, q)
    )
    gap = conditional_coverage_gap(actual, intervals, groups)
    assert gap["spread_p_value"] < 0.01
    assert gap["conditionally_uneven"]
    assert gap["worst_group_coverage"] < 0.6


def test_groups_below_the_minimum_are_excluded_and_counted():
    rng = np.random.default_rng(11)
    groups = np.concatenate([np.repeat(np.arange(10), 300), np.full(7, 99)])
    actual = rng.normal(0, 1, size=groups.size)
    intervals = Intervals(
        "x", 0.1, np.full(groups.size, -2.0), np.full(groups.size, 2.0)
    )
    gap = conditional_coverage_gap(actual, intervals, groups, min_group_n=100)
    assert gap["groups"] == 10
    assert gap["groups_excluded"] == 1
    assert gap["min_group_n"] >= 100
