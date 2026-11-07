"""The weighted quantile, the kernel, and what localized conformal buys.

The claim under test is not "localized conformal works". It is the narrower
and checkable one that the weighted construction *contains* the methods
already in the module -- uniform weights give split conformal, indicator
weights give Mondrian -- and that between those two settings it conditions on
more than one axis at a time, which is the thing every other method here
fails to do.

Several tests build data whose conditional structure is known by
construction, because on real load you cannot tell a method that fixed
conditional coverage from one that got lucky on the fold.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridcast.conformal import (
    BLOCKS_PER_DAY,
    BlockCovariates,
    CovariateKernel,
    LocalizedConformal,
    MondrianConformal,
    SplitConformal,
    conditional_coverage_gap,
    conformal_quantile,
    coverage_by_group,
    effective_sample_size,
    weighted_conformal_quantile,
)

ALPHA = 0.1


# -- the primitive ---------------------------------------------------------


@pytest.mark.parametrize("n", [9, 10, 37, 200, 1001])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_uniform_weights_reproduce_the_unweighted_quantile(n, alpha):
    """The invariant the whole generalisation rests on.

    With every weight equal and the test point carrying the same weight,
    ``p_i = 1/(n+1)`` and the threshold is crossed at the
    ``ceil((n+1)(1-alpha))``-th score -- which is the definition
    ``conformal_quantile`` implements. Exact equality, not approximate: these
    are the same order statistic, so any drift means the weighted path is
    picking a different one.
    """
    scores = np.abs(np.random.default_rng(n).normal(size=n))
    assert weighted_conformal_quantile(
        scores, np.ones(n), alpha
    ) == conformal_quantile(scores, alpha)


def test_scaling_every_weight_changes_nothing_except_through_the_test_point():
    """Doubling all calibration weights is not a no-op, and should not be.

    ``w_test`` is fixed at the kernel's maximum, so scaling the calibration
    weights up shifts the normalisation and lets the threshold be reached
    earlier. That is the intended behaviour -- more total weight means more
    evidence -- but it is worth pinning, because the obvious "normalise the
    weights to sum to 1 first" refactor silently destroys it and reintroduces
    the under-coverage that including the test point exists to prevent.
    """
    scores = np.linspace(1.0, 100.0, 50)
    small = weighted_conformal_quantile(scores, np.full(50, 0.01), ALPHA)
    large = weighted_conformal_quantile(scores, np.full(50, 100.0), ALPHA)
    assert small > large


def test_too_little_weight_cannot_support_the_claim():
    """Below ``w_test (1-alpha)/alpha`` of weight the honest answer is infinite.

    At 90% that is nine test-point-equivalents. Eight points at weight 1.0
    cannot get there however large their scores are, and the method says so
    rather than returning the biggest thing it happens to have seen.
    """
    scores = np.arange(1.0, 9.0)
    assert weighted_conformal_quantile(scores, np.ones(8), ALPHA) == float("inf")
    assert np.isfinite(
        weighted_conformal_quantile(scores, np.ones(8), ALPHA, fallback="max")
    )
    # Nine is exactly enough, which is the same boundary the unweighted
    # quantile has at 1/alpha - 1.
    assert np.isfinite(weighted_conformal_quantile(np.arange(1.0, 10.0), np.ones(9), ALPHA))


def test_zero_and_nonfinite_weights_are_dropped_not_counted():
    """A calibration point with zero weight must not occupy a slot.

    If zeros stayed in, a kernel that excludes most of the window would still
    have its quantile dragged by the count of excluded points, and the
    exclusion would do nothing.
    """
    scores = np.array([1.0, 2.0, 3.0, 1000.0, np.nan])
    weights = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
    kept = weighted_conformal_quantile(scores, weights, 0.5)
    assert kept == weighted_conformal_quantile(np.array([1.0, 2.0, 3.0]), np.ones(3), 0.5)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="align"):
        weighted_conformal_quantile(np.ones(5), np.ones(4), ALPHA)
    with pytest.raises(ValueError, match="alpha"):
        weighted_conformal_quantile(np.ones(5), np.ones(5), 1.5)
    with pytest.raises(ValueError, match="w_test"):
        weighted_conformal_quantile(np.ones(5), np.ones(5), ALPHA, w_test=0.0)


def test_effective_sample_size_counts_information_not_rows():
    """The number that says when a sharp kernel has run out of data."""
    assert effective_sample_size(np.ones(500)) == pytest.approx(500.0)
    # One point carrying almost all the weight is worth about one point,
    # however many near-zero neighbours surround it.
    dominated = np.concatenate([[1.0], np.full(999, 1e-6)])
    assert effective_sample_size(dominated) == pytest.approx(1.0, abs=0.01)
    assert effective_sample_size(np.zeros(10)) == 0.0


# -- the kernel ------------------------------------------------------------


def _covariates(blocks, kinds, temps=None):
    return BlockCovariates(np.asarray(blocks), np.asarray(kinds), temps)


def test_block_distance_wraps_around_midnight():
    """23:45 and 00:00 are adjacent, not 95 blocks apart.

    Without the wrap the kernel teaches the model that the last block of the
    day resembles nothing, which is exactly the block that already has the
    fewest observations.
    """
    kernel = CovariateKernel(block_bandwidth=4.0, kind_penalty=1.0)
    test = _covariates([0], ["weekday"])
    cal = _covariates([95, 1, 48], ["weekday"] * 3)
    weights = kernel.between(test, cal)[0]
    assert weights[0] == pytest.approx(weights[1])  # one block either side
    assert weights[2] < weights[0] / 100  # noon is genuinely far from midnight


def test_kind_penalty_zero_is_a_hard_partition():
    """The limiting case that makes Mondrian a special case rather than a rival."""
    kernel = CovariateKernel(block_bandwidth=1e6, kind_penalty=0.0)
    test = _covariates([10], ["holiday"])
    cal = _covariates([10, 10], ["holiday", "weekday"])
    weights = kernel.between(test, cal)[0]
    assert weights[0] > 0.0
    assert weights[1] == 0.0


def test_kind_penalty_one_ignores_day_kind():
    kernel = CovariateKernel(block_bandwidth=1e6, kind_penalty=1.0)
    weights = kernel.between(
        _covariates([10], ["holiday"]), _covariates([10, 10], ["holiday", "weekday"])
    )[0]
    assert weights[0] == pytest.approx(weights[1])


def test_missing_temperature_neither_favours_nor_excludes():
    """A block with no weather is compared on the other two axes.

    The alternative -- propagating NaN -- would zero the weight and silently
    drop every row the archive is missing, on top of the days it already
    costs.
    """
    kernel = CovariateKernel(block_bandwidth=1e6, temp_bandwidth=2.0, kind_penalty=1.0)
    test = _covariates([0], ["weekday"], np.array([30.0]))
    cal = _covariates([0, 0], ["weekday"] * 2, np.array([30.0, np.nan]))
    weights = kernel.between(test, cal)[0]
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(1.0)


def test_kernel_rejects_impossible_settings():
    with pytest.raises(ValueError, match="bandwidths"):
        CovariateKernel(block_bandwidth=0.0)
    with pytest.raises(ValueError, match="kind_penalty"):
        CovariateKernel(kind_penalty=1.5)


def test_covariates_must_align():
    with pytest.raises(ValueError, match="align"):
        BlockCovariates(np.arange(5), np.array(["weekday"] * 4))
    with pytest.raises(ValueError, match="temperature"):
        BlockCovariates(np.arange(5), np.array(["weekday"] * 5), np.ones(3))


# -- the method ------------------------------------------------------------


def _heteroscedastic(n_days=150, seed=7):
    """Load errors whose scale depends on two axes at once.

    Spread rises through the evening ramp *and* is larger on holidays. That
    is the situation the real backtest reports and no single-axis method
    handles: condition on the block and the holidays stay uncovered,
    condition on the day kind and the ramp does.

    Kept to a few months rather than three years because the kernel is
    quadratic in the calibration window: the weight matrix for a full
    dataset is 19,000 x 19,000, which is a minute of test time per
    assertion and no more informative than this.
    """
    rng = np.random.default_rng(seed)
    blocks = np.tile(np.arange(BLOCKS_PER_DAY), n_days)
    kinds = np.repeat(
        rng.choice(["weekday", "sunday", "holiday"], n_days, p=[0.75, 0.18, 0.07]),
        BLOCKS_PER_DAY,
    )
    ramp = 1.0 + 3.0 * np.exp(-0.5 * ((blocks - 76) / 8.0) ** 2)
    festival = np.where(kinds == "holiday", 2.5, 1.0)
    scale = 100.0 * ramp * festival
    residuals = rng.normal(0.0, scale)
    return BlockCovariates(blocks, kinds), residuals


def test_localized_conditions_on_both_axes_where_mondrian_fixes_one():
    """The result the method exists for, on data whose structure is known.

    Mondrian on block of day is given the advantage here -- it conditions on
    the stronger of the two effects -- and still leaves the holiday groups
    short, because nothing in it knows holidays exist. Localized conformal
    sees both through one kernel.
    """
    cov, residuals = _heteroscedastic()
    half = residuals.size // 2
    cal_cov, test_cov = cov.take(np.arange(half)), cov.take(np.arange(half, cov.block.size))
    cal_res, test_res = residuals[:half], residuals[half:]
    point = np.zeros(test_res.size)

    mondrian = MondrianConformal(ALPHA).fit(cal_res, groups=cal_cov.block)
    m_bands = mondrian.predict(point, groups=test_cov.block)
    local = LocalizedConformal(
        ALPHA, kernel=CovariateKernel(block_bandwidth=6.0, kind_penalty=0.02)
    ).fit(cal_res, covariates=cal_cov)
    l_bands = local.predict(point, covariates=test_cov)

    m_kind = conditional_coverage_gap(test_res, m_bands, test_cov.day_kind, min_group_n=50)
    l_kind = conditional_coverage_gap(test_res, l_bands, test_cov.day_kind, min_group_n=50)
    # Both stay honest on the axis Mondrian partitions...
    m_block = conditional_coverage_gap(test_res, m_bands, test_cov.block, min_group_n=50)
    l_block = conditional_coverage_gap(test_res, l_bands, test_cov.block, min_group_n=50)
    assert l_block["spread"] < m_block["spread"] * 1.5
    # ...and only the kernel closes the one it does not.
    assert l_kind["spread"] < m_kind["spread"]
    assert l_kind["worst_group_coverage"] > m_kind["worst_group_coverage"]


def test_localized_beats_pooled_on_the_axis_pooling_ignores():
    """If it does not beat split conformal conditionally, the weighting bought nothing."""
    cov, residuals = _heteroscedastic()
    half = residuals.size // 2
    cal_cov, test_cov = cov.take(np.arange(half)), cov.take(np.arange(half, cov.block.size))
    point = np.zeros(residuals.size - half)

    split = SplitConformal(ALPHA).fit(residuals[:half]).predict(point)
    local = (
        LocalizedConformal(ALPHA, kernel=CovariateKernel(block_bandwidth=6.0))
        .fit(residuals[:half], covariates=cal_cov)
        .predict(point, covariates=test_cov)
    )
    s_gap = conditional_coverage_gap(residuals[half:], split, test_cov.block, min_group_n=50)
    l_gap = conditional_coverage_gap(residuals[half:], local, test_cov.block, min_group_n=50)
    assert l_gap["spread"] < s_gap["spread"]


def test_a_flat_kernel_degenerates_to_split_conformal():
    """Widen every bandwidth and the method must stop doing anything.

    A generalisation that cannot reproduce the thing it generalises is a
    different method wearing the name. Every radius should collapse to the
    one pooled width, give or take the single order statistic that the test
    point's weight shifts.
    """
    cov, residuals = _heteroscedastic(n_days=120)
    flat = CovariateKernel(block_bandwidth=1e9, temp_bandwidth=1e9, kind_penalty=1.0)
    local = LocalizedConformal(ALPHA, kernel=flat).fit(residuals, covariates=cov)
    bands = local.predict(np.zeros(len(cov)), covariates=cov)
    pooled = SplitConformal(ALPHA).fit(residuals).q_
    assert np.allclose(bands.radius, bands.radius[0])
    assert bands.radius[0] == pytest.approx(pooled, rel=0.02)


def test_a_starved_kernel_reports_itself_rather_than_inventing_a_number():
    """Sharpen past the data and the method must admit it.

    This is the failure mode that makes kernel methods dangerous: narrow
    bandwidths look like better conditional coverage right up to the point
    where each interval rests on four residuals. ``starved_`` is how that
    becomes visible instead of becoming a confident wrong answer.
    """
    cov, residuals = _heteroscedastic(n_days=40)
    sharp = CovariateKernel(block_bandwidth=0.05, temp_bandwidth=0.05, kind_penalty=0.0)
    local = LocalizedConformal(ALPHA, kernel=sharp).fit(residuals, covariates=cov)
    local.predict(np.zeros(len(cov)), covariates=cov)
    assert local.starved_ > 0
    assert local.ess_.mean() < 50
    # And a wide kernel on the same data is not starved at all.
    wide = LocalizedConformal(
        ALPHA, kernel=CovariateKernel(block_bandwidth=12.0, kind_penalty=0.3)
    ).fit(residuals, covariates=cov)
    wide.predict(np.zeros(len(cov)), covariates=cov)
    assert wide.starved_ == 0
    assert wide.ess_.mean() > local.ess_.mean()


def test_chunking_does_not_change_the_answer():
    """The chunk size is a memory decision, not a modelling one."""
    cov, residuals = _heteroscedastic(n_days=60)
    kernel = CovariateKernel(block_bandwidth=7.0)
    point = np.zeros(len(cov))
    one = LocalizedConformal(ALPHA, kernel=kernel, chunk=100000).fit(
        residuals, covariates=cov
    ).predict(point, covariates=cov)
    many = LocalizedConformal(ALPHA, kernel=kernel, chunk=37).fit(
        residuals, covariates=cov
    ).predict(point, covariates=cov)
    assert np.array_equal(one.radius, many.radius)


def test_marginal_coverage_still_lands_near_nominal():
    """Conditional coverage is the goal; marginal coverage is the sanity check."""
    cov, residuals = _heteroscedastic()
    half = residuals.size // 2
    local = LocalizedConformal(
        ALPHA, kernel=CovariateKernel(block_bandwidth=6.0, kind_penalty=0.05)
    ).fit(residuals[:half], covariates=cov.take(np.arange(half)))
    bands = local.predict(
        np.zeros(residuals.size - half),
        covariates=cov.take(np.arange(half, residuals.size)),
    )
    assert 0.86 < bands.evaluate(residuals[half:])["coverage"] < 0.94


def test_predict_without_covariates_or_fit_is_an_error():
    cov, residuals = _heteroscedastic(n_days=20)
    with pytest.raises(ValueError, match="covariates"):
        LocalizedConformal(ALPHA).fit(residuals)
    with pytest.raises(ValueError, match="fit before predict"):
        LocalizedConformal(ALPHA).predict(np.zeros(3), covariates=cov.take(np.arange(3)))
    fitted = LocalizedConformal(ALPHA).fit(residuals, covariates=cov)
    with pytest.raises(ValueError, match="covariates"):
        fitted.predict(np.zeros(len(cov)))
    with pytest.raises(ValueError, match="align"):
        fitted.predict(np.zeros(3), covariates=cov)


def test_group_coverage_rows_cover_every_day_kind_present():
    """Guards the plumbing the backtest reports from: no silently dropped group."""
    cov, residuals = _heteroscedastic(n_days=100)
    bands = (
        LocalizedConformal(ALPHA)
        .fit(residuals, covariates=cov)
        .predict(np.zeros(len(cov)), covariates=cov)
    )
    rows = coverage_by_group(residuals, bands, cov.day_kind)
    assert {row["group"] for row in rows} == set(np.unique(cov.day_kind))
    assert sum(row["n"] for row in rows) == residuals.size
