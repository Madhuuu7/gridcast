"""The rupee model: its asymmetries, and the places it must refuse to guess.

Most of these pin behaviour that a simpler implementation would get wrong in
a direction that flatters the project -- pricing an unbounded interval as
free, pricing under-drawal like over-drawal, or reporting a floor on the bill
that no outcome could have produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridcast.settlement import (
    BLOCK_HOURS,
    CRORE,
    DeviationSettlement,
    annualise,
    budget_calibration,
    cost_summary,
)

SCHEDULE = 5000.0


def _flat(value: float, n: int = 4) -> np.ndarray:
    return np.full(n, value)


# -- the charge ------------------------------------------------------------


def test_a_perfect_schedule_costs_nothing():
    s = DeviationSettlement()
    assert s.charge(_flat(SCHEDULE), _flat(SCHEDULE)).tolist() == [0.0] * 4


def test_the_volume_limit_is_the_lower_of_the_two_rules():
    """12% of schedule, or 150 MW, whichever binds.

    At Delhi's scale the MW cap always binds -- 12% of 5,000 MW is 600 --
    and the fraction only matters for a small schedule. Both branches are
    checked because a model that only ever exercises one of them is a
    constant with extra steps.
    """
    s = DeviationSettlement()
    assert s.volume_limit(np.array([5000.0]))[0] == pytest.approx(150.0)
    assert s.volume_limit(np.array([500.0]))[0] == pytest.approx(60.0)


def test_deviation_inside_the_limit_is_priced_at_the_normal_rate():
    s = DeviationSettlement(normal_rate=12.0)
    # 100 MW over, inside the 150 MW limit, for a quarter hour.
    charge = s.charge(np.array([SCHEDULE]), np.array([SCHEDULE + 100.0]))[0]
    expected = 100.0 * BLOCK_HOURS * 1000.0 * 12.0
    assert charge == pytest.approx(expected)


def test_over_drawal_beyond_the_limit_costs_more_than_under_drawal():
    """The asymmetry the regulation exists for, and the reason this module
    does not take an absolute value before pricing.

    Drawing more than declared threatens the grid; drawing less does not.
    A model that prices both the same removes the reason under-forecasting
    is the expensive direction.
    """
    s = DeviationSettlement()
    over = s.charge(np.array([SCHEDULE]), np.array([SCHEDULE + 400.0]))[0]
    under = s.charge(np.array([SCHEDULE]), np.array([SCHEDULE - 400.0]))[0]
    assert over > under
    # Under-drawal beyond the limit is free, so only the first 150 MW count.
    assert under == pytest.approx(150.0 * BLOCK_HOURS * 1000.0 * s.normal_rate)
    # Over-drawal: 150 at normal, 250 at double.
    assert over == pytest.approx(
        (150.0 + 250.0 * 2.0) * BLOCK_HOURS * 1000.0 * s.normal_rate
    )


def test_under_drawal_can_be_charged_when_the_regime_says_so():
    s = DeviationSettlement(charge_excess_under_drawal=True)
    over = s.charge(np.array([SCHEDULE]), np.array([SCHEDULE + 400.0]))[0]
    under = s.charge(np.array([SCHEDULE]), np.array([SCHEDULE - 400.0]))[0]
    assert under > 0
    assert under < over  # still cheaper: no 2x multiplier


@pytest.mark.parametrize(
    "frequency, expected",
    [(49.85, 2.0), (49.92, 1.5), (50.0, 1.0), (50.04, 0.75), (None, 1.0)],
)
def test_frequency_slabs(frequency, expected):
    assert DeviationSettlement().frequency_multiplier(frequency) == expected


def test_no_frequency_data_means_the_normal_rate_not_a_guessed_slab():
    """This project has no source for 15-minute grid frequency.

    Picking a slab and reporting the result as a bill would be the single
    most misleading thing this module could do, so the absence is the
    default and ``frequency_hz`` exists to show sensitivity rather than to
    be quietly filled in.
    """
    s = DeviationSettlement()
    assert s.frequency_multiplier(None) == 1.0
    assert s.frequency_multiplier(float("nan")) == 1.0


def test_a_non_finite_input_is_not_free():
    """An unbounded interval edge priced as zero makes it look costless.

    NaN propagates so that ``nansum`` ignores the block and
    ``budget_calibration`` can exclude it, rather than both silently
    treating "we do not know" as "nothing to pay".
    """
    s = DeviationSettlement()
    charge = s.charge(
        np.array([SCHEDULE, SCHEDULE]), np.array([np.inf, np.nan])
    )
    assert np.isnan(charge).all()
    assert np.nansum(charge) == 0.0  # ignored, not counted as a real zero


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="align"):
        DeviationSettlement().charge(np.ones(3), np.ones(4))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"normal_rate": 0.0}, "normal_rate"),
        ({"volume_limit_fraction": 0.0}, "volume_limit_fraction"),
        ({"volume_limit_fraction": 1.5}, "volume_limit_fraction"),
        ({"volume_limit_mw": -1.0}, "volume_limit_mw"),
    ],
)
def test_impossible_parameters_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DeviationSettlement(**kwargs)


# -- aggregation -----------------------------------------------------------


def test_annualising_scales_rather_than_sums():
    """The folds deliberately do not cover a contiguous year.

    They are spread across three so an interval meets a change of season,
    which means the total over the backtest is not a year's worth and
    summing it would under-report by a factor of four.
    """
    one_year_of_blocks = 365 * 96
    per_block = np.full(1000, 100.0)
    assert annualise(per_block, 1000) == pytest.approx(100.0 * one_year_of_blocks)
    assert annualise(np.array([]), 0) == 0.0


def test_cost_summary_reports_the_shape_of_the_exposure_not_just_a_total():
    rng = np.random.default_rng(0)
    scheduled = np.full(2000, SCHEDULE)
    actual = scheduled + rng.normal(0.0, 300.0, 2000)
    out = cost_summary(scheduled, actual)
    assert out["blocks"] == 2000
    assert 0.4 < out["blocks_over_drawing"] < 0.6
    assert out["blocks_beyond_limit"] > 0.5  # 300 MW sigma against a 150 MW limit
    assert out["crore_per_year"] == pytest.approx(
        out["rupees_per_year"] / CRORE
    )
    assert out["mean_abs_deviation_mw"] == pytest.approx(
        np.mean(np.abs(actual - scheduled))
    )


def test_the_volume_band_saturates_so_cost_improves_far_less_than_mae():
    """A structural feature of the regulation, and a surprise worth keeping.

    Once both forecasts routinely deviate past the 150 MW band, the
    within-band charge is identical for both and only the over-drawal
    excess separates them. So halving the error does not halve the bill,
    and a project that quoted "39% better MAE, therefore 39% cheaper" would
    be wrong by construction rather than by a rounding error.
    """
    rng = np.random.default_rng(11)
    n = 20_000
    scheduled = np.full(n, SCHEDULE)
    sharp = scheduled + rng.normal(0.0, 310.0, n)
    blunt = scheduled + rng.normal(0.0, 510.0, n)

    mae_ratio = np.mean(np.abs(blunt - scheduled)) / np.mean(np.abs(sharp - scheduled))
    cost_ratio = (
        cost_summary(scheduled, blunt)["rupees_total"]
        / cost_summary(scheduled, sharp)["rupees_total"]
    )
    assert mae_ratio > 1.5
    # The cost advantage is real but markedly smaller than the error advantage.
    assert 1.0 < cost_ratio < mae_ratio


def test_a_better_point_forecast_costs_less():
    """The claim the rupee figure is actually allowed to support.

    Absolute rupees here are an upper bound for reasons the module
    docstring sets out, but the ordering between two forecasts survives all
    of them.
    """
    rng = np.random.default_rng(1)
    scheduled = np.full(5000, SCHEDULE)
    good = scheduled + rng.normal(0.0, 150.0, 5000)
    bad = scheduled + rng.normal(0.0, 450.0, 5000)
    assert cost_summary(scheduled, good)["rupees_per_year"] < (
        cost_summary(scheduled, bad)["rupees_per_year"]
    )


# -- what the interval buys ------------------------------------------------


def test_cost_is_not_monotone_in_load_so_a_straddling_interval_floors_at_zero():
    """The mistake this function exists to avoid.

    The cheapest outcome is the schedule itself. An interval that contains
    the schedule therefore contains an outcome costing nothing, and taking
    the cost at the interval's lower edge as the floor would report a
    minimum bill that no outcome inside the interval would have produced.
    """
    scheduled = np.array([SCHEDULE])
    out = budget_calibration(
        np.array([SCHEDULE - 400.0]),
        np.array([SCHEDULE + 400.0]),
        np.array([SCHEDULE]),       # landed exactly on schedule: free
        scheduled,
    )
    assert out["cost_coverage"] == 1.0
    assert out["mean_realised_cost"] == 0.0


def test_a_well_calibrated_interval_brackets_the_bill():
    rng = np.random.default_rng(3)
    n = 4000
    scheduled = np.full(n, SCHEDULE)
    actual = scheduled + rng.normal(0.0, 200.0, n)
    radius = 1.645 * 200.0
    out = budget_calibration(
        scheduled - radius, scheduled + radius, actual, scheduled
    )
    assert out["cost_coverage"] > 0.85
    assert out["blocks_unbounded"] == 0


def test_an_interval_that_is_too_narrow_under_reports_the_bill():
    rng = np.random.default_rng(4)
    n = 4000
    scheduled = np.full(n, SCHEDULE)
    actual = scheduled + rng.normal(0.0, 400.0, n)
    tight = 50.0
    out = budget_calibration(
        scheduled - tight, scheduled + tight, actual, scheduled
    )
    assert out["cost_coverage"] < 0.3
    assert out["worst_underestimate"] > 0


def test_unbounded_blocks_are_excluded_from_the_score_not_counted_as_covered():
    """An adaptive method that has lost track produces infinite bounds.

    Scoring those as covered would reward exactly the failure the bound is
    reporting.
    """
    scheduled = np.full(3, SCHEDULE)
    lower = np.array([SCHEDULE - 100.0, -np.inf, SCHEDULE - 100.0])
    upper = np.array([SCHEDULE + 100.0, np.inf, SCHEDULE + 100.0])
    actual = np.array([SCHEDULE + 50.0, SCHEDULE + 9000.0, SCHEDULE - 50.0])
    out = budget_calibration(lower, upper, actual, scheduled)
    assert out["blocks_scored"] == 2
    assert out["blocks_unbounded"] == 1
    assert out["cost_coverage"] == 1.0
