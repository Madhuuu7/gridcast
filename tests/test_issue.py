"""Forecasting a day that has not happened, and refusing when that is a lie.

The backtest can always build a design matrix, because every day it looks at
already happened. The live path cannot, and most of what can go wrong is a
feed that silently stops: a weather forecast that ends before the target day,
or a load archive two weeks stale whose 48-hour lag is itself missing. Both
produce a confident flat line rather than an error, because the boosted model
consumes a missing feature without complaint, so both are checked here by
being provoked.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from gridcast.data.dataset import (
    BLOCKS_PER_DAY,
    build_features,
    forecast_design,
    incomplete_blocks,
)
from gridcast.issue import Inputs, issue

START = pd.Timestamp("2026-01-01")
DAYS = 200


def _blocks(days: int = DAYS, *, end_short: int = 0) -> pd.DataFrame:
    """A synthetic load series with a daily and weekly shape.

    Not random noise around a constant: the features are lags and weekly
    means, so a series without that structure would let a broken lag look
    as good as a working one.
    """
    index = pd.date_range(START, periods=days * BLOCKS_PER_DAY, freq="15min")
    block = np.arange(len(index)) % BLOCKS_PER_DAY
    dow = index.day_of_week.to_numpy()
    load = (
        3500.0
        + 900.0 * np.sin(2 * np.pi * (block - 20) / BLOCKS_PER_DAY)
        + 400.0 * np.exp(-0.5 * ((block - 76) / 7.0) ** 2)
        - 250.0 * (dow >= 5)
        + np.random.default_rng(0).normal(0.0, 60.0, len(index))
    )
    frame = pd.DataFrame({"delhi": load}, index=index)
    return frame.iloc[: len(frame) - end_short] if end_short else frame


def _weather(index_end: pd.Timestamp, *, days_ahead: int = 3) -> pd.DataFrame:
    hours = pd.date_range(
        START, index_end + pd.Timedelta(days=days_ahead), freq="h"
    )
    doy = hours.day_of_year.to_numpy()
    hour = hours.hour.to_numpy()
    temp = 24.0 + 10.0 * np.sin(2 * np.pi * (doy - 110) / 365.0) + 5.0 * np.sin(
        2 * np.pi * (hour - 9) / 24.0
    )
    return pd.DataFrame(
        {
            "temperature_2m": temp,
            "relative_humidity_2m": np.full(len(hours), 55.0),
            "apparent_temperature": temp + 1.0,
            "cloud_cover": np.full(len(hours), 30.0),
            "wind_speed_10m": np.full(len(hours), 8.0),
        },
        index=hours,
    )


# -- the design for a future day ------------------------------------------


def test_a_future_day_gets_the_same_columns_in_the_same_order():
    """The failure this shares a code path to prevent.

    Two feature builders drifting apart does not raise. The model is fitted
    on one column order and served from another, and it returns confident
    nonsense.
    """
    blocks = _blocks()
    last = blocks.index.max()
    weather = _weather(last)
    target = (last + pd.Timedelta(days=1)).date()

    X, _ = build_features(blocks, weather)
    design = forecast_design(blocks, weather, target)
    assert list(design.columns) == list(X.columns)
    assert len(design) == BLOCKS_PER_DAY
    assert pd.DatetimeIndex(design.index).normalize().unique().tolist() == [
        pd.Timestamp(target)
    ]


def test_a_complete_future_day_has_no_gaps():
    blocks = _blocks()
    last = blocks.index.max()
    design = forecast_design(blocks, _weather(last), (last + pd.Timedelta(days=1)).date())
    assert incomplete_blocks(design) == 0


def test_a_gap_in_the_archive_is_flagged_rather_than_refused():
    """A scheduler has to publish 96 blocks whatever the archive did.

    So a block whose 48-hour lag fell in a hole is still forecast, and the
    count travels with the forecast instead of the day being dropped.
    """
    blocks = _blocks()
    last = blocks.index.max()
    target = (last + pd.Timedelta(days=1)).date()
    # Punch out the readings that the first few blocks' 48-hour lag needs.
    holed = blocks.copy()
    hole = pd.Timestamp(target) - pd.Timedelta(hours=48)
    holed.loc[hole : hole + pd.Timedelta(minutes=45), "delhi"] = np.nan

    design = forecast_design(holed, _weather(last), target)
    assert len(design) == BLOCKS_PER_DAY
    assert incomplete_blocks(design) == 4


def test_weather_that_stops_before_the_target_day_is_an_error():
    """The failure that produced a real forecast for a day with no weather.

    Every weather column was null, the boosted model served it without
    complaint, and the output looked exactly like a forecast. A whole empty
    column is a feed that does not reach the day, not a gap in it.
    """
    blocks = _blocks()
    last = blocks.index.max()
    short = _weather(last, days_ahead=0)
    short = short[short.index < pd.Timestamp((last + pd.Timedelta(days=1)).date())]
    with pytest.raises(ValueError, match="does not reach the target day"):
        forecast_design(blocks, short, (last + pd.Timedelta(days=1)).date())


def test_a_day_already_in_the_archive_is_not_a_forecast():
    blocks = _blocks()
    with pytest.raises(ValueError, match="already covered"):
        forecast_design(blocks, _weather(blocks.index.max()), blocks.index.max().date())


# -- issuing ---------------------------------------------------------------


def _inputs(**kwargs) -> Inputs:
    blocks = _blocks(**kwargs)
    last = blocks.index.max()
    return Inputs(blocks, _weather(last), last.date())


def test_issuing_produces_a_full_day_with_ordered_bounds():
    inputs = _inputs()
    target = inputs.data_end + timedelta(days=1)
    forecast = issue(inputs, target, now=datetime.now(timezone.utc))

    assert forecast.point_mw.size == BLOCKS_PER_DAY
    assert np.all(forecast.lower_mw <= forecast.point_mw)
    assert np.all(forecast.point_mw <= forecast.upper_mw)
    assert forecast.target_date == target
    assert forecast.method == "localized conformal"
    assert forecast.kernel is not None
    # The windows must not overlap: calibrating on rows the model trained on
    # gives residuals that are too small and intervals that are too narrow.
    assert forecast.train_end < forecast.calibration_start


def test_the_forecast_follows_the_shape_of_the_series():
    """A flat line is what a broken lag produces, so check it is not flat."""
    inputs = _inputs()
    forecast = issue(inputs, inputs.data_end + timedelta(days=1))
    spread = forecast.point_mw.max() - forecast.point_mw.min()
    assert spread > 800.0
    # The fixture's dominant term is a 900 MW sine peaking at block 44, so
    # that is where the forecast must peak. Asserting an evening peak here
    # would be testing a load curve this series does not have.
    assert 36 <= int(np.argmax(forecast.point_mw)) <= 52


def test_a_stale_archive_is_refused_rather_than_extrapolated():
    """Three days past the archive and the nearest lag is itself missing.

    The model would forecast from the weekly profile alone and say nothing
    about having done so.
    """
    inputs = _inputs()
    with pytest.raises(ValueError, match="days before"):
        issue(inputs, inputs.data_end + timedelta(days=9))


def test_a_day_in_the_archive_is_refused():
    inputs = _inputs()
    with pytest.raises(ValueError, match="hindcast"):
        issue(inputs, inputs.data_end)


def test_too_little_history_is_refused():
    inputs = _inputs(days=40)
    with pytest.raises(RuntimeError, match="not enough history"):
        issue(inputs, inputs.data_end + timedelta(days=1))


@pytest.mark.parametrize(
    "method", ["split conformal", "mondrian conformal", "gaussian", "adaptive conformal"]
)
def test_every_offered_method_issues_a_usable_day(method):
    """Including the online one, which has no outcomes to update on.

    Adaptive conformal walks the calibration window and carries the level
    it ends on. Handing it the target day with unknown outcomes instead
    would score every block a miss, drive the level to zero and emit an
    infinite band -- a method reporting that it had lost track when it had
    merely not been told anything.
    """
    inputs = _inputs()
    forecast = issue(
        inputs, inputs.data_end + timedelta(days=1), method=method
    )
    assert forecast.method == method
    assert np.isfinite(forecast.lower_mw).all()
    assert np.isfinite(forecast.upper_mw).all()
    assert np.all(forecast.upper_mw - forecast.lower_mw > 0)


def test_an_unknown_method_is_named_rather_than_defaulted():
    inputs = _inputs()
    with pytest.raises(ValueError, match="unknown interval method"):
        issue(inputs, inputs.data_end + timedelta(days=1), method="wishful")
