"""Issuing a real forecast for a day that has not happened.

Everything else here replays history. This module does the thing the rest of
the project is evidence *about*: at the 10:00 gate it fits on what is known,
predicts all 96 blocks of tomorrow, and writes the result somewhere it cannot
be edited afterwards.

Three differences from the backtest, each of which makes the forecast worse
and the claim stronger:

**The weather is a forecast.** The backtest is handed the temperature the day
actually had. Here there is no such thing yet, so the design matrix carries a
weather *prediction* with its own error -- about 1.6C mean absolute at this
lead. This is the single largest reason a live number will be worse than a
backtested one, and the reason it is worth publishing.

**There is no fold.** The model trains on everything up to the calibration
window and calibrates on everything up to yesterday. No choice of split is
available after the fact, because the split is "now".

**Nothing can be retried into looking better.** A failed run is a gap in the
record, and a late run is refused by the store rather than backdated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .conformal import (
    AdaptiveConformal,
    CovariateKernel,
    GaussianIntervals,
    LocalizedConformal,
    MondrianConformal,
    SplitConformal,
)
from .data.dataset import (
    BLOCKS_PER_DAY,
    block_of_day,
    build_features,
    forecast_design,
    incomplete_blocks,
    to_blocks,
)
from .data.sldc import SldcClient, load_cached
from .data.weather import fetch_weather, fetch_weather_forecast
from .evaluate import day_kinds
from .models import BoostedForecaster
from .store import IssuedForecast
from .store.repository import IST

log = logging.getLogger("gridcast.issue")

#: Default published method. The sweep's Winkler optimum among the offline
#: methods, and the only one that reaches the noise floor on time of day.
DEFAULT_METHOD = "localized conformal"

#: Days of recent history used to calibrate. Long enough for the conformal
#: quantile to mean something, short enough that the residuals still
#: describe the current season.
CALIBRATION_DAYS = 60


@dataclass(frozen=True)
class Inputs:
    """Everything the gate had, loaded once."""

    blocks: pd.DataFrame
    weather: pd.DataFrame
    data_end: date

    @property
    def data_start(self) -> date:
        return self.blocks.index.min().date()


def gather(
    cache: Path | str = "data/raw",
    *,
    refresh_days: int = 4,
    offline: bool = False,
) -> Inputs:
    """Load history, top up the last few days, and splice on the forecast.

    ``refresh_days`` re-fetches the tail rather than trusting the cache:
    SLDC publishes a day and then revises it, and a stale final day is the
    one the 48-hour lag for tomorrow's early blocks comes from.

    The weather series is the archive with the forecast laid over its end,
    so a single frame spans history and tomorrow and ``forecast_design``
    does not need to know where the seam is.
    """
    cache = Path(cache)
    client = SldcClient(cache_dir=cache / "sldc")
    if not offline:
        today = datetime.now(IST).date()
        start = today - timedelta(days=refresh_days)
        log.info("refreshing SLDC %s..%s", start, today)
        client.fetch_range(start, today, refresh=True)

    load = load_cached(cache / "sldc")
    if load.empty:
        raise RuntimeError(
            "no cached load data; run scripts/fetch_data.py before issuing"
        )
    blocks = to_blocks(load)
    data_end = blocks.index.max().date()

    archive = fetch_weather(
        blocks.index.min().date(), data_end, cache_dir=cache / "weather"
    )
    if offline:
        return Inputs(blocks, archive, data_end)

    predicted = fetch_weather_forecast(past_days=7, forecast_days=3)
    # The forecast wins on the overlap. Its recent hours come from the same
    # model run as tomorrow's, so the seam carries no step change -- joining
    # a reanalysis directly onto a forecast puts a discontinuity in the
    # middle of the features that the model reads as weather.
    archive = archive[~archive.index.isin(predicted.index)]
    weather = pd.concat([archive, predicted]).sort_index()
    return Inputs(blocks, weather, data_end)


def _interval_method(name: str, alpha: float):
    kernel = CovariateKernel()
    builders = {
        "localized conformal": lambda: LocalizedConformal(alpha, kernel=kernel),
        "split conformal": lambda: SplitConformal(alpha),
        "mondrian conformal": lambda: MondrianConformal(alpha),
        "gaussian": lambda: GaussianIntervals(alpha),
        "adaptive conformal": lambda: AdaptiveConformal(alpha),
    }
    if name not in builders:
        raise ValueError(
            f"unknown interval method {name!r}; one of {sorted(builders)}"
        )
    return builders[name]()


def issue(
    inputs: Inputs,
    target_date: date,
    *,
    method: str = DEFAULT_METHOD,
    alpha: float = 0.1,
    calibration_days: int = CALIBRATION_DAYS,
    max_staleness_days: int = 2,
    now: datetime | None = None,
) -> IssuedForecast:
    """Fit, calibrate and forecast one day. Does not write anything.

    Separated from the writing so the whole computation can be tested
    without a database and so a failure to persist cannot leave a
    half-written run behind.
    """
    now = now or datetime.now(timezone.utc)
    if target_date <= inputs.data_end:
        raise ValueError(
            f"{target_date} is already in the load archive (which ends "
            f"{inputs.data_end}); issuing a forecast for it would be a "
            f"hindcast, and score_day exists to replay those instead"
        )
    # The design is built from same-block lags of 48 hours and up. Issue
    # three days past the end of the archive and the nearest lag is itself
    # missing, so the model forecasts from the weekly profile alone and
    # reports nothing about it -- a confident flat line. The gate is a
    # day-ahead gate; anything further is a different product.
    staleness = (target_date - inputs.data_end).days
    if staleness > max_staleness_days:
        raise ValueError(
            f"the load archive ends {inputs.data_end}, which is {staleness} "
            f"days before {target_date}. The 48-hour lag that every block of "
            f"the design depends on would itself be missing. Refresh the "
            f"archive (scripts/fetch_data.py) before issuing, or pass "
            f"--target a day within {max_staleness_days} of it"
        )

    X, y = build_features(inputs.blocks, inputs.weather)
    index = pd.DatetimeIndex(X.index)
    cal_from = pd.Timestamp(inputs.data_end - timedelta(days=calibration_days))
    train = index < cal_from
    cal = index >= cal_from
    if int(train.sum()) < 5000 or int(cal.sum()) < 500:
        raise RuntimeError(
            f"not enough history to issue: {int(train.sum())} training rows "
            f"and {int(cal.sum())} calibration rows"
        )

    model = BoostedForecaster().fit(X[train], y[train])
    residuals = y[cal].to_numpy() - model.predict(X[cal])

    design = forecast_design(inputs.blocks, inputs.weather, target_date)
    point = model.predict(design)
    degraded = incomplete_blocks(design)

    bands = _fit_and_predict(
        _interval_method(method, alpha),
        residuals=residuals,
        cal_index=index[cal],
        cal_X=X[cal],
        cal_actual=y[cal].to_numpy(),
        cal_point=model.predict(X[cal]),
        point=point,
        design=design,
    )

    note = None
    if degraded:
        note = (
            f"{degraded} of {BLOCKS_PER_DAY} blocks were forecast with a "
            f"missing feature, from gaps in the SLDC archive"
        )
    return IssuedForecast(
        target_date=target_date,
        issued_at=now,
        model=getattr(model, "name", "boosted"),
        method=method,
        alpha=alpha,
        point_mw=np.asarray(point, dtype=float),
        lower_mw=bands.lower,
        upper_mw=bands.upper,
        train_start=index[train].min().date(),
        train_end=index[train].max().date(),
        calibration_start=index[cal].min().date(),
        calibration_end=index[cal].max().date(),
        train_rows=int(train.sum()),
        kernel=(
            {
                "block_bandwidth": CovariateKernel().block_bandwidth,
                "temp_bandwidth": CovariateKernel().temp_bandwidth,
                "kind_penalty": CovariateKernel().kind_penalty,
            }
            if method == "localized conformal"
            else None
        ),
        note=note,
    )


def _fit_and_predict(
    estimator, *, residuals, cal_index, cal_X, cal_actual, cal_point, point, design
):
    """Hand each method the extra arguments it needs, and nothing else."""
    from .conformal import BlockCovariates, Intervals, conformal_quantile

    target_index = pd.DatetimeIndex(design.index)
    if isinstance(estimator, LocalizedConformal):
        def covariates(index, frame):
            temp = (
                frame["temperature_2m"].to_numpy(dtype=float)
                if "temperature_2m" in frame.columns
                else None
            )
            return BlockCovariates(block_of_day(index), day_kinds(index), temp)

        estimator.fit(residuals, covariates=covariates(cal_index, cal_X))
        return estimator.predict(point, covariates=covariates(target_index, design))

    if isinstance(estimator, MondrianConformal):
        estimator.fit(residuals, groups=block_of_day(cal_index))
        return estimator.predict(point, groups=block_of_day(target_index))

    if isinstance(estimator, AdaptiveConformal):
        # An online method updates on outcomes, and tomorrow has none. What
        # it can do is walk the calibration window, where the outcomes are
        # known, and carry the working level it ends on into the day being
        # issued -- which is exactly what the method would be doing at the
        # gate anyway.
        #
        # Feeding it the target day with NaN outcomes instead would score
        # every block a miss, drive the level toward zero and emit an
        # infinite band: a method reporting that it had lost track, when in
        # fact it had merely not been told anything.
        warmup = residuals[: len(residuals) // 2]
        walked = residuals[len(residuals) // 2 :]
        estimator.run(
            cal_actual[len(residuals) // 2 :],
            cal_point[len(residuals) // 2 :],
            warmup,
        )
        level = min(max(estimator.alpha_t_, 1e-6), 1.0 - 1e-6)
        radius = conformal_quantile(np.abs(residuals), level, fallback="max")
        point_a = np.asarray(point, dtype=float)
        widths = np.full(point_a.shape, radius)
        return Intervals(
            estimator.name, estimator.alpha, point_a - widths, point_a + widths, widths
        )

    estimator.fit(residuals)
    return estimator.predict(point)
