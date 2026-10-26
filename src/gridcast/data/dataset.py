"""Five-minute readings into day-ahead training rows, without leaking.

## The scheduling problem this is shaped around

Indian grid scheduling runs on **15-minute blocks** -- 96 a day -- and a state
utility declares its day-ahead schedule before the day-ahead market gates
close, around 10:00 the previous day. Deviation from that schedule is settled
block by block under the DSM mechanism, so the thing worth forecasting is each
of tomorrow's 96 blocks, and the thing worth knowing is how wrong each one
might be.

So: a forecast for every block of day D+1, issued at 10:00 on day D.

## The leakage trap, and why it gets its own guard

At 10:00 on day D, the most recent load reading is D 10:00. For the first
block of D+1 that is 14 hours of distance; for the last block it is 38. The
feature everyone reaches for first -- "load at the same block yesterday", a
24-hour lag -- is therefore **available for the morning blocks and in the
future for the afternoon ones**. Build it with a naive ``shift(96)`` and the
model trains on information it will not have, the backtest looks excellent,
and the system is worse than the naive baseline in production.

It is not a subtle failure to produce and it is nearly invisible to read, so
every lag here declares its offset and ``assert_causal`` checks, for every row
in the built frame, that the value it reads was observable at that row's issue
time. ``tests/test_leakage.py`` asserts that a 24-hour lag is *rejected* --
the guard is tested by showing it fires, not by trusting it.

The practical consequence is that the shortest uniformly safe same-block lag
is 48 hours, not 24. Everything more recent than that enters as a day-level
summary of what was known by 10:00: the last reading, the morning so far, and
yesterday in full.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .calendar import day_kind, holiday_name

#: Blocks per day under the Indian ABT scheduling convention.
BLOCKS_PER_DAY = 96
BLOCK = pd.Timedelta(minutes=15)

#: When tomorrow's schedule is fixed. Everything used to forecast a block of
#: day D+1 must have been observable by this hour on day D.
ISSUE_HOUR = 10

#: Same-block lags, in hours. 48 is the shortest that is safe for every block
#: of the day; see the module docstring.
SAME_BLOCK_LAGS = (48, 72, 168, 336)


@dataclass(frozen=True)
class FeatureSpec:
    """One lag, and how far back it reaches."""

    name: str
    hours: float

    def source_times(self, targets: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return targets - pd.Timedelta(hours=self.hours)


def issue_times(targets: pd.DatetimeIndex, issue_hour: int = ISSUE_HOUR) -> pd.DatetimeIndex:
    """When the forecast for each target block was made.

    The day before the target, at the gate hour. A block at 00:15 on the 5th
    and a block at 23:45 on the 5th share an issue time of 10:00 on the 4th,
    which is the whole source of the leakage asymmetry.
    """
    return pd.DatetimeIndex(targets).normalize() - pd.Timedelta(days=1) + pd.Timedelta(
        hours=issue_hour
    )


def assert_causal(
    targets: pd.DatetimeIndex, specs: list[FeatureSpec], issue_hour: int = ISSUE_HOUR
) -> None:
    """Fail loudly if any feature reads something not yet observable.

    Checked against the *worst* row rather than the first: a 24-hour lag is
    perfectly legitimate for the 00:15 block and illegal for the 23:45 one, so
    a spot check on the head of the frame passes while the dataset is poisoned.
    """
    issues = issue_times(targets, issue_hour)
    for spec in specs:
        sources = spec.source_times(targets)
        late = sources > issues
        if bool(np.any(late)):
            worst = int(np.argmax(sources - issues))
            raise ValueError(
                f"feature {spec.name!r} (lag {spec.hours}h) is not causal: for the "
                f"block at {targets[worst]} it reads {sources[worst]}, which is "
                f"{(sources[worst] - issues[worst])} after the {issues[worst]} "
                f"issue time. The shortest safe same-block lag is 48h -- see "
                f"data/dataset.py."
            )


# -- five-minute readings to blocks ---------------------------------------


def to_blocks(load: pd.DataFrame, *, min_readings: int = 2) -> pd.DataFrame:
    """Average the five-minute readings into 15-minute scheduling blocks.

    A block is three readings. ``min_readings`` is why this is not a one-line
    resample: a block with one reading out of three is an average of whatever
    survived, and telemetry gaps cluster, so a single-reading block is
    materially less trustworthy than a full one. Below the threshold the block
    is missing and says so.
    """
    if load.empty:
        return load.copy()
    resampled = load.resample(BLOCK)
    counts = resampled.count()
    blocks = resampled.mean()
    return blocks.where(counts >= min_readings)


def block_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """0..95, the ABT block number."""
    return ((index.hour * 60 + index.minute) // 15).to_numpy()


# -- features --------------------------------------------------------------


def build_features(
    blocks: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    *,
    target: str = "delhi",
    issue_hour: int = ISSUE_HOUR,
    lags: tuple[int, ...] = SAME_BLOCK_LAGS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the day-ahead design matrix and its target.

    Returns ``(X, y)`` indexed by the target block's timestamp, with rows
    dropped only where the target or a required lag is missing -- never
    filled, because an invented load value is a training example that says
    something false about the grid.
    """
    if target not in blocks:
        raise ValueError(f"no column {target!r} in the block frame")

    series = blocks[target].astype(float)
    # A regular grid first: lags are positional only if nothing is missing,
    # and real data has gaps. Reindexing to the full range makes a 48-hour
    # lag mean 48 hours rather than "192 rows ago, whenever those were".
    full = pd.date_range(series.index.min(), series.index.max(), freq=BLOCK)
    series = series.reindex(full)

    specs = [FeatureSpec(f"lag_{hours}h", hours) for hours in lags]
    assert_causal(full, specs, issue_hour)

    features = pd.DataFrame(index=full)
    steps_per_hour = 4
    for spec in specs:
        features[spec.name] = series.shift(int(spec.hours * steps_per_hour))

    # The same block, averaged over the last three same-weekdays. A single
    # week-old reading is noisy; three of them is the weekly profile.
    weekly = pd.concat(
        [series.shift(168 * steps_per_hour * k) for k in (1, 2, 3)], axis=1
    )
    features["same_block_3wk_mean"] = weekly.mean(axis=1)
    features["same_block_3wk_std"] = weekly.std(axis=1)

    # Day-level summaries of everything known by the gate. These are computed
    # per issue day and broadcast to all 96 target blocks, which is what makes
    # them safe: the value is identical for the 00:15 and the 23:45 block
    # because both were forecast at the same moment.
    day_level = _issue_day_features(series, full, issue_hour)
    features = features.join(day_level)

    # Calendar.
    index_days = pd.DatetimeIndex(full).date
    block = block_of_day(pd.DatetimeIndex(full))
    features["block"] = block
    # The block number as an angle: block 95 and block 0 are fifteen minutes
    # apart, and a raw 0..95 column tells a linear model they are maximally
    # distant.
    features["block_sin"] = np.sin(2 * np.pi * block / BLOCKS_PER_DAY)
    features["block_cos"] = np.cos(2 * np.pi * block / BLOCKS_PER_DAY)
    features["dow"] = pd.DatetimeIndex(full).day_of_week
    features["is_weekend"] = (features["dow"] >= 5).astype(int)
    doy = pd.DatetimeIndex(full).day_of_year
    features["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    kinds = pd.Series([day_kind(d) for d in index_days], index=full, name="day_kind")
    for kind in ("holiday", "near-festival", "sunday", "saturday"):
        features[f"kind_{kind.replace('-', '_')}"] = (kinds == kind).astype(int)

    if weather is not None and not weather.empty:
        features = features.join(_weather_features(weather, full))

    target_series = series.rename("load")
    frame = features.join(target_series)
    # Every column, not just the lags: a linear model cannot consume a NaN,
    # and leaving some columns ragged would mean the ridge and the boosted
    # model were silently trained on different rows -- which makes the
    # comparison between them meaningless. Anything still missing here is a
    # genuine hole with no causal fallback, and those rows are dropped.
    frame = frame.dropna()
    y = frame.pop("load")
    return frame, y


def _issue_day_features(
    series: pd.Series, targets: pd.DatetimeIndex, issue_hour: int
) -> pd.DataFrame:
    """Summaries of what was known at the gate, per issue day.

    Built on the issue day's own calendar and then mapped onto the target
    blocks, rather than lagged positionally. That mapping is the point: it is
    impossible to accidentally include an afternoon reading from the issue day
    because the window is cut at the gate hour by construction.
    """
    frame = series.to_frame("load")
    frame["day"] = pd.DatetimeIndex(frame.index).normalize()
    gate = pd.DatetimeIndex(frame.index).hour < issue_hour

    morning = frame[gate].groupby("day")["load"]
    whole_day = frame.groupby("day")["load"]

    summary = pd.DataFrame(
        {
            "issue_morning_mean": morning.mean(),
            "issue_morning_max": morning.max(),
            "issue_morning_min": morning.min(),
            "prev_day_mean": whole_day.mean().shift(1),
            "prev_day_max": whole_day.max().shift(1),
            "prev_day_min": whole_day.min().shift(1),
            "prev_week_mean": whole_day.mean().rolling(7).mean().shift(1),
        }
    )
    # The last reading before the gate, which is the freshest number a
    # scheduler actually has in hand.
    summary["issue_last_reading"] = frame[gate].groupby("day")["load"].last()

    # Some issue days have no readings before the gate at all -- 172 blocks'
    # worth over three years -- and the first week has no previous week. A
    # real scheduler still has to publish a schedule on those days, so
    # dropping them would quietly excuse the model from the hardest cases and
    # flatter every error metric. Instead each gap falls back to the most
    # recent thing that *was* known, and a flag records that it happened, so
    # the model can learn that a blind issue morning is a worse-informed
    # forecast rather than a normal one.
    summary["issue_blind"] = summary["issue_morning_mean"].isna().astype(int)
    expanding = summary["prev_day_mean"].expanding().mean()
    for column, fallback in (
        ("issue_morning_mean", summary["prev_day_mean"]),
        ("issue_morning_max", summary["prev_day_max"]),
        ("issue_morning_min", summary["prev_day_min"]),
        ("issue_last_reading", summary["prev_day_mean"]),
        ("prev_week_mean", expanding),
    ):
        summary[column] = summary[column].fillna(fallback).fillna(expanding)

    issue_day = pd.DatetimeIndex(targets).normalize() - pd.Timedelta(days=1)
    mapped = summary.reindex(issue_day)
    mapped.index = targets
    return mapped


def _weather_features(weather: pd.DataFrame, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """Weather at the target block, plus the target day's shape.

    Interpolated from hourly to 15-minute. Daily max and min go in alongside
    the instantaneous value because cooling load has memory: a 44C afternoon
    leaves the evening hotter than the thermometer alone suggests, and the
    overnight minimum sets how much heat the building stock shed.
    """
    from .weather import to_blocks as weather_to_blocks

    at_block = weather_to_blocks(weather, targets)
    out = at_block.copy()

    if "temperature_2m" in at_block:
        temp = at_block["temperature_2m"]
        day = pd.DatetimeIndex(targets).normalize()
        grouped = temp.groupby(day)
        out["temp_day_max"] = grouped.transform("max")
        out["temp_day_min"] = grouped.transform("min")
        # Cooling and heating degrees around the comfort band, kept separate
        # because the response is asymmetric: Delhi's cooling slope is far
        # steeper than its heating slope, and a single |t - 22| term forces
        # one coefficient to describe both.
        out["cooling_degrees"] = (temp - 24.0).clip(lower=0)
        out["heating_degrees"] = (16.0 - temp).clip(lower=0)
    return out


def summarise(blocks: pd.DataFrame, target: str = "delhi") -> dict[str, object]:
    """A few facts about the data, for the README and the console."""
    series = blocks[target].dropna()
    if series.empty:
        return {"blocks": 0}
    days = pd.DatetimeIndex(series.index).normalize().nunique()
    expected = BLOCKS_PER_DAY * days
    return {
        "blocks": int(series.size),
        "days": int(days),
        "completeness": float(series.size / expected) if expected else 0.0,
        "start": str(series.index.min()),
        "end": str(series.index.max()),
        "min_mw": float(series.min()),
        "max_mw": float(series.max()),
        "mean_mw": float(series.mean()),
        "peak_day": str(pd.DatetimeIndex([series.idxmax()])[0].date()),
    }
