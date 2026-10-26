"""The causality guard, tested by making it fire.

This is the most important test file in the project. Every other failure mode
here produces a visibly wrong number; leakage produces a *better* one. A
backtest that quietly uses tomorrow's data reports a lower MAPE, a tighter
interval and a happier chart, and there is nothing in the output to suggest
anything is wrong -- the error only appears in production, months later, as a
model that cannot reproduce its own evaluation.

So the guard is not trusted. Each test below hands ``assert_causal`` something
illegal and checks that it refuses, and the central one is
``test_a_24_hour_lag_is_rejected``: 24 hours is the lag every load forecasting
tutorial reaches for first, it is perfectly legal for the morning blocks, and
it is a time machine for the afternoon ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast.data.dataset import (
    BLOCKS_PER_DAY,
    FeatureSpec,
    ISSUE_HOUR,
    assert_causal,
    block_of_day,
    build_features,
    issue_times,
    to_blocks,
)


def a_day_of_targets(day: str = "2025-06-10") -> pd.DatetimeIndex:
    return pd.date_range(f"{day} 00:00", f"{day} 23:45", freq="15min")


# -- what the gate means ---------------------------------------------------


def test_every_block_of_a_day_shares_one_issue_time():
    """The asymmetry that makes leakage possible. A forecast for 00:15 and a
    forecast for 23:45 are published together, so they know exactly the same
    things -- even though one is 14 hours out and the other 38."""
    targets = a_day_of_targets()
    issues = issue_times(targets)
    assert len(set(issues)) == 1
    assert issues[0] == pd.Timestamp("2025-06-09 10:00")
    assert (targets[0] - issues[0]) == pd.Timedelta(hours=14)
    assert (targets[-1] - issues[-1]) == pd.Timedelta(hours=37, minutes=45)


def test_the_issue_hour_moves_the_gate():
    targets = a_day_of_targets()
    assert issue_times(targets, issue_hour=6)[0] == pd.Timestamp("2025-06-09 06:00")
    assert issue_times(targets, issue_hour=18)[0] == pd.Timestamp("2025-06-09 18:00")


# -- the guard -------------------------------------------------------------


def test_a_24_hour_lag_is_rejected():
    """The headline. Yesterday's same block is in the future for every target
    block after the gate hour, which is 58 of the 96."""
    targets = a_day_of_targets()
    with pytest.raises(ValueError, match="not causal"):
        assert_causal(targets, [FeatureSpec("lag_24h", 24)])


def test_the_rejection_names_the_worst_block_and_the_overshoot():
    """An error that says only "leakage" sends someone hunting. This one has
    to say which block, by how much, and what to do instead."""
    targets = a_day_of_targets()
    with pytest.raises(ValueError) as excinfo:
        assert_causal(targets, [FeatureSpec("lag_24h", 24)])
    message = str(excinfo.value)
    assert "lag_24h" in message
    assert "2025-06-10 23:45" in message  # the worst offender, not the first
    assert "48h" in message  # what to use instead


def test_a_24_hour_lag_is_legal_for_the_early_blocks_alone():
    """Why a spot check on the head of the frame passes while the dataset is
    poisoned: for targets before the gate hour, the 24-hour lag is genuinely
    observable."""
    early = pd.date_range("2025-06-10 00:00", "2025-06-10 09:45", freq="15min")
    assert_causal(early, [FeatureSpec("lag_24h", 24)])  # no raise

    late = pd.date_range("2025-06-10 10:15", "2025-06-10 23:45", freq="15min")
    with pytest.raises(ValueError, match="not causal"):
        assert_causal(late, [FeatureSpec("lag_24h", 24)])


@pytest.mark.parametrize("hours", [1, 6, 12, 24, 36, 37.5])
def test_every_lag_shorter_than_38_hours_is_rejected(hours):
    """38 hours is the distance from the gate to the last block of the target
    day. Anything shorter reaches past it for at least one block."""
    with pytest.raises(ValueError, match="not causal"):
        assert_causal(a_day_of_targets(), [FeatureSpec(f"lag_{hours}h", hours)])


@pytest.mark.parametrize("hours", [38, 48, 72, 168, 336])
def test_the_lags_the_pipeline_actually_uses_are_accepted(hours):
    assert_causal(a_day_of_targets(), [FeatureSpec(f"lag_{hours}h", hours)])


def test_the_guard_checks_every_row_not_just_the_first():
    """A guard that short-circuits on the first row is worse than none: it
    passes on exactly the input that is wrong."""
    targets = a_day_of_targets()
    specs = [FeatureSpec("fine", 48), FeatureSpec("sneaky", 24)]
    with pytest.raises(ValueError, match="sneaky"):
        assert_causal(targets, specs)


def test_a_later_gate_permits_a_shorter_lag():
    """The 38-hour floor is a consequence of the 10:00 gate, not a law. Move
    the gate to 23:00 and a 25-hour lag becomes legal -- which is worth
    pinning, because it shows the guard is computing the constraint rather
    than hard-coding a number."""
    targets = a_day_of_targets()
    with pytest.raises(ValueError):
        assert_causal(targets, [FeatureSpec("lag_25h", 25)], issue_hour=10)
    assert_causal(targets, [FeatureSpec("lag_25h", 25)], issue_hour=23)


# -- the built frame -------------------------------------------------------


def synthetic_blocks(days: int = 60) -> pd.DataFrame:
    """A smooth daily cycle with a weekly wobble: enough structure that a
    leaked feature would obviously help, which is the point."""
    index = pd.date_range("2025-01-01", periods=days * BLOCKS_PER_DAY, freq="15min")
    block = block_of_day(index)
    daily = 4000 + 1200 * np.sin(2 * np.pi * (block - 20) / BLOCKS_PER_DAY)
    weekly = 200 * np.asarray(index.day_of_week >= 5)
    noise = np.random.default_rng(0).normal(0, 40, size=index.size)
    return pd.DataFrame({"delhi": daily - weekly + noise}, index=index)


def test_build_features_produces_only_causal_lags():
    """The integration check: whatever the builder names its columns, every
    lag column it creates must survive the guard."""
    X, _y = build_features(synthetic_blocks())
    lag_specs = [
        FeatureSpec(name, float(name.removeprefix("lag_").removesuffix("h")))
        for name in X.columns
        if name.startswith("lag_") and name.endswith("h")
    ]
    assert lag_specs, "no lag columns were built"
    assert_causal(pd.DatetimeIndex(X.index), lag_specs)
    assert min(spec.hours for spec in lag_specs) >= 38


def test_the_day_level_features_are_identical_across_a_target_day():
    """What makes the issue-day summaries safe. Every block of a target day
    was forecast at the same moment, so a summary of what was known then must
    not vary within the day -- if it does, it is reading the target day's own
    load as the day unfolds.
    """
    X, _y = build_features(synthetic_blocks())
    day = X.loc["2025-02-10"]
    assert len(day) > 50
    for column in (
        "issue_morning_mean",
        "issue_last_reading",
        "prev_day_mean",
        "prev_week_mean",
    ):
        assert day[column].nunique() == 1, column


def test_the_target_is_never_among_the_features():
    X, y = build_features(synthetic_blocks())
    assert "load" not in X.columns
    # And no column is a copy of the target, which is the other way a target
    # leaks: not by name but by value.
    for column in X.columns:
        values = X[column].to_numpy(dtype=float)
        if np.std(values) > 0:
            correlation = abs(np.corrcoef(values, y.to_numpy())[0, 1])
            assert correlation < 0.999, f"{column} is effectively the target"


def test_features_and_target_stay_aligned():
    X, y = build_features(synthetic_blocks())
    assert len(X) == len(y)
    assert X.index.equals(y.index)
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_a_gap_in_the_history_removes_rows_rather_than_inventing_them():
    """Reindexing a time series to a regular grid and forward-filling is the
    other quiet corruption: it manufactures load readings that were never
    taken, and they are indistinguishable from real ones downstream."""
    blocks = synthetic_blocks()
    blocks.loc["2025-01-20":"2025-01-22", "delhi"] = np.nan
    X, y = build_features(blocks)
    missing_window = y.loc["2025-01-20":"2025-01-22"]
    assert missing_window.empty
    # The rows that depended on that window as a lag are gone too, not filled.
    assert y.loc["2025-01-22":"2025-01-24"].size < 3 * BLOCKS_PER_DAY
