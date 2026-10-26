"""Rolling backtest: fit, calibrate, forecast, and check what was claimed.

## Why the data is cut three ways and not two

Conformal prediction needs residuals from data the model has never seen. Score
the model on its own training set and the residuals are too small -- a boosted
tree fits the training data far better than it fits tomorrow -- so the
intervals come out confidently narrow and under-cover on exactly the data
anyone cares about. So each fold is:

    [--------- train ---------][-- calibrate --][-- test --]
     fit the point forecaster   residuals here   claims checked here

all strictly in time order, and the next fold slides forward. Never shuffled:
a random split of a time series puts 09:30 in the training set and 09:45 in
the test set, which is interpolation dressed up as forecasting, and it makes
every model look superb.

## The one extra split, for the normalized method

``NormalizedConformal`` needs a model of the error magnitude, and that model
also has to be fitted somewhere honest. Fitting it on the same residuals that
produce the conformal quantile would let it peek at the scores it is about to
be judged by, so the calibration window is halved: the first half fits the
sigma model, the second half supplies the conformal quantile. It costs half
the calibration data and keeps the guarantee intact.

## What gets reported

Marginal coverage, which every method will roughly hit, and then conditional
coverage by block of day, by month and by kind of day -- which is where they
come apart. A method that covers 90% overall by covering 98% of the night and
79% of the evening ramp has not done its job, and only the conditional view
shows it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .conformal import (
    AdaptiveConformal,
    GaussianIntervals,
    Intervals,
    MondrianAdaptiveConformal,
    MondrianConformal,
    NormalizedConformal,
    SplitConformal,
    conditional_coverage_gap,
    coverage_by_group,
)
from .data.dataset import block_of_day
from .models import mae, mape, rmse, skill


@dataclass(frozen=True)
class Fold:
    """One train / calibrate / test cut, by timestamp."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    cal_end: pd.Timestamp
    test_end: pd.Timestamp

    def masks(self, index: pd.DatetimeIndex):
        train = (index >= self.train_start) & (index < self.train_end)
        cal = (index >= self.train_end) & (index < self.cal_end)
        test = (index >= self.cal_end) & (index < self.test_end)
        return train, cal, test

    def describe(self) -> str:
        return (
            f"fold {self.index}: train {self.train_start.date()}..{self.train_end.date()} "
            f"| cal ..{self.cal_end.date()} | test ..{self.test_end.date()}"
        )


def make_folds(
    index: pd.DatetimeIndex,
    *,
    n_folds: int = 8,
    test_days: int = 45,
    cal_days: int = 60,
    min_train_days: int = 180,
    expanding: bool = True,
    spread: bool = True,
) -> list[Fold]:
    """Cut the series into rolling folds, in time order.

    ``spread`` places the folds evenly across the whole usable range rather
    than stacking them against the end of the data, and it is the default for
    a reason specific to this problem: six consecutive 30-day folds all land
    in the same season, so every fold tests the same weather, the same
    holidays and the same air-conditioning regime. The interesting question
    for an interval method is whether it survives a *seasonal* change, and
    that question is unanswerable from four months of summer.

    Spreading also avoids a trap this dataset lays in particular. The SLDC
    archive has a cluster of entirely missing days in August and September
    2026; a stack of recent folds puts one of them almost entirely inside a
    hole, and that fold's "coverage" is then computed from a quarter of the
    blocks it should have had.

    ``expanding`` keeps each training window anchored at the start of the
    data, which is what a deployment does -- it does not throw away last year.
    """
    start, end = index.min().normalize(), index.max().normalize()
    earliest_test_end = (
        start + pd.Timedelta(days=min_train_days + cal_days + test_days)
    )
    if earliest_test_end > end:
        raise ValueError(
            f"only {(end - start).days} days of data: not enough for "
            f"{min_train_days} train + {cal_days} cal + {test_days} test"
        )

    if spread:
        test_ends = pd.date_range(earliest_test_end, end, periods=n_folds)
    else:
        test_ends = [end - pd.Timedelta(days=test_days * k) for k in range(n_folds)][::-1]

    folds: list[Fold] = []
    for test_end in test_ends:
        test_end = pd.Timestamp(test_end).normalize()
        cal_end = test_end - pd.Timedelta(days=test_days)
        train_end = cal_end - pd.Timedelta(days=cal_days)
        train_start = start if expanding else train_end - pd.Timedelta(days=365)
        if (train_end - max(train_start, start)).days < min_train_days:
            continue
        folds.append(
            Fold(len(folds), max(train_start, start), train_end, cal_end, test_end)
        )
    return folds


@dataclass
class BacktestResult:
    """Everything a fold produced, tidy enough to group any way later."""

    alpha: float
    model: str
    frame: pd.DataFrame
    folds: list[Fold] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def methods(self) -> list[str]:
        return sorted({c[len("lower_") :] for c in self.frame if c.startswith("lower_")})

    def intervals(self, method: str) -> Intervals:
        return Intervals(
            method,
            self.alpha,
            self.frame[f"lower_{method}"].to_numpy(),
            self.frame[f"upper_{method}"].to_numpy(),
            (
                self.frame[f"upper_{method}"].to_numpy()
                - self.frame[f"lower_{method}"].to_numpy()
            )
            / 2.0,
        )

    def point_metrics(self) -> dict[str, float]:
        actual = self.frame["actual"].to_numpy()
        predicted = self.frame["predicted"].to_numpy()
        out = {
            "n": int(actual.size),
            "mape": mape(actual, predicted),
            "mae": mae(actual, predicted),
            "rmse": rmse(actual, predicted),
        }
        if "baseline" in self.frame:
            base = self.frame["baseline"].to_numpy()
            out["baseline_mape"] = mape(actual, base)
            out["baseline_mae"] = mae(actual, base)
            out["skill"] = skill(actual, predicted, base)
        return out

    def interval_metrics(self) -> pd.DataFrame:
        actual = self.frame["actual"].to_numpy()
        rows = []
        for method in self.methods:
            intervals = self.intervals(method)
            row = {"method": method}
            row.update(intervals.evaluate(actual))
            row.update(
                conditional_coverage_gap(
                    actual, intervals, self.frame["block"].to_numpy()
                )
            )
            rows.append(row)
        return pd.DataFrame(rows).sort_values("winkler").reset_index(drop=True)

    def coverage_by(self, method: str, by: str) -> pd.DataFrame:
        return pd.DataFrame(
            coverage_by_group(
                self.frame["actual"].to_numpy(),
                self.intervals(method),
                self.frame[by].to_numpy(),
            )
        )


def _sigma_model(X: pd.DataFrame, residuals: np.ndarray):
    """A small model of the absolute residual: how hard is this block?

    Deliberately smaller than the point model -- 150 trees on a few weeks of
    calibration data. Error magnitude is a smoother function of the features
    than load itself, and a large model here would fit the noise in the
    residuals, which is the one thing it must not do.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.08, min_samples_leaf=40,
        early_stopping=False, random_state=0,
    )
    model.fit(X.to_numpy(dtype=float), np.abs(residuals))
    return model


def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    *,
    alpha: float = 0.1,
    folds: list[Fold] | None = None,
    baseline_factory=None,
    gamma: float = 0.02,
    verbose: bool = True,
) -> BacktestResult:
    """Fit, calibrate and forecast over every fold; return every block."""
    index = pd.DatetimeIndex(X.index)
    folds = folds or make_folds(index)
    started = time.perf_counter()
    pieces: list[pd.DataFrame] = []
    model_name = getattr(model_factory(), "name", "model")

    for fold in folds:
        train, cal, test = fold.masks(index)
        if not (train.any() and cal.any() and test.any()):
            continue
        if verbose:
            print(
                f"  {fold.describe()} | {train.sum():,} train, {cal.sum():,} cal, "
                f"{test.sum():,} test",
                flush=True,
            )

        model = model_factory().fit(X[train], y[train])
        cal_pred = model.predict(X[cal])
        cal_resid = y[cal].to_numpy() - cal_pred
        test_pred = model.predict(X[test])
        actual = y[test].to_numpy()

        cal_blocks = block_of_day(index[cal])
        test_blocks = block_of_day(index[test])

        frame = pd.DataFrame(
            {
                "actual": actual,
                "predicted": test_pred,
                "block": test_blocks,
                "fold": fold.index,
            },
            index=index[test],
        )
        if baseline_factory is not None:
            frame["baseline"] = baseline_factory().fit(X[train], y[train]).predict(X[test])

        # -- the methods, all from the same residuals -------------------
        for method in (GaussianIntervals(alpha), SplitConformal(alpha)):
            intervals = method.fit(cal_resid).predict(test_pred)
            frame[f"lower_{method.name}"] = intervals.lower
            frame[f"upper_{method.name}"] = intervals.upper

        mondrian = MondrianConformal(alpha).fit(cal_resid, groups=cal_blocks)
        bands = mondrian.predict(test_pred, groups=test_blocks)
        frame[f"lower_{mondrian.name}"] = bands.lower
        frame[f"upper_{mondrian.name}"] = bands.upper

        # Normalized: halve the calibration window so the sigma model and the
        # conformal quantile never see the same rows.
        cal_index = np.flatnonzero(cal)
        half = cal_index.size // 2
        if half > 50:
            first, second = cal_index[:half], cal_index[half:]
            sigma_model = _sigma_model(X.iloc[first], cal_resid[:half])
            sigma_cal = sigma_model.predict(X.iloc[second].to_numpy(dtype=float))
            sigma_test = sigma_model.predict(X[test].to_numpy(dtype=float))
            normalized = NormalizedConformal(alpha).fit(
                cal_resid[half:], sigma=sigma_cal
            )
            bands = normalized.predict(test_pred, sigma=sigma_test)
            frame[f"lower_{normalized.name}"] = bands.lower
            frame[f"upper_{normalized.name}"] = bands.upper

        adaptive = AdaptiveConformal(alpha, gamma=gamma)
        bands = adaptive.run(actual, test_pred, cal_resid)
        frame[f"lower_{adaptive.name}"] = bands.lower
        frame[f"upper_{adaptive.name}"] = bands.upper
        frame["alpha_t"] = adaptive.alpha_trace_

        # Per-group online levels: the combination the other two motivate.
        # Grouped by hour rather than by block, because 96 independent
        # learners each see one observation a day and adapt far too slowly
        # over a 45-day test window; 24 groups is the compromise the backtest
        # settled on.
        cal_hours = index[cal].hour.to_numpy()
        test_hours = index[test].hour.to_numpy()
        both = MondrianAdaptiveConformal(alpha, gamma=max(gamma * 2.5, 0.05))
        bands = both.run(actual, test_pred, cal_resid, test_hours, cal_hours)
        frame[f"lower_{both.name}"] = bands.lower
        frame[f"upper_{both.name}"] = bands.upper

        pieces.append(frame)

    combined = pd.concat(pieces).sort_index()
    combined["month"] = pd.DatetimeIndex(combined.index).month
    combined["hour"] = pd.DatetimeIndex(combined.index).hour
    combined["dow"] = pd.DatetimeIndex(combined.index).dayofweek
    return BacktestResult(
        alpha=alpha,
        model=model_name,
        frame=combined,
        folds=folds,
        elapsed_s=time.perf_counter() - started,
    )
