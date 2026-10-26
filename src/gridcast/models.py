"""The point forecasters, weakest first.

Three models, and the ordering is the argument. A load forecasting project
that reports only its best model's error has said nothing: load is enormously
predictable -- the same Tuesday evening ramp happens every Tuesday -- so any
model looks good in isolation. The question is always *how much better than
the obvious thing*, and the obvious thing here is "last week, same block".

``SeasonalNaive``
    Tomorrow's 10:15 block is the average of the last three same-weekday
    10:15 blocks. No fitting, no features, and it is genuinely hard to beat
    on a quiet week. It is the number every other model has to justify
    itself against.

``RidgeForecaster``
    Closed-form ridge on the lag and calendar features, written out rather
    than imported: standardise, solve ``(X'X + lambda I) beta = X'y``, keep
    the intercept out of the penalty. Linear in temperature, which is its
    main weakness -- cooling load is not linear in anything.

``BoostedForecaster``
    Histogram gradient boosting, which handles the temperature non-linearity
    and the interaction between block-of-day and season without being told
    about them.

All three expose the same two methods, because the conformal layer on top
must not care which one it is wrapping -- that is the entire point of
conformal prediction, and a model-specific interval routine would quietly
give it up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


class Forecaster(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


# -- the baseline ----------------------------------------------------------


@dataclass
class SeasonalNaive:
    """Last week's same block, averaged over the last three weeks.

    The column it reads is built by the feature pipeline, so this model is a
    projection rather than a computation -- which is the honest way to make it
    comparable: it sees exactly the same leak-free information every other
    model does.
    """

    column: str = "same_block_3wk_mean"
    fallback: str = "lag_168h"
    name: str = "seasonal naive"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive":
        if self.column not in X and self.fallback not in X:
            raise ValueError(
                f"neither {self.column!r} nor {self.fallback!r} is in the features"
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.column in X:
            values = X[self.column].to_numpy(dtype=float)
            if self.fallback in X:
                spare = X[self.fallback].to_numpy(dtype=float)
                values = np.where(np.isnan(values), spare, values)
            return values
        return X[self.fallback].to_numpy(dtype=float)


# -- ridge, written out ----------------------------------------------------


@dataclass
class RidgeForecaster:
    """Ridge regression by the normal equations, standardised.

    Standardising matters more than it looks: the features run from a 0/1
    weekend flag to a 7,000 MW lag, and a single ``lambda`` applied to raw
    columns penalises the small-scale ones into irrelevance. Standardising
    first makes one penalty mean the same thing for every column.

    The intercept is fitted by centring ``y`` rather than by adding a column
    of ones, so it is never shrunk -- penalising the intercept would bias
    every prediction toward zero MW, which is not a sensible prior for a city.
    """

    alpha: float = 10.0
    name: str = "ridge"
    columns: list[str] = field(default_factory=list, repr=False)
    mean_: np.ndarray | None = field(default=None, repr=False)
    scale_: np.ndarray | None = field(default=None, repr=False)
    coef_: np.ndarray | None = field(default=None, repr=False)
    intercept_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeForecaster":
        self.columns = list(X.columns)
        matrix = X.to_numpy(dtype=float)
        target = y.to_numpy(dtype=float)

        self.mean_ = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        # A constant column has zero spread; dividing by it is a NaN that
        # propagates through the whole solve. Leave it at 1 and let the
        # centring take it to zero, where it contributes nothing.
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        standardised = (matrix - self.mean_) / self.scale_

        y_mean = float(target.mean())
        centred = target - y_mean

        gram = standardised.T @ standardised
        gram[np.diag_indices_from(gram)] += self.alpha
        self.coef_ = np.linalg.solve(gram, standardised.T @ centred)
        self.intercept_ = y_mean
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit first")
        matrix = X[self.columns].to_numpy(dtype=float)
        standardised = (matrix - self.mean_) / self.scale_
        return standardised @ self.coef_ + self.intercept_

    def coefficients(self) -> pd.Series:
        """Standardised coefficients, which are comparable across columns."""
        if self.coef_ is None:
            raise RuntimeError("fit first")
        return pd.Series(self.coef_, index=self.columns).sort_values(
            key=np.abs, ascending=False
        )


# -- gradient boosting -----------------------------------------------------


@dataclass
class BoostedForecaster:
    """Histogram gradient boosting over the same features.

    Chosen over a plain GBM for speed -- it bins the features once and trains
    in seconds on 100k rows, which keeps a rolling backtest of 24 refits
    inside a coffee break on a laptop with no GPU.

    ``early_stopping`` is off on purpose. Scikit-learn's internal early stop
    carves a validation slice out *at random*, and a random slice of a time
    series is interpolation: neighbouring blocks are nearly identical, so the
    held-out rows are effectively in the training set and the stop fires far
    too late. The backtest already holds out by time, which is the only split
    that means anything here.
    """

    max_iter: int = 400
    learning_rate: float = 0.06
    max_depth: int | None = None
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    random_state: int = 0
    name: str = "boosted"
    model_: object | None = field(default=None, repr=False)
    columns: list[str] = field(default_factory=list, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BoostedForecaster":
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.columns = list(X.columns)
        self.model_ = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=self.random_state,
        )
        self.model_.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit first")
        return self.model_.predict(X[self.columns].to_numpy(dtype=float))


# -- errors ----------------------------------------------------------------


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, the industry's unit for load.

    Safe against a zero denominator because a zero load reading is a
    telemetry gap that should never have reached here -- but if one does,
    returning inf is better than silently dividing.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if np.any(actual <= 0):
        return float("inf")
    return float(np.mean(np.abs(actual - predicted) / actual) * 100.0)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    diff = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def skill(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> float:
    """How much of the baseline's error this model removes, as a fraction.

    The number that matters when every model looks good: 0 means no better
    than last week's same block, 1 means perfect, negative means worse than
    doing nothing.
    """
    base = mae(actual, baseline)
    return float(1.0 - mae(actual, predicted) / base) if base > 0 else 0.0
