"""The HTTP surface: forecast a day, and show how honest the interval is.

## The one design decision worth defending

A gate request cannot fit a model. Histogram boosting on 90,000 rows takes
several seconds, and a service that spends five seconds per request is not a
service. So the model is fitted **once at startup**, on everything up to a
cutoff, and every date it will serve a forecast for lies *after* that cutoff.

That is not a caching convenience, it is what keeps the demo honest. If the
service refitted per request on all available data, a forecast for last
Tuesday would be made by a model that had already seen last Tuesday, and the
interval drawn next to the actual load would be meaningless -- it would cover
every time, and the page would be a liar with a chart.

The split is therefore:

    [------------- fit -------------][-- calibrate --][-- servable --]
                                      residuals here   forecasts here

and ``/api/health`` reports all three boundaries so the arrangement can be
checked from outside rather than taken on trust.

## What it serves

A day's 96 blocks with the point forecast, every interval method's band, and
-- where the day is in the past -- the actual load, so the console can show
which blocks the interval missed. Plus the backtest study's output, so the
page can quote measured coverage rather than nominal coverage.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
import logging
import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..conformal import (
    AdaptiveConformal,
    GaussianIntervals,
    MondrianConformal,
    SplitConformal,
    conformal_quantile,
)
from ..data.calendar import day_kind, holiday_name
from ..data.dataset import BLOCKS_PER_DAY, ISSUE_HOUR, block_of_day, build_features, summarise, to_blocks
from ..data.sldc import load_cached
from ..data.weather import fetch_weather
from ..models import BoostedForecaster, SeasonalNaive, mae, mape

log = logging.getLogger("gridcast.api")


def _find_root() -> Path:
    """The repo root, by looking for the console rather than counting parents.

    Counting is right for a source checkout and wrong for an installed
    package, where three levels up is a site-packages directory with no
    ``web/`` in it -- the page 404s on the deployed service while every local
    test passes.
    """
    override = os.environ.get("GRIDCAST_ROOT")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "web" / "index.html").is_file():
            return parent
    return here.parents[3]


ROOT = _find_root()
WEB = ROOT / "web"
BACKTEST = ROOT / "reports" / "backtest.json"

#: Days at the end of the record the service will forecast. Everything before
#: this is fit and calibration.
SERVABLE_DAYS = int(os.environ.get("GRIDCAST_SERVABLE_DAYS", "45"))
CALIBRATION_DAYS = int(os.environ.get("GRIDCAST_CALIBRATION_DAYS", "45"))
ALPHA = float(os.environ.get("GRIDCAST_ALPHA", "0.1"))


class ForecastService:
    """Everything fitted once: the model, and every interval method."""

    def __init__(self, cache: Path | str = "data/raw", alpha: float = ALPHA):
        self.cache = Path(cache)
        self.alpha = alpha
        self.ready = False
        self.error: str | None = None

    def warm(self) -> None:
        if self.ready:
            return
        load = load_cached(self.cache / "sldc")
        if load.empty:
            self.error = (
                "no cached load data; run scripts/fetch_data.py before starting "
                "the service"
            )
            return

        self.blocks = to_blocks(load)
        first = self.blocks.index.min().date()
        last = self.blocks.index.max().date()
        weather = fetch_weather(first, last, cache_dir=self.cache / "weather")
        self.X, self.y = build_features(self.blocks, weather)
        index = pd.DatetimeIndex(self.X.index)

        self.serve_from = last - timedelta(days=SERVABLE_DAYS)
        self.cal_from = self.serve_from - timedelta(days=CALIBRATION_DAYS)
        self.data_end = last
        self.data_start = first

        train = index < pd.Timestamp(self.cal_from)
        cal = (index >= pd.Timestamp(self.cal_from)) & (
            index < pd.Timestamp(self.serve_from)
        )
        if train.sum() < 5000 or cal.sum() < 500:
            self.error = (
                f"not enough history: {int(train.sum())} training rows and "
                f"{int(cal.sum())} calibration rows"
            )
            return

        self.model = BoostedForecaster().fit(self.X[train], self.y[train])
        self.baseline = SeasonalNaive().fit(self.X[train], self.y[train])

        cal_pred = self.model.predict(self.X[cal])
        self.cal_resid = self.y[cal].to_numpy() - cal_pred
        self.cal_blocks = block_of_day(index[cal])
        self.cal_n = int(cal.sum())
        self.train_n = int(train.sum())

        self.methods = {
            "gaussian": GaussianIntervals(self.alpha).fit(self.cal_resid),
            "split conformal": SplitConformal(self.alpha).fit(self.cal_resid),
            "mondrian conformal": MondrianConformal(self.alpha).fit(
                self.cal_resid, groups=self.cal_blocks
            ),
        }
        self.ready = True

    # -- serving ---------------------------------------------------------

    def forecast(self, day: date) -> dict[str, Any]:
        if not self.ready:
            raise HTTPException(status_code=503, detail=self.error or "not warm yet")
        if day < self.serve_from or day > self.data_end:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{day} is outside the servable window "
                    f"{self.serve_from}..{self.data_end}. Earlier dates were used "
                    f"to fit or calibrate the model, so a forecast for them would "
                    f"be made by a model that had already seen them."
                ),
            )
        index = pd.DatetimeIndex(self.X.index)
        mask = index.normalize() == pd.Timestamp(day)
        if not mask.any():
            raise HTTPException(
                status_code=404,
                detail=f"no usable blocks for {day}; the SLDC archive has gaps",
            )

        features = self.X[mask]
        actual = self.y[mask].to_numpy()
        point = self.model.predict(features)
        baseline = self.baseline.predict(features)
        blocks = block_of_day(pd.DatetimeIndex(features.index))

        bands: dict[str, Any] = {}
        for name, method in self.methods.items():
            if isinstance(method, MondrianConformal):
                intervals = method.predict(point, groups=blocks)
            else:
                intervals = method.predict(point)
            bands[name] = {
                "lower": [_finite(v) for v in intervals.lower],
                "upper": [_finite(v) for v in intervals.upper],
                "coverage": float(
                    np.mean((actual >= intervals.lower) & (actual <= intervals.upper))
                ),
                "mean_width": float(np.mean(intervals.upper - intervals.lower)),
            }

        # The adaptive method is online, so it is replayed over the days from
        # the start of the servable window up to this one -- it has no
        # meaning as a single-day snapshot.
        bands["adaptive conformal"] = self._adaptive_for(day, point, actual, index)

        return {
            "date": str(day),
            "day_kind": day_kind(day),
            "holiday": holiday_name(day),
            "issued": f"{day - timedelta(days=1)} {ISSUE_HOUR:02d}:00",
            "blocks": [int(b) for b in blocks],
            "times": [ts.strftime("%H:%M") for ts in pd.DatetimeIndex(features.index)],
            "actual": [float(v) for v in actual],
            "point": [float(v) for v in point],
            "baseline": [float(v) for v in baseline],
            "intervals": bands,
            "alpha": self.alpha,
            "nominal": 1.0 - self.alpha,
            "metrics": {
                "mape": mape(actual, point),
                "mae": mae(actual, point),
                "baseline_mape": mape(actual, baseline),
                "peak_actual": float(actual.max()),
                "peak_point": float(point.max()),
                "peak_error": float(point.max() - actual.max()),
            },
        }

    def _adaptive_for(self, day, point, actual, index) -> dict[str, Any]:
        """Replay the online method from the start of the servable window.

        An online method's interval on a Thursday depends on how it did on
        the Wednesday. Serving a single day in isolation would show the
        warm-up interval every time, which is not what the method does in
        use.
        """
        upto = (index.normalize() >= pd.Timestamp(self.serve_from)) & (
            index.normalize() <= pd.Timestamp(day)
        )
        window_pred = self.model.predict(self.X[upto])
        window_actual = self.y[upto].to_numpy()
        adaptive = AdaptiveConformal(self.alpha, gamma=0.1)
        intervals = adaptive.run(window_actual, window_pred, self.cal_resid)
        on_day = pd.DatetimeIndex(self.X[upto].index).normalize() == pd.Timestamp(day)
        lower = intervals.lower[on_day]
        upper = intervals.upper[on_day]
        return {
            "lower": [_finite(v) for v in lower],
            "upper": [_finite(v) for v in upper],
            "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
            "mean_width": float(np.mean(upper - lower)),
            "alpha_t": float(adaptive.alpha_t_),
        }


def _finite(x: float) -> float | None:
    """JSON has no infinity, and an unbounded interval is a real outcome."""
    value = float(x)
    return None if not np.isfinite(value) else value


def create_app(cache: Path | str = "data/raw") -> FastAPI:
    service = ForecastService(cache)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Fit at startup rather than on the first request, so a cold Render
        # instance pays the ten seconds before it accepts traffic instead of
        # making the first visitor wait for it. A failure here is recorded
        # and surfaced by /api/health rather than crashing the process: a
        # service that can explain why it is empty beats one that will not
        # start.
        try:
            service.warm()
        except Exception as exc:  # pragma: no cover - startup diagnostics
            service.error = f"{type(exc).__name__}: {exc}"
            log.exception("warm-up failed")
        yield

    app = FastAPI(
        title="gridcast",
        version="0.1.0",
        summary="Day-ahead load forecasting for the Delhi grid, with prediction "
        "intervals whose coverage is measured rather than assumed.",
        lifespan=lifespan,
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        body: dict[str, Any] = {
            "status": "ok" if service.ready else "not ready",
            "error": service.error,
        }
        if service.ready:
            body.update(
                {
                    "data": {"start": str(service.data_start), "end": str(service.data_end)},
                    "fitted_up_to": str(service.cal_from),
                    "calibrated_up_to": str(service.serve_from),
                    "servable": {
                        "from": str(service.serve_from),
                        "to": str(service.data_end),
                    },
                    "train_rows": service.train_n,
                    "calibration_rows": service.cal_n,
                    "alpha": service.alpha,
                }
            )
        return body

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        if not service.ready:
            raise HTTPException(status_code=503, detail=service.error or "not warm")
        facts = summarise(service.blocks)
        facts["features"] = len(service.X.columns)
        facts["feature_rows"] = len(service.X)
        return facts

    @app.get("/api/days")
    def days() -> list[dict[str, Any]]:
        """The dates the service may legitimately forecast."""
        if not service.ready:
            raise HTTPException(status_code=503, detail=service.error or "not warm")
        index = pd.DatetimeIndex(service.X.index).normalize()
        available = sorted({d.date() for d in index if d.date() >= service.serve_from})
        return [
            {
                "date": str(d),
                "kind": day_kind(d),
                "holiday": holiday_name(d),
                "blocks": int((index == pd.Timestamp(d)).sum()),
            }
            for d in available
        ]

    @app.get("/api/forecast")
    def forecast(
        day: date = Query(..., alias="date", description="a date in the servable window")
    ) -> dict[str, Any]:
        return service.forecast(day)

    @app.get("/api/backtest")
    def backtest() -> JSONResponse:
        """The study's output, so the console quotes measured coverage."""
        if not BACKTEST.is_file():
            raise HTTPException(
                status_code=404,
                detail="no backtest report yet; run scripts/backtest.py",
            )
        return JSONResponse(json.loads(BACKTEST.read_text(encoding="utf-8")))

    @app.get("/")
    def console() -> FileResponse:
        index = WEB / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="console not built")
        return FileResponse(index)

    if WEB.is_dir():
        app.mount("/web", StaticFiles(directory=WEB), name="web")

    app.state.service = service
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
