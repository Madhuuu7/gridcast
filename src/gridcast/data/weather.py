"""Delhi weather from Open-Meteo's archive: free, key-less, hourly, reanalysis.

Temperature is the single most useful exogenous variable a load forecaster
has. Delhi's summer peak is air conditioning, and the relationship is strongly
non-linear and asymmetric: demand climbs steeply above about 25C, flattens in
the mild band, and climbs again -- less steeply -- in the winter heating
range. A model without temperature can still learn the daily and weekly shape,
but it cannot know that tomorrow is six degrees hotter than today, which is
exactly the day it will be most wrong and most expensive to be wrong on.

Two things worth being explicit about, because both affect what the forecast
is allowed to claim:

**This is reanalysis, not a forecast.** The archive returns what the weather
actually was. A genuine day-ahead system would use a weather *forecast*, which
carries its own error, so using actuals makes the point forecast look better
than production would be. The backtest says so, and the honest framing is that
the intervals are conditional on the weather being known -- which is also why
a forecast-error term is in the README's limits rather than hidden.

**The archive lags by about five days.** ERA5 is not real time. Recent days
come from a separate near-real-time model, so the last week can shift slightly
when the final reanalysis lands. For a backtest over two years that is noise;
for a live service it means the most recent rows are provisional.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

log = logging.getLogger("gridcast.weather")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

#: Safdarjung, the reference station for Delhi's weather and roughly the
#: load centre's centroid.
DELHI_LAT, DELHI_LON = 28.6139, 77.2090

#: What the model gets. Temperature does the work; humidity matters because
#: Delhi's monsoon weeks feel hotter than the thermometer says and cooling
#: load follows the feel. Cloud cover is a proxy for daylight, which moves
#: the evening lighting ramp.
HOURLY = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "cloud_cover",
    "wind_speed_10m",
)


def fetch_weather(
    start: date,
    end: date,
    *,
    cache_dir: str | Path = "data/raw/weather",
    latitude: float = DELHI_LAT,
    longitude: float = DELHI_LON,
    refresh: bool = False,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Hourly Delhi weather over [start, end], cached as one CSV per range.

    Open-Meteo serves a multi-year range in a single response, so unlike the
    SLDC pull this is one request rather than one per day.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"delhi-{start:%Y%m%d}-{end:%Y%m%d}.csv"
    if path.is_file() and not refresh:
        return pd.read_csv(path, index_col="timestamp", parse_dates=True)

    response = httpx.get(
        ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": f"{start:%Y-%m-%d}",
            "end_date": f"{end:%Y-%m-%d}",
            "hourly": ",".join(HOURLY),
            # Asking for IST rather than converting after: the load data is
            # in local time with no DST to reconcile, and a timezone mismatch
            # between the two sources is a bug that looks like a modelling
            # result -- a model that is quietly 5.5 hours out still fits
            # something, just the wrong thing.
            "timezone": "Asia/Kolkata",
        },
        timeout=timeout,
        headers={"User-Agent": "gridcast/0.1"},
    )
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly")
    if not hourly or not hourly.get("time"):
        raise ValueError(f"no hourly weather returned for {start}..{end}")

    frame = pd.DataFrame(hourly).rename(columns={"time": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.to_csv(path)
    return frame


def to_blocks(weather: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly weather onto a finer index, by interpolation.

    Linear interpolation between hourly values, forward/back filled at the
    ends. Temperature is smooth at this resolution, so interpolating it is
    honest; a step function would put a false discontinuity at the top of
    every hour and the model would learn it as structure.
    """
    if weather.empty:
        return pd.DataFrame(index=index)
    combined = weather.reindex(weather.index.union(index)).sort_index()
    combined = combined.interpolate(method="time", limit_direction="both")
    return combined.reindex(index)
