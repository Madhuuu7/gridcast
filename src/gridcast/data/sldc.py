"""Delhi SLDC load data: 5-minute readings for Delhi and its five discoms.

The State Load Despatch Centre publishes a page per day at

    https://www.delhisldc.org/Loaddata.aspx?mode=DD/MM/YYYY

with 288 rows -- one per five-minute slot -- carrying the Delhi total and the
five distribution licensees (BRPL, BYPL, NDPL/TPDDL, NDMC and the Military
Engineer Services). It goes back to at least 2019 and runs to yesterday. No
key, no registration, no terms forbidding it: there is no robots.txt at all,
and this fetches a few hundred pages once and caches them forever.

Three things about the data that the parser has to take a position on, because
getting any of them wrong quietly poisons every model downstream:

**Zero is not a reading.** A slot whose meter did not report comes back as
``0.000``, not as blank. Delhi's minimum load is above 2,000 MW even at 4am in
February, so a zero is a telemetry gap and is read as missing. Treating it as
a real value would teach a model that demand sometimes collapses to nothing,
and would put a fake 5,000 MW error in every backtest that straddles one.

**A day can be short.** Today's page has only the slots so far, and some
historical days are missing slots in the middle. Rows are returned as they
come and the gaps are explicit, so the feature builder can decide -- rather
than a reindex silently inventing values.

**The discoms do not have to sum to the total.** They are separate meters, and
in the archive the sum is off by a few MW routinely and by a lot
occasionally. The total is what this forecasts; the discom columns come along
because the split is interesting and cheap to keep.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

log = logging.getLogger("gridcast.sldc")

BASE_URL = "https://www.delhisldc.org/Loaddata.aspx"

#: The columns the page publishes, in the order it publishes them.
COLUMNS = ("delhi", "brpl", "bypl", "ndpl", "ndmc", "mes")

#: Identify the fetcher. A government site should be able to see who is
#: pulling a few hundred pages and why, rather than a bare python-httpx.
USER_AGENT = (
    "gridcast/0.1 (day-ahead load forecasting study; "
    "https://github.com/Madhuuu7/gridcast)"
)

_ROW = re.compile(
    r"<tr[^>]*>\s*"
    r"<td[^>]*>\s*<font[^>]*>\s*(\d{2}:\d{2})\s*</font>\s*</td>"  # timeslot
    r"((?:\s*<td[^>]*>\s*<font[^>]*>\s*[-\d.]*\s*</font>\s*</td>){1,8})",
    re.IGNORECASE,
)
_CELL = re.compile(r"<td[^>]*>\s*<font[^>]*>\s*([-\d.]*)\s*</font>\s*</td>", re.I)


@dataclass(frozen=True)
class DayResult:
    """What came back for one date, including why it is thin if it is."""

    day: date
    frame: pd.DataFrame
    from_cache: bool = False
    note: str = ""

    @property
    def slots(self) -> int:
        return len(self.frame)

    @property
    def complete(self) -> bool:
        return self.slots == 288


def parse_day(html: str, day: date) -> pd.DataFrame:
    """Pull the load table out of one Loaddata.aspx page.

    Returns a frame indexed by timestamp with one float column per column in
    ``COLUMNS``. Zeros and blanks become NaN -- see the module docstring for
    why a zero cannot be taken at face value.
    """
    start = html.find("DGGridAv")
    if start == -1:
        return _empty()
    end = html.find("</table>", start)
    table = html[start : end if end != -1 else len(html)]

    rows: list[dict[str, object]] = []
    for match in _ROW.finditer(table):
        slot = match.group(1)
        values = _CELL.findall(match.group(2))
        hour, minute = int(slot[:2]), int(slot[3:])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            continue
        row: dict[str, object] = {
            "timestamp": pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)
        }
        for name, raw in zip(COLUMNS, values):
            row[name] = _reading(raw)
        rows.append(row)

    if not rows:
        return _empty()
    frame = pd.DataFrame(rows).set_index("timestamp").sort_index()
    # A repeated timeslot means the page carried the same slot twice; keep the
    # last, which is what the SLDC's own revisions imply.
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[list(COLUMNS)].astype("Float64").astype("float64")


def _reading(raw: str) -> float:
    """A cell to a float, with zero and blank both meaning 'no reading'."""
    text = (raw or "").strip()
    if not text or text in {"-", "."}:
        return float("nan")
    try:
        value = float(text)
    except ValueError:
        return float("nan")
    # Delhi's minimum ever load is well above 1,000 MW, so a zero or a
    # negative is telemetry, not demand.
    return float("nan") if value <= 0.0 else value


def _empty() -> pd.DataFrame:
    frame = pd.DataFrame(
        {column: pd.Series(dtype="float64") for column in COLUMNS},
    )
    frame.index = pd.DatetimeIndex([], name="timestamp")
    return frame


class SldcClient:
    """Fetches days, caching each one as a CSV so a day is pulled once.

    The cache is the point. A backtest over two years is 730 pages; without a
    cache every experiment re-fetches them, which is slow for whoever runs
    this and rude to a public service that is not charging for it.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/raw/sldc",
        *,
        timeout: float = 45.0,
        retries: int = 3,
        pause: float = 0.3,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.retries = retries
        self.pause = pause

    def cache_path(self, day: date) -> Path:
        return self.cache_dir / f"{day:%Y-%m-%d}.csv"

    def fetch_day(self, day: date, *, refresh: bool = False) -> DayResult:
        path = self.cache_path(day)
        if path.is_file() and not refresh:
            frame = pd.read_csv(path, index_col="timestamp", parse_dates=True)
            return DayResult(day, frame, from_cache=True)

        last_error = ""
        for attempt in range(1, self.retries + 1):
            try:
                with httpx.Client(
                    timeout=self.timeout,
                    headers={"User-Agent": USER_AGENT},
                    follow_redirects=True,
                ) as client:
                    response = client.get(BASE_URL, params={"mode": f"{day:%d/%m/%Y}"})
                response.raise_for_status()
                frame = parse_day(response.text, day)
                if frame.empty:
                    return DayResult(day, frame, note="the page carried no load table")
                frame.to_csv(path)
                return DayResult(day, frame)
            except (httpx.HTTPError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                # Linear backoff: the site is slow rather than rate limiting,
                # so waiting longer each time is enough and exponential would
                # just stall a 700-day pull.
                time.sleep(self.pause * attempt * 2)
        return DayResult(day, _empty(), note=f"gave up after {self.retries}: {last_error}")

    def fetch_range(
        self,
        start: date,
        end: date,
        *,
        workers: int = 4,
        refresh: bool = False,
        on_day=None,
    ) -> pd.DataFrame:
        """Every day in [start, end], concatenated, missing days skipped.

        Four workers, not forty: this is one public server and the whole pull
        is a few hundred pages. It finishes in minutes either way, and the
        difference between polite and not is whether anyone minds.
        """
        days = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
        frames: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(lambda d: self.fetch_day(d, refresh=refresh), days):
                if on_day is not None:
                    on_day(result)
                if not result.frame.empty:
                    frames.append(result.frame)
                if not result.from_cache:
                    time.sleep(self.pause)
        if not frames:
            return _empty()
        combined = pd.concat(frames).sort_index()
        return combined[~combined.index.duplicated(keep="last")]


def load_cached(cache_dir: str | Path = "data/raw/sldc") -> pd.DataFrame:
    """Everything already on disk, without touching the network."""
    paths = sorted(Path(cache_dir).glob("*.csv"))
    if not paths:
        return _empty()
    frames = [pd.read_csv(p, index_col="timestamp", parse_dates=True) for p in paths]
    combined = pd.concat(frames).sort_index()
    return combined[~combined.index.duplicated(keep="last")]
