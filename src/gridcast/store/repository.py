"""Reading and writing the record.

The write side is deliberately narrow. There is a function to record a run
and a function to record observations, and there is no function to amend a
forecast, because the value of the whole table is that nothing can. Anything
that wants to change a published forecast has to publish another one, under
its own issue time, where a reader can see both.

The read side is a join between two things written at different times, which
is the only reason any of it is interesting: the forecast rows were written
before the observation rows existed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from .models import BLOCKS_PER_DAY, ForecastBlock, ForecastRun, Observation

#: Indian Standard Time. Fixed offset: the grid does not observe DST.
IST = timezone(timedelta(hours=5, minutes=30))


def ist_day_start_utc(day: date) -> datetime:
    """00:00 IST on ``day``, expressed in UTC."""
    return datetime(day.year, day.month, day.day, tzinfo=IST).astimezone(timezone.utc)


def block_timestamp(day: date, block: int) -> datetime:
    """Start of a scheduling block, in UTC."""
    return ist_day_start_utc(day) + timedelta(minutes=15 * block)


def code_version() -> str:
    """The short git sha, or ``unknown``.

    Recorded on every run because a scorecard spanning months is
    uninterpretable if the thing being scored changed silently partway
    through. ``unknown`` is honest when the deployed artefact has no git
    directory, which on Render it does not.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class ForecastAlreadyIssued(RuntimeError):
    """A forecast for this target day and method already exists.

    Raised rather than overwritten. The caller decides whether to publish a
    second, visible run or to do nothing, and neither of those is a silent
    update.
    """


class HindcastRejected(ValueError):
    """The forecast was issued after the day it claims to forecast began."""


@dataclass(frozen=True)
class IssuedForecast:
    """What the daily job hands the store."""

    target_date: date
    issued_at: datetime
    model: str
    method: str
    alpha: float
    point_mw: np.ndarray
    lower_mw: np.ndarray
    upper_mw: np.ndarray
    train_start: date
    train_end: date
    calibration_start: date
    calibration_end: date
    train_rows: int
    kernel: dict | None = None
    gate: str = "10:00 IST"
    note: str | None = None

    def __post_init__(self) -> None:
        sizes = {
            self.point_mw.size, self.lower_mw.size, self.upper_mw.size
        }
        if sizes != {BLOCKS_PER_DAY}:
            raise ValueError(
                f"a day is {BLOCKS_PER_DAY} blocks; got sizes {sorted(sizes)}"
            )
        if self.issued_at.tzinfo is None:
            raise ValueError("issued_at must be timezone-aware")


def record_forecast(
    session: Session, forecast: IssuedForecast, *, allow_reissue: bool = False
) -> ForecastRun:
    """Write one issuance. Never updates an existing one.

    ``allow_reissue`` permits a *second* run for the same target day, which
    is visible in the table as two rows with different issue times. It does
    not permit touching the first.
    """
    target_start = ist_day_start_utc(forecast.target_date)
    issued = forecast.issued_at.astimezone(timezone.utc)
    if issued >= target_start:
        raise HindcastRejected(
            f"forecast for {forecast.target_date} was issued at {issued:%Y-%m-%d %H:%M} "
            f"UTC, which is after that day began at {target_start:%Y-%m-%d %H:%M} UTC"
        )

    existing = session.scalar(
        select(func.count())
        .select_from(ForecastRun)
        .where(
            ForecastRun.target_date == forecast.target_date,
            ForecastRun.method == forecast.method,
        )
    )
    if existing and not allow_reissue:
        raise ForecastAlreadyIssued(
            f"{forecast.method} already has a forecast for {forecast.target_date}; "
            f"pass allow_reissue=True to publish a second one alongside it"
        )

    run = ForecastRun(
        issued_at=issued,
        target_date=forecast.target_date,
        target_start_utc=target_start,
        gate=forecast.gate,
        model=forecast.model,
        method=forecast.method,
        alpha=forecast.alpha,
        kernel=forecast.kernel,
        train_start=forecast.train_start,
        train_end=forecast.train_end,
        calibration_start=forecast.calibration_start,
        calibration_end=forecast.calibration_end,
        train_rows=forecast.train_rows,
        code_version=code_version(),
        note=forecast.note,
    )
    session.add(run)
    session.flush()

    def _bound(value: float) -> float | None:
        # An unbounded side serialises as NULL, not as a large number. A
        # console that reads a missing bound as zero reports every block as
        # covered, which is a bug this project has already had once.
        return None if not np.isfinite(value) else float(value)

    session.add_all(
        ForecastBlock(
            run_id=run.id,
            target_ts=block_timestamp(forecast.target_date, block),
            block=block,
            point_mw=float(forecast.point_mw[block]),
            lower_mw=_bound(forecast.lower_mw[block]),
            upper_mw=_bound(forecast.upper_mw[block]),
        )
        for block in range(BLOCKS_PER_DAY)
    )
    session.flush()
    return run


def record_observations(
    session: Session,
    observations: dict[datetime, float],
    *,
    series: str = "delhi",
    fetched_at: datetime | None = None,
) -> int:
    """Upsert observed load. Returns the number of rows written.

    Observations, unlike forecasts, *are* allowed to change: SLDC revises
    readings, and a scorecard built on a reading that was later corrected
    should move with the correction. The asymmetry is the point -- what was
    claimed is frozen, what happened is whatever the operator now says it
    was.
    """
    fetched_at = fetched_at or datetime.now(timezone.utc)
    rows = [
        {
            "ts": ts.astimezone(timezone.utc),
            "series": series,
            "mw": float(mw),
            "fetched_at": fetched_at,
        }
        # A zero is a meter that did not report, not a reading, and the
        # table's CHECK would reject it anyway -- filtered here so one bad
        # slot does not abort a whole day's insert.
        for ts, mw in sorted(observations.items())
        if np.isfinite(mw) and mw > 0
    ]
    if not rows:
        return 0

    dialect = session.get_bind().dialect.name
    insert = pg_insert if dialect == "postgresql" else sqlite_insert
    statement = insert(Observation).values(rows)
    statement = statement.on_conflict_do_update(
        index_elements=[Observation.ts, Observation.series],
        set_={
            "mw": statement.excluded.mw,
            "fetched_at": statement.excluded.fetched_at,
        },
    )
    session.execute(statement)
    return len(rows)


def latest_run(
    session: Session, target_date: date, method: str | None = None
) -> ForecastRun | None:
    """The most recent issuance for a day."""
    query = select(ForecastRun).where(ForecastRun.target_date == target_date)
    if method:
        query = query.where(ForecastRun.method == method)
    return session.scalars(
        query.order_by(ForecastRun.issued_at.desc()).limit(1)
    ).first()


def first_run(
    session: Session, target_date: date, method: str | None = None
) -> ForecastRun | None:
    """The *earliest* issuance for a day, which is the one that counts.

    If a day was forecast twice, scoring the later one flatters the record:
    the second issuance was made with more information, even when that
    information was only "the first one looked wrong". The scorecard reads
    this.
    """
    query = select(ForecastRun).where(ForecastRun.target_date == target_date)
    if method:
        query = query.where(ForecastRun.method == method)
    return session.scalars(
        query.order_by(ForecastRun.issued_at.asc()).limit(1)
    ).first()


def issued_days(session: Session, method: str | None = None) -> list[date]:
    """Every target day with a forecast on record, oldest first."""
    query = select(ForecastRun.target_date).distinct()
    if method:
        query = query.where(ForecastRun.method == method)
    return list(session.scalars(query.order_by(ForecastRun.target_date)))


def observations_for(
    session: Session, target_date: date, *, series: str = "delhi"
) -> dict[int, float]:
    """Observed load for a day, keyed by block."""
    start = ist_day_start_utc(target_date)
    end = start + timedelta(days=1)
    rows = session.execute(
        select(Observation.ts, Observation.mw).where(
            Observation.series == series,
            Observation.ts >= start,
            Observation.ts < end,
        )
    ).all()
    out: dict[int, float] = {}
    for ts, mw in rows:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        block = int((ts - start).total_seconds() // 900)
        if 0 <= block < BLOCKS_PER_DAY:
            out[block] = float(mw)
    return out
