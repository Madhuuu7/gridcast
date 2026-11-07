"""The tables, and the one property that makes the record worth keeping.

A backtest is a claim about the past made in the present, and the honest
reader's objection to every one of them is the same: you chose the folds, the
features and the methods knowing how the story ended. The answer is not a
better backtest. It is a forecast written down *before* the day it describes,
and never touched again.

So the schema's job is not storage, it is evidence. Three rules give it that,
and each is enforced here rather than trusted to the job that writes:

**A forecast is immutable.** There is no update path for a forecast block.
Issuing again for the same target day creates a *new* run, visibly, with its
own issue time; it does not overwrite the first. A record that can be revised
after the outcome is known proves nothing at all, and the easiest way to be
sure nobody revises it is to give the code no way to.

**A forecast must predate its target.** ``issued_at`` is checked against the
start of ``target_date`` in the local grid timezone by a database constraint,
not by the caller. A row that fails it is not a late forecast, it is a
hindcast wearing a forecast's clothes, and the difference is the entire
claim.

**What produced it is part of it.** Model, interval method, alpha, kernel
settings, the calibration window and the code version are stored on the run.
Six months of accumulated scorecard is uninterpretable if the thing being
scored changed silently in week three, and "the model improved in March"
should be a visible fact rather than a gap in the numbers.

Observations are separate and arrive later, which is the whole point: the
scorecard is a join between something written on Tuesday and something that
happened on Wednesday.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, TypeDecorator


class UtcDateTime(TypeDecorator):
    """A timestamp that is timezone-aware UTC on both dialects.

    Postgres stores the offset; SQLite has no timestamp type at all and
    hands back a naive string, so ``DateTime(timezone=True)`` quietly means
    two different things in the two environments this runs in. The symptom
    is not an error -- it is a comparison between an aware and a naive
    datetime raising ``TypeError`` in whichever environment you test second,
    or worse, an hour-and-a-half of silent drift once somebody "fixes" it
    by stripping the tzinfo.

    So: convert to UTC on the way in, attach UTC on the way out, and let
    every caller assume aware UTC everywhere.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                f"refusing to store the naive timestamp {value!r}: it is "
                f"ambiguous, and the 5h30m it would silently shift by is "
                f"larger than any error this project measures"
            )
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else (
            value.astimezone(timezone.utc)
        )

#: Indian Standard Time, as a fixed offset. The grid does not observe DST and
#: neither does the scheduling day, so a fixed offset is correct here and a
#: timezone database lookup would only add a dependency and a failure mode.
IST_OFFSET_HOURS = 5.5

#: Scheduling blocks in a day.
BLOCKS_PER_DAY = 96


class Base(DeclarativeBase):
    pass


class ForecastRun(Base):
    """One issuance: everything that was true about the model at the gate."""

    __tablename__ = "forecast_run"

    id: Mapped[int] = mapped_column(primary_key=True)

    #: When the forecast was issued, in UTC. Compared against the target
    #: day's start by a constraint below.
    issued_at: Mapped[datetime] = mapped_column(UtcDateTime, index=True)
    #: The day being forecast, in grid-local terms.
    target_date: Mapped[date] = mapped_column(Date, index=True)
    #: 00:00 IST on ``target_date``, as UTC. Derived, and stored anyway: the
    #: "a forecast must predate its target" rule is a comparison between a
    #: timestamp and a date, and expressing that in portable SQL means
    #: materialising one side. Computing it in Postgres needs INTERVAL
    #: arithmetic that SQLite does not have, and this project runs on SQLite
    #: locally and Postgres deployed, so the constraint would exist in only
    #: one of the two places that matter.
    target_start_utc: Mapped[datetime] = mapped_column(UtcDateTime)
    #: The scheduling gate this issuance is for, kept as text because it is
    #: a market convention rather than a computed value.
    gate: Mapped[str] = mapped_column(String(32), default="10:00 IST")

    model: Mapped[str] = mapped_column(String(64))
    method: Mapped[str] = mapped_column(String(64))
    alpha: Mapped[float] = mapped_column(Float)
    #: Kernel settings when the method is localized conformal, else null.
    #: JSON rather than columns because the tuning surface is the method's
    #: business and will change; the scorecard only needs to know that two
    #: runs differed.
    kernel: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    train_start: Mapped[date] = mapped_column(Date)
    train_end: Mapped[date] = mapped_column(Date)
    calibration_start: Mapped[date] = mapped_column(Date)
    calibration_end: Mapped[date] = mapped_column(Date)
    #: Rows the model was fitted on. A run on a short window is not wrong,
    #: but it is different, and the scorecard should be able to say so.
    train_rows: Mapped[int] = mapped_column(Integer)

    #: Git describe or sha of the code that produced this. Unknown is
    #: recorded as such rather than left null, because a null here reads as
    #: "not applicable" when it means "nobody checked".
    code_version: Mapped[str] = mapped_column(String(64), default="unknown")
    #: Why this run exists, when it is not the ordinary daily one --
    #: a backfill, a re-issue after a failure, a manual trigger.
    note: Mapped[str | None] = mapped_column(String(256), nullable=True)

    blocks: Mapped[list["ForecastBlock"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        # The rule the whole table exists to enforce. A run issued at or
        # after midnight IST on its target day is a hindcast wearing a
        # forecast's clothes, and the difference is the entire claim this
        # project makes. In SQL rather than in the caller, so that no future
        # job written in a hurry can route around it.
        CheckConstraint(
            "issued_at < target_start_utc",
            name="ck_forecast_run_issued_before_target",
        ),
        CheckConstraint("alpha > 0 AND alpha < 1", name="ck_forecast_run_alpha"),
        CheckConstraint("train_rows >= 0", name="ck_forecast_run_train_rows"),
        CheckConstraint(
            "calibration_start <= calibration_end AND train_start <= train_end",
            name="ck_forecast_run_windows_ordered",
        ),
        Index("ix_forecast_run_target_method", "target_date", "method"),
    )


class ForecastBlock(Base):
    """One of the 96 blocks of one issuance. Written once, never updated."""

    __tablename__ = "forecast_block"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("forecast_run.id", ondelete="CASCADE"), index=True
    )

    #: Start of the block, UTC. Redundant with run.target_date and block, and
    #: stored anyway because every join the scorecard does is on time.
    target_ts: Mapped[datetime] = mapped_column(UtcDateTime, index=True)
    block: Mapped[int] = mapped_column(Integer)

    point_mw: Mapped[float] = mapped_column(Float)
    #: Null means an unbounded side. An adaptive method that has lost track
    #: produces one, and it has to survive the round trip as "no bound"
    #: rather than as a number -- a console that reads a missing bound as
    #: zero will happily report that every block was covered.
    lower_mw: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_mw: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped[ForecastRun] = relationship(back_populates="blocks")

    __table_args__ = (
        UniqueConstraint("run_id", "block", name="uq_forecast_block_run_block"),
        CheckConstraint(
            f"block >= 0 AND block < {BLOCKS_PER_DAY}", name="ck_forecast_block_range"
        ),
        CheckConstraint(
            "lower_mw IS NULL OR upper_mw IS NULL OR lower_mw <= upper_mw",
            name="ck_forecast_block_ordered",
        ),
    )


class Observation(Base):
    """What the grid actually drew. Arrives after the forecast, by design."""

    __tablename__ = "observation"

    ts: Mapped[datetime] = mapped_column(UtcDateTime, primary_key=True)
    mw: Mapped[float] = mapped_column(Float)
    #: Which series this is. The Delhi total is what gets forecast; the five
    #: discoms are fetched and stored so the hierarchy is available later.
    series: Mapped[str] = mapped_column(String(16), primary_key=True, default="delhi")
    #: When it was pulled. An observation revised by SLDC after the fact is a
    #: real thing, and the scorecard should be able to notice.
    fetched_at: Mapped[datetime] = mapped_column(UtcDateTime)

    __table_args__ = (
        # A meter that did not report returns 0.000, and Delhi never draws
        # less than about 1,400 MW. Letting a zero in would teach both the
        # model and the scorecard that demand sometimes collapses to nothing.
        CheckConstraint("mw > 0", name="ck_observation_positive"),
        Index("ix_observation_series_ts", "series", "ts"),
    )
