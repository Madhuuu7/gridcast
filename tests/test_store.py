"""The rules that make a stored forecast evidence rather than a row.

A table of predictions is worth nothing on its own -- anyone can write last
week's load into a column called ``forecast``. What makes the record mean
something is a short list of things the code is unable to do, so most of
these tests are attempts to do them.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy.exc import IntegrityError, StatementError

from gridcast.store import (
    Base,
    ForecastAlreadyIssued,
    ForecastBlock,
    ForecastRun,
    HindcastRejected,
    IssuedForecast,
    Observation,
    block_timestamp,
    create_db_engine,
    database_url,
    first_run,
    ist_day_start_utc,
    issued_days,
    latest_run,
    observations_for,
    record_forecast,
    record_observations,
    session_factory,
    session_scope,
)
from gridcast.store.models import BLOCKS_PER_DAY

TARGET = date(2026, 6, 15)
#: 10:00 IST on the day before the target, in UTC -- the scheduling gate.
GATE = datetime(2026, 6, 14, 4, 30, tzinfo=timezone.utc)


@pytest.fixture
def session():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = session_factory(engine)
    with session_scope(factory) as s:
        yield s
    engine.dispose()


def _forecast(**overrides) -> IssuedForecast:
    point = np.linspace(3000.0, 6000.0, BLOCKS_PER_DAY)
    defaults = dict(
        target_date=TARGET,
        issued_at=GATE,
        model="boosted",
        method="localized conformal",
        alpha=0.1,
        point_mw=point,
        lower_mw=point - 500.0,
        upper_mw=point + 500.0,
        train_start=date(2026, 1, 1),
        train_end=date(2026, 4, 15),
        calibration_start=date(2026, 4, 16),
        calibration_end=date(2026, 6, 13),
        train_rows=9_600,
        kernel={"block_bandwidth": 8.0, "kind_penalty": 0.05},
    )
    return IssuedForecast(**{**defaults, **overrides})


# -- the rule the table exists for -----------------------------------------


def test_a_forecast_issued_after_its_day_began_is_rejected():
    """The difference between a forecast and a hindcast is a timestamp.

    Not a warning, not a flag on the row. A record that accepts these is a
    record where every entry has to be individually defended, which is the
    same as having no record.
    """
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with session_scope(session_factory(engine)) as s:
        # One minute after midnight IST on the target day.
        late = ist_day_start_utc(TARGET) + timedelta(minutes=1)
        with pytest.raises(HindcastRejected, match="after that day began"):
            record_forecast(s, _forecast(issued_at=late))
    engine.dispose()


def test_midnight_ist_is_the_boundary_and_it_is_exclusive(session):
    """Issued at exactly 00:00 IST is already too late."""
    with pytest.raises(HindcastRejected):
        record_forecast(session, _forecast(issued_at=ist_day_start_utc(TARGET)))
    one_second_earlier = ist_day_start_utc(TARGET) - timedelta(seconds=1)
    run = record_forecast(session, _forecast(issued_at=one_second_earlier))
    assert run.id is not None


def test_the_database_rejects_a_hindcast_even_when_python_does_not(session):
    """The constraint is in SQL, so bypassing the helper does not help.

    The Python guard is the friendly error. This is the one that holds when
    somebody writes a second insert path in six months and forgets.
    """
    session.add(
        ForecastRun(
            issued_at=ist_day_start_utc(TARGET) + timedelta(hours=3),
            target_date=TARGET,
            target_start_utc=ist_day_start_utc(TARGET),
            model="boosted", method="split conformal", alpha=0.1,
            train_start=date(2026, 1, 1), train_end=date(2026, 4, 15),
            calibration_start=date(2026, 4, 16), calibration_end=date(2026, 6, 13),
            train_rows=100,
        )
    )
    with pytest.raises(IntegrityError, match="issued_before_target"):
        session.flush()
    session.rollback()


def test_there_is_no_way_to_amend_a_published_forecast(session):
    """Re-issuing publishes a second run; it never edits the first."""
    first = record_forecast(session, _forecast())
    with pytest.raises(ForecastAlreadyIssued, match="already has a forecast"):
        record_forecast(session, _forecast())

    later = record_forecast(
        session,
        _forecast(issued_at=GATE + timedelta(hours=2), note="re-issued after a bad fit"),
        allow_reissue=True,
    )
    assert later.id != first.id
    # And the scorecard reads the earlier one, because the later issuance
    # was made with more information -- even if that information was only
    # "the first one looked wrong".
    assert first_run(session, TARGET).id == first.id
    assert latest_run(session, TARGET).id == later.id


def test_a_day_is_ninety_six_blocks_and_nothing_else():
    short = np.ones(95)
    with pytest.raises(ValueError, match="96 blocks"):
        _forecast(point_mw=short, lower_mw=short, upper_mw=short)


def test_issued_at_must_carry_a_timezone():
    with pytest.raises(ValueError, match="timezone-aware"):
        _forecast(issued_at=datetime(2026, 6, 14, 4, 30))


# -- round trips that have gone wrong before -------------------------------


def test_an_unbounded_side_round_trips_as_null_not_as_a_number(session):
    """An adaptive method that has lost track produces an infinite bound.

    This project has already shipped the bug once, in the console: a missing
    bound compared as a number counted every such block as covered. Null is
    the only representation that cannot be silently compared.
    """
    point = np.full(BLOCKS_PER_DAY, 4000.0)
    lower = point - 500.0
    upper = point + 500.0
    upper[7] = np.inf
    lower[7] = -np.inf
    run = record_forecast(
        session, _forecast(point_mw=point, lower_mw=lower, upper_mw=upper)
    )
    session.flush()
    stored = {b.block: b for b in run.blocks}
    assert stored[7].lower_mw is None
    assert stored[7].upper_mw is None
    assert stored[8].lower_mw == pytest.approx(3500.0)


def test_block_timestamps_are_fifteen_minutes_apart_from_midnight_ist(session):
    run = record_forecast(session, _forecast())
    session.flush()
    blocks = sorted(run.blocks, key=lambda b: b.block)
    assert len(blocks) == BLOCKS_PER_DAY
    start = ist_day_start_utc(TARGET)
    assert blocks[0].target_ts == start
    assert blocks[-1].target_ts == start + timedelta(minutes=15 * 95)
    assert block_timestamp(TARGET, 40) == start + timedelta(hours=10)


def test_timestamps_come_back_aware_on_sqlite_too(session):
    """SQLite has no timestamp type and returns a naive string.

    Postgres returns an aware datetime for the same column, so without the
    ``UtcDateTime`` decorator this project compares aware against naive in
    whichever environment it is tested second. The drift it would hide is
    5h30m -- larger than any error the forecast makes.
    """
    run = record_forecast(session, _forecast())
    session.flush()
    session.expire_all()
    reloaded = session.get(ForecastRun, run.id)
    assert reloaded.issued_at.tzinfo is not None
    assert reloaded.issued_at == GATE
    assert reloaded.blocks[0].target_ts.tzinfo is not None


def test_a_naive_timestamp_is_refused_rather_than_guessed(session):
    """There is no sensible default timezone for a grid scheduling record.

    SQLAlchemy wraps the type's own ValueError in a StatementError, so the
    assertion is on the message rather than the class -- the point is that
    the write fails loudly at the boundary and not that a particular
    exception type escapes.
    """
    session.add(
        Observation(
            ts=datetime(2026, 6, 15, 0, 0),
            series="delhi",
            mw=4000.0,
            fetched_at=datetime.now(timezone.utc),
        )
    )
    with pytest.raises(StatementError, match="naive timestamp"):
        session.flush()
    session.rollback()


def test_deleting_a_run_takes_its_blocks_with_it(session):
    """SQLite ignores foreign keys unless the pragma is set, so check it is."""
    run = record_forecast(session, _forecast())
    session.flush()
    assert session.query(ForecastBlock).count() == BLOCKS_PER_DAY
    session.delete(run)
    session.flush()
    assert session.query(ForecastBlock).count() == 0


# -- observations ----------------------------------------------------------


def _readings(day: date, n: int = 96, mw: float = 4000.0) -> dict:
    start = ist_day_start_utc(day)
    return {start + timedelta(minutes=15 * i): mw + i for i in range(n)}


def test_observations_upsert_because_the_operator_revises_them(session):
    """Forecasts are frozen; what happened is whatever SLDC now says.

    The asymmetry is deliberate and is the only place in the schema where an
    update is allowed.
    """
    assert record_observations(session, _readings(TARGET, n=4)) == 4
    session.flush()
    revised = {ist_day_start_utc(TARGET): 9999.0}
    assert record_observations(session, revised) == 1
    session.flush()
    assert observations_for(session, TARGET)[0] == pytest.approx(9999.0)
    assert session.query(Observation).count() == 4


def test_a_zero_is_a_meter_that_did_not_report(session):
    """Delhi never draws nothing. Taken at face value it teaches the model
    that demand sometimes collapses, and the scorecard that the forecast
    missed by 4,000 MW."""
    start = ist_day_start_utc(TARGET)
    written = record_observations(
        session,
        {
            start: 4000.0,
            start + timedelta(minutes=15): 0.0,
            start + timedelta(minutes=30): float("nan"),
            start + timedelta(minutes=45): -12.0,
        },
    )
    session.flush()
    assert written == 1
    assert sorted(observations_for(session, TARGET)) == [0]


def test_observations_are_keyed_by_block_within_the_ist_day(session):
    record_observations(session, _readings(TARGET))
    # A reading from the next IST day must not leak into this one.
    record_observations(
        session, {ist_day_start_utc(TARGET) + timedelta(days=1): 7777.0}
    )
    session.flush()
    today = observations_for(session, TARGET)
    assert len(today) == BLOCKS_PER_DAY
    assert max(today) == BLOCKS_PER_DAY - 1
    assert 7777.0 not in today.values()


def test_issued_days_lists_each_target_once(session):
    record_forecast(session, _forecast())
    record_forecast(
        session, _forecast(issued_at=GATE + timedelta(hours=1)), allow_reissue=True
    )
    record_forecast(
        session,
        _forecast(target_date=TARGET + timedelta(days=1), issued_at=GATE + timedelta(days=1)),
    )
    session.flush()
    assert issued_days(session) == [TARGET, TARGET + timedelta(days=1)]
    assert issued_days(session, method="nonexistent") == []


# -- connection handling ---------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected_prefix",
    [
        ("postgres://u:p@host/db", "postgresql+psycopg://"),
        ("postgresql://u:p@host/db", "postgresql+psycopg://"),
        ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://"),
        ("sqlite+pysqlite:///x.db", "sqlite+pysqlite://"),
    ],
)
def test_managed_postgres_schemes_are_normalised(raw, expected_prefix):
    """Render and Neon hand out ``postgres://``, which SQLAlchemy 2 rejects.

    The failure is a NoSuchModuleError at startup, which names neither the
    provider nor the fix, so it is handled here rather than in a deploy
    checklist.
    """
    assert database_url(raw).startswith(expected_prefix)
    assert database_url(raw).endswith("/db") or raw.startswith("sqlite")


def test_the_url_does_not_come_from_the_environment_when_passed_explicitly(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://env/db")
    assert "explicit" in database_url("sqlite+pysqlite:///explicit.db")
    assert database_url().startswith("postgresql+psycopg://")


def test_a_stale_schema_stops_the_app_rather_than_serving(tmp_path):
    """``create_all`` adds missing tables but never missing columns.

    A database from before a column existed looks healthy and fails on the
    first row that needs it. That bug cost an afternoon on an earlier
    project; here the version is checked at boot and the app refuses.
    """
    from gridcast.store.engine import SchemaOutOfDate, require_current_schema

    engine = create_db_engine(f"sqlite+pysqlite:///{tmp_path / 'empty.db'}")
    try:
        with pytest.raises(SchemaOutOfDate, match="no schema"):
            require_current_schema(engine)
    finally:
        engine.dispose()
