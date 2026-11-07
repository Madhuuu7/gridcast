"""The running record, including the days it is not allowed to quietly skip."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from gridcast.scorecard import build_record, cost_of_record, interval_value, score_day
from gridcast.store import (
    Base,
    IssuedForecast,
    create_db_engine,
    ist_day_start_utc,
    record_forecast,
    record_observations,
    session_factory,
    session_scope,
)
from gridcast.store.models import BLOCKS_PER_DAY

DAY = date(2026, 6, 15)


@pytest.fixture
def session():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with session_scope(session_factory(engine)) as s:
        yield s
    engine.dispose()


def _issue(session, day=DAY, *, point=4000.0, radius=300.0, method="localized conformal",
           issued_at=None, allow_reissue=False):
    p = np.full(BLOCKS_PER_DAY, point)
    return record_forecast(
        session,
        IssuedForecast(
            target_date=day,
            issued_at=issued_at or (ist_day_start_utc(day) - timedelta(hours=14)),
            model="boosted",
            method=method,
            alpha=0.1,
            point_mw=p,
            lower_mw=p - radius,
            upper_mw=p + radius,
            train_start=day - timedelta(days=200),
            train_end=day - timedelta(days=61),
            calibration_start=day - timedelta(days=60),
            calibration_end=day - timedelta(days=2),
            train_rows=9600,
        ),
        allow_reissue=allow_reissue,
    )


def _observe(session, day=DAY, *, mw=4100.0, blocks=BLOCKS_PER_DAY):
    start = ist_day_start_utc(day)
    record_observations(
        session,
        {start + timedelta(minutes=15 * i): mw for i in range(blocks)},
    )


def test_a_day_nobody_forecast_scores_as_nothing(session):
    assert score_day(session, DAY) is None


def test_a_forecast_with_no_observations_yet_is_pending_not_perfect(session):
    """Tomorrow's forecast has no outcome. It must not read as a clean sheet."""
    _issue(session)
    session.flush()
    score = score_day(session, DAY)
    assert score.scored == 0
    assert score.missing == BLOCKS_PER_DAY
    assert score.coverage is None
    assert score.mae_mw is None


def test_scoring_a_complete_day(session):
    _issue(session, point=4000.0, radius=300.0)
    _observe(session, mw=4100.0)
    session.flush()
    score = score_day(session, DAY)
    assert score.scored == BLOCKS_PER_DAY
    assert score.missing == 0
    assert score.coverage == 1.0          # 100 MW out, inside a 300 MW band
    assert score.mae_mw == pytest.approx(100.0)
    assert score.mean_width_mw == pytest.approx(600.0)
    assert score.rupees > 0


def test_a_partial_day_is_reported_as_partial_not_dropped(session):
    """SLDC publishes incomplete days, and those are the interesting ones.

    A scorecard that silently skips them is a scorecard that skips the days
    the grid had trouble, which are exactly the days a reader would want
    counted.
    """
    _issue(session)
    _observe(session, blocks=40)
    session.flush()
    score = score_day(session, DAY)
    assert score.scored == 40
    assert score.missing == BLOCKS_PER_DAY - 40
    assert score.coverage is not None


def test_one_day_cannot_measure_coverage_precisely_and_says_so(session):
    """96 blocks give a 90% estimate about +/-6 points of room.

    Showing "87.5%" for yesterday as though it were a measurement is the
    error this project spends a whole module avoiding in the backtest.
    """
    _issue(session, point=4000.0, radius=120.0)
    start = ist_day_start_utc(DAY)
    # 12 of 96 blocks outside the band: coverage 87.5%.
    record_observations(
        session,
        {
            start + timedelta(minutes=15 * i): (4000.0 + (400.0 if i < 12 else 50.0))
            for i in range(BLOCKS_PER_DAY)
        },
    )
    session.flush()
    score = score_day(session, DAY)
    assert score.coverage == pytest.approx(0.875)
    assert score.coverage_se == pytest.approx(0.066, abs=0.01)


def test_the_earliest_issuance_is_the_one_scored(session):
    """A second issuance was made with more information, even when that
    information was only that the first looked wrong."""
    gate = ist_day_start_utc(DAY) - timedelta(hours=14)
    _issue(session, point=4000.0, radius=100.0, issued_at=gate)
    _issue(
        session, point=4100.0, radius=100.0,
        issued_at=gate + timedelta(hours=2), allow_reissue=True,
    )
    _observe(session, mw=4100.0)
    session.flush()
    score = score_day(session, DAY)
    # Scored against the 4000 forecast, which misses, not the 4100 one.
    assert score.mae_mw == pytest.approx(100.0)
    assert score.coverage == 1.0
    assert score.issued_at.startswith(gate.isoformat()[:13])


def test_unbounded_blocks_are_counted_and_do_not_poison_the_width(session):
    p = np.full(BLOCKS_PER_DAY, 4000.0)
    lower = p - 300.0
    upper = p + 300.0
    upper[5] = np.inf
    lower[5] = -np.inf
    record_forecast(
        session,
        IssuedForecast(
            target_date=DAY,
            issued_at=ist_day_start_utc(DAY) - timedelta(hours=14),
            model="boosted", method="adaptive conformal", alpha=0.1,
            point_mw=p, lower_mw=lower, upper_mw=upper,
            train_start=DAY - timedelta(days=200), train_end=DAY - timedelta(days=61),
            calibration_start=DAY - timedelta(days=60),
            calibration_end=DAY - timedelta(days=2), train_rows=9600,
        ),
    )
    _observe(session, mw=4100.0)
    session.flush()
    score = score_day(session, DAY)
    assert score.unbounded_blocks == 1
    # The infinite block is covered, but it must not make the mean width
    # infinite and wipe out the figure for the other 95.
    assert np.isfinite(score.mean_width_mw)
    assert score.mean_width_mw == pytest.approx(600.0)


# -- the accumulating totals ----------------------------------------------


def test_totals_pool_blocks_rather_than_averaging_days(session):
    """A 96-block day and a 4-block day are not equal evidence.

    Averaging the per-day coverages would treat them as though they were,
    and the whole reason to keep a record is that the standard error
    shrinks as blocks accumulate.
    """
    other = DAY + timedelta(days=1)
    _issue(session, DAY, point=4000.0, radius=300.0)
    _observe(session, DAY, mw=4100.0)                      # 96 blocks, all covered
    _issue(session, other, point=4000.0, radius=50.0,
           issued_at=ist_day_start_utc(other) - timedelta(hours=14))
    _observe(session, other, mw=4900.0, blocks=4)          # 4 blocks, none covered
    session.flush()

    record = build_record(session)
    totals = record.totals()
    assert totals["days"] == 2
    assert totals["blocks"] == 100
    # Pooled: 96/100, not the day-average of (1.0 + 0.0) / 2 = 0.5.
    assert totals["coverage"] == pytest.approx(0.96)
    assert totals["coverage"] != pytest.approx(0.5)


def test_an_empty_record_reports_nothing_rather_than_zero_coverage(session):
    """Zero coverage and no data are different claims."""
    totals = build_record(session).totals()
    assert totals["days"] == 0
    assert totals["coverage"] is None


def test_the_record_grows_and_its_uncertainty_shrinks(session):
    """The property that makes a live record worth more than a backtest.

    Coverage has to be imperfect for this to say anything: a Wald interval
    around 1.0 has zero width, so a record of flawless days would show the
    uncertainty shrinking only because there was none to begin with.
    """
    for offset in range(10):
        day = DAY + timedelta(days=offset)
        _issue(session, day, point=4000.0, radius=300.0,
               issued_at=ist_day_start_utc(day) - timedelta(hours=14))
        start = ist_day_start_utc(day)
        # Ten of 96 blocks land outside the band, every day.
        record_observations(
            session,
            {
                start + timedelta(minutes=15 * i): (4600.0 if i < 10 else 4100.0)
                for i in range(BLOCKS_PER_DAY)
            },
        )
    session.flush()
    full = build_record(session).totals()

    one_day = build_record(session)
    one_day.days = one_day.days[:1]
    assert full["blocks"] == 10 * BLOCKS_PER_DAY
    assert full["coverage"] == pytest.approx(86 / 96)
    assert 0.0 < full["coverage_se"] < one_day.totals()["coverage_se"]


def test_cost_and_interval_value_run_over_the_whole_record(session):
    _issue(session, point=4000.0, radius=300.0)
    _observe(session, mw=4400.0)
    session.flush()
    cost = cost_of_record(session)
    assert cost["blocks"] == BLOCKS_PER_DAY
    assert cost["rupees_total"] > 0

    value = interval_value(session)
    assert value["blocks_scored"] == BLOCKS_PER_DAY
    # Actual landed 400 MW out against a 300 MW band: the bill exceeded what
    # the interval implied, and the calibration figure must show that.
    assert value["cost_coverage"] == 0.0
    assert value["worst_underestimate"] > 0


def test_filtering_by_method_keeps_two_methods_apart(session):
    _issue(session, method="localized conformal", point=4000.0, radius=300.0)
    _issue(session, method="split conformal", point=4000.0, radius=50.0)
    _observe(session, mw=4100.0)
    session.flush()
    wide = build_record(session, method="localized conformal").totals()
    narrow = build_record(session, method="split conformal").totals()
    assert wide["coverage"] == 1.0
    assert narrow["coverage"] == 0.0
