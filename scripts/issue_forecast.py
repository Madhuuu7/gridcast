"""Issue tomorrow's schedule, record yesterday's outcome, print the record.

    python scripts/issue_forecast.py                  # forecast tomorrow
    python scripts/issue_forecast.py --observe        # record what happened
    python scripts/issue_forecast.py --scorecard      # what the record says
    python scripts/issue_forecast.py --target 2026-11-09 --method "split conformal"

Run daily, a little after the 10:00 IST gate. The forecast is refused by the
store if it arrives after midnight IST on the day it describes, so a job that
silently slips past the boundary produces a visible gap rather than a quietly
backdated row -- which is the behaviour that makes the record worth reading.

Two steps, deliberately in one command and in this order: record the
observations for days already forecast, then issue the next one. The
observations settle what was claimed before anything new is claimed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridcast.data.sldc import SldcClient  # noqa: E402
from gridcast.issue import DEFAULT_METHOD, gather, issue  # noqa: E402
from gridcast.scorecard import build_record, interval_value  # noqa: E402
from gridcast.store import (  # noqa: E402
    ForecastAlreadyIssued,
    HindcastRejected,
    create_db_engine,
    issued_days,
    record_forecast,
    record_observations,
    require_current_schema,
    session_factory,
    session_scope,
)
from gridcast.store.repository import IST  # noqa: E402


def observe(session, cache: Path, *, days_back: int = 7) -> int:
    """Record actual load for recent days, including revisions.

    Re-fetches rather than reading the cache, because SLDC revises: a block
    that read 4,210 MW on the night can read 4,260 a week later, and a
    scorecard built on the first number is scoring a reading the operator
    has since withdrawn.
    """
    client = SldcClient(cache_dir=cache / "sldc")
    today = datetime.now(IST).date()
    start = today - timedelta(days=days_back)
    frame = client.fetch_range(start, today, refresh=True)
    if frame.empty or "delhi" not in frame:
        print(f"  no load published for {start}..{today}")
        return 0

    series = frame["delhi"].astype(float)
    readings = {
        timestamp.to_pydatetime().replace(tzinfo=IST): float(value)
        for timestamp, value in series.items()
    }
    written = record_observations(session, readings)
    print(f"  recorded {written:,} readings for {start}..{today}")
    return written


def print_scorecard(session, method: str | None) -> None:
    record = build_record(session, method=method)
    totals = record.totals()
    if not totals["days"]:
        print("  nothing on record yet")
        return

    print(f"  {'day':12s} {'method':20s} {'blocks':>7s} {'coverage':>10s} "
          f"{'MAE':>8s} {'rupees':>12s}")
    for day in record.days[-14:]:
        coverage = (
            f"{day.coverage:.1%}" if day.coverage is not None else "pending"
        )
        mae = f"{day.mae_mw:7.1f}" if day.mae_mw is not None else "      -"
        rupees = f"{day.rupees:,.0f}" if day.rupees is not None else "-"
        print(f"  {day.target_date.isoformat():12s} {day.method:20s} "
              f"{day.scored:7d} {coverage:>10s} {mae} {rupees:>12s}")

    nominal_gap = ""
    if totals["coverage"] is not None and totals["coverage_se"]:
        nominal_gap = (
            f" +/- {totals['coverage_se']:.1%}"
        )
    print(f"\n  {totals['days']} days, {totals['blocks']:,} blocks scored")
    if totals["coverage"] is not None:
        print(f"  pooled coverage {totals['coverage']:.1%}{nominal_gap}, "
              f"MAE {totals['mae_mw']:.1f} MW")
    value = interval_value(session, method=method)
    if value["blocks_scored"]:
        print(f"  the interval bracketed the settled cost on "
              f"{value['cost_coverage']:.1%} of blocks")
    if totals["blocks"] < 500:
        print("\n  Too few blocks to read the coverage figure as a measurement:")
        print("  a 90% rate over a few hundred blocks carries several points")
        print("  of binomial noise on its own. The record is the point; it")
        print("  gets informative as it accumulates.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    parser.add_argument("--target", type=date.fromisoformat, default=None)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--observe", action="store_true",
        help="record actual load for recent days and stop",
    )
    parser.add_argument(
        "--scorecard", action="store_true", help="print the record and stop"
    )
    parser.add_argument(
        "--reissue", action="store_true",
        help="publish a second forecast for a day that already has one",
    )
    parser.add_argument("--offline", action="store_true", help="no network")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    engine = create_db_engine()
    try:
        require_current_schema(engine)
    except Exception as exc:
        print(f"database not ready: {exc}")
        return 2
    factory = session_factory(engine)

    if args.scorecard:
        with session_scope(factory) as session:
            print_scorecard(session, None)
        return 0

    if args.observe:
        with session_scope(factory) as session:
            observe(session, args.cache)
            print_scorecard(session, None)
        return 0

    with session_scope(factory) as session:
        if issued_days(session):
            print("recording what happened:")
            observe(session, args.cache)

    print("\nissuing:")
    inputs = gather(args.cache, offline=args.offline)
    target = args.target or (datetime.now(IST).date() + timedelta(days=1))
    print(f"  load archive ends {inputs.data_end}, target {target}")

    try:
        forecast = issue(
            inputs, target, method=args.method, alpha=args.alpha
        )
    except ValueError as exc:
        print(f"  cannot issue: {exc}")
        return 1

    with session_scope(factory) as session:
        try:
            run = record_forecast(session, forecast, allow_reissue=args.reissue)
        except HindcastRejected as exc:
            print(f"  refused: {exc}")
            print("  This is the store working. A forecast issued after its")
            print("  day began is not a forecast, and the gap in the record")
            print("  is the honest outcome of a job that ran too late.")
            return 1
        except ForecastAlreadyIssued as exc:
            print(f"  refused: {exc}")
            return 1
        print(
            f"  issued run {run.id}: {forecast.method}, "
            f"{forecast.point_mw.min():,.0f}..{forecast.point_mw.max():,.0f} MW, "
            f"mean band {float((forecast.upper_mw - forecast.lower_mw).mean()):,.0f} MW"
        )
        if forecast.note:
            print(f"  note: {forecast.note}")

    with session_scope(factory) as session:
        print("\nthe record so far:")
        print_scorecard(session, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
