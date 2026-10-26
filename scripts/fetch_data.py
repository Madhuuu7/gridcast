"""Pull the load and weather history once, into the local cache.

    python scripts/fetch_data.py                 # the default three years
    python scripts/fetch_data.py --years 1
    python scripts/fetch_data.py --start 2024-01-01 --end 2024-12-31

Every day is cached as its own CSV, so this is resumable: interrupt it, run it
again, and it picks up the days it does not have. Re-running it when the cache
is warm touches the network only for the days at the end that are new.

It is deliberately unhurried -- four workers and a pause between requests. The
SLDC is a public service that is not charging for this, the whole pull is a
few hundred small pages, and the difference between finishing in twelve
minutes and finishing in four is not worth being a nuisance over.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridcast.data.sldc import SldcClient  # noqa: E402
from gridcast.data.weather import fetch_weather  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    # The archive runs to yesterday; today's page is partial by definition.
    end = args.end or (date.today() - timedelta(days=2))
    start = args.start or (end - timedelta(days=int(365 * args.years)))
    print(f"load:    {start} .. {end}  ({(end - start).days + 1} days)")

    client = SldcClient(cache_dir=args.cache / "sldc")

    started = time.perf_counter()
    state = {"done": 0, "fetched": 0, "thin": 0, "missing": 0}

    def report(result) -> None:
        state["done"] += 1
        if not result.from_cache:
            state["fetched"] += 1
        if result.frame.empty:
            state["missing"] += 1
        elif not result.complete:
            state["thin"] += 1
        if state["done"] % 25 == 0 or state["done"] == 1:
            rate = state["done"] / max(time.perf_counter() - started, 1e-9)
            print(
                f"  {state['done']:5d} days | {state['fetched']:5d} fetched | "
                f"{state['thin']:3d} partial | {state['missing']:3d} missing | "
                f"{rate:.1f}/s",
                flush=True,
            )

    load = client.fetch_range(
        start, end, workers=args.workers, refresh=args.refresh, on_day=report
    )
    print(
        f"load:    {len(load):,} five-minute rows, "
        f"{load.index.min()} .. {load.index.max()}"
    )
    print(
        f"         {state['missing']} days unavailable, "
        f"{state['thin']} partial, in {time.perf_counter() - started:.0f}s"
    )

    print(f"weather: {start} .. {end}")
    weather = fetch_weather(
        start, end, cache_dir=args.cache / "weather", refresh=args.refresh
    )
    print(f"weather: {len(weather):,} hourly rows, {weather.columns.tolist()}")

    processed = args.cache.parent / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    load.to_csv(processed / "load_5min.csv")
    weather.to_csv(processed / "weather_hourly.csv")
    print(f"wrote    {processed / 'load_5min.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
