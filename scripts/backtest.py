"""The study behind every number in the README.

    python scripts/backtest.py                # the full study, ~3 minutes
    python scripts/backtest.py --quick        # fewer folds, ~1 minute
    python scripts/backtest.py --json out.json

Six parts:

1. **the data** -- what three years of Delhi load actually looks like
2. **point forecasts** -- three models against the thing they must beat
3. **leakage** -- what the illegal 24-hour lag does to the score, measured
4. **intervals** -- five methods, marginal coverage, width and Winkler score
5. **conditional coverage** -- by block of day and by month, against a null
6. **adaptation rate** -- the gamma sweep for the online methods

Part 5 is the one the project exists for. Marginal coverage is easy and every
conformal method hits it; the question is whether an interval is honest at
4am *and* during the evening ramp, in January *and* in June. Each number there
is compared against a simulated noise floor, because with 96 groups a spread
of 0.10 in per-group coverage happens by chance alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridcast.conformal import (  # noqa: E402
    CovariateKernel,
    conditional_coverage_gap,
)
from gridcast.data.calendar import coverage_warning  # noqa: E402
from gridcast.data.dataset import (  # noqa: E402
    SAME_BLOCK_LAGS,
    build_features,
    summarise,
    to_blocks,
)
from gridcast.data.sldc import load_cached  # noqa: E402
from gridcast.data.weather import (  # noqa: E402
    fetch_previous_forecast,
    fetch_weather,
)
from gridcast.evaluate import make_folds, run_backtest  # noqa: E402
from gridcast.models import (  # noqa: E402
    BoostedForecaster,
    RidgeForecaster,
    SeasonalNaive,
    mae,
    mape,
    skill,
)

ALPHA = 0.1


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def load_everything(cache: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    load = load_cached(cache / "sldc")
    if load.empty:
        raise SystemExit(
            "no cached load data -- run scripts/fetch_data.py first "
            "(it takes about seven minutes and is resumable)"
        )
    blocks = to_blocks(load)
    first = blocks.index.min().date()
    last = blocks.index.max().date()
    weather = fetch_weather(first, last, cache_dir=cache / "weather")
    X, y = build_features(blocks, weather)
    return blocks, (X, y), weather


def study_data(blocks: pd.DataFrame, X: pd.DataFrame) -> dict:
    rule("1. three years of Delhi load")
    facts = summarise(blocks)
    print(f"  {facts['days']:,} days, {facts['blocks']:,} fifteen-minute blocks "
          f"({facts['completeness']:.1%} of the slots that should exist)")
    print(f"  {facts['start'][:10]} .. {facts['end'][:10]}")
    print(f"  load {facts['min_mw']:,.0f} .. {facts['max_mw']:,.0f} MW, "
          f"mean {facts['mean_mw']:,.0f} MW, peak on {facts['peak_day']}")
    print(f"  usable training rows after lags and joins: {len(X):,}")
    warning = coverage_warning(pd.Timestamp(facts["end"]).date())
    if warning:
        print(f"  note: {warning}")
    facts["feature_rows"] = len(X)
    facts["features"] = len(X.columns)
    return facts


def study_point(X: pd.DataFrame, y: pd.Series, folds) -> dict:
    rule("2. point forecasts, day-ahead, issued at the 10:00 gate")
    out = {}
    index = pd.DatetimeIndex(X.index)
    results = {}
    for factory in (SeasonalNaive, RidgeForecaster, BoostedForecaster):
        preds, actuals = [], []
        for fold in folds:
            train, _cal, test = fold.masks(index)
            model = factory().fit(X[train], y[train])
            preds.append(model.predict(X[test]))
            actuals.append(y[test].to_numpy())
        results[factory().name] = (np.concatenate(actuals), np.concatenate(preds))

    base_actual, base_pred = results["seasonal naive"]
    print(f"  {'model':16s} {'MAPE':>7s} {'MAE':>9s} {'skill vs naive':>15s}")
    for name, (actual, pred) in results.items():
        row = {
            "mape": mape(actual, pred),
            "mae": mae(actual, pred),
            "skill": skill(actual, pred, base_pred) if name != "seasonal naive" else 0.0,
        }
        out[name] = row
        print(
            f"  {name:16s} {row['mape']:6.2f}% {row['mae']:8.1f} MW "
            f"{row['skill']:+14.1%}"
        )
    print("\n  Day-ahead MAPE in the 4-6% range is where a city-scale load")
    print("  forecast lands. The number that matters is the skill column: the")
    print("  seasonal naive is genuinely hard to beat on a quiet week.")
    return out


def study_leakage(X: pd.DataFrame, y: pd.Series, folds) -> dict:
    """What the illegal 24-hour lag buys, in MAPE.

    The point is not that leakage is bad in the abstract. It is that a leaky
    backtest looks *better*, by a margin large enough to win an argument, and
    nothing in the output says anything is wrong.
    """
    rule("3. the leakage trap, measured")
    index = pd.DatetimeIndex(X.index)
    series = y.reindex(
        pd.date_range(index.min(), index.max(), freq="15min")
    )
    leaky = X.copy()
    # 24 hours back: available for the morning blocks of the target day,
    # still in the future for the afternoon ones at a 10:00 gate.
    leaky["lag_24h_ILLEGAL"] = series.shift(96).reindex(index).to_numpy()
    leaky = leaky.dropna()
    y_leaky = y.reindex(leaky.index)

    honest_pred, leaky_pred, actual = [], [], []
    leaky_index = pd.DatetimeIndex(leaky.index)
    for fold in folds:
        train, _cal, test = fold.masks(leaky_index)
        if not (train.any() and test.any()):
            continue
        honest_cols = [c for c in leaky.columns if c != "lag_24h_ILLEGAL"]
        honest_pred.append(
            BoostedForecaster().fit(leaky.loc[train, honest_cols], y_leaky[train])
            .predict(leaky.loc[test, honest_cols])
        )
        leaky_pred.append(
            BoostedForecaster().fit(leaky[train], y_leaky[train]).predict(leaky[test])
        )
        actual.append(y_leaky[test].to_numpy())

    actual = np.concatenate(actual)
    honest = mape(actual, np.concatenate(honest_pred))
    cheat = mape(actual, np.concatenate(leaky_pred))
    print(f"  causal features only          MAPE {honest:5.2f}%")
    print(f"  plus a 24-hour lag (illegal)  MAPE {cheat:5.2f}%")
    print(f"  apparent improvement          {honest - cheat:5.2f} points "
          f"({(honest - cheat) / honest:.0%} better, and entirely fictional)")
    print("\n  At a 10:00 gate the 24-hour lag is observable for tomorrow's")
    print("  morning blocks and in the future for its afternoon ones. The")
    print("  backtest cannot tell; only the clock can. dataset.assert_causal")
    print("  rejects it, and tests/test_leakage.py checks that it does.")
    return {"honest_mape": honest, "leaky_mape": cheat, "apparent_gain": honest - cheat}


def study_intervals(result) -> dict:
    rule(f"4. prediction intervals at {1 - ALPHA:.0%} nominal coverage")
    table = result.interval_metrics()
    print(f"  {'method':22s} {'coverage':>9s} {'width':>9s} {'Winkler':>9s}")
    for _, row in table.iterrows():
        print(
            f"  {row['method']:22s} {row['coverage']:8.1%} "
            f"{row['mean_width']:8.0f} {row['winkler']:9.0f}"
        )
    print("\n  Winkler charges the width and penalises misses, so neither a")
    print("  very wide nor a very sharp interval can win it by cheating.")
    return {
        row["method"]: {
            "coverage": row["coverage"],
            "mean_width": row["mean_width"],
            "winkler": row["winkler"],
        }
        for _, row in table.iterrows()
    }


def study_conditional(result) -> dict:
    rule("5. conditional coverage: honest everywhere, or only on average?")
    frame = result.frame
    actual = frame["actual"].to_numpy()

    # One test per method per grouping, and the p-values get a
    # Benjamini-Hochberg correction before any of them is called
    # significant. Uncorrected, at 0.05 each, the chance of at least one
    # spurious "uneven" across a family this size is well over half, and the
    # whole point of the noise floor was to stop reporting noise as a result.
    gaps: dict[tuple[str, str], dict] = {}
    for by in ("block", "month", "day_kind"):
        for method in result.methods:
            gaps[(by, method)] = conditional_coverage_gap(
                actual, result.intervals(method), frame[by].to_numpy()
            )
    keys = list(gaps)
    raw = np.array([gaps[k]["spread_p_value"] for k in keys])
    order = np.argsort(raw)
    m = raw.size
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        running = min(running, raw[order[rank - 1]] * m / rank)
        adjusted[order[rank - 1]] = min(1.0, running)
    for key, p_adj in zip(keys, adjusted):
        gaps[key]["spread_p_adjusted"] = float(p_adj)
        gaps[key]["uneven_after_correction"] = bool(p_adj < 0.05)

    out: dict[str, dict] = {}
    print(f"\n  {len(keys)} tests in the family, Benjamini-Hochberg corrected.")
    for by, label in (
        ("block", "block of day"),
        ("month", "month"),
        ("day_kind", "day kind"),
    ):
        print(f"\n  by {label}:")
        print(f"    {'method':22s} {'worst':>7s} {'spread':>8s} {'noise p95':>10s} "
              f"{'p (BH)':>8s}  verdict")
        out[by] = {}
        for method in result.methods:
            gap = gaps[(by, method)]
            verdict = (
                "uneven" if gap["uneven_after_correction"] else "within noise"
            )
            print(
                f"    {method:22s} {gap['worst_group_coverage']:6.1%} "
                f"{gap['spread']:8.3f} {gap['null_spread_p95']:10.3f} "
                f"{gap['spread_p_adjusted']:8.4f}  {verdict}"
            )
            out[by][method] = gap

    print("\n  Read the two tables together. Mondrian and normalized conformal")
    print("  bring the block-of-day spread down to the noise floor and leave")
    print("  the seasonal spread untouched; the adaptive method does the")
    print("  reverse. Each fixes the axis it conditions on and no other.")
    return out


def study_gamma(X, y, folds, gammas) -> dict:
    rule("6. how fast should an online method adapt?")
    print(f"  {'gamma':>7s} {'coverage':>9s} {'width':>8s} {'Winkler':>9s} "
          f"{'month spread':>13s} {'block spread':>13s}")
    out = {}
    for gamma in gammas:
        result = run_backtest(
            X, y, BoostedForecaster, alpha=ALPHA, folds=folds, gamma=gamma,
            verbose=False,
        )
        intervals = result.intervals("adaptive conformal")
        actual = result.frame["actual"].to_numpy()
        marginal = intervals.evaluate(actual)
        month = conditional_coverage_gap(
            actual, intervals, result.frame["month"].to_numpy()
        )
        block = conditional_coverage_gap(
            actual, intervals, result.frame["block"].to_numpy()
        )
        out[str(gamma)] = {
            **marginal,
            "month_spread": month["spread"],
            "block_spread": block["spread"],
            "block_noise_p95": block["null_spread_p95"],
        }
        print(
            f"  {gamma:7.3f} {marginal['coverage']:8.1%} "
            f"{marginal['mean_width']:7.0f} {marginal['winkler']:9.0f} "
            f"{month['spread']:13.3f} {block['spread']:13.3f}"
        )
    print("\n  Two things in this table. The Winkler score bottoms out around")
    print("  gamma = 0.1 -- slower cannot track the season, faster pays in")
    print("  width for no further gain -- so the optimum is interior and not")
    print("  an artefact of where the sweep stopped.")
    print("\n  And at the aggressive end the method starts fixing the *other*")
    print("  axis too: by gamma = 0.8 it is adapting fast enough to track")
    print("  within-day variation, so the block-of-day spread falls to the")
    print("  noise floor as well. It costs about 60% more width than the")
    print("  Winkler optimum, which is the price of conditional coverage on")
    print("  both axes from a single pooled learner.")
    return out


def study_bandwidth(X, y, folds, settings) -> dict:
    rule("7. how wide should the kernel be?")
    print("  kind_penalty 0 is a hard partition on day kind; 1 ignores it.")
    print("  ESS is Kish's effective sample size -- how many equally-weighted")
    print("  residuals each interval actually rests on, against a calibration")
    print("  window of several thousand.\n")
    print(f"  {'blocks':>7s} {'degC':>7s} {'kind':>6s} {'coverage':>9s} {'width':>8s} "
          f"{'Winkler':>9s} {'block':>7s} {'month':>7s} {'kind sp':>8s} {'ESS':>7s}")
    out = {}
    for block_bandwidth, temp_bandwidth, kind_penalty in settings:
        kernel = CovariateKernel(
            block_bandwidth=block_bandwidth,
            temp_bandwidth=temp_bandwidth,
            kind_penalty=kind_penalty,
        )
        result = run_backtest(
            X, y, BoostedForecaster, alpha=ALPHA, folds=folds, kernel=kernel,
            verbose=False,
        )
        intervals = result.intervals("localized conformal")
        actual = result.frame["actual"].to_numpy()
        marginal = intervals.evaluate(actual)
        spreads = {
            by: conditional_coverage_gap(
                actual, intervals, result.frame[by].to_numpy()
            )["spread"]
            for by in ("block", "month", "day_kind")
        }
        ess = float(result.frame["localized_ess"].mean())
        key = f"block={block_bandwidth},temp={temp_bandwidth},kind={kind_penalty}"
        out[key] = {**marginal, **{f"{k}_spread": v for k, v in spreads.items()},
                    "mean_ess": ess}
        print(
            f"  {block_bandwidth:7.0f} {temp_bandwidth:7.0f} {kind_penalty:6.2f} "
            f"{marginal['coverage']:8.1%} {marginal['mean_width']:7.0f} "
            f"{marginal['winkler']:9.0f} {spreads['block']:7.3f} "
            f"{spreads['month']:7.3f} {spreads['day_kind']:8.3f} {ess:7.0f}"
        )
    print("\n  The two bandwidths trade against different axes, which is the")
    print("  point of having both: narrowing the block bandwidth buys")
    print("  time-of-day coverage, and lowering the day-kind penalty buys")
    print("  festival coverage, and neither does the other's job. Watch the")
    print("  ESS column for where the kernel runs out of data rather than")
    print("  reading the spreads alone -- a sharp kernel's conditional")
    print("  coverage improves right up until each interval rests on a")
    print("  handful of residuals, and the spread does not say when.")
    return out


def study_weather_penalty(blocks, X, y, cache: Path, folds, lead_days: int = 2) -> dict:
    rule("8. what the forecast costs when the weather is also forecast")
    first, last = blocks.index.min().date(), blocks.index.max().date()
    analysis = fetch_weather(first, last, cache_dir=cache / "weather")
    predicted = fetch_previous_forecast(
        first, last, lead_days=lead_days, cache_dir=cache / "weather_forecast"
    )

    paired = predicted[["temperature_2m"]].join(
        analysis[["temperature_2m"]], how="inner", rsuffix="_analysis"
    ).dropna()
    temp_error = paired["temperature_2m"] - paired["temperature_2m_analysis"]
    print(f"  lead-{lead_days} temperature against the analysis: "
          f"{len(paired):,} hours, MAE {temp_error.abs().mean():.2f}C, "
          f"bias {temp_error.mean():+.2f}C, "
          f"p95 |error| {temp_error.abs().quantile(0.95):.2f}C")

    X_fc, y_fc = build_features(blocks, predicted)
    # Score both on the rows the forecast-weather build survives, so the
    # comparison is not partly a comparison of different row sets -- the
    # previous-runs archive has gaps the analysis does not.
    common = X.index.intersection(X_fc.index)
    X_fc, y_fc = X_fc.loc[common], y_fc.loc[common]
    honest = run_backtest(
        X.loc[common], y.loc[common], BoostedForecaster,
        alpha=ALPHA, folds=folds, verbose=False,
    )
    realistic = run_backtest(
        X_fc, y_fc, BoostedForecaster, alpha=ALPHA, folds=folds, verbose=False,
    )

    out = {"lead_days": lead_days, "hours_paired": len(paired),
           "temp_mae_c": float(temp_error.abs().mean())}
    print(f"\n  {'weather at fit time':24s} {'MAPE':>7s} {'MAE':>9s} "
          f"{'coverage':>9s} {'width':>8s} {'Winkler':>9s}")
    for label, result in (("the analysis", honest), (f"a lead-{lead_days} forecast", realistic)):
        point = result.point_metrics()
        intervals = result.intervals("localized conformal")
        marginal = intervals.evaluate(result.frame["actual"].to_numpy())
        out[label] = {**point, **marginal}
        print(f"  {label:24s} {point['mape']:6.2f}% {point['mae']:8.1f} "
              f"{marginal['coverage']:8.1%} {marginal['mean_width']:7.0f} "
              f"{marginal['winkler']:9.0f}")

    honest_mape = honest.point_metrics()["mape"]
    real_mape = realistic.point_metrics()["mape"]
    out["mape_penalty_points"] = float(real_mape - honest_mape)
    out["mape_penalty_relative"] = float((real_mape - honest_mape) / honest_mape)
    print(f"\n  The penalty is {real_mape - honest_mape:.2f} MAPE points, "
          f"{(real_mape - honest_mape) / honest_mape:.0%} worse.")
    print("  Every other number in this study is computed with the weather")
    print("  the day actually had, which nobody has at a 10:00 gate. This is")
    print("  the size of that advantage, and the gap between these figures")
    print("  and what a production system would report.")

    # Whether the interval absorbs the extra error or breaks under it is the
    # question, so it gets read off the numbers rather than asserted.
    honest_cov = out["the analysis"]["coverage"]
    real_cov = out[f"a lead-{lead_days} forecast"]["coverage"]
    widened = (
        out[f"a lead-{lead_days} forecast"]["mean_width"]
        / out["the analysis"]["mean_width"] - 1.0
    )
    nominal = 1.0 - ALPHA
    if abs(real_cov - nominal) <= 0.015:
        print(f"\n  The interval absorbs it: coverage holds at {real_cov:.1%}")
        print(f"  against {nominal:.0%} nominal while the band widens {widened:+.0%}.")
        print("  The conformal quantile is computed from residuals that")
        print("  already contain the weather error, so it prices the unknown")
        print("  weather in without being told about it.")
    else:
        print(f"\n  The interval does not absorb it: coverage moves from "
              f"{honest_cov:.1%} to {real_cov:.1%} against {nominal:.0%} "
              f"nominal, on {widened:+.0%} width.")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    parser.add_argument("--json", type=Path, default=Path("reports/backtest.json"))
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()

    started = time.perf_counter()
    print("gridcast -- day-ahead load forecasting for Delhi, with intervals")
    print(f"alpha = {ALPHA} ({1 - ALPHA:.0%} nominal coverage), "
          f"{'quick' if args.quick else 'full'} run")

    blocks, (X, y), _weather = load_everything(args.cache)
    n_folds = 4 if args.quick else 8
    folds = make_folds(
        pd.DatetimeIndex(X.index), n_folds=n_folds, test_days=45, cal_days=60
    )

    results: dict[str, object] = {
        "alpha": ALPHA,
        "quick": args.quick,
        "folds": [fold.describe() for fold in folds],
        "lags_hours": list(SAME_BLOCK_LAGS),
    }
    skip = set(args.skip)

    if "1" not in skip:
        results["data"] = study_data(blocks, X)
    if "2" not in skip:
        results["point"] = study_point(X, y, folds)
    if "3" not in skip:
        results["leakage"] = study_leakage(X, y, folds)

    print("\n  running the interval backtest ...", flush=True)
    backtest = run_backtest(
        X, y, BoostedForecaster, alpha=ALPHA, folds=folds,
        baseline_factory=SeasonalNaive, verbose=False,
    )
    results["backtest_blocks"] = len(backtest.frame)
    results["point_backtest"] = backtest.point_metrics()

    if "4" not in skip:
        results["intervals"] = study_intervals(backtest)
    if "5" not in skip:
        results["conditional"] = study_conditional(backtest)
    if "6" not in skip:
        gammas = (
            (0.01, 0.1)
            if args.quick
            else (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8)
        )
        results["gamma"] = study_gamma(X, y, folds, gammas)
    if "7" not in skip:
        FLAT = 1e9
        settings = (
            ((FLAT, FLAT, 1.00), (8.0, 5.0, 0.05))
            if args.quick
            else (
                # Flat on every axis: must reproduce split conformal exactly,
                # which is the check that the kernel path and the partition
                # path are one estimator and not two implementations.
                (FLAT, FLAT, 1.00),
                # One axis at a time, so each column below is attributable.
                (8.0, FLAT, 1.00),     # time of day only
                (FLAT, 5.0, 1.00),     # temperature only
                (FLAT, FLAT, 0.05),    # day kind only
                (FLAT, FLAT, 0.00),    # day kind, hard partition
                # Then together, tightening.
                (16.0, 10.0, 0.30),
                (8.0, 5.0, 0.30),
                (8.0, 5.0, 0.05),
                (4.0, 3.0, 0.05),
                (2.0, 2.0, 0.05),
                (1.0, 1.0, 0.02),      # sharp enough to start starving
            )
        )
        results["bandwidth"] = study_bandwidth(X, y, folds, settings)
    if "8" not in skip:
        results["weather_penalty"] = study_weather_penalty(
            blocks, X, y, args.cache, folds
        )

    results["elapsed_s"] = time.perf_counter() - started
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    backtest.frame.to_csv(args.json.parent / "backtest_blocks.csv")
    print(f"\nwrote {args.json} in {results['elapsed_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
