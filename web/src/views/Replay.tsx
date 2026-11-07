/* Replay: any out-of-sample day, every interval method, against the outcome.
 *
 * The service will only serve dates after its calibration window, because a
 * forecast for a day the model trained on covers everything and proves
 * nothing. That rule is why the date list is short, and it is stated rather
 * than left to be discovered.
 */

import { useEffect, useMemo, useState } from "react";
import { ScheduleStrip, type StripBlock } from "../components/ScheduleStrip";
import { load, type DayEntry, type Forecast } from "../lib/api";
import { mw, percent } from "../lib/format";

/** The one day a published build carries a forecast for. Picking a
 *  different date on a static page has no answer, and the view says that
 *  rather than silently showing this one. */
const RECORDED_DAY = "2026-10-13";

export function Replay({ days }: { days: DayEntry[] }) {
  const [date, setDate] = useState(
    () =>
      days.find((d) => d.date === RECORDED_DAY)?.date ??
      days[Math.floor(days.length / 2)]?.date ??
      "",
  );
  const [method, setMethod] = useState("localized conformal");
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!date) return;
    let live = true;
    setBusy(true);
    load<Forecast>(
      `api/forecast?date=${date}`,
      date === RECORDED_DAY ? "forecast.json" : undefined,
    ).then((result) => {
      if (!live) return;
      setBusy(false);
      if (result.state === "ready") {
        setForecast(result.value);
        setProblem(null);
      } else {
        setForecast(null);
        setProblem(result.reason);
      }
    });
    return () => {
      live = false;
    };
  }, [date]);

  const methods = useMemo(
    () => (forecast ? Object.keys(forecast.intervals).sort() : []),
    [forecast],
  );

  useEffect(() => {
    if (methods.length && !methods.includes(method)) setMethod(methods[0]);
  }, [methods, method]);

  const blocks: StripBlock[] = useMemo(() => {
    if (!forecast) return [];
    const band = forecast.intervals[method];
    if (!band) return [];
    return forecast.point.map((point, i) => ({
      block: forecast.blocks?.[i] ?? i,
      point,
      lower: band.lower[i],
      upper: band.upper[i],
      actual: forecast.actual[i] ?? null,
    }));
  }, [forecast, method]);

  const band = forecast?.intervals[method];
  const selected = days.find((d) => d.date === date);

  return (
    <section>
      <h2>Replay a day</h2>
      <p className="lede">
        Any day after the calibration window, under every interval method the
        service fits. The model never saw these days, which is the only
        condition under which the coverage figure means anything.
      </p>

      <div className="controls">
        <label className="field">
          <span>Day</span>
          <select value={date} onChange={(e) => setDate(e.target.value)}>
            {days.map((d) => (
              <option key={d.date} value={d.date}>
                {d.date}
                {d.holiday ? ` — ${d.holiday}` : d.kind !== "weekday" ? ` — ${d.kind}` : ""}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Interval method</span>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            disabled={!methods.length}
          >
            {methods.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
      </div>

      {problem && (
        <div className="state error" style={{ marginTop: 18 }}>
          <h3>That day cannot be served</h3>
          <p>{problem}</p>
        </div>
      )}

      {forecast && band && (
        <>
          <div className="figures" style={{ marginTop: 22 }}>
            <div>
              <span className="value">{percent(band.coverage)}</span>
              <span className="caption">blocks inside the band</span>
              <span className="sub">one day is 96 blocks, ± ~3 points</span>
            </div>
            <div>
              <span className="value">{mw(band.mean_width)}</span>
              <span className="caption">mean band width, MW</span>
            </div>
            <div>
              <span className="value">{mw(forecast.metrics.mae, 1)}</span>
              <span className="caption">mean absolute error, MW</span>
              <span className="sub">
                {percent(forecast.metrics.mape / 100, 2)} of load
              </span>
            </div>
            <div>
              <span className="value">
                {mw(forecast.metrics.peak_error, 1)}
              </span>
              <span className="caption">peak miss, MW</span>
              <span className="sub">
                the block a scheduler is judged on
              </span>
            </div>
            <div>
              <span className="value">{forecast.day_kind}</span>
              <span className="caption">day kind</span>
              {(forecast.holiday ?? selected?.holiday) && (
                <span className="sub">
                  {forecast.holiday ?? selected?.holiday}
                </span>
              )}
            </div>
          </div>
          <ScheduleStrip blocks={blocks} animateIn={!busy} />
        </>
      )}
    </section>
  );
}
