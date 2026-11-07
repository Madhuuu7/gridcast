/* The ledger: every day this system declared, and how it went.
 *
 * Two tables, and the distinction between them is the point. The live record
 * is written one row a day before the day happens and cannot be backfilled.
 * The replayed record runs the same pipeline across the whole out-of-sample
 * window at once, which is not the same claim and is labelled as such — but
 * it is how you see a month of operation on the first day, and every number
 * in it is a real out-of-sample number.
 */

import { useEffect, useState } from "react";
import { load, type RecordBody } from "../lib/api";
import { mw, percent, rupees, shortDate } from "../lib/format";

type ReplayDay = {
  target_date: string;
  method: string;
  day_kind: string;
  holiday: string | null;
  scored: number;
  unbounded: number;
  coverage: number | null;
  mae_mw: number;
  mape: number;
  mean_width_mw: number;
  rupees: number;
};

type Replayed = {
  replayed: true;
  method: string;
  window: { from: string; to: string };
  totals: {
    days: number;
    blocks: number;
    coverage: number | null;
    coverage_se: number | null;
    mae_mw: number | null;
    rupees: number;
  };
  days: ReplayDay[];
};

export function Record({ record }: { record: RecordBody }) {
  const [replay, setReplay] = useState<Replayed | null>(null);

  useEffect(() => {
    load<Replayed>("api/record/replay", "record-replay.json").then((r) => {
      if (r.state === "ready") setReplay(r.value);
    });
  }, []);

  const live = record.days;

  return (
    <section>
      <h2>The record</h2>
      <p className="lede">
        Every forecast in the first table was written before the day it
        describes and has not been touched since. The store refuses one that
        arrives after its target day has begun, so a missed run leaves a gap
        rather than a late entry — which is also why this table starts at one
        row and grows by one a day.
      </p>

      <h3 style={{ marginTop: 26 }}>Issued in advance</h3>
      {live.length ? (
        <div className="scroll">
          <table>
            <thead>
              <tr>
                <th>Day</th>
                <th>Interval method</th>
                <th className="num">Blocks</th>
                <th className="num">Inside band</th>
                <th className="num">MAE, MW</th>
                <th className="num">Mean band, MW</th>
                <th className="num">Deviation charge</th>
              </tr>
            </thead>
            <tbody>
              {[...live].reverse().map((day) => (
                <tr key={`${day.target_date}-${day.method}`}>
                  <td>{shortDate(day.target_date)}</td>
                  <td>{day.method}</td>
                  <td className="num">
                    {day.scored}
                    {day.missing > 0 && (
                      <span className="pending"> of {day.scored + day.missing}</span>
                    )}
                  </td>
                  <td className="num">
                    {day.coverage === null ? (
                      <span className="pending">settles tomorrow</span>
                    ) : (
                      percent(day.coverage)
                    )}
                  </td>
                  <td className="num">{mw(day.mae_mw, 1)}</td>
                  <td className="num">{mw(day.mean_width_mw)}</td>
                  <td className="num">{rupees(day.rupees)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="state">
          Nothing issued yet. <code>scripts/issue_forecast.py</code> writes one
          row a day.
        </p>
      )}

      {replay && replay.days.length > 0 && (
        <>
          <h3 style={{ marginTop: 40 }}>
            Replayed across the out-of-sample window
          </h3>
          <p className="lede">
            The same pipeline over every day from {replay.window.from} to{" "}
            {replay.window.to}, scored the same way. These were computed now
            rather than issued in advance, so they carry less weight than the
            table above — but the model never saw these days, and the numbers
            are what a month of operation looks like.
          </p>

          <div className="figures">
            <div>
              <span className="value">{replay.totals.days}</span>
              <span className="caption">days</span>
            </div>
            <div>
              <span className="value">
                {replay.totals.blocks.toLocaleString("en-IN")}
              </span>
              <span className="caption">blocks scored</span>
            </div>
            <div>
              <span className="value">{percent(replay.totals.coverage)}</span>
              <span className="caption">inside the band, pooled</span>
              {replay.totals.coverage_se !== null && (
                <span className="sub">
                  ± {percent(replay.totals.coverage_se)} at 95%
                </span>
              )}
            </div>
            <div>
              <span className="value">{mw(replay.totals.mae_mw, 1)}</span>
              <span className="caption">mean absolute error, MW</span>
            </div>
            <div>
              <span className="value">{rupees(replay.totals.rupees)}</span>
              <span className="caption">deviation charge, upper bound</span>
            </div>
          </div>

          <div className="scroll" style={{ marginTop: 24 }}>
            <table>
              <thead>
                <tr>
                  <th>Day</th>
                  <th>Kind</th>
                  <th className="num">Blocks</th>
                  <th className="num">Inside band</th>
                  <th className="num">MAE, MW</th>
                  <th className="num">MAPE</th>
                  <th className="num">Mean band, MW</th>
                  <th className="num">Deviation charge</th>
                </tr>
              </thead>
              <tbody>
                {[...replay.days].reverse().map((day) => (
                  <tr key={day.target_date}>
                    <td>{shortDate(day.target_date)}</td>
                    <td>{day.holiday ?? day.day_kind}</td>
                    <td className="num">{day.scored}</td>
                    <td className="num">{percent(day.coverage)}</td>
                    <td className="num">{mw(day.mae_mw, 1)}</td>
                    <td className="num">{percent(day.mape / 100, 2)}</td>
                    <td className="num">{mw(day.mean_width_mw)}</td>
                    <td className="num">{rupees(day.rupees)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="note">
            Deviation charges price each block's error under the CERC
            mechanism: the first 150 MW at the normal rate, over-drawal beyond
            it at double, under-drawal beyond it free. The rate is the ₹12/kWh
            cap, so these are an upper bound — a real utility revises its
            schedule intra-day and is not settled on a 38-hour-old forecast.
            Compare methods by the ratio, not the total.
          </p>
        </>
      )}
    </section>
  );
}
