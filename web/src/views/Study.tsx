/* The backtest: the measured study behind every claim on the site.
 *
 * Two tables, deliberately in this order. Marginal coverage first, where
 * every method looks fine, then conditional coverage, where they come apart.
 * Reading them the other way round makes the second look like a caveat
 * rather than the result.
 */

import type { Backtest } from "../lib/api";
import { mw, percent } from "../lib/format";

type Gap = {
  worst_group_coverage: number;
  spread: number;
  null_spread_p95: number;
  spread_p_adjusted: number;
  uneven_after_correction: boolean;
};

const AXES: [string, string][] = [
  ["block", "by block of day"],
  ["month", "by month"],
  ["day_kind", "by kind of day"],
];

export function Study({ backtest }: { backtest: Backtest }) {
  const intervals = backtest.intervals as
    | Record<string, { coverage: number; mean_width: number; winkler: number }>
    | undefined;
  const conditional = backtest.conditional as
    | Record<string, Record<string, Gap>>
    | undefined;
  const point = backtest.point as Record<string, any> | undefined;
  const bandwidth = backtest.bandwidth as Record<string, any> | undefined;

  const ranked = intervals
    ? Object.entries(intervals).sort((a, b) => a[1].winkler - b[1].winkler)
    : [];

  return (
    <section>
      <h2>The study</h2>
      <p className="lede">
        Eight folds spread across three years rather than stacked at the end,
        because the question an interval has to answer is whether it survives
        a change of season. Everything below is out of sample.
      </p>

      {point && (
        <>
          <h3>Point forecast</h3>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="num">MAPE</th>
                  <th className="num">MAE, MW</th>
                  <th className="num">Skill vs naive</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(point).map(([name, row]: [string, any]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td className="num">{percent(row.mape / 100, 2)}</td>
                    <td className="num">{mw(row.mae, 1)}</td>
                    {/* mape arrives as a percentage and skill as a
                        fraction, which is the kind of mismatch that
                        silently renders 52% as 1%. */}
                    <td className="num">
                      {row.skill ? percent(row.skill, 0) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {ranked.length > 0 && (
        <>
          <h3 style={{ marginTop: 34 }}>Marginal coverage: everyone passes</h3>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Interval method</th>
                  <th className="num">Coverage</th>
                  <th className="num">Mean width, MW</th>
                  <th className="num">Winkler</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map(([name, row], i) => (
                  <tr key={name} className={i === 0 ? "best" : undefined}>
                    <td>{name}</td>
                    <td className="num">{percent(row.coverage)}</td>
                    <td className="num">{mw(row.mean_width)}</td>
                    <td className="num">{Math.round(row.winkler)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">
            Winkler charges the width and penalises every miss, so neither an
            absurdly wide interval nor an absurdly sharp one wins it by
            cheating. Lower is better, and it is in MW.
          </p>
        </>
      )}

      {conditional && (
        <>
          <h3 style={{ marginTop: 34 }}>
            Conditional coverage: where they come apart
          </h3>
          <p className="lede">
            An interval that covers 90% overall by covering 98% of quiet
            nights and 79% of the evening ramp is wrong exactly when being
            wrong is expensive. Spread is the gap between the best and worst
            group; the noise column is what that spread reaches by chance
            alone at these group sizes, so a method is only called uneven when
            it beats its own null.
          </p>
          <p className="lede">
            <strong>Most rows below say uneven, and that is the result.</strong>{" "}
            Marginal coverage is the figure forecasting papers report and
            every method passes it; this is the harder question almost nobody
            asks, and it is supposed to be hard to pass. A method reaching
            <em> within noise</em> on an axis means its coverage across that
            axis is statistically indistinguishable from perfect — on 32,000
            out-of-sample blocks, with a Benjamini–Hochberg correction across
            the whole family of tests.
          </p>
          {AXES.filter(([key]) => conditional[key]).map(([key, label]) => (
            <div key={key} style={{ marginBottom: 26 }}>
              <h3>{label}</h3>
              <div className="scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Method</th>
                      <th className="num">Worst group</th>
                      <th className="num">Spread</th>
                      <th className="num">Noise, p95</th>
                      <th className="num">p (BH)</th>
                      <th>Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(conditional[key])
                      .sort((a, b) => a[1].spread - b[1].spread)
                      .map(([name, gap]) => (
                        <tr key={name}>
                          <td>{name}</td>
                          <td className="num">
                            {percent(gap.worst_group_coverage)}
                          </td>
                          <td className="num">{gap.spread.toFixed(3)}</td>
                          <td className="num">
                            {gap.null_spread_p95.toFixed(3)}
                          </td>
                          <td className="num">
                            {gap.spread_p_adjusted.toFixed(4)}
                          </td>
                          <td
                            className={
                              gap.uneven_after_correction
                                ? "verdict-uneven"
                                : "verdict-noise"
                            }
                          >
                            {gap.uneven_after_correction
                              ? "uneven"
                              : "within noise"}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </>
      )}

      {bandwidth && (
        <>
          <h3 style={{ marginTop: 34 }}>How wide should the kernel be?</h3>
          <p className="lede">
            A flat kernel is split conformal exactly, and a sharp one runs out
            of data. The effective sample size is how many equally weighted
            residuals each interval actually rests on, against a calibration
            window of several thousand — conditional coverage keeps improving
            right up until that number collapses, and the spread alone does
            not say when.
          </p>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Kernel</th>
                  <th className="num">Coverage</th>
                  <th className="num">Width</th>
                  <th className="num">Winkler</th>
                  <th className="num">Block spread</th>
                  <th className="num">Day-kind spread</th>
                  <th className="num">Effective n</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(bandwidth).map(([key, row]: [string, any]) => (
                  <tr key={key}>
                    <td className="num" style={{ textAlign: "left" }}>
                      {key
                        .replace(/=1000000000(\.0)?/g, "=flat")
                        .replace(/,/g, "  ")}
                    </td>
                    <td className="num">{percent(row.coverage)}</td>
                    <td className="num">{mw(row.mean_width)}</td>
                    <td className="num">{Math.round(row.winkler)}</td>
                    <td className="num">{row.block_spread?.toFixed(3)}</td>
                    <td className="num">{row.day_kind_spread?.toFixed(3)}</td>
                    <td className="num">{Math.round(row.mean_ess ?? 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
