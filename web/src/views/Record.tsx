/* The ledger: every day this system declared, and how it went.
 *
 * The one table on the site that was not chosen after the fact. Pending
 * days stay listed rather than being hidden until they settle, because a
 * record that only shows resolved entries is a record with an editor.
 */

import type { RecordBody } from "../lib/api";
import { mw, percent, rupees, shortDate } from "../lib/format";

export function Record({ record }: { record: RecordBody }) {
  const { totals, days } = record;

  if (!days.length) {
    return (
      <section>
        <div className="state">
          <h3>The ledger is empty</h3>
          <p>
            It fills one row a day, each written before the day it describes.
            There is no backfill: a forecast issued after its target day has
            begun is refused by the store, so a missed run leaves a gap rather
            than a late entry.
          </p>
        </div>
      </section>
    );
  }

  const thin = totals.blocks < 500;

  return (
    <section>
      <h2>The record</h2>
      <p className="lede">
        Every forecast below was written before the day it describes and has
        not been touched since. The backtest elsewhere on this site was scored
        by the person who chose its folds; this was not.
      </p>

      <div className="figures">
        <div>
          <span className="value">{totals.days}</span>
          <span className="caption">days declared</span>
        </div>
        <div>
          <span className="value">{totals.blocks.toLocaleString("en-IN")}</span>
          <span className="caption">blocks settled</span>
        </div>
        <div>
          <span className="value">{percent(totals.coverage)}</span>
          <span className="caption">inside the band, pooled</span>
          {totals.coverage_se !== null && (
            <span className="sub">± {percent(totals.coverage_se)} at 95%</span>
          )}
        </div>
        <div>
          <span className="value">{mw(totals.mae_mw, 1)}</span>
          <span className="caption">mean absolute error, MW</span>
        </div>
        {record.interval_value && record.interval_value.blocks_scored > 0 && (
          <div>
            <span className="value">
              {percent(record.interval_value.cost_coverage)}
            </span>
            <span className="caption">blocks whose bill the band bracketed</span>
          </div>
        )}
      </div>

      <div className="scroll" style={{ marginTop: 26 }}>
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
            {[...days].reverse().map((day) => (
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
                    <span className="pending">pending</span>
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

      {thin && (
        <p className="note">
          Too few blocks to read the coverage figure as a measurement. A 90%
          rate over a few hundred blocks carries several points of binomial
          noise on its own, which is why the interval above it is printed. The
          number becomes informative as the ledger grows, and that is the
          point of keeping one.
        </p>
      )}

      {record.cost && (
        <p className="note">
          Deviation charges price each block's error under the CERC mechanism:
          the first 150 MW at the normal rate, over-drawal beyond it at double,
          under-drawal beyond it free. The rate is the ₹12/kWh cap, so these
          are an upper bound — a real utility revises its schedule intra-day
          and is not settled on a 38-hour-old forecast. Compare methods by the
          ratio, not the total.
        </p>
      )}
    </section>
  );
}
