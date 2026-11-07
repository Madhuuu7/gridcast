/* The declaration: the schedule this system has committed to, and its
 * outcome once the grid has drawn it.
 *
 * This is the only view whose contents nobody here chose. The forecast was
 * written before the day began and the store refuses to let it be edited,
 * so a bad day stays on the page.
 */

import { ScheduleStrip } from "../components/ScheduleStrip";
import type { Schedule } from "../lib/api";
import { issuedAt, longDate, mw, percent, rupees } from "../lib/format";

export function Tomorrow({ schedule }: { schedule: Schedule }) {
  if (!schedule.issued) {
    return (
      <section>
        <div className="state">
          <h3>No schedule has been issued yet</h3>
          <p>
            The daily job writes one forecast for the next day, a little after
            the 10:00 IST gate, and the store refuses it if it arrives after
            that day has begun. Run <code>python scripts/issue_forecast.py</code>{" "}
            to publish the first one.
          </p>
          <p className="note">Reported by the service: {schedule.reason}</p>
        </div>
      </section>
    );
  }

  const blocks = schedule.blocks;
  const points = blocks.map((b) => b.point);
  const widths = blocks
    .filter((b) => b.lower !== null && b.upper !== null)
    .map((b) => (b.upper as number) - (b.lower as number));
  const peak = blocks.reduce((a, b) => (b.point > a.point ? b : a), blocks[0]);
  const score = schedule.score;
  const settled = score && score.scored > 0;

  return (
    <section>
      <div className="declaration">
        <div>
          <h2 className="target">
            {longDate(schedule.target_date)}
            {schedule.holiday && (
              <span className="kind">{schedule.holiday}</span>
            )}
            {!schedule.holiday && schedule.day_kind !== "weekday" && (
              <span className="kind">{schedule.day_kind}</span>
            )}
          </h2>
        </div>
        <dl>
          <dt>issued</dt>
          <dd>{issuedAt(schedule.issued_at)} IST</dd>
          <dt>gate</dt>
          <dd>{schedule.gate}</dd>
          <dt>interval</dt>
          <dd>
            {schedule.method} at {percent(1 - schedule.alpha, 0)}
          </dd>
          <dt>code</dt>
          <dd>{schedule.code_version}</dd>
        </dl>
      </div>

      <p className="lede">
        Ninety-six separate commitments, one per fifteen-minute settlement
        block. The height of each bar is what the forecast claims it does not
        know.
      </p>

      <div className="figures">
        <div>
          <span className="value">{mw(Math.min(...points))}</span>
          <span className="caption">lowest declared block, MW</span>
        </div>
        <div>
          <span className="value">{mw(Math.max(...points))}</span>
          <span className="caption">peak, MW</span>
          <span className="sub">
            block {peak.block}, {String(Math.floor(peak.block / 4)).padStart(2, "0")}
            :{String((peak.block % 4) * 15).padStart(2, "0")}
          </span>
        </div>
        <div>
          <span className="value">
            {mw(widths.reduce((a, b) => a + b, 0) / (widths.length || 1))}
          </span>
          <span className="caption">mean band, MW</span>
          <span className="sub">
            {mw(Math.min(...widths))} to {mw(Math.max(...widths))}
          </span>
        </div>
        <div>
          <span className="value">
            {settled ? percent(score!.coverage) : "pending"}
          </span>
          <span className="caption">
            {settled ? "blocks inside the band" : "awaiting the outcome"}
          </span>
          {settled && score!.coverage_se !== null && (
            <span className="sub">± {percent(score!.coverage_se)}</span>
          )}
        </div>
        {settled && (
          <div>
            <span className="value">{rupees(score!.rupees)}</span>
            <span className="caption">deviation charge, upper bound</span>
          </div>
        )}
      </div>

      <ScheduleStrip blocks={blocks} />

      {schedule.note && <p className="note">{schedule.note}</p>}

      <p className="note">
        Calibrated on {schedule.calibration.from} to {schedule.calibration.to},
        fitted on {schedule.train_rows.toLocaleString("en-IN")} blocks.
        {schedule.kernel &&
          ` Kernel bandwidths: ${schedule.kernel.block_bandwidth} blocks, ${schedule.kernel.temp_bandwidth}°C, day-kind weight ${schedule.kernel.kind_penalty}.`}
      </p>
    </section>
  );
}
