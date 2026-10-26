/**
 * A headless check on the console: no browser, no test framework.
 *
 *   node scripts/check_console.mjs
 *   node scripts/check_console.mjs --forecast reports/day.json --report reports/backtest.json
 *
 * Two kinds of check.
 *
 * **Static.** Every id the script reaches for must exist in the markup. A
 * typo there is silent in a browser -- the element is null, the handler
 * throws inside an async function, and the page simply stops updating with
 * nothing in the terminal. The Python tests cannot see it and neither can
 * anyone who does not open the page. It also asserts the page loads nothing
 * from the network: no CDN, no font host, no analytics.
 *
 * **Behavioural.** The page's pure helpers run in a Node vm with no DOM at
 * all, against a canned forecast or -- with `--forecast` -- a real one from
 * the API. These are the functions that turn numbers into sentences and
 * geometry, which is where a chart goes quietly wrong: a band drawn through
 * a null, a miss counted on the wrong side, an SVG path containing NaN.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const html = readFileSync(resolve(root, "web/index.html"), "utf8");

let passed = 0;
const failures = [];

function check(name, fn) {
  try { fn(); passed += 1; }
  catch (err) { failures.push({ name, message: err.message }); }
}
function assert(condition, message) {
  if (!condition) throw new Error(message || "assertion failed");
}
function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(
      (message ? message + ": " : "") +
        `expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`
    );
  }
}
function assertIncludes(haystack, needle, message) {
  assert(
    String(haystack).includes(needle),
    (message ? message + ": " : "") +
      `expected ${JSON.stringify(needle)} in ${JSON.stringify(String(haystack).slice(0, 200))}`
  );
}

// -- evaluate the helpers without a DOM -----------------------------------

const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
assert(scriptMatch, "no inline <script> in web/index.html");
const source = scriptMatch[1];
const sandbox = { module: { exports: {} }, console };
vm.createContext(sandbox);
new vm.Script(source, { filename: "index.html#script" }).runInContext(sandbox);
const G = sandbox.module.exports;
assert(G && typeof G.fmtMw === "function", "the helpers did not export");

// -- static ----------------------------------------------------------------

const declaredIds = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
const referencedIds = new Set([
  ...[...source.matchAll(/\$\("([^"]+)"\)/g)].map((m) => m[1]),
  ...[...source.matchAll(/getElementById\("([^"]+)"\)/g)].map((m) => m[1]),
]);

check("every id the script reaches for exists in the markup", () => {
  const missing = [...referencedIds].filter((id) => !declaredIds.has(id));
  assertEqual(missing.join(", "), "", "ids referenced but never declared");
  assert(referencedIds.size > 10, "suspiciously few ids referenced");
});

check("the page loads nothing from the network", () => {
  const external = [...html.matchAll(/(?:src|href)="([^"]+)"/g)]
    .map((m) => m[1])
    .filter((url) => /^(https?:)?\/\//.test(url));
  assertEqual(external.join(", "), "", "external resources found");
  assert(!/<link\b/.test(html), "no stylesheet links expected");
});

check("every API path the console calls is one the server serves", () => {
  const paths = [...source.matchAll(/api\("(\/api[^"?]*)/g)].map((m) => m[1]);
  const served = ["/api/health", "/api/days", "/api/summary", "/api/forecast", "/api/backtest"];
  const unknown = paths.filter((p) => !served.includes(p));
  assertEqual(unknown.join(", "), "", "console calls an endpoint the API does not serve");
  assert(paths.includes("/api/forecast"), "the console never forecasts");
});

check("the sections the renderers unhide start hidden", () => {
  for (const id of ["statsection", "chartsection", "daymethodsection", "measuredsection", "datasection"]) {
    assert(declaredIds.has(id), `missing section #${id}`);
    assert(new RegExp(`id="${id}"[^>]*hidden`).test(html), `#${id} should start hidden`);
  }
});

// -- fixtures ---------------------------------------------------------------

function cannedForecast(overrides = {}) {
  const n = 96;
  const times = [], actual = [], point = [], lower = [], upper = [], blocks = [];
  for (let i = 0; i < n; i++) {
    const hh = String(Math.floor(i / 4)).padStart(2, "0");
    const mm = String((i % 4) * 15).padStart(2, "0");
    times.push(`${hh}:${mm}`);
    blocks.push(i);
    const base = 4500 + 1200 * Math.sin((2 * Math.PI * (i - 20)) / n);
    actual.push(base);
    point.push(base - 60);
    lower.push(base - 400);
    upper.push(base + 300);
  }
  // two deliberate misses, one each side
  actual[10] = lower[10] - 150;
  actual[70] = upper[70] + 220;
  return Object.assign(
    {
      date: "2026-10-05",
      day_kind: "weekday",
      holiday: null,
      issued: "2026-10-04 10:00",
      blocks, times, actual, point,
      baseline: point.map((v) => v - 200),
      alpha: 0.1,
      nominal: 0.9,
      metrics: { mape: 2.18, mae: 114.8, baseline_mape: 2.84,
                 peak_actual: 6251, peak_point: 6232, peak_error: -19 },
      intervals: {
        "adaptive conformal": { lower, upper, coverage: 0.926, mean_width: 700 },
        "split conformal": {
          lower: lower.map((v) => v - 400), upper: upper.map((v) => v + 400),
          coverage: 1.0, mean_width: 1500,
        },
      },
    },
    overrides
  );
}

function cannedReport() {
  return {
    intervals: {
      "adaptive conformal": { coverage: 0.901, mean_width: 924, winkler: 1067 },
      "split conformal": { coverage: 0.906, mean_width: 1327, winkler: 1710 },
      gaussian: { coverage: 0.886, mean_width: 1088, winkler: 1597 },
    },
    conditional: {
      block: {
        "mondrian conformal": { worst_group_coverage: 0.843, spread: 0.106,
          null_spread_p95: 0.101, spread_p_adjusted: 0.023, uneven_after_correction: true },
        "split conformal": { worst_group_coverage: 0.791, spread: 0.181,
          null_spread_p95: 0.101, spread_p_adjusted: 0.0003, uneven_after_correction: true },
      },
      month: {
        "adaptive conformal": { worst_group_coverage: 0.863, spread: 0.048,
          null_spread_p95: 0.058, spread_p_adjusted: 0.113, uneven_after_correction: false },
        "split conformal": { worst_group_coverage: 0.770, spread: 0.230,
          null_spread_p95: 0.058, spread_p_adjusted: 0.0003, uneven_after_correction: true },
      },
    },
    data: { start: "2023-10-25", end: "2026-10-24", days: 1083, blocks: 101660,
            completeness: 0.978, min_mw: 1458, max_mw: 8680, mean_mw: 4401,
            peak_day: "2026-06-29", features: 35 },
  };
}

const argFor = (flag) => {
  const i = process.argv.indexOf(flag);
  return i > -1 ? JSON.parse(readFileSync(process.argv[i + 1], "utf8")) : null;
};
const liveForecast = argFor("--forecast");
const liveReport = argFor("--report");
const forecast = liveForecast || cannedForecast();
const report = liveReport || cannedReport();
const live = Boolean(liveForecast);

// -- formatting -------------------------------------------------------------

check("megawatts read as megawatts", () => {
  assertEqual(G.fmtMw(1457.8), "1,458");
  assertEqual(G.fmtMw(null), "--");
  assertEqual(G.fmtPct(0.9012), "90.1%");
  assertEqual(G.fmtPct(0.9012, 0), "90%");
  assertEqual(G.fmtP(0.0003), "<0.001");
});

// -- misses -----------------------------------------------------------------

check("a miss is counted on the side it actually happened", () => {
  const canned = cannedForecast();
  const missed = G.misses(canned, "adaptive conformal");
  assertEqual(missed.length, 2, "the fixture has exactly two misses");
  const below = missed.find((m) => m.side === "below");
  const above = missed.find((m) => m.side === "above");
  assert(below && above, "one miss each side");
  assert(below.by > 100 && above.by > 100, "the distance is positive on both sides");
  assertEqual(below.i, 10);
  assertEqual(above.i, 70);
});

check("a block inside the band is never a miss", () => {
  const canned = cannedForecast();
  assertEqual(G.misses(canned, "split conformal").length, 0);
});

check("an unbounded side cannot produce a miss", () => {
  // An adaptive method can report an infinite bound, which serialises as
  // null. A null compared with < or > is false in JavaScript, so the naive
  // version silently counts every such block as covered -- this pins that
  // nulls are handled on purpose rather than by accident.
  const canned = cannedForecast();
  canned.intervals["adaptive conformal"].lower = canned.intervals["adaptive conformal"].lower.map(() => null);
  const missed = G.misses(canned, "adaptive conformal");
  assert(missed.every((m) => m.side === "above"), "a null lower bound cannot be missed below");
});

check("an unknown method yields no misses rather than throwing", () => {
  assertEqual(G.misses(cannedForecast(), "no such method").length, 0);
});

// -- the stat strip ---------------------------------------------------------

check("the stat strip carries the error, the width and the coverage", () => {
  const method = Object.keys(forecast.intervals)[0];
  const tiles = G.statTiles(forecast, method);
  const joined = tiles.map((t) => t.label + "=" + t.value).join("|");
  assert(tiles.length >= 6, "too few tiles");
  assertIncludes(joined, "MAPE");
  assertIncludes(joined, "blocks covered");
  assertIncludes(joined, "blocks missed");
  assertIncludes(joined, "interval width");
});

// -- geometry ---------------------------------------------------------------

check("the chart domain contains the actuals and the band", () => {
  const method = Object.keys(forecast.intervals)[0];
  const domain = G.domain(forecast, method);
  assert(Number.isFinite(domain.lo) && Number.isFinite(domain.hi), "domain is not finite");
  assert(domain.lo < domain.hi, "domain is inverted");
  assert(domain.lo <= Math.min(...forecast.actual), "an actual falls below the chart");
  assert(domain.hi >= Math.max(...forecast.actual), "an actual falls above the chart");
});

check("a path skips nulls instead of drawing through zero", () => {
  const scales = G.scales({ lo: 0, hi: 100 }, 5, 500, 200, { l: 10, r: 10, t: 10, b: 10 });
  const d = G.linePath([10, null, 30, undefined, 50], scales);
  assertEqual((d.match(/[ML]/g) || []).length, 3, "nulls should not be drawn");
  assert(d.startsWith("M"));
  assert(!/NaN|Infinity/.test(d), "the path contains a non-number");
});

check("a band with no finite pair is empty, and a real one closes", () => {
  const scales = G.scales({ lo: 0, hi: 100 }, 3, 300, 100, { l: 0, r: 0, t: 0, b: 0 });
  assertEqual(G.bandPath([null, null], [null, null], scales), "");
  const band = G.bandPath([10, 20], [40, 50], scales);
  assert(band.endsWith("Z"), "a band must be closed");
  assert(!/NaN/.test(band));
});

check("the real forecast draws without a NaN anywhere", () => {
  const width = 1000, height = 330, pad = { l: 58, r: 12, t: 12, b: 32 };
  for (const method of Object.keys(forecast.intervals)) {
    const domain = G.domain(forecast, method);
    const scales = G.scales(domain, forecast.actual.length, width, height, pad);
    const band = forecast.intervals[method];
    for (const d of [
      G.linePath(forecast.actual, scales),
      G.linePath(forecast.point, scales),
      G.bandPath(band.lower, band.upper, scales),
    ]) {
      assert(!/NaN|Infinity|undefined/.test(d), `${method}: bad path`);
    }
  }
});

// -- prose ------------------------------------------------------------------

check("the chart note says what the band did", () => {
  const canned = cannedForecast();
  const note = G.chartNote(canned, "adaptive conformal");
  assertIncludes(note, "missed 2 blocks");
  assertIncludes(note, "worst at");
  assertIncludes(note, "noisy estimate");
  const clean = G.chartNote(canned, "split conformal");
  assertIncludes(clean, "covered every block");
});

check("the conditional note quotes the noise floor, not just the spread", () => {
  const note = G.conditionalNote(report, "block");
  assertIncludes(note, "by chance");
  assertIncludes(note, "95th percentile");
  assert(/0\.\d{3}/.test(note), "the note should carry the actual floor");
});

check("a method within the noise is named as such", () => {
  const note = G.conditionalNote(report, "month");
  assertIncludes(note, "within that noise");
  assertIncludes(note, "adaptive conformal");
});

// -- tables -----------------------------------------------------------------

check("the marginal table is ordered by Winkler, best first", () => {
  const rows = G.marginalRows(report);
  assert(rows.length >= 2);
  for (let i = 1; i < rows.length; i++) {
    assert(rows[i].winkler >= rows[i - 1].winkler, "not sorted by Winkler");
  }
});

check("the conditional table is ordered by spread and keeps its verdict", () => {
  const rows = G.conditionalRows(report, "block");
  assert(rows.length >= 2);
  for (let i = 1; i < rows.length; i++) {
    assert(rows[i].spread >= rows[i - 1].spread, "not sorted by spread");
  }
  assert(rows.every((r) => typeof r.uneven === "boolean"), "every row needs a verdict");
});

check("the conditional table prefers the corrected p-value", () => {
  // Eighteen tests are run at once; the console must show the corrected
  // figure, not the raw one, or it reports a significance the study does not
  // claim. Checked against the report's own numbers rather than a literal,
  // so this holds for a live report as well as the fixture.
  const section = report.conditional.block;
  const name = Object.keys(section)[0];
  const row = G.conditionalRows(report, "block").find((r) => r.method === name);
  const gap = section[name];
  assertEqual(row.p, gap.spread_p_adjusted);
  if (gap.spread_p_value !== undefined && gap.spread_p_value !== gap.spread_p_adjusted) {
    assert(row.p !== gap.spread_p_value, "the raw p-value leaked into the table");
  }
});

check("missing sections degrade to empty rather than throwing", () => {
  assertEqual(G.marginalRows(null).length, 0);
  assertEqual(G.conditionalRows(null, "block").length, 0);
  assertEqual(G.conditionalRows(report, "nope").length, 0);
  assertEqual(G.conditionalNote(report, "nope"), "");
  assertEqual(G.dataFactRows(null).length, 0);
});

check("the data panel states the period and the completeness", () => {
  const rows = G.dataFactRows(report.data || cannedReport().data);
  const joined = rows.map((r) => r.label + "=" + r.value).join("|");
  assertIncludes(joined, "period");
  assertIncludes(joined, "slots present");
  assertIncludes(joined, "MW");
});

// -- report -----------------------------------------------------------------

const label = live ? "live forecast" : "canned fixtures";
if (failures.length) {
  console.error(`console check FAILED on the ${label}`);
  for (const failure of failures) {
    console.error(`  x ${failure.name}\n      ${failure.message}`);
  }
  console.error(`\n${passed} passed, ${failures.length} failed`);
  process.exit(1);
}
console.log(`console check passed on the ${label}: ${passed} checks`);
