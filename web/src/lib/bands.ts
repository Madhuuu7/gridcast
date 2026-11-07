/* Reading a band: the two judgements the console gets to make, kept out of
 * the component so they can be tested on their own.
 *
 * `missed` carries a bug this project has already shipped once. The old
 * console compared the actual against the band with `actual < lower`, and an
 * unbounded side serialises as JSON `null`. In JavaScript `x < null` is
 * `false` and `x > null` is `true` for any positive number, so a block with
 * no lower bound was silently counted as covered and the miss markers simply
 * stopped appearing. The fix is not a tighter comparison, it is refusing to
 * compare at all when a bound is absent, which is why the null check comes
 * first and returns a third answer rather than a boolean.
 */

export type BandBlock = {
  block: number;
  point: number;
  lower: number | null;
  upper: number | null;
  actual: number | null;
};

export type Verdict = "covered" | "missed" | "unbounded" | "pending";

export function verdictFor(b: BandBlock): Verdict {
  if (b.actual === null || !Number.isFinite(b.actual)) return "pending";
  if (
    b.lower === null ||
    b.upper === null ||
    !Number.isFinite(b.lower) ||
    !Number.isFinite(b.upper)
  ) {
    // An interval with no bound covers everything, so calling this a miss
    // would be wrong -- but calling it "covered" and folding it into a
    // coverage figure credits the method for having given up.
    return "unbounded";
  }
  return b.actual >= b.lower && b.actual <= b.upper ? "covered" : "missed";
}

export function missed(b: BandBlock): boolean {
  return verdictFor(b) === "missed";
}

/** Coverage over the blocks that can be judged, and the count that cannot. */
export function coverageOf(blocks: BandBlock[]): {
  coverage: number | null;
  scored: number;
  unbounded: number;
  pending: number;
} {
  let covered = 0;
  let scored = 0;
  let unbounded = 0;
  let pending = 0;
  for (const b of blocks) {
    switch (verdictFor(b)) {
      case "covered":
        covered++;
        scored++;
        break;
      case "missed":
        scored++;
        break;
      case "unbounded":
        unbounded++;
        break;
      default:
        pending++;
    }
  }
  return {
    coverage: scored ? covered / scored : null,
    scored,
    unbounded,
    pending,
  };
}

/**
 * Round gridlines across a range.
 *
 * Four equal divisions of the data range give labels like 4,175, which is a
 * number nobody has a feel for. A scheduler reads MW off an axis.
 */
export function niceTicks(min: number, max: number, target = 4): number[] {
  const span = max - min;
  if (!(span > 0) || !Number.isFinite(span)) return [min];
  const rough = span / target;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = ([1, 2, 2.5, 5, 10].find((m) => m * magnitude >= rough) ?? 10) * magnitude;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max; v += step) out.push(v);
  return out;
}
