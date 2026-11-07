/* The console's own arithmetic, including the bug it has already shipped. */

import { describe, expect, it } from "vitest";
import { coverageOf, missed, niceTicks, verdictFor, type BandBlock } from "./bands";
import { mixRamp, rampAt, rupees } from "./format";

const block = (over: Partial<BandBlock> = {}): BandBlock => ({
  block: 0,
  point: 4000,
  lower: 3700,
  upper: 4300,
  actual: 4100,
  ...over,
});

describe("reading a band", () => {
  it("covers an outcome inside the band", () => {
    expect(verdictFor(block())).toBe("covered");
    expect(missed(block())).toBe(false);
  });

  it("misses an outcome outside it, on either side", () => {
    expect(verdictFor(block({ actual: 4600 }))).toBe("missed");
    expect(verdictFor(block({ actual: 3100 }))).toBe("missed");
  });

  it("treats the bounds as inclusive", () => {
    expect(verdictFor(block({ actual: 3700 }))).toBe("covered");
    expect(verdictFor(block({ actual: 4300 }))).toBe("covered");
  });

  it("does not compare against a missing bound", () => {
    // The shipped bug: `4100 > null` is true and `4100 < null` is false, so
    // a naive comparison called every unbounded block covered and the miss
    // markers quietly disappeared from the page.
    expect(verdictFor(block({ lower: null }))).toBe("unbounded");
    expect(verdictFor(block({ upper: null }))).toBe("unbounded");
    expect(verdictFor(block({ lower: null, upper: null }))).toBe("unbounded");
    // And an unbounded block is not a miss either: the interval does
    // contain the outcome. It is simply not evidence about coverage.
    expect(missed(block({ lower: null, actual: 99999 }))).toBe(false);
  });

  it("treats an infinite bound the same as a missing one", () => {
    expect(verdictFor(block({ lower: -Infinity }))).toBe("unbounded");
    expect(verdictFor(block({ upper: Infinity }))).toBe("unbounded");
  });

  it("reports a block with no outcome as pending, not as covered", () => {
    expect(verdictFor(block({ actual: null }))).toBe("pending");
    expect(verdictFor(block({ actual: NaN }))).toBe("pending");
  });
});

describe("coverage over a day", () => {
  it("divides by the blocks it could actually judge", () => {
    const day: BandBlock[] = [
      block({ actual: 4100 }), // covered
      block({ actual: 4100 }), // covered
      block({ actual: 9000 }), // missed
      block({ lower: null, actual: 9000 }), // unbounded
      block({ actual: null }), // pending
    ];
    const out = coverageOf(day);
    expect(out.scored).toBe(3);
    expect(out.unbounded).toBe(1);
    expect(out.pending).toBe(1);
    // Two of three, not two of five: an unbounded block and an unsettled
    // one are not evidence, and counting them as covered would be the
    // flattering mistake.
    expect(out.coverage).toBeCloseTo(2 / 3, 10);
  });

  it("returns null rather than zero when nothing can be judged", () => {
    expect(coverageOf([]).coverage).toBeNull();
    expect(coverageOf([block({ actual: null })]).coverage).toBeNull();
  });
});

describe("axis ticks", () => {
  it("lands on round numbers", () => {
    expect(niceTicks(1261, 5147)).toEqual([2000, 3000, 4000, 5000]);
    expect(niceTicks(0, 1)).toEqual([0, 0.25, 0.5, 0.75, 1]);
  });

  it("stays inside the range", () => {
    for (const t of niceTicks(2247, 3680)) {
      expect(t).toBeGreaterThanOrEqual(2247);
      expect(t).toBeLessThanOrEqual(3680);
    }
  });

  it("survives a degenerate range", () => {
    expect(niceTicks(100, 100)).toEqual([100]);
    expect(niceTicks(5, 1)).toEqual([5]);
  });
});

describe("the strip's ramp", () => {
  it("is coolest overnight and warmest through the afternoon heat", () => {
    const night = rampAt(8); // 02:00
    const afternoon = rampAt(56); // 14:00
    expect(night).toBeLessThan(0.05);
    expect(afternoon).toBeGreaterThan(0.95);
  });

  it("wraps without a seam at midnight", () => {
    expect(Math.abs(rampAt(0) - rampAt(96))).toBeLessThan(1e-9);
  });

  it("mixes within the two endpoint colours", () => {
    expect(mixRamp(0, "#3e6e8e", "#e09b2d")).toBe("rgb(62, 110, 142)");
    expect(mixRamp(1, "#3e6e8e", "#e09b2d")).toBe("rgb(224, 155, 45)");
    // Out-of-range input clamps rather than producing an invalid colour.
    expect(mixRamp(5, "#3e6e8e", "#e09b2d")).toBe("rgb(224, 155, 45)");
  });
});

describe("money in the units the desk uses", () => {
  it("reads in lakh and crore, not millions", () => {
    expect(rupees(4.5e7)).toBe("₹4.50 cr");
    expect(rupees(3.2e5)).toBe("₹3.20 lakh");
    expect(rupees(900)).toBe("₹900");
    expect(rupees(null)).toBe("—");
  });
});
