/* Formatting, with the units a scheduler actually uses.
 *
 * Indian grid work reads money in lakh and crore, not in millions, and load
 * to the nearest MW because the settlement block is a quarter hour. A
 * console that prints "₹43.5M" is using the wrong vocabulary for the desk
 * it is sitting on.
 */

const BLOCK_MINUTES = 15;

export function blockLabel(block: number): string {
  const minutes = block * BLOCK_MINUTES;
  const hh = String(Math.floor(minutes / 60)).padStart(2, "0");
  const mm = String(minutes % 60).padStart(2, "0");
  return `${hh}:${mm}`;
}

export function mw(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function rupees(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const crore = value / 1e7;
  if (Math.abs(crore) >= 1) return `₹${crore.toFixed(2)} cr`;
  const lakh = value / 1e5;
  if (Math.abs(lakh) >= 1) return `₹${lakh.toFixed(2)} lakh`;
  return `₹${Math.round(value).toLocaleString("en-IN")}`;
}

export function longDate(iso: string): string {
  const parsed = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleDateString("en-IN", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}

export function shortDate(iso: string): string {
  const parsed = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
  });
}

export function issuedAt(iso: string): string {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleString("en-IN", {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Asia/Kolkata",
  });
}

/** The strip's ramp position for a block: midnight cool, evening warm. */
export function rampAt(block: number): number {
  // Not linear in the block number. The thing the colour stands for is
  // cooling load, which tracks the day's heat rather than the clock, so the
  // ramp peaks in the late afternoon and falls back overnight.
  const phase = (block - 8) / 96;
  return 0.5 - 0.5 * Math.cos(2 * Math.PI * phase);
}

export function mixRamp(t: number, cool: string, hot: string): string {
  const parse = (hex: string) => [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
  const [r1, g1, b1] = parse(cool);
  const [r2, g2, b2] = parse(hot);
  const clamp = Math.min(1, Math.max(0, t));
  const to = (a: number, b: number) => Math.round(a + (b - a) * clamp);
  return `rgb(${to(r1, r2)}, ${to(g1, g2)}, ${to(b1, b2)})`;
}
