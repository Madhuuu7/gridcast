/* The service's shapes, and a fetch that distinguishes kinds of absence.
 *
 * Three things can be missing here and they mean different things: the
 * service has no data at all, the study has not been run, nothing has been
 * issued yet. A console that renders all three as a spinner is lying about
 * two of them, so `load` returns the reason and each view says which one it
 * hit.
 */

export type Health = {
  status: string;
  error: string | null;
  data?: { start: string; end: string };
  servable?: { from: string; to: string };
  train_rows?: number;
  calibration_rows?: number;
  alpha?: number;
};

export type DayEntry = {
  date: string;
  kind: string;
  holiday: string | null;
  blocks: number;
};

export type Band = {
  lower: (number | null)[];
  upper: (number | null)[];
  coverage: number;
  mean_width: number;
};

export type Forecast = {
  date: string;
  day_kind: string;
  holiday: string | null;
  issued: string;
  point: number[];
  baseline: number[];
  actual: (number | null)[];
  blocks: number[];
  times: string[];
  intervals: Record<string, Band>;
  alpha: number;
  nominal: number;
  metrics: {
    mape: number;
    mae: number;
    baseline_mape: number;
    peak_actual: number;
    peak_point: number;
    peak_error: number;
  };
};

export type ScheduleBlock = {
  block: number;
  point: number;
  lower: number | null;
  upper: number | null;
  actual: number | null;
};

export type Schedule =
  | { issued: false; reason: string }
  | {
      issued: true;
      target_date: string;
      issued_at: string;
      gate: string;
      method: string;
      model: string;
      alpha: number;
      kernel: Record<string, number> | null;
      code_version: string;
      note: string | null;
      day_kind: string;
      holiday: string | null;
      calibration: { from: string; to: string };
      train_rows: number;
      blocks: ScheduleBlock[];
      score: DayScore | null;
    };

export type DayScore = {
  target_date: string;
  method: string;
  model: string;
  issued_at: string;
  code_version: string;
  scored: number;
  missing: number;
  coverage: number | null;
  coverage_se: number | null;
  mae_mw: number | null;
  mape: number | null;
  mean_width_mw: number | null;
  rupees: number | null;
  unbounded_blocks: number;
};

export type RecordBody = {
  totals: {
    days: number;
    blocks: number;
    coverage: number | null;
    coverage_se: number | null;
    mae_mw: number | null;
    rupees: number;
  };
  days: DayScore[];
  interval_value?: { cost_coverage: number; blocks_scored: number };
  cost?: { rupees_per_year: number; crore_per_year: number };
  error?: string;
};

export type Backtest = Record<string, any>;

/** What a finished fetch can be: there is no "loading" after it resolves. */
export type Resolved<T> =
  | { state: "ready"; value: T }
  | { state: "absent"; reason: string };

export type Loaded<T> = { state: "loading" } | Resolved<T>;

export async function load<T>(path: string): Promise<Resolved<T>> {
  try {
    const response = await fetch(path, { headers: { accept: "application/json" } });
    if (response.status === 404) {
      return { state: "absent", reason: "not-built" };
    }
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      return {
        state: "absent",
        reason: (body as { detail?: string }).detail ?? `HTTP ${response.status}`,
      };
    }
    return { state: "ready", value: (await response.json()) as T };
  } catch (error) {
    return {
      state: "absent",
      reason: error instanceof Error ? error.message : "the service did not answer",
    };
  }
}
