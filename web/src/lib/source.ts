/* Where the desk gets its data when there is no service behind it.
 *
 * The desk is published two ways. Served by the forecasting service it
 * reads a live model and a live ledger. Published as a static page there
 * is no service and no database.
 *
 * What survives that honestly differs by view, and the page says which is
 * which. The study is a deterministic backtest -- fixed folds, a seeded
 * model, the same 32,391 out-of-sample blocks -- so a recording of it is
 * the result rather than a picture of one. A replayed day is likewise
 * exactly what the service computes for that date.
 *
 * The ledger is the view that cannot be faithfully frozen, and it is the
 * one that matters most: its whole claim is that it grows a row a day and
 * nobody can edit it. A snapshot of it is a snapshot, dated on the page,
 * with the live one named.
 *
 * Probe once, cache the answer. Retrying a dead endpoint on every tab
 * change would make the static page feel broken rather than static.
 */

export type Mode = "live" | "recorded";

let resolved: Mode | null = null;
// The promise, not only the answer. Every view asks on first paint, and
// they all ask before any of them has an answer to cache -- so a page with
// six panels fired six probes at a dead endpoint and left six 404s in the
// console of anyone who opened it.
let asking: Promise<Mode> | null = null;

/** Ask the runtime whether it is there. */
export async function detect(): Promise<Mode> {
  if (resolved) return resolved;
  if (!asking) {
    asking = (async () => {
      try {
        const response = await fetch("health", {
          headers: { accept: "application/json" },
          signal: AbortSignal.timeout(4000),
        });
        resolved = response.ok ? "live" : "recorded";
      } catch {
        resolved = "recorded";
      }
      asking = null;
      return resolved;
    })();
  }
  return asking;
}

/** Force a mode, for tests. */
export function setMode(mode: Mode | null): void {
  resolved = mode;
  asking = null;
}

/**
 * Fetch from the runtime, or from the recording if there is no runtime.
 *
 * Paths are relative so the page works under a sub-path -- GitHub Pages
 * serves a project site from `/<repo>/`, and a leading slash would send
 * every request to the domain root.
 */
export async function read<T>(live: string, recorded: string): Promise<T> {
  const mode = await detect();
  const path = mode === "live" ? live : `demo/${recorded}`;
  const response = await fetch(path, { headers: { accept: "application/json" } });
  if (!response.ok) {
    throw new Error(
      mode === "live"
        ? `the runtime answered ${response.status} for ${live}`
        : `the recording ${recorded} is missing from this build`,
    );
  }
  return (await response.json()) as T;
}
