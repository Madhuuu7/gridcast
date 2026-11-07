import { useEffect, useState } from "react";
import {
  load,
  type Backtest,
  type DayEntry,
  type Health,
  type Loaded,
  type RecordBody,
  type Schedule,
} from "./lib/api";
import { detect, type Mode } from "./lib/source";
import { Record as RecordView } from "./views/Record";
import { Replay } from "./views/Replay";
import { Study } from "./views/Study";
import { Tomorrow } from "./views/Tomorrow";

type View = "declaration" | "record" | "replay" | "study";

const VIEWS: [View, string][] = [
  ["declaration", "The declaration"],
  ["record", "The record"],
  ["replay", "Replay a day"],
  ["study", "The study"],
];

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark" | null>(() => {
    try {
      const stored = localStorage.getItem("gridcast-theme");
      return stored === "light" || stored === "dark" ? stored : null;
    } catch {
      return null;
    }
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme) root.setAttribute("data-theme", theme);
    else root.removeAttribute("data-theme");
    try {
      if (theme) localStorage.setItem("gridcast-theme", theme);
      else localStorage.removeItem("gridcast-theme");
    } catch {
      /* private browsing: the page still renders, it just will not remember */
    }
  }, [theme]);

  const prefersDark =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
  const effective = theme ?? (prefersDark ? "dark" : "light");
  return [effective, () => setTheme(effective === "dark" ? "light" : "dark")] as const;
}

export default function App() {
  const [view, setView] = useState<View>("declaration");
  const [mode, setMode] = useState<Mode | null>(null);
  const [theme, toggleTheme] = useTheme();
  const [health, setHealth] = useState<Loaded<Health>>({ state: "loading" });
  const [schedule, setSchedule] = useState<Loaded<Schedule>>({ state: "loading" });
  const [record, setRecord] = useState<Loaded<RecordBody>>({ state: "loading" });
  const [days, setDays] = useState<Loaded<DayEntry[]>>({ state: "loading" });
  const [backtest, setBacktest] = useState<Loaded<Backtest>>({ state: "loading" });

  useEffect(() => {
    void detect().then(setMode);
    load<Health>("api/health", "health.json").then(setHealth);
    load<Schedule>("api/schedule", "schedule.json").then(setSchedule);
    load<RecordBody>("api/record", "record.json").then(setRecord);
    load<DayEntry[]>("api/days", "days.json").then(setDays);
    load<Backtest>("api/backtest", "backtest.json").then(setBacktest);
  }, []);

  const servable =
    health.state === "ready" && health.value.servable
      ? `${health.value.servable.from} to ${health.value.servable.to}`
      : null;

  return (
    <div className="shell">
      <header className="masthead">
        <h1 className="wordmark">gridcast</h1>
        <p>
          Day-ahead load forecasts for the Delhi grid, declared in 96
          fifteen-minute settlement blocks, with intervals whose coverage is
          measured rather than assumed.
        </p>
        <div className="masthead-right">
          {servable && <span className="fig">serving {servable}</span>}
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          >
            {theme === "dark" ? "Light" : "Dark"}
          </button>
        </div>
      </header>

      <nav className="views" aria-label="Views">
        {VIEWS.map(([key, label]) => (
          <button
            key={key}
            onClick={() => setView(key)}
            aria-current={view === key}
          >
            {label}
          </button>
        ))}
      </nav>

      {mode === "recorded" && (
        <div className="state" style={{ marginBottom: 24 }}>
          <p style={{ margin: 0 }}>
            <strong>No service behind this page.</strong> The study is a
            deterministic backtest — fixed folds, the same 32,391
            out-of-sample blocks — so what you see is the result, not a
            picture of one, and a replayed day is exactly what the service
            computes for that date. The ledger is a snapshot: its whole
            point is that it grows a row a day and cannot be edited, which
            a static page cannot show you happening. Run{" "}
            <code>uvicorn gridcast.api.app:app</code> for the live one.
          </p>
        </div>
      )}

      {health.state === "ready" && health.value.status !== "ok" && (
        <div className="state error">
          <h3>The service has no data to serve</h3>
          <p>{health.value.error}</p>
        </div>
      )}

      {view === "declaration" &&
        (schedule.state === "ready" ? (
          <Tomorrow schedule={schedule.value} />
        ) : (
          <Waiting what="the declaration" loaded={schedule} />
        ))}

      {view === "record" &&
        (record.state === "ready" ? (
          <RecordView record={record.value} />
        ) : (
          <Waiting what="the record" loaded={record} />
        ))}

      {view === "replay" &&
        (days.state === "ready" ? (
          days.value.length ? (
            <Replay days={days.value} />
          ) : (
            <section>
              <div className="state">
                <h3>No servable days</h3>
                <p>
                  Every day in the archive was used to fit or calibrate the
                  model. Fetch more history and the window opens.
                </p>
              </div>
            </section>
          )
        ) : (
          <Waiting what="the day list" loaded={days} />
        ))}

      {view === "study" &&
        (backtest.state === "ready" ? (
          <Study backtest={backtest.value} />
        ) : (
          <section>
            <div className="state">
              <h3>The study has not been run here</h3>
              <p>
                It takes about twenty minutes of CPU and writes{" "}
                <code>reports/backtest.json</code>. Deployed instances skip it:
                the numbers are already in the README and re-deriving them on a
                512 MB free instance at every deploy would prove nothing new.
              </p>
              <p className="note">
                Run <code>python scripts/backtest.py</code> locally to fill this
                page.
              </p>
            </div>
          </section>
        ))}

      <footer>
        Load from the Delhi SLDC, weather from Open-Meteo, both free and
        key-less. The forecast for a future day uses a weather forecast, which
        costs about 0.44 MAPE points against knowing the weather in advance —
        measured, not assumed.
      </footer>
    </div>
  );
}

function Waiting({ what, loaded }: { what: string; loaded: Loaded<unknown> }) {
  if (loaded.state !== "absent") {
    return (
      <section>
        <p className="state">Loading {what}.</p>
      </section>
    );
  }
  return (
    <section>
      <div className="state error">
        <h3>Could not load {what}</h3>
        <p>{loaded.reason}</p>
      </div>
    </section>
  );
}
