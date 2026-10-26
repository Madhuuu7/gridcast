# gridcast

Day-ahead load forecasting for the Delhi grid, on three years of real
five-minute SLDC readings. A forecast for all 96 fifteen-minute scheduling
blocks of tomorrow, issued at the 10:00 gate, with a prediction interval whose
coverage is **measured rather than assumed**.

The point forecast is table stakes — load is enormously predictable and every
model looks good in isolation. The product is the interval, because deviation
from a declared schedule is settled per block, and a scheduler needs to know
whether the truth is within 50 MW or within 500.

The finding, in one sentence: **every interval method here hits its 90% target
on average, and they are honest about completely different things.**

---

## The numbers

From `python scripts/backtest.py` — 8 folds spread across three years, 32,340
out-of-sample blocks, 90% nominal coverage.

### Point forecast

| model | MAPE | MAE | skill vs naive |
|---|---|---|---|
| seasonal naive (last 3 same weekdays) | 12.34% | 523 MW | — |
| ridge, closed form | 6.66% | 311 MW | +39% |
| histogram gradient boosting | **5.42%** | **251 MW** | **+52%** |

### Intervals: marginal

| method | coverage | mean width | Winkler ↓ |
|---|---|---|---|
| **adaptive conformal** | 90.1% | **924 MW** | **1067** |
| mondrian-adaptive | 91.4% | 1086 MW | 1311 |
| gaussian (±1.645σ) | **88.6%** | 1088 MW | 1597 |
| mondrian conformal | 90.6% | 1272 MW | 1604 |
| split conformal | 90.6% | 1327 MW | 1710 |
| normalized conformal | 91.1% | 1470 MW | 1788 |

Winkler charges the width and penalises misses, so neither an absurdly wide
interval nor an absurdly sharp one wins it by cheating.

### Intervals: conditional — where they come apart

An interval that covers 90% overall by covering 98% of quiet nights and 79% of
the evening ramp is wrong exactly when being wrong is expensive. "Spread" is
the gap between the best and worst group's coverage; "noise" is the 95th
percentile of that spread under a **simulated null where coverage is perfect
in every group**, because with 96 groups a spread of 0.10 happens by chance.

| method | by block of day | by month |
|---|---|---|
| gaussian | 0.220 (worst block 74.9%) | 0.274 (worst month 72.6%) |
| split conformal | 0.181 | 0.230 |
| normalized conformal | **0.107** | 0.255 |
| mondrian conformal | **0.106** | 0.235 |
| adaptive conformal | 0.176 | **0.048 — within noise** |
| *noise floor (p95)* | *0.101* | *0.058* |

**Each method fixes the axis it conditions on, and no other.** Mondrian and
normalized conformal bring time-of-day coverage down to the noise floor and
leave the seasonal spread untouched; adaptive conformal does the reverse and is
the only method whose seasonal variation is statistically indistinguishable
from chance.

Combining them (`mondrian-adaptive`, a separate online level per hour) helps
on both axes and fixes neither — 0.138 and 0.135 — because 24 independent
learners each see one observation per day and adapt 24× more slowly. That is a
measured compromise, not a win.

**Every method fails on day type.** Grouped by weekday / weekend / holiday /
festival-adjacent, the best worst-group coverage is 84.6% and the worst is
70.8%, all far beyond the 0.022 noise floor. None of these methods conditions
on what kind of day it is, and Diwali is when the model is most wrong and
least honest about it. That is the obvious next thing to build.

### How fast should an online method adapt?

| gamma | coverage | width | Winkler | month spread | block spread |
|---|---|---|---|---|---|
| 0.005 | 90.2% | 1039 | 1276 | 0.092 | — |
| 0.05 | 90.0% | 859 | 968 | 0.027 | — |
| **0.1** | 90.0% | **828** | **918** | 0.018 | 0.192 |
| 0.35 | 90.0% | 962 | 1045 | 0.006 | 0.167 |
| 0.8 | 90.0% | 1335 | 1435 | 0.005 | **0.101** |

The Winkler optimum is interior at γ ≈ 0.1, so it is not an artefact of where
the sweep stopped. And at the aggressive end something interesting happens:
by γ = 0.8 the method adapts fast enough to track *within-day* variation too,
so the block-of-day spread also falls to the noise floor — conditional
coverage on both axes from one pooled learner, at 60% more width.

### The leakage trap, measured

| | MAPE |
|---|---|
| causal features only | 6.46% |
| plus a 24-hour lag | **5.98%** |

A 24-hour lag is the first feature every load-forecasting tutorial reaches
for. At a 10:00 gate it is observable for tomorrow's *morning* blocks and
still in the future for its afternoon ones. It buys an 8% apparent
improvement, entirely fictional, and **nothing in the backtest output says
anything is wrong** — a leaky model simply looks better.

`dataset.assert_causal` checks every row against its own issue time, and
`tests/test_leakage.py` asserts that a 24-hour lag is *rejected*. The guard is
tested by making it fire, not by trusting it. The shortest uniformly safe
same-block lag at this gate is **48 hours**, not 24.

---

## The data

Both sources are free, key-less and real.

**Delhi SLDC** (`delhisldc.org`) publishes 288 five-minute readings a day for
the Delhi total and its five discoms — BRPL, BYPL, NDPL, NDMC and MES. Fetched
back to October 2023: 1,083 days, 101,660 fifteen-minute blocks, 97.8% of the
slots that should exist, load from 1,458 to 8,680 MW. The 2026 peak was on
29 June.

**Open-Meteo's archive** supplies hourly Delhi weather — temperature, humidity,
apparent temperature, cloud cover, wind — for a multi-year range in one
request.

Three judgements the parser makes, each of which quietly poisons everything
downstream if made wrong:

- **A zero is not a reading.** A meter that did not report returns `0.000`, and
  Delhi never draws less than ~1,400 MW. Taken at face value it would teach a
  model that demand sometimes collapses to nothing.
- **A day can be short.** Gaps stay explicit instead of being reindexed into
  invented values. 13 days of the 1,096 are missing from the archive entirely —
  not a fetch failure, they are simply not published — and they cluster in
  August–September 2026, which is why the backtest folds are spread across the
  whole record rather than stacked at the end.
- **The discoms do not sum to the total.** Separate meters, off by ~45 MW on an
  average day. The total is what gets forecast.

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -e ".[dev]"

python scripts/fetch_data.py        # ~7 minutes, resumable, cached per day
pytest                              # 72 tests
python scripts/backtest.py          # the study above, ~3 minutes
node scripts/check_console.mjs      # the console, headless
uvicorn gridcast.api.app:app        # the console at http://127.0.0.1:8000
```

`fetch_data.py` caches one CSV per day, so interrupting and re-running it
picks up where it stopped, and later runs only fetch new days.

The console serves one day at a time: actual against forecast, the chosen band
behind them, and a dot on every block the band missed. It will only forecast
dates **after** the model's calibration window — the service fits once at
startup and publishes all three boundaries at `/api/health`, because a
forecast for a day the model trained on would cover every block and prove
nothing.

---

## How the forecast is set up

Indian grid scheduling runs on 15-minute blocks, and a state utility declares
its day-ahead schedule before the market gates close around 10:00 the previous
day. So: a forecast for every block of day D+1, issued at 10:00 on day D.

At that moment the freshest load reading is D 10:00 — 14 hours before the
first target block and 38 before the last. Features therefore come in two
kinds: same-block lags of 48 hours or more, and day-level summaries of
everything known by the gate (yesterday in full, this morning so far, the last
reading before the gate). The day-level ones are identical across all 96
blocks of a target day, which is what makes them safe, and a test asserts
exactly that.

Days whose issue morning has no readings at all — 172 blocks' worth — fall
back to the most recent thing that *was* known, with a flag recording it. A
real scheduler still has to publish a schedule on those days; dropping them
would excuse the model from the hardest cases and flatter every metric.

---

## Decisions worth arguing about

**Conformal, not Gaussian.** `±1.645σ` assumes symmetry, Gaussian tails and
homoscedasticity; load errors have none of the three. Split conformal assumes
only exchangeability and gives a finite-sample guarantee — coverage in
`[1-α, 1-α+1/(n+1)]` — for any underlying model. Measured here, the Gaussian
interval under-covers at 88.6% and loses on Winkler to every conformal variant.

**The conformal quantile is not `np.quantile`.** It is the
`ceil((n+1)(1-α))`-th smallest score. On 10,000 calibration points the
difference is invisible; on the ~200 a per-group Mondrian split leaves, it is
worth half a percent of coverage. Below `1/α - 1` points no finite quantile
supports the claim, and the honest answer is an infinite interval.

**Conditional coverage gets a null.** The project's main claim is that some
methods are uneven across groups. With 96 groups of ~350 blocks, a 90%
coverage estimate carries a 1.6% standard error, so the *minimum* of 96 such
estimates sits near 0.84 by chance alone. Reporting that as a failure would be
reporting the binomial. Every spread is compared against a simulated null with
the same group sizes, and the 18 resulting p-values get a Benjamini–Hochberg
correction before any is called significant.

**Groups below 100 observations are excluded from that analysis.** The 23:45
block appears 9 times in three years — the last block of the day routinely
lacks the readings to form — and a coverage estimate from 9 observations is
not a coverage estimate. Left in, it dominated both the observed spread and
the noise floor.

**Folds are spread, not stacked.** Six consecutive folds all land in one
season, and the interesting question for an interval is whether it survives a
seasonal change.

**The sigma model gets its own half of the calibration window.** Normalized
conformal needs a model of the error magnitude; fitting it on the same
residuals that produce the conformal quantile would let it peek at the scores
it is judged by.

**Weather is reanalysis, not forecast.** The archive returns what the weather
actually was. A production system would use a weather *forecast*, with its own
error, so the point forecast here is better than production would be. The
intervals are conditional on the weather being known.

---

## Bugs worth keeping in the history

**A noise floor that depended on group order.** The floor is simulated from the
group sizes, and callers hand them over sorted by each method's coverage. With
a fixed seed that paired a different random draw with each size, so identical
inputs produced a floor of 0.18 for one method and 0.27 for another — and the
floor is the guard the project's main claim rests on. Sorted now, with an
order-invariance test.

**A test assumption that was simply wrong.** I asserted that Gaussian
intervals under-cover on fat tails. At 90% they *over*-cover: a few large
Student-t draws inflate the sample standard deviation, so `1.645σ` reaches
past the 90th percentile of the bulk. They under-cover at 99%, where the tails
actually bite. Both directions are now pinned, and the under-coverage seen on
real data comes from a third mechanism — seasonal shift between calibration
and test.

**A null bound is not a covered block.** An unbounded adaptive interval
serialises as JSON `null`, and `null < x` is false in JavaScript, so the
console's miss-detector silently counted every such block as covered.

**Fifteen hundred rows that a boosted model ate and a linear one could not.**
Ridge returned `NaN` for every prediction because 1.3% of rows had a missing
feature that `HistGradientBoostingRegressor` handles natively. Had only the
boosted model been checked, the two would have been silently trained on
different rows and the comparison between them would have meant nothing.

---

## What this does not do

- **No weather forecast error.** See above; this is the largest gap between
  these numbers and a production system's.
- **Nothing conditions on day type**, and that is where every method fails
  worst (70.8% coverage in the worst group). A Mondrian split on day kind is
  the obvious next experiment.
- **One city, one target.** The discom-level series are fetched and stored but
  not modelled.
- **The online methods are replayed, not streamed.** `AdaptiveConformal` walks
  the test set in order, which is the correct simulation of an online method,
  but the service has no persistent state between restarts.
- **Render's free tier has an ephemeral disk**, so the fetched data is rebuilt
  on each deploy — the blueprint fetches a short window at build time rather
  than the full three years.
- **13 days are missing from the SLDC archive** and cannot be recovered.

---

## Layout

```
src/gridcast/
  data/sldc.py        the SLDC fetcher, parser and per-day cache
  data/weather.py     Open-Meteo archive
  data/calendar.py    Indian holidays, including the lunar ones
  data/dataset.py     blocks, features, and the causality guard
  models.py           seasonal naive, ridge from scratch, boosting
  conformal.py        six interval methods and the conditional-coverage null
  evaluate.py         rolling train/calibrate/test backtest
  api/app.py          the HTTP surface
web/index.html        the console, self-contained
scripts/fetch_data.py     the one-time data pull
scripts/backtest.py       the study behind every number above
scripts/check_console.mjs the console's headless check
```

## License

MIT.
