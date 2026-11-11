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
on average, and they are honest about completely different things.** Which
turns out to have a shape. A method that conditions on one variable fixes that
variable's coverage and no other, because a partition *is* a function of one
variable; replace the partition with a kernel and it fixes every axis its
calibration data covers, and still none that it does not.

And the whole study has the weakness every backtest has — the folds, the
features and the methods were all chosen by someone who already knew how the
story ended. So the system also **declares a schedule for tomorrow every day
and cannot edit it afterwards**: a workflow runs at the gate, and the store
refuses anything that arrives after the day it is about has begun. That ledger
is the only measurement here that nobody arranged.

**It is paused, and the published ledger is a snapshot.** The free Postgres it
wrote to has expired, so there is nowhere durable to record a declaration; the
workflow now skips each morning with a warning rather than pretending
otherwise, and resumes on its own when `DATABASE_URL` points at a database
again. The rows on the site are the ones it managed to write, dated on the
page. Everything else here — the backtest, the coverage, the replay — is
reproducible from this repository and does not depend on that.

---

## The numbers

From `python scripts/backtest.py` — 8 folds spread across three years, 32,391
out-of-sample blocks, 90% nominal coverage.

### Point forecast

| model | MAPE | MAE | skill vs naive |
|---|---|---|---|
| seasonal naive (last 3 same weekdays) | 11.84% | 476 MW | — |
| ridge, closed form | 6.42% | 270 MW | +43% |
| histogram gradient boosting | **5.07%** | **233 MW** | **+51%** |

### Intervals: marginal

| method | coverage | mean width | Winkler ↓ |
|---|---|---|---|
| **adaptive conformal** *(online)* | 90.1% | **860 MW** | **1007** |
| mondrian-adaptive *(online)* | 91.8% | 1012 MW | 1206 |
| gaussian (±1.645σ) | 90.9% | 1104 MW | 1482 |
| **localized conformal** | 92.1% | **1250 MW** | **1492** |
| mondrian conformal | 92.4% | 1274 MW | 1507 |
| split conformal | 92.9% | 1335 MW | 1577 |
| normalized conformal | 94.4% | 1405 MW | 1609 |

Winkler charges the width and penalises misses, so neither an absurdly wide
interval nor an absurdly sharp one wins it by cheating.

The two online methods at the top update on the outcome of every block as it
lands, which is a different and easier problem. Among the offline methods the
kernel and the Gaussian are within 10 Winkler of each other — and the
Gaussian is there by luck of the window rather than by construction. On the
previous data cut it sat at 88.6% coverage and 1597 Winkler, well behind;
nothing about `±1.645σ` changed, the fortnight of extra data did. The
conformal methods all over-cover on this window instead, which is the failure
direction that costs width rather than credibility.

Marginal coverage is also the easy test. The next section is the one that
separates them.

### Intervals: conditional — where they come apart

An interval that covers 90% overall by covering 98% of quiet nights and 79% of
the evening ramp is wrong exactly when being wrong is expensive. "Spread" is
the gap between the best and worst group's coverage; "noise" is the 95th
percentile of that spread under a **simulated null where coverage is perfect
in every group**, because with 96 groups a spread of 0.10 happens by chance.

| method | by block of day | by month | by day kind |
|---|---|---|---|
| gaussian | 0.163 | 0.218 | 0.168 |
| split conformal | 0.123 | 0.189 | 0.175 |
| normalized conformal | **0.067 — within noise** | 0.164 | 0.175 |
| mondrian conformal | **0.068 — within noise** | 0.198 | 0.180 |
| adaptive conformal | 0.171 | **0.041 — within noise** | 0.061 |
| mondrian-adaptive | 0.110 | 0.099 | 0.261 |
| **localized conformal** | **0.065 — within noise** | 0.216 | **0.112** |
| *noise floor (p95)* | *0.101* | *0.056* | *0.024* |

**Each partition fixes the axis it partitions on, and no other.** Mondrian and
normalized conformal pull time-of-day coverage close to the noise floor and
leave the seasonal spread untouched; adaptive conformal does the reverse and is
the only method whose seasonal variation is statistically indistinguishable
from chance.

Combining them (`mondrian-adaptive`, a separate online level per hour) helps
on both axes and fixes neither — 0.138 and 0.135 — because 24 independent
learners each see one observation per day and adapt 24× more slowly. That is a
measured compromise, not a win.

This is structural rather than a tuning accident: **a partition conditions on
one variable because a partition is a function of one variable.** A kernel is
not. Weighting calibration residuals by similarity across time of day, day
kind and temperature at once lets one estimator work on more than one axis,
and that shows up as the only row in the table that is good in two columns:
`localized conformal` is inside the noise floor on block of day (0.065
against 0.101) *and* the best offline method on day kind (0.112, against
0.168 for the next one).

Three methods reach the noise floor on block of day — Mondrian and normalized
conformal get there too, which is exactly what conditioning on time of day is
supposed to do. What neither of them does is anything for day kind: 0.180 and
0.175, no better than ignoring the question. Being good on one axis is
available to any partition. Being good on two is what the kernel buys.

**It does nothing at all for the seasons**, and that is the honest shape of
the result. 0.229 against split conformal's 0.230. A kernel borrows strength
from calibration data that exists, and next season's weather is not in a
60-day window at any bandwidth. Only a method that updates during the test
window tracks a season it has never seen.

**Day type is still not solved, only improved.** The worst day-kind group goes
from 76.9% under split conformal to 82.1%, against a 90% claim and a 0.024
noise floor. Diwali remains the day the model is most wrong about.

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

### How wide should the kernel be?

One axis at a time, so each column is attributable. `flat` is a bandwidth
wide enough to ignore that axis entirely; `kind` is the weight a mismatched
day kind keeps, so 1.00 ignores day kind and 0.00 is a hard partition on it.
ESS is Kish's effective sample size — how many equally weighted residuals
each interval actually rests on, out of a calibration window of several
thousand.

| block | °C | kind | coverage | width | Winkler ↓ | block spread | day-kind spread | ESS |
|---|---|---|---|---|---|---|---|---|
| flat | flat | 1.00 | 92.9% | 1335 | 1577 | 0.123 | 0.175 | 5074 |
| 8 | flat | 1.00 | 92.5% | 1291 | 1524 | 0.072 | 0.179 | 1505 |
| flat | 5 | 1.00 | 93.2% | 1348 | 1583 | 0.125 | 0.181 | 3475 |
| flat | flat | 0.05 | 92.8% | 1320 | 1555 | 0.144 | 0.091 | 2698 |
| flat | flat | 0.00 | 92.8% | 1317 | 1553 | 0.141 | **0.055** | 2302 |
| 8 | 5 | 0.05 | 92.6% | 1273 | 1496 | 0.081 | 0.097 | 616 |
| **4** | **3** | **0.05** | 92.1% | **1250** | **1492** | **0.065** | 0.112 | 219 |
| 2 | 2 | 0.05 | 92.5% | 1261 | 1501 | 0.066 | 0.113 | 77 |
| 1 | 1 | 0.02 | 92.8% | 1297 | 1536 | 0.074 | 0.173 | 19 |

The first row is the check that matters: **a flat kernel reproduces split
conformal exactly** — 92.9%, 1335 MW, 1577 — every digit of the row in the
marginal table above. The two are one estimator at two bandwidths, not two
methods that happen to agree, and that is what licenses reading the rest of
the table as a single tuning surface.

Each axis moves only its own column. Narrowing the block bandwidth alone
takes block spread from 0.123 to 0.072 and makes day kind slightly *worse*;
lowering the day-kind weight alone takes day kind from 0.175 to 0.091 and
leaves block spread at 0.144.

**The optimum is interior, and the ESS column is why.** Winkler bottoms out
at 1492 with about 219 effective residuals per interval, and then rises again
— not because sharper conditioning is wrong, but because a quantile drawn
from 19 residuals is noise wearing the authority of ten thousand. Conditional
coverage keeps looking like it is improving on the way down, and the spread
column alone never says when the data ran out. That is the trap this column
exists to make visible.

### What it costs when the weather is also a forecast

Every number above is computed with the temperature the day actually had,
which nobody has at a 10:00 gate. Open-Meteo keeps its past model runs, so
the whole backtest can be rerun against what was *predicted* two days ahead
instead.

The weather forecast itself, over 26,124 hours: **1.58 °C mean absolute
error**, +0.15 °C bias, and a 95th percentile of 5.0 °C.

| weather at fit time | MAPE | MAE | coverage | width | Winkler |
|---|---|---|---|---|---|
| the analysis | 5.07% | 233 MW | 92.1% | 1250 | 1492 |
| a lead-2 forecast | 5.49% | 256 MW | 91.3% | 1363 | 1630 |

**0.42 MAPE points, 8% worse.** That is the size of the advantage every other
figure in this README is getting, and the gap between these numbers and a
production system's.

The more interesting half is the coverage column. The point forecast degrades
and **the interval absorbs it**: coverage holds at 91.3% against a 90% claim
while the band widens 9%. The conformal quantile is computed from residuals
that already contain the weather error, so it prices in an uncertainty nobody
told it about. A Gaussian interval fitted to the same residuals would do this
too; what it would not do is stay calibrated, since the extra error is not
Gaussian either.

### The leakage trap, measured

| | MAPE |
|---|---|
| causal features only | 5.17% |
| plus a 24-hour lag | **4.59%** |

A 24-hour lag is the first feature every load-forecasting tutorial reaches
for. At a 10:00 gate it is observable for tomorrow's *morning* blocks and
still in the future for its afternoon ones. It buys an 11% apparent
improvement, entirely fictional, and **nothing in the backtest output says
anything is wrong** — a leaky model simply looks better.

`dataset.assert_causal` checks every row against its own issue time, and
`tests/test_leakage.py` asserts that a 24-hour lag is *rejected*. The guard is
tested by making it fire, not by trusting it. The shortest uniformly safe
same-block lag at this gate is **48 hours**, not 24.

---

## The part that is not a backtest

Everything above was scored by the person who chose the folds. That is the
standing objection to every backtest ever published, and no amount of extra
rigour inside one answers it.

So the service also issues a real schedule. Once a day, a little after the
10:00 IST gate, it fits on what is known, forecasts all 96 blocks of
tomorrow, and writes the result to Postgres. Three rules in the schema, not
in the job that writes:

- **A forecast must predate its target.** `issued_at` is checked in SQL
  against midnight IST on the target day. A run that slips past the boundary
  is refused, and the record gets a visible gap rather than a backdated row.
- **A forecast is immutable.** There is no update path. Re-issuing creates a
  second run with its own timestamp, and the scorecard reads the *earliest* —
  a later issuance was made with more information, even when that information
  was only that the first one looked wrong.
- **What produced it is part of it.** Model, method, alpha, kernel settings,
  calibration window and git sha, all on the run. Months of scorecard are
  uninterpretable if the thing being scored changed silently in week three.

Observations are the deliberate exception and do get updated, because SLDC
revises readings. What was claimed is frozen; what happened is whatever the
operator now says it was.

```bash
python scripts/issue_forecast.py            # declare tomorrow
python scripts/issue_forecast.py --observe  # record what the grid drew
python scripts/issue_forecast.py --scorecard
```

The first forecast on record is for **8 November 2026, which is Diwali** —
the day this project has spent its whole conditional-coverage section saying
every method is worst on. It widened its band from 777 MW to 2,819 MW across
the day and forecast a mean of 3,014 MW against a recent weekday's 3,600.
Whether that was right is on the ledger either way.

A ledger that grows one row a day starts at one row, so the console also
replays the same pipeline across the whole out-of-sample window and scores
it the same way — **41 days, 3,822 blocks, 96.2% coverage, 191 MW MAE**. That
is shown separately and labelled, because it was computed after the fact and
the one-row table was not. It is the right number to read for how the system
performs; the issued table is the right one to read for whether it can be
trusted.

## Forecast error in rupees

MAE in MW is the right unit for comparing methods and the wrong one for
explaining why anyone should care. Deviation from a declared schedule is
settled per block, so the error has a price.

The model follows the CERC mechanism as it applied to a state drawee:
deviation up to the lower of 12% of schedule and 150 MW at the normal rate,
over-drawal beyond that at double, under-drawal beyond it free — drawing less
than you declared is not a grid security problem, and a model that takes an
absolute value before pricing throws away the asymmetry that makes
under-forecasting the expensive direction.

Measuring it turned up the thing worth reporting. The boosted model beats the
seasonal naive by **39% on MAE and only 17% on cost**, because once both
forecasts routinely deviate past the 150 MW band the within-band charge is
identical for both and only the over-drawal excess separates them. **The
volume limit saturates.** "39% better, therefore 39% cheaper" would have been
wrong by construction.

The absolute figure is an upper bound and is labelled as one everywhere it
appears: a real utility revises its schedule intra-day and is not settled on
a 38-hour-old forecast, 15-minute grid frequency is not fetched so no slab is
applied, and the rate used is the ₹12/kWh cap rather than a typical settled
rate. All three distortions apply equally to any two methods, so **the ratio
is the number to read.**

---

## The data

Both sources are free, key-less and real.

**Delhi SLDC** (`delhisldc.org`) publishes 288 five-minute readings a day for
the Delhi total and its five discoms — BRPL, BYPL, NDPL, NDMC and MES. Fetched
back to October 2023: 1,095 days, 102,753 fifteen-minute blocks, 97.7% of the
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
  invented values. 13 days of the 1,097 are missing from the archive entirely —
  not a fetch failure, they are simply not published — and they cluster in
  August–September 2026, which is why the backtest folds are spread across the
  whole record rather than stacked at the end.
- **A day can also lie.** Two days came back with one reading repeated in all
  286 rows, which parses cleanly and is entirely false. A day with a single
  distinct value is treated as unpublished, like an absent one.
- **The discoms do not sum to the total.** Separate meters, off by ~45 MW on an
  average day. The total is what gets forecast.

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -e ".[dev]"

python scripts/fetch_data.py        # ~7 minutes, resumable, cached per day
alembic upgrade head                # SQLite locally, Postgres on DATABASE_URL
pytest                              # 185 tests
python scripts/backtest.py          # the study above, ~15 minutes

npm --prefix web install            # the console
npm --prefix web test               # 15 more tests, on the console's arithmetic
npm --prefix web run build

python scripts/issue_forecast.py    # declare tomorrow, once the data is fresh
uvicorn gridcast.api.app:app        # the desk at http://127.0.0.1:8000
```

`fetch_data.py` caches one CSV per day, so interrupting and re-running it
picks up where it stopped, and later runs only fetch new days. The backtest
is slower than it used to be because the kernel is quadratic in the
calibration window and the two sweeps each re-run the whole thing per
setting; `--quick` cuts it to four folds, and `--skip 6 7 8` drops the
sweeps.

During development, `npm --prefix web run dev` serves the console on :5173
and proxies `/api` to :8000, so the front end reloads without rebuilding.

The desk has four views. **The declaration** is the schedule currently on
record for tomorrow, drawn as 96 discrete bars whose vertical extent is the
prediction interval, because a utility is settled on each block separately
and a smooth ribbon would imply a continuity the product does not have.
**The record** is the ledger of past declarations against what the grid drew.
**Replay a day** runs any out-of-sample day under every interval method, and
**the study** is the backtest above.

Replay will only serve dates **after** the model's calibration window — the
service fits once at startup and publishes all three boundaries at
`/api/health`, because a forecast for a day the model trained on would cover
every block and prove nothing.

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

**A kernel instead of a partition, and what that costs.** Weighted conformal
(Tibshirani et al., 2019) normalises over the calibration weights *including
the test point's own*, which is the weighted version of taking the
`ceil((n+1)(1-α))`-th order statistic rather than `np.quantile` — leave the
test point out and it under-covers, by most at exactly the small effective
sample sizes that weighting creates. The honest cost is that the guarantee
does not transfer: weighted conformal is exact when the weights come from a
*known* covariate shift, and here they encode a similarity judgement instead,
so the coverage claim is empirical and the backtest is the only evidence for
it. That is the same trade `AdaptiveConformal` makes and it is stated for the
same reason. The bandwidths were also chosen on the backtest they are
reported on, which is in-sample tuning of three hyperparameters; the sweep is
printed in full so a reader can see the optimum is interior rather than at
the edge of where someone stopped looking.

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

**Weather is reanalysis in the headline numbers, and forecast where it
counts.** The archive returns what the weather actually was, so every figure
in the tables above is better than production would be. Rather than leave
that as a caveat it is measured — 0.44 MAPE points, above — and the live
daily job uses a genuine forecast, because at a 10:00 gate there is nothing
else to use.

**The record is append-only, and that is a schema decision rather than a
convention.** It would have been easier to upsert a forecast per day and let
the job be idempotent. But an idempotent writer is one that can quietly
replace yesterday's claim, and then the ledger means nothing — so re-issuing
inserts a visible second run and the scorecard reads the first.

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

**A calibration window measured in days, over an archive with holes.** The
service took the 45 days before the servable window. Fresh data moved that
window onto the August–September 2026 gap, and it came back holding 1,397 of
its 4,320 blocks — 14 per block group against a Mondrian minimum of 20, so
all 95 groups fell back to the pooled quantile and the per-block method
silently became split conformal. Nothing raised. The bands just stopped
varying, which is invisible unless you are looking for it. Counted in rows
instead, a thin stretch makes the window reach further back rather than hold
less data.

**Weather that stops is not weather that is missing.** `forecast_design`
checked for null features before trusting a day, which is useless here:
`to_blocks` interpolates with `limit_direction="both"` and forward-fills past
the end of the series, so a weather feed that stopped last Tuesday produced a
complete-looking design matrix carrying last Tuesday's temperature for
tomorrow. The first offline run issued a forecast for a day with no weather
at all and the boosted model served it without complaint. The span is checked
before the join now, and a whole empty column is an error rather than a gap.

**A ledger that hid every pending forecast.** The scorecard's totals returned
zero days whenever no *blocks* were scored — so a forecast that had been
published and was waiting for its outcome did not appear at all, which is the
state the record spends most of its life in.

---

## What this does not do

- **Day type is improved, not solved.** The worst day-kind group still gets
  82.5% coverage against a 90% claim. Diwali is the hardest day of the year
  and three years hold about thirty holidays, so there is not much more to
  learn from this dataset — the fix is more years, not a sharper kernel.
- **Nothing fixes seasonal coverage offline.** Only the online methods track
  it, for the structural reason given above, and the online methods have no
  persistent state between service restarts: `AdaptiveConformal` walks the
  test set in order in the backtest, and at the gate it carries the level it
  ended the calibration window on, which is a weaker thing.
- **The deviation charge is an upper bound**, for the three reasons listed
  with it. Grid frequency is the missing input and this project has no free
  source for it at 15-minute resolution.
- **One city, one target.** The five discom series are fetched and stored but
  not modelled, so there is no hierarchy and no reconciliation.
- **The weather forecast is Open-Meteo's, not the scheduler's.** A real SLDC
  has access to IMD products this does not.
- **13 days are missing from the SLDC archive** and cannot be recovered. They
  cluster in August–September 2026, which is also why the API's calibration
  window counts rows rather than days.
- **The ledger is young.** A coverage figure over a few hundred blocks
  carries several points of binomial noise, and the console says so on the
  page rather than printing a percentage that looks like a measurement.

---

## Layout

```
src/gridcast/
  data/sldc.py        the SLDC fetcher, parser and per-day cache
  data/weather.py     Open-Meteo: the archive, and past forecast runs
  data/calendar.py    Indian holidays, including the lunar ones
  data/dataset.py     blocks, features, the causality guard, tomorrow's design
  models.py           seasonal naive, ridge from scratch, boosting
  conformal.py        seven interval methods, weighted quantile, the null
  evaluate.py         rolling train/calibrate/test backtest
  settlement.py       forecast error priced under the deviation mechanism
  issue.py            fit, forecast tomorrow, and refuse when that is a lie
  scorecard.py        the ledger: claims joined to outcomes
  store/              the schema that will not let a forecast be revised
  api/app.py          the HTTP surface
web/src/              the desk: Vite, React, TypeScript, custom SVG
alembic/              migrations; the app refuses to boot behind the head
scripts/fetch_data.py       the one-time data pull
scripts/backtest.py         the study behind every number above
scripts/issue_forecast.py   the daily job
```

## License

MIT.
