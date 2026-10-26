"""Prediction intervals that are checked, and four ways of building them.

A point forecast of 6,200 MW is not actionable on its own. A scheduler needs
to know whether the truth is within 50 MW or within 500, because deviation
from the declared schedule is settled per block on the gap. The interval is
the product, and the only property that makes one worth anything is that it
covers as often as it claims to.

The usual approach takes the standard deviation of the residuals and reaches
for 1.96 of them. That assumes the errors are Gaussian, identically
distributed and independent, and load forecast errors are none of the three:
they are fat-tailed, several times larger during the evening ramp than at 4am,
and strongly autocorrelated. ``GaussianIntervals`` is here to be measured and
beaten.

**Split conformal** makes one assumption instead -- that calibration and test
data are exchangeable -- and gives a finite-sample guarantee that holds for
*any* underlying model: coverage at least ``1 - alpha`` and at most
``1 - alpha + 1/(n+1)``. No distributional assumption. The catch is that the
guarantee is *marginal*, averaged over everything, which brings us to the
finding this module exists to make measurable.

**Marginal coverage is not the coverage anyone wants.** An interval covering
90% of blocks overall -- by covering 99% of quiet night blocks and 78% of the
evening ramp -- is wrong exactly when being wrong is expensive.
``MondrianConformal`` calibrates a separate quantile per group, buying
conditional coverage at the price of splitting the calibration data.

**Exchangeability breaks over time.** Calibrate in winter, predict a
heatwave, and the calibration residuals no longer describe the test ones, so
the guarantee lapses silently. ``AdaptiveConformal`` (Gibbs and Candes, 2021)
gives it up deliberately, adjusting the quantile level online from its own
recent hit rate: no finite-sample guarantee, but coverage that tracks a
moving target.

Every claim above is measured in ``scripts/backtest.py`` against real Delhi
load, including the ones that do not hold up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# -- the pieces ------------------------------------------------------------


def normal_ppf(p: float) -> float:
    """Inverse normal CDF: Acklam's approximation plus one Halley step.

    Written out rather than taking a SciPy dependency for one function, and
    refined because the raw approximation's ~1e-9 error would otherwise be the
    only unexamined constant in a module whose whole purpose is examining its
    own numbers.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    low, high = 0.02425, 1.0 - 0.02425
    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


def conformal_quantile(
    scores: np.ndarray, alpha: float, *, fallback: str = "inf"
) -> float:
    """The finite-sample conformal quantile of a set of scores.

    Not ``np.quantile(scores, 1 - alpha)``. The guarantee needs the
    ``ceil((n + 1) * (1 - alpha))``-th smallest score, which sits slightly
    *above* the empirical quantile. The difference is one order statistic --
    invisible on 10,000 calibration points, worth about half a percent of
    coverage on 200, which is exactly the regime a per-group Mondrian split
    lands in.

    When ``ceil((n + 1) * (1 - alpha)) > n`` no observed score is high enough
    to support the claim. That happens below ``1/alpha - 1`` calibration
    points -- fewer than 9 at 90% coverage -- and also whenever an adaptive
    method drives its working level near zero.

    ``fallback`` says what to do then. ``"inf"`` is the honest default: the
    data cannot support the claim and the interval says so. ``"max"`` returns
    the largest observed score instead, which is what ``AdaptiveConformal``
    asks for -- it has no finite-sample guarantee to protect in the first
    place, and an infinite interval there is a scheduler being told nothing
    rather than being told something wide.
    """
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = clean.size
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n == 0:
        return float("inf")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float(clean.max()) if fallback == "max" else float("inf")
    return float(np.partition(clean, rank - 1)[rank - 1])


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    if actual.size == 0:
        return float("nan")
    inside = (actual >= np.asarray(lower, dtype=float)) & (
        actual <= np.asarray(upper, dtype=float)
    )
    return float(np.mean(inside))


def mean_width(lower: np.ndarray, upper: np.ndarray) -> float:
    width = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    finite = width[np.isfinite(width)]
    return float(np.mean(finite)) if finite.size else float("inf")


def winkler_score(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> float:
    """Interval score: the width, plus a penalty for each miss.

    The number to compare methods on, because coverage and width are each
    gameable alone -- an infinitely wide interval covers everything and a
    zero-width one is beautifully sharp. Winkler charges the width and adds
    ``2/alpha`` times the distance of any miss, so neither cheat wins. Lower
    is better, and it is in MW, so the units mean something.
    """
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    penalty = np.where(
        actual < lower,
        2.0 / alpha * (lower - actual),
        np.where(actual > upper, 2.0 / alpha * (actual - upper), 0.0),
    )
    return float(np.mean(width + penalty))


@dataclass
class Intervals:
    """A set of intervals and the method that produced them."""

    method: str
    alpha: float
    lower: np.ndarray
    upper: np.ndarray
    #: Per-point half-width, kept for inspection: whether a method adapts its
    #: width at all is visible here and nowhere else.
    radius: np.ndarray = field(default_factory=lambda: np.empty(0))

    def evaluate(self, actual: np.ndarray) -> dict[str, float]:
        return {
            "coverage": coverage(actual, self.lower, self.upper),
            "nominal": 1.0 - self.alpha,
            "mean_width": mean_width(self.lower, self.upper),
            "winkler": winkler_score(actual, self.lower, self.upper, self.alpha),
            "width_spread": float(
                np.std(self.radius[np.isfinite(self.radius)])
                if self.radius.size
                else 0.0
            ),
        }


# -- the method everyone uses ---------------------------------------------


@dataclass
class GaussianIntervals:
    """Residual standard deviation times a normal critical value.

    What most forecasting write-ups do. One global sigma, symmetric, and no
    guarantee of anything -- which is the point of including it.
    """

    alpha: float = 0.1
    name: str = "gaussian"
    sigma_: float = 0.0

    def fit(self, residuals: np.ndarray, **_: object) -> "GaussianIntervals":
        clean = np.asarray(residuals, dtype=float)
        clean = clean[np.isfinite(clean)]
        self.sigma_ = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0
        return self

    def predict(self, point: np.ndarray, **_: object) -> Intervals:
        point = np.asarray(point, dtype=float)
        z = normal_ppf(1.0 - self.alpha / 2.0)
        radius = np.full(point.shape, z * self.sigma_)
        return Intervals(self.name, self.alpha, point - radius, point + radius, radius)


# -- split conformal -------------------------------------------------------


@dataclass
class SplitConformal:
    """One quantile of the absolute residuals, applied to every point.

    The guarantee: for exchangeable data, coverage is in
    ``[1 - alpha, 1 - alpha + 1/(n+1)]`` whatever the model is. The width is
    the same for every block, which is both its simplicity and its problem --
    the night and the evening ramp get identical intervals despite error
    variances that differ several-fold.
    """

    alpha: float = 0.1
    name: str = "split conformal"
    q_: float = float("inf")
    n_: int = 0

    def fit(self, residuals: np.ndarray, **_: object) -> "SplitConformal":
        scores = np.abs(np.asarray(residuals, dtype=float))
        self.q_ = conformal_quantile(scores, self.alpha)
        self.n_ = int(np.isfinite(scores).sum())
        return self

    def predict(self, point: np.ndarray, **_: object) -> Intervals:
        point = np.asarray(point, dtype=float)
        radius = np.full(point.shape, self.q_)
        return Intervals(self.name, self.alpha, point - radius, point + radius, radius)


@dataclass
class NormalizedConformal:
    """Conformal on residuals scaled by a predicted error magnitude.

    The score is ``|y - yhat| / sigma(x)`` where ``sigma`` is a second model
    trained to predict the absolute residual. The interval then widens where
    the forecast is expected to be hard, which is the adaptivity split
    conformal lacks, while keeping the marginal guarantee intact -- the score
    is still exchangeable, it is just a different score.

    ``floor`` matters: if ``sigma(x)`` is ever near zero the score explodes,
    one calibration point dominates the quantile, and every interval inflates.
    Flooring it at a fraction of the mean keeps one quiet block from setting
    the width of the whole year.
    """

    alpha: float = 0.1
    name: str = "normalized conformal"
    floor_fraction: float = 0.1
    q_: float = float("inf")
    floor_: float = 1.0

    def fit(
        self, residuals: np.ndarray, *, sigma: np.ndarray | None = None, **_: object
    ) -> "NormalizedConformal":
        residuals = np.asarray(residuals, dtype=float)
        if sigma is None:
            raise ValueError("normalized conformal needs a sigma estimate")
        sigma = np.asarray(sigma, dtype=float)
        self.floor_ = max(float(np.nanmean(sigma)) * self.floor_fraction, 1e-6)
        scaled = np.abs(residuals) / np.maximum(sigma, self.floor_)
        self.q_ = conformal_quantile(scaled, self.alpha)
        return self

    def predict(
        self, point: np.ndarray, *, sigma: np.ndarray | None = None, **_: object
    ) -> Intervals:
        point = np.asarray(point, dtype=float)
        if sigma is None:
            raise ValueError("normalized conformal needs a sigma estimate")
        radius = self.q_ * np.maximum(np.asarray(sigma, dtype=float), self.floor_)
        return Intervals(self.name, self.alpha, point - radius, point + radius, radius)


@dataclass
class MondrianConformal:
    """A separate conformal quantile per group: conditional coverage.

    Groups are whatever the caller says they are -- block of day, season, day
    kind, or a combination. Each group gets its own quantile from its own
    calibration residuals, so each group's coverage is guaranteed separately
    rather than on average.

    The cost is arithmetic: 96 block groups split 10,000 calibration points
    into 104 each, and at 90% coverage a group needs at least 9 points to say
    anything at all. ``min_group`` is the floor below which a group falls back
    to the pooled quantile, which is a deliberate, stated retreat from
    conditional coverage rather than an infinite interval nobody can use.
    """

    alpha: float = 0.1
    name: str = "mondrian conformal"
    min_group: int = 20
    q_: dict[object, float] = field(default_factory=dict)
    pooled_: float = float("inf")
    small_groups_: list[object] = field(default_factory=list)

    def fit(
        self, residuals: np.ndarray, *, groups: np.ndarray | None = None, **_: object
    ) -> "MondrianConformal":
        if groups is None:
            raise ValueError("mondrian conformal needs group labels")
        scores = np.abs(np.asarray(residuals, dtype=float))
        groups = np.asarray(groups)
        self.pooled_ = conformal_quantile(scores, self.alpha)
        self.q_ = {}
        self.small_groups_ = []
        for key in np.unique(groups):
            mask = groups == key
            if int(mask.sum()) < self.min_group:
                self.small_groups_.append(key)
                self.q_[key] = self.pooled_
                continue
            self.q_[key] = conformal_quantile(scores[mask], self.alpha)
        return self

    def predict(
        self, point: np.ndarray, *, groups: np.ndarray | None = None, **_: object
    ) -> Intervals:
        if groups is None:
            raise ValueError("mondrian conformal needs group labels")
        point = np.asarray(point, dtype=float)
        radius = np.array(
            [self.q_.get(key, self.pooled_) for key in np.asarray(groups)], dtype=float
        )
        return Intervals(self.name, self.alpha, point - radius, point + radius, radius)


@dataclass
class AdaptiveConformal:
    """Online conformal that chases its own miss rate (Gibbs & Candes, 2021).

    Keeps a working level ``alpha_t`` and updates it after every observation:

        alpha_{t+1} = alpha_t + gamma * (alpha - 1{miss})

    A miss pushes ``alpha_t`` down, widening the next interval; a run of hits
    pushes it up, tightening. Over any long stretch the realised miss rate
    converges to ``alpha`` *regardless of distribution shift*, which is the
    guarantee split conformal loses the moment a heatwave arrives.

    What it gives up is the finite-sample marginal guarantee: at any single
    moment the interval can be badly wrong, and when ``alpha_t`` is driven to
    zero the interval is infinite -- which is the method telling you it has
    lost track, and is worth surfacing rather than clipping away quietly.

    ``gamma`` is the whole tuning surface. Too small and it cannot keep up
    with a seasonal change; too large and it oscillates, over-widening after
    every miss. The backtest sweeps it.
    """

    alpha: float = 0.1
    gamma: float = 0.02
    window: int = 2000
    name: str = "adaptive conformal"
    alpha_t_: float = field(default=0.0)
    history_: list[float] = field(default_factory=list, repr=False)
    alpha_trace_: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.alpha_t_ = self.alpha

    def run(
        self, actual: np.ndarray, point: np.ndarray, warmup: np.ndarray
    ) -> Intervals:
        """Walk the test set in order, updating after each block.

        ``warmup`` is the calibration residuals the run starts from; after
        that the method feeds on its own test residuals, which is what makes
        it online. The test set must be in time order -- shuffle it and the
        adaptation is meaningless.
        """
        actual = np.asarray(actual, dtype=float)
        point = np.asarray(point, dtype=float)
        scores = list(np.abs(np.asarray(warmup, dtype=float)))

        lower = np.empty(actual.size)
        upper = np.empty(actual.size)
        radius = np.empty(actual.size)
        self.alpha_t_ = self.alpha
        self.alpha_trace_ = []

        for i in range(actual.size):
            level = min(max(self.alpha_t_, 1e-6), 1.0 - 1e-6)
            # "max" rather than "inf": a run of misses drives alpha_t toward
            # zero, where no finite quantile of the window supports the level.
            # The widest observed residual is a usable answer; infinity is
            # not, and this method never had the finite-sample guarantee that
            # infinity would be protecting.
            q = conformal_quantile(
                np.asarray(scores[-self.window :]), level, fallback="max"
            )
            radius[i] = q
            lower[i] = point[i] - q
            upper[i] = point[i] + q
            missed = not (lower[i] <= actual[i] <= upper[i])
            self.alpha_t_ += self.gamma * (self.alpha - float(missed))
            self.alpha_trace_.append(self.alpha_t_)
            scores.append(abs(actual[i] - point[i]))

        return Intervals(self.name, self.alpha, lower, upper, radius)


@dataclass
class MondrianAdaptiveConformal:
    """Per-group adaptive levels: the two failures, fixed together.

    The measurement that motivates this one is in the README. Mondrian
    conformal makes coverage even across the *blocks of the day* and leaves
    it uneven across *months*; adaptive conformal does the reverse. Neither
    is surprising once stated -- Mondrian conditions on the block and nothing
    else, adaptive tracks time and nothing else -- but it means picking
    either one alone leaves a hole, and which hole you get depends on which
    axis you happened to test.

    So: run an independent adaptive level per group. Each block of day keeps
    its own ``alpha_t`` and its own window of recent scores, updated only
    when that block comes round. A block whose errors are growing widens on
    its own schedule without dragging the quiet night blocks wider with it.

    The cost is data. Ninety-six independent online learners see one
    observation each per day, so each adapts 96 times more slowly in
    wall-clock terms than a single pooled learner, and ``gamma`` has to be
    larger to compensate. Coarser groups -- six four-hour periods rather than
    96 blocks -- trade conditional resolution for adaptation speed, and the
    backtest reports both.
    """

    alpha: float = 0.1
    gamma: float = 0.05
    window: int = 400
    name: str = "mondrian-adaptive"
    alpha_t_: dict[object, float] = field(default_factory=dict, repr=False)
    scores_: dict[object, list[float]] = field(default_factory=dict, repr=False)

    def run(
        self,
        actual: np.ndarray,
        point: np.ndarray,
        warmup: np.ndarray,
        groups: np.ndarray,
        warmup_groups: np.ndarray,
    ) -> Intervals:
        actual = np.asarray(actual, dtype=float)
        point = np.asarray(point, dtype=float)
        groups = np.asarray(groups)
        warmup = np.abs(np.asarray(warmup, dtype=float))
        warmup_groups = np.asarray(warmup_groups)

        self.scores_ = {}
        pooled = list(warmup)
        for key in np.unique(warmup_groups):
            self.scores_[key] = list(warmup[warmup_groups == key])
        self.alpha_t_ = {key: self.alpha for key in np.unique(groups)}

        lower = np.empty(actual.size)
        upper = np.empty(actual.size)
        radius = np.empty(actual.size)

        for i in range(actual.size):
            key = groups[i]
            scores = self.scores_.setdefault(key, list(pooled))
            level = min(max(self.alpha_t_.get(key, self.alpha), 1e-6), 1.0 - 1e-6)
            q = conformal_quantile(
                np.asarray(scores[-self.window :]), level, fallback="max"
            )
            radius[i] = q
            lower[i] = point[i] - q
            upper[i] = point[i] + q
            missed = not (lower[i] <= actual[i] <= upper[i])
            self.alpha_t_[key] = self.alpha_t_.get(key, self.alpha) + self.gamma * (
                self.alpha - float(missed)
            )
            scores.append(abs(actual[i] - point[i]))

        return Intervals(self.name, self.alpha, lower, upper, radius)


# -- conditional coverage --------------------------------------------------


def coverage_by_group(
    actual: np.ndarray,
    intervals: Intervals,
    groups: np.ndarray,
    *,
    labels: dict[object, str] | None = None,
) -> "list[dict[str, object]]":
    """Coverage within each group: the number marginal coverage hides.

    Returned sorted by coverage so the worst-served group is first, because
    that is the one a scheduler is exposed to.
    """
    actual = np.asarray(actual, dtype=float)
    groups = np.asarray(groups)
    rows: list[dict[str, object]] = []
    for key in np.unique(groups):
        mask = groups == key
        rows.append(
            {
                "group": labels.get(key, str(key)) if labels else key,
                "n": int(mask.sum()),
                "coverage": coverage(
                    actual[mask], intervals.lower[mask], intervals.upper[mask]
                ),
                "mean_width": mean_width(
                    intervals.lower[mask], intervals.upper[mask]
                ),
            }
        )
    return sorted(rows, key=lambda row: row["coverage"])


def conditional_coverage_noise_floor(
    group_sizes: np.ndarray,
    alpha: float,
    *,
    trials: int = 4000,
    seed: int = 0,
    observed_spread: float | None = None,
) -> dict[str, float]:
    """What spread in per-group coverage looks like when nothing is wrong.

    This is the guard that keeps the project's main claim honest. Split 5,000
    test blocks across 96 block-of-day groups and each group holds ~50 points;
    the standard error on a 90% coverage estimate from 50 points is 4%, so the
    *minimum* of 96 such estimates sits around 0.80 by chance alone. Reporting
    "the worst block only gets 80% coverage" from that is reporting the
    binomial, not the method.

    So: simulate the same group sizes under perfect conditional coverage and
    return the distribution of the spread. An observed spread inside this
    band is noise; only one above it is evidence that a method's coverage
    really does vary by group.
    """
    # Sorted, so the answer depends on the multiset of group sizes and not on
    # the order they arrive in. Callers hand these over sorted by coverage,
    # which differs per method; with a fixed seed that pairs a different
    # random draw with each size and made this function return 0.18 for one
    # method and 0.27 for another on identical group sizes.
    sizes = np.sort(np.asarray(group_sizes, dtype=int))
    sizes = sizes[sizes > 0]
    if sizes.size == 0:
        return {}
    rng = np.random.default_rng(seed)
    nominal = 1.0 - alpha
    spreads = np.empty(trials)
    worst = np.empty(trials)
    gaps = np.empty(trials)
    for t in range(trials):
        covs = rng.binomial(sizes, nominal) / sizes
        spreads[t] = covs.max() - covs.min()
        worst[t] = covs.min()
        gaps[t] = np.mean(np.abs(covs - nominal))
    out = {
        "null_spread_median": float(np.median(spreads)),
        "null_spread_p95": float(np.quantile(spreads, 0.95)),
        "null_worst_median": float(np.median(worst)),
        "null_worst_p05": float(np.quantile(worst, 0.05)),
        "null_mean_abs_gap_p95": float(np.quantile(gaps, 0.95)),
    }
    if observed_spread is not None:
        # How often perfect conditional coverage would produce a spread this
        # large by itself. A threshold on p95 alone calls a spread of 0.106
        # against a 0.101 floor "uneven" with the same confidence as 0.22,
        # which is not a distinction worth hiding.
        out["spread_p_value"] = float(
            (np.sum(spreads >= observed_spread) + 1) / (trials + 1)
        )
    return out


def conditional_coverage_gap(
    actual: np.ndarray,
    intervals: Intervals,
    groups: np.ndarray,
    *,
    min_group_n: int = 100,
) -> dict[str, float]:
    """How far the worst and best groups sit from the nominal rate.

    One number for "is this interval honest everywhere, or only on average".
    A marginal coverage of 0.90 built from groups spanning 0.78 to 0.99 is a
    different product from one built from groups spanning 0.88 to 0.92, and
    the marginal figure alone cannot tell them apart.
    """
    all_rows = coverage_by_group(actual, intervals, groups)
    if not all_rows:
        return {}
    # A coverage estimate from a handful of observations is not a coverage
    # estimate. The 23:45 block appears 9 times in this dataset -- the last
    # block of the day routinely lacks the readings to form -- and its
    # "coverage" is a number from a 9-flip sample that swings between 0.78
    # and 1.00 on its own. Left in, it dominates both the observed spread and
    # the noise floor and makes every method look conditionally broken.
    rows = [row for row in all_rows if row["n"] >= min_group_n]
    excluded = len(all_rows) - len(rows)
    if not rows:
        return {"groups": 0, "groups_excluded": excluded}
    covs = np.array([row["coverage"] for row in rows], dtype=float)
    sizes = np.array([row["n"] for row in rows], dtype=int)
    nominal = 1.0 - intervals.alpha
    spread = float(np.nanmax(covs) - np.nanmin(covs))
    out = {
        "worst_group_coverage": float(np.nanmin(covs)),
        "best_group_coverage": float(np.nanmax(covs)),
        "spread": spread,
        "mean_abs_gap": float(np.nanmean(np.abs(covs - nominal))),
        "groups_below_nominal": int(np.sum(covs < nominal)),
        "groups": int(covs.size),
        "groups_excluded": excluded,
        "median_group_n": int(np.median(sizes)),
        "min_group_n": int(np.min(sizes)),
    }
    floor = conditional_coverage_noise_floor(
        sizes, intervals.alpha, observed_spread=spread
    )
    out.update(floor)
    if floor:
        # The only honest headline: is the variation bigger than the binomial
        # noise these group sizes would produce on their own?
        out["spread_beyond_noise"] = spread - floor["null_spread_p95"]
        out["conditionally_uneven"] = bool(floor.get("spread_p_value", 1.0) < 0.05)
    return out
