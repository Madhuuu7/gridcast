"""The running record: what was promised, what happened, and the gap.

Everything else in this project measures itself on a backtest. This module
measures the one thing a backtest structurally cannot, because a backtest is
scored by the person who chose its folds. Here the forecast rows were written
before the observation rows existed, so the only decisions available after
the fact are which days to include -- and the answer to that is all of them.

Three rules keep it from drifting back into a backtest:

**The earliest issuance counts.** If a day was forecast twice, the second was
made with more information, even when that information was only that the
first looked wrong. ``first_run`` is what the scorecard reads.

**A day with partial observations is reported as partial.** Not dropped, not
silently averaged over the blocks that happen to exist. SLDC publishes
incomplete days, and a scorecard that quietly skips them is a scorecard that
skips the days the grid had trouble.

**Coverage is reported with its own uncertainty.** One day is 96 blocks, so a
90% coverage estimate from a single day carries a standard error of about
three points. Printing "87.5%" for yesterday as though it were a measurement
is the same error this project spends a whole module avoiding in the
backtest, and the fix is the same: say what the number could be by chance.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
from sqlalchemy.orm import Session

from .settlement import DeviationSettlement, budget_calibration, cost_summary
from .store.models import BLOCKS_PER_DAY
from .store.repository import first_run, issued_days, observations_for


@dataclass(frozen=True)
class DayScore:
    """One target day, scored against what the grid actually drew."""

    target_date: date
    method: str
    model: str
    issued_at: str
    code_version: str
    #: Blocks with both a forecast and an observation.
    scored: int
    #: Blocks forecast but never observed. Reported, not hidden.
    missing: int
    coverage: float | None
    #: Half-width of a 95% Wald interval on that coverage, in points. One
    #: day cannot measure coverage to better than about +/-6 points, and a
    #: console that shows 87.5% without this invites a conclusion the data
    #: does not support.
    coverage_se: float | None
    mae_mw: float | None
    mape: float | None
    mean_width_mw: float | None
    rupees: float | None
    unbounded_blocks: int = 0

    def as_dict(self) -> dict:
        out = asdict(self)
        out["target_date"] = self.target_date.isoformat()
        return out


@dataclass
class Record:
    """Every scored day, and the totals that only accumulate."""

    days: list[DayScore] = field(default_factory=list)

    @property
    def scored_blocks(self) -> int:
        return sum(day.scored for day in self.days)

    def totals(self) -> dict[str, float | int | None]:
        """Pooled over every block on record, not averaged over days.

        A day with 96 scored blocks and a day with 4 are not equal evidence,
        and averaging the per-day coverages would treat them as though they
        were. Pooling also means the standard error shrinks as the record
        grows, which is the entire reason for keeping one.
        """
        scored = self.scored_blocks
        if not scored:
            # Days issued but not yet settled still count as days. Reporting
            # zero here hid a forecast that had been published and was
            # waiting for its outcome, which is precisely the state the
            # record spends most of its life in.
            return {
                "days": len(self.days), "blocks": 0, "coverage": None,
                "coverage_se": None, "mae_mw": None, "rupees": 0.0,
                "unbounded_blocks": 0,
            }
        covered = sum(
            (day.coverage or 0.0) * day.scored for day in self.days if day.coverage is not None
        )
        coverage = covered / scored
        weighted_mae = sum(
            (day.mae_mw or 0.0) * day.scored for day in self.days if day.mae_mw is not None
        )
        return {
            "days": len(self.days),
            "blocks": scored,
            "coverage": coverage,
            "coverage_se": _wald_half_width(coverage, scored),
            "mae_mw": weighted_mae / scored,
            "rupees": float(sum(day.rupees or 0.0 for day in self.days)),
            "unbounded_blocks": sum(day.unbounded_blocks for day in self.days),
        }

    def as_dict(self) -> dict:
        return {"totals": self.totals(), "days": [d.as_dict() for d in self.days]}


def _wald_half_width(p: float, n: int, z: float = 1.96) -> float | None:
    """Half-width of a 95% interval on a proportion, in percentage points.

    Wald rather than Wilson, and only because it is the one a reader can
    reproduce in their head from the two numbers printed beside it. It is
    poor near 0 and 1; coverage near 0.9 on 96 or more blocks is not near
    either.
    """
    if n <= 0:
        return None
    return z * math.sqrt(max(p * (1.0 - p), 0.0) / n)


def score_day(
    session: Session,
    target_date: date,
    *,
    method: str | None = None,
    settlement: DeviationSettlement | None = None,
) -> DayScore | None:
    """Score one issued day against the observations now on record."""
    run = first_run(session, target_date, method)
    if run is None:
        return None
    observed = observations_for(session, target_date)
    blocks = sorted(run.blocks, key=lambda b: b.block)

    point: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    actual: list[float] = []
    unbounded = 0
    for block in blocks:
        if block.block not in observed:
            continue
        point.append(block.point_mw)
        actual.append(observed[block.block])
        if block.lower_mw is None or block.upper_mw is None:
            unbounded += 1
            lower.append(-np.inf)
            upper.append(np.inf)
        else:
            lower.append(block.lower_mw)
            upper.append(block.upper_mw)

    scored = len(point)
    common = dict(
        target_date=target_date,
        method=run.method,
        model=run.model,
        issued_at=run.issued_at.isoformat(),
        code_version=run.code_version,
        scored=scored,
        missing=BLOCKS_PER_DAY - scored,
        unbounded_blocks=unbounded,
    )
    if scored == 0:
        return DayScore(
            **common, coverage=None, coverage_se=None, mae_mw=None,
            mape=None, mean_width_mw=None, rupees=None,
        )

    point_a = np.array(point)
    actual_a = np.array(actual)
    lower_a = np.array(lower)
    upper_a = np.array(upper)

    inside = (actual_a >= lower_a) & (actual_a <= upper_a)
    coverage = float(np.mean(inside))
    width = upper_a - lower_a
    finite_width = width[np.isfinite(width)]
    charge = (settlement or DeviationSettlement()).charge(point_a, actual_a)

    return DayScore(
        **common,
        coverage=coverage,
        coverage_se=_wald_half_width(coverage, scored),
        mae_mw=float(np.mean(np.abs(actual_a - point_a))),
        mape=float(np.mean(np.abs(actual_a - point_a) / actual_a)),
        mean_width_mw=float(np.mean(finite_width)) if finite_width.size else None,
        rupees=float(np.nansum(charge)),
    )


def build_record(
    session: Session,
    *,
    method: str | None = None,
    settlement: DeviationSettlement | None = None,
) -> Record:
    """Score every day that has an issued forecast, oldest first."""
    scores = []
    for day in issued_days(session, method):
        score = score_day(session, day, method=method, settlement=settlement)
        if score is not None:
            scores.append(score)
    return Record(days=scores)


def interval_value(
    session: Session,
    *,
    method: str | None = None,
    settlement: DeviationSettlement | None = None,
) -> dict[str, float]:
    """Does the published interval bracket the published bill?

    The scorecard's version of ``budget_calibration``: pooled over every
    block on record rather than over a backtest, and therefore the only
    measurement here that nobody could have tuned.
    """
    settlement = settlement or DeviationSettlement()
    point: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    actual: list[float] = []
    for day in issued_days(session, method):
        run = first_run(session, day, method)
        if run is None:
            continue
        observed = observations_for(session, day)
        for block in run.blocks:
            if block.block not in observed:
                continue
            point.append(block.point_mw)
            actual.append(observed[block.block])
            lower.append(-np.inf if block.lower_mw is None else block.lower_mw)
            upper.append(np.inf if block.upper_mw is None else block.upper_mw)
    if not point:
        return {"cost_coverage": float("nan"), "blocks_scored": 0}
    return budget_calibration(
        np.array(lower), np.array(upper), np.array(actual), np.array(point),
        settlement=settlement,
    )


def cost_of_record(
    session: Session,
    *,
    method: str | None = None,
    settlement: DeviationSettlement | None = None,
) -> dict[str, float]:
    """What the published schedules would have been settled at."""
    point: list[float] = []
    actual: list[float] = []
    for day in issued_days(session, method):
        run = first_run(session, day, method)
        if run is None:
            continue
        observed = observations_for(session, day)
        for block in run.blocks:
            if block.block in observed:
                point.append(block.point_mw)
                actual.append(observed[block.block])
    if not point:
        return {"blocks": 0, "rupees_total": 0.0}
    return cost_summary(np.array(point), np.array(actual), settlement=settlement)
