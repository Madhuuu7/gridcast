"""Forecast error in rupees, and an honest account of what that number is not.

"MAE 251 MW" and "Winkler 1067" are the right units for comparing methods and
the wrong units for explaining why anyone should care. A state utility that
deviates from its declared schedule is settled per fifteen-minute block under
the Deviation Settlement Mechanism, so the error has a price, and the price is
what makes the interval a product rather than a plot.

## The model

Per block, with ``deviation = actual - scheduled`` in MW and a block lasting a
quarter hour:

- Deviation up to a **volume limit** -- the lower of 12% of schedule and
  150 MW, under the CERC DSM Regulations 2014 as they applied to a state
  drawee -- is settled at the normal rate.
- **Over-drawal beyond** that limit attracts an additional charge, because
  over-drawal is what threatens the grid.
- **Under-drawal beyond** the limit is not charged. Drawing less than you
  asked for is not a security problem, and the asymmetry is deliberate in the
  regulation.
- The normal rate is **capped at Rs 12/kWh** by the CERC's February 2023
  guidelines, after ancillary-service costs pushed some blocks past Rs 40.
- Frequency modulates the rate: over-drawal is dearer when the grid is slow.
  The multipliers here follow the 2022 regime's operative band -- 150% of
  normal between 49.90 and 49.95 Hz, 200% below 49.90, 75% between 50.03 and
  50.05.

## What this number is not

**It is an upper bound, and knowingly so.** Three reasons, each of which cuts
the real figure:

1. **A real utility revises its schedule intra-day.** The day-ahead schedule
   is not what gets settled; it is revised as the day approaches and the
   error being priced here is the error of a forecast nobody is obliged to
   stand behind for 38 hours. This is the largest of the three.
2. **Frequency is not fetched.** Settlement needs the block's average grid
   frequency and this project has no source for it, so the default is the
   normal rate with no slab applied. ``frequency_hz`` exists so the
   sensitivity can be shown rather than assumed away.
3. **The regulation has moved.** The 2024 draft replaces the 12%/150 MW cap
   with an operative band and a 10%/100 MW limit. Which regime applies to a
   given year is a question this module takes as a parameter rather than
   answering.

So the absolute rupees are not a bill. **The ratio between two methods is the
number to read**, because all three distortions apply equally to both, and a
model that halves the deviation energy halves the charge under any of these
regimes.

Every parameter is a field with a documented default rather than a constant
buried in a formula, so a reader who knows the regulation better can correct
it in one place and re-derive every figure in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Hours in a fifteen-minute scheduling block. MW times this is MWh.
BLOCK_HOURS = 0.25

#: Rupees per crore and per lakh, for reporting.
CRORE = 1e7
LAKH = 1e5


@dataclass(frozen=True)
class DeviationSettlement:
    """The charge model, as parameters rather than as a formula.

    Defaults follow the CERC DSM Regulations 2014 volume limit as applied to
    a state drawee, with the normal rate at the February 2023 cap. See the
    module docstring for what each is and why the absolute figure is an
    upper bound.
    """

    #: Rupees per kWh. The default is the CERC's February 2023 **cap** on the
    #: normal rate, not a typical settled rate -- at 50.00 Hz the applicable
    #: charge tracks a market-linked reference that has generally run well
    #: below it. The cap is used because it is the one figure with a hard
    #: citation, and because an upper bound that is labelled as one is more
    #: useful than a plausible number that is not sourced.
    normal_rate: float = 12.0
    #: Volume limit, the lower of this fraction of schedule and the MW cap.
    volume_limit_fraction: float = 0.12
    volume_limit_mw: float = 150.0
    #: Multiplier on the normal rate for over-drawal beyond the volume limit.
    excess_over_drawal_multiplier: float = 2.0
    #: Under-drawal beyond the limit is not charged: drawing less than you
    #: asked for is not a grid security problem.
    charge_excess_under_drawal: bool = False

    def __post_init__(self) -> None:
        if self.normal_rate <= 0:
            raise ValueError(f"normal_rate must be positive, got {self.normal_rate}")
        if not 0.0 < self.volume_limit_fraction <= 1.0:
            raise ValueError(
                f"volume_limit_fraction is a fraction of schedule, got "
                f"{self.volume_limit_fraction}"
            )
        if self.volume_limit_mw <= 0:
            raise ValueError("volume_limit_mw must be positive")

    def frequency_multiplier(self, frequency_hz: float | None) -> float:
        """The slab multiplier for a block's average grid frequency.

        ``None`` means no frequency data, which is the honest default here --
        the alternative is picking a slab and calling the result a bill.
        """
        if frequency_hz is None or not np.isfinite(frequency_hz):
            return 1.0
        if frequency_hz < 49.90:
            return 2.0
        if frequency_hz < 49.95:
            return 1.5
        if frequency_hz <= 50.03:
            return 1.0
        if frequency_hz <= 50.05:
            return 0.75
        return 0.75

    def volume_limit(self, scheduled_mw: np.ndarray) -> np.ndarray:
        """The permissible deviation per block, in MW."""
        scheduled = np.abs(np.asarray(scheduled_mw, dtype=float))
        return np.minimum(scheduled * self.volume_limit_fraction, self.volume_limit_mw)

    def charge(
        self,
        scheduled_mw: np.ndarray,
        actual_mw: np.ndarray,
        *,
        frequency_hz: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Rupees payable per block.

        Positive deviation is over-drawal: the grid supplied more than was
        declared. The sign matters, because the regulation treats the two
        directions differently and a model that takes an absolute value
        before pricing has thrown away the asymmetry that makes
        under-forecasting worse than over-forecasting.
        """
        scheduled = np.asarray(scheduled_mw, dtype=float)
        actual = np.asarray(actual_mw, dtype=float)
        if scheduled.shape != actual.shape:
            raise ValueError(
                f"scheduled and actual must align, got {scheduled.shape} and "
                f"{actual.shape}"
            )
        deviation = actual - scheduled
        limit = self.volume_limit(scheduled)

        within = np.clip(np.abs(deviation), 0.0, limit)
        beyond = np.maximum(np.abs(deviation) - limit, 0.0)
        over_drawing = deviation > 0

        excess = np.where(
            over_drawing,
            beyond * self.excess_over_drawal_multiplier,
            beyond if self.charge_excess_under_drawal else 0.0,
        )

        if isinstance(frequency_hz, np.ndarray):
            multiplier = np.array(
                [self.frequency_multiplier(float(f)) for f in frequency_hz]
            )
        else:
            multiplier = self.frequency_multiplier(frequency_hz)

        # MW -> MWh -> kWh, then rupees.
        energy_kwh = (within + excess) * BLOCK_HOURS * 1000.0
        charge = energy_kwh * self.normal_rate * multiplier
        # NaN, not zero. An unbounded interval edge and a missing reading
        # both arrive here as non-finite, and "no number" priced as "no
        # charge" makes an interval that has lost track look free. Callers
        # use nansum, or test finiteness and exclude the block.
        return np.where(np.isfinite(charge), charge, np.nan)


def annualise(rupees_per_block: np.ndarray, n_blocks: int) -> float:
    """Scale a backtest's total charge to a year.

    A year is 35,040 blocks. Scaling rather than summing, because the folds
    deliberately do not cover a contiguous year -- they are spread across
    three, so that an interval is tested against a change of season.
    """
    total = float(np.nansum(rupees_per_block))
    if n_blocks <= 0:
        return 0.0
    return total * (365 * 96) / n_blocks


def cost_summary(
    scheduled_mw: np.ndarray,
    actual_mw: np.ndarray,
    *,
    settlement: DeviationSettlement | None = None,
    frequency_hz: float | np.ndarray | None = None,
) -> dict[str, float]:
    """Per-block and annualised cost of a point forecast's error."""
    settlement = settlement or DeviationSettlement()
    charge = settlement.charge(scheduled_mw, actual_mw, frequency_hz=frequency_hz)
    deviation = np.asarray(actual_mw, dtype=float) - np.asarray(
        scheduled_mw, dtype=float
    )
    limit = settlement.volume_limit(scheduled_mw)
    deviation_mwh = float(np.nansum(np.abs(deviation)) * BLOCK_HOURS)
    return {
        "blocks": int(charge.size),
        # Rupees per MWh of deviation energy. The figure that survives the
        # volume limit saturating: once both forecasts are routinely past
        # the 150 MW band, the within-band charge is identical for both and
        # only this rate separates them.
        "rupees_per_mwh_deviation": (
            float(np.nansum(charge)) / deviation_mwh if deviation_mwh else 0.0
        ),
        "rupees_total": float(np.nansum(charge)),
        "rupees_per_block": float(np.nanmean(charge)),
        "rupees_per_year": annualise(charge, charge.size),
        "crore_per_year": annualise(charge, charge.size) / CRORE,
        "blocks_over_drawing": float(np.mean(deviation > 0)),
        "blocks_beyond_limit": float(np.mean(np.abs(deviation) > limit)),
        "mean_abs_deviation_mw": float(np.nanmean(np.abs(deviation))),
    }


def budget_calibration(
    lower_mw: np.ndarray,
    upper_mw: np.ndarray,
    actual_mw: np.ndarray,
    scheduled_mw: np.ndarray,
    *,
    settlement: DeviationSettlement | None = None,
) -> dict[str, float]:
    """Does a 90% interval on load give a 90% interval on the bill?

    This is what the interval is *for*, in rupees. The schedule is the point
    forecast, so the interval does not change the charge -- it changes what
    the scheduler knew about the charge in advance. An interval that covers
    90% of outcomes but whose implied cost range misses the realised cost
    half the time is useless for budgeting, and nothing in coverage or
    Winkler would say so.

    The implied range is the cost at each end of the interval. Note that
    cost is not monotone in load: the cheapest outcome is the schedule
    itself, so when the interval straddles the schedule the lower bound of
    the cost range is zero rather than the cost at the interval's lower
    edge. Getting that wrong reports a floor on the bill that no outcome
    would have produced.
    """
    settlement = settlement or DeviationSettlement()
    lower = np.asarray(lower_mw, dtype=float)
    upper = np.asarray(upper_mw, dtype=float)
    actual = np.asarray(actual_mw, dtype=float)
    scheduled = np.asarray(scheduled_mw, dtype=float)

    at_lower = settlement.charge(scheduled, lower)
    at_upper = settlement.charge(scheduled, upper)
    realised = settlement.charge(scheduled, actual)

    straddles = (lower <= scheduled) & (scheduled <= upper)
    cost_floor = np.where(straddles, 0.0, np.minimum(at_lower, at_upper))
    cost_ceiling = np.maximum(at_lower, at_upper)

    usable = np.isfinite(cost_floor) & np.isfinite(cost_ceiling)
    inside = (realised >= cost_floor - 1e-6) & (realised <= cost_ceiling + 1e-6)
    return {
        "cost_coverage": float(np.mean(inside[usable])) if usable.any() else float("nan"),
        "blocks_scored": int(usable.sum()),
        "blocks_unbounded": int((~usable).sum()),
        "mean_cost_ceiling": float(np.nanmean(cost_ceiling[usable])),
        "mean_realised_cost": float(np.nanmean(realised[usable])),
        "worst_underestimate": float(
            np.nanmax(np.where(inside, 0.0, realised - cost_ceiling)[usable])
        )
        if usable.any()
        else float("nan"),
    }
