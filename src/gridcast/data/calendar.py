"""The Indian calendar, as load sees it.

Two kinds of day move Delhi's demand curve enough that a model without them
is systematically wrong on exactly the days a scheduler cares about.

**Fixed-date national holidays** -- Republic Day, Independence Day, Gandhi
Jayanti -- are easy: they are the same date every year and they look like a
Sunday with offices shut.

**Festivals that move with the lunar calendar** are the interesting ones.
Diwali, Holi and Eid shift by days or weeks each year, and their load
signature is nothing like a Sunday: Diwali night is a lighting peak on top of
a day of reduced industrial draw, Holi flattens the morning, and the days
*around* Diwali carry a distinct shape of their own. There is no formula; the
dates are enumerated. They are listed here through 2027 and the list says
plainly what happens when it runs out, because the failure mode of a stale
holiday table is a model that silently mispredicts one day a year and nobody
notices until it is the day of the peak.

The dates are the commonly observed Delhi dates. Where a festival spans two
days (Diwali's Chhoti and Badi, Holi's Holika Dahan and Dhulandi) both are
listed, because both move load.
"""

from __future__ import annotations

from datetime import date

#: Fixed-date national holidays: (month, day).
FIXED = {
    (1, 26): "republic day",
    (8, 15): "independence day",
    (10, 2): "gandhi jayanti",
    (12, 25): "christmas",
    (5, 1): "labour day",
}

#: Moving festivals, enumerated because they follow lunar calendars. Delhi
#: observance dates.
MOVING: dict[date, str] = {
    # 2023
    date(2023, 3, 7): "holika dahan",
    date(2023, 3, 8): "holi",
    date(2023, 4, 22): "eid al-fitr",
    date(2023, 6, 29): "eid al-adha",
    date(2023, 8, 30): "raksha bandhan",
    date(2023, 9, 7): "janmashtami",
    date(2023, 9, 19): "ganesh chaturthi",
    date(2023, 10, 24): "dussehra",
    date(2023, 11, 12): "diwali",
    date(2023, 11, 13): "govardhan puja",
    date(2023, 11, 27): "guru nanak jayanti",
    # 2024
    date(2024, 3, 24): "holika dahan",
    date(2024, 3, 25): "holi",
    date(2024, 3, 29): "good friday",
    date(2024, 4, 11): "eid al-fitr",
    date(2024, 6, 17): "eid al-adha",
    date(2024, 8, 19): "raksha bandhan",
    date(2024, 8, 26): "janmashtami",
    date(2024, 9, 7): "ganesh chaturthi",
    date(2024, 10, 12): "dussehra",
    date(2024, 10, 31): "diwali",
    date(2024, 11, 1): "govardhan puja",
    date(2024, 11, 15): "guru nanak jayanti",
    # 2025
    date(2025, 3, 13): "holika dahan",
    date(2025, 3, 14): "holi",
    date(2025, 3, 31): "eid al-fitr",
    date(2025, 4, 18): "good friday",
    date(2025, 6, 7): "eid al-adha",
    date(2025, 8, 9): "raksha bandhan",
    date(2025, 8, 16): "janmashtami",
    date(2025, 8, 27): "ganesh chaturthi",
    date(2025, 10, 2): "dussehra",
    date(2025, 10, 20): "diwali",
    date(2025, 10, 22): "govardhan puja",
    date(2025, 11, 5): "guru nanak jayanti",
    # 2026
    date(2026, 3, 3): "holika dahan",
    date(2026, 3, 4): "holi",
    date(2026, 3, 20): "eid al-fitr",
    date(2026, 4, 3): "good friday",
    date(2026, 5, 27): "eid al-adha",
    date(2026, 8, 28): "raksha bandhan",
    date(2026, 9, 4): "janmashtami",
    date(2026, 9, 14): "ganesh chaturthi",
    date(2026, 10, 20): "dussehra",
    date(2026, 11, 8): "diwali",
    date(2026, 11, 10): "govardhan puja",
    date(2026, 11, 24): "guru nanak jayanti",
    # 2027
    date(2027, 3, 21): "holi",
    date(2027, 3, 10): "eid al-fitr",
    date(2027, 5, 17): "eid al-adha",
    date(2027, 10, 9): "dussehra",
    date(2027, 10, 29): "diwali",
}

#: The last date the moving table covers. Past this the calendar is guessing,
#: and callers are told rather than silently given "ordinary Tuesday".
COVERED_UNTIL = max(MOVING)


def holiday_name(day: date) -> str | None:
    """The holiday on this date, or None. Fixed dates win ties."""
    fixed = FIXED.get((day.month, day.day))
    if fixed:
        return fixed
    return MOVING.get(day)


def is_holiday(day: date) -> bool:
    return holiday_name(day) is not None


def is_festival_window(day: date, *, days: int = 1) -> bool:
    """Whether the date is within ``days`` of a moving festival.

    Diwali week is not one day. Industrial load falls for several days around
    it and residential lighting rises, so the day before and after carry a
    shape of their own -- distinct from both an ordinary day and from the
    festival itself.
    """
    from datetime import timedelta

    for offset in range(-days, days + 1):
        if (day + timedelta(days=offset)) in MOVING:
            return True
    return False


def day_kind(day: date) -> str:
    """One label per day, which is what the model actually consumes.

    Deliberately coarse. A separate indicator per festival would be a column
    with four or five examples in three years of data, which is a licence to
    overfit; "a festival happened here" is a claim the data can support.
    """
    if is_holiday(day):
        return "holiday"
    if is_festival_window(day, days=1):
        return "near-festival"
    if day.weekday() == 6:
        return "sunday"
    if day.weekday() == 5:
        return "saturday"
    return "weekday"


def coverage_warning(last_day: date) -> str | None:
    """A message if the data runs past what the holiday table knows."""
    if last_day <= COVERED_UNTIL:
        return None
    return (
        f"the holiday calendar is enumerated only to {COVERED_UNTIL}; "
        f"days after that are treated as ordinary, which will be wrong on "
        f"roughly a dozen days a year -- extend MOVING in data/calendar.py"
    )
