"""The HTTP surface, and the property that keeps the demo honest.

The service fits its model once at startup and will only forecast dates
*after* the calibration window. That is not a caching detail -- it is the
thing that makes the page mean anything. A service that refitted on all
available data per request would draw an interval around a day the model had
already memorised, it would cover every time, and the chart would be a lie.

``test_a_date_inside_the_training_window_is_refused`` is therefore the test
that matters here; the rest is contract.

These tests need the cached SLDC data, so they skip rather than fail when it
is absent -- a fresh clone has no data until ``scripts/fetch_data.py`` runs,
and a test suite that fails on a fresh clone teaches people to ignore it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gridcast.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw"
HARNESS = ROOT / "scripts" / "check_console.mjs"

needs_data = pytest.mark.skipif(
    not (CACHE / "sldc").is_dir() or not any((CACHE / "sldc").glob("*.csv")),
    reason="no cached SLDC data; run scripts/fetch_data.py",
)


@pytest.fixture(scope="module")
def client():
    app = create_app(CACHE)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def health(client):
    return client.get("/api/health").json()


# -- the window ------------------------------------------------------------


@needs_data
def test_health_reports_all_three_boundaries(health):
    """Fit, calibrate and serve, published so the arrangement can be checked
    from outside rather than taken on trust."""
    assert health["status"] == "ok", health.get("error")
    fitted = date.fromisoformat(health["fitted_up_to"])
    calibrated = date.fromisoformat(health["calibrated_up_to"])
    servable_from = date.fromisoformat(health["servable"]["from"])
    assert fitted < calibrated <= servable_from
    assert health["train_rows"] > 10_000
    assert health["calibration_rows"] > 500


@needs_data
def test_a_date_inside_the_training_window_is_refused(client, health):
    """The test this file exists for. A forecast for a day the model trained
    on would cover every block and prove nothing."""
    inside = date.fromisoformat(health["fitted_up_to"]) - timedelta(days=30)
    response = client.get(f"/api/forecast?date={inside}")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "outside the servable window" in detail
    assert "already seen" in detail


@needs_data
def test_a_date_inside_the_calibration_window_is_refused(client, health):
    """Calibration days are out of the model's training set but their
    residuals set the interval width, so serving them would still flatter the
    coverage."""
    fitted = date.fromisoformat(health["fitted_up_to"])
    calibrated = date.fromisoformat(health["calibrated_up_to"])
    midway = fitted + (calibrated - fitted) / 2
    assert client.get(f"/api/forecast?date={midway}").status_code == 422


@needs_data
def test_a_date_beyond_the_data_is_refused(client, health):
    beyond = date.fromisoformat(health["data"]["end"]) + timedelta(days=5)
    assert client.get(f"/api/forecast?date={beyond}").status_code == 422


@needs_data
def test_every_offered_day_can_actually_be_forecast(client):
    """The console populates its dropdown from /api/days, so a date listed
    there that 422s or 404s is a dead control."""
    days = client.get("/api/days").json()
    assert len(days) > 10
    for entry in (days[0], days[len(days) // 2], days[-1]):
        response = client.get(f"/api/forecast?date={entry['date']}")
        assert response.status_code == 200, (entry, response.json())


@needs_data
def test_a_malformed_date_is_rejected(client):
    assert client.get("/api/forecast?date=not-a-date").status_code == 422
    assert client.get("/api/forecast").status_code == 422


# -- the forecast ----------------------------------------------------------


@pytest.fixture(scope="module")
def forecast(client):
    days = client.get("/api/days").json()
    if not days:
        pytest.skip("no servable days")
    return client.get(f"/api/forecast?date={days[len(days) // 2]['date']}").json()


@needs_data
def test_a_forecast_covers_the_scheduling_day(forecast):
    n = len(forecast["point"])
    assert 80 <= n <= 96  # the archive has gaps; a short day is real
    assert len(forecast["actual"]) == n
    assert len(forecast["times"]) == n
    assert forecast["times"][0] == "00:00"
    assert forecast["issued"].endswith("10:00")


@needs_data
def test_the_forecast_is_in_the_right_units_and_the_right_shape(forecast):
    """Delhi draws between about 1,400 and 8,700 MW. A forecast outside that
    is not a small error, it is a units or alignment bug."""
    for value in forecast["point"]:
        assert 1000 < value < 10_000
    assert forecast["metrics"]["mape"] < 15
    assert forecast["metrics"]["mae"] < 1500


@needs_data
def test_every_method_returns_a_band_of_the_right_length(forecast):
    assert set(forecast["intervals"]) >= {
        "gaussian", "split conformal", "mondrian conformal", "adaptive conformal"
    }
    n = len(forecast["point"])
    for name, band in forecast["intervals"].items():
        assert len(band["lower"]) == n, name
        assert len(band["upper"]) == n, name
        assert 0.0 <= band["coverage"] <= 1.0, name
        assert band["mean_width"] > 0, name
        for lo, hi, point in zip(band["lower"], band["upper"], forecast["point"]):
            if lo is not None and hi is not None:
                assert lo <= point <= hi, name


@needs_data
def test_the_mondrian_band_varies_across_the_day_and_the_static_ones_do_not(
    forecast,
):
    """How the methods differ, visible in one response: a single pooled
    quantile is flat by construction, a per-block one is not."""
    def width(name):
        band = forecast["intervals"][name]
        return [hi - lo for lo, hi in zip(band["lower"], band["upper"])
                if lo is not None and hi is not None]

    assert len(set(round(w, 6) for w in width("split conformal"))) == 1
    assert len(set(round(w, 6) for w in width("mondrian conformal"))) > 5


@needs_data
def test_the_response_is_valid_json_without_infinities(client, forecast):
    """An adaptive method can report an unbounded interval, and `Infinity` is
    not valid JSON -- JSON.parse rejects it outright."""
    raw = json.dumps(forecast)
    assert "Infinity" not in raw
    assert "NaN" not in raw


@needs_data
def test_a_holiday_is_labelled_as_one(client):
    """The console shows the day kind, and a festival is exactly when the
    forecast is worth looking at."""
    days = client.get("/api/days").json()
    kinds = {entry["kind"] for entry in days}
    assert kinds <= {"weekday", "saturday", "sunday", "holiday", "near-festival"}
    assert "weekday" in kinds


# -- the rest --------------------------------------------------------------


@needs_data
def test_the_summary_describes_the_whole_record(client):
    facts = client.get("/api/summary").json()
    assert facts["days"] > 900
    assert 0.9 < facts["completeness"] <= 1.0
    assert 1000 < facts["min_mw"] < facts["mean_mw"] < facts["max_mw"] < 10_000


@needs_data
def test_the_console_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@needs_data
def test_the_openapi_schema_describes_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/api/health", "/api/summary", "/api/days", "/api/forecast", "/api/backtest"):
        assert path in paths


def test_the_console_is_self_contained():
    """No CDN, no font host, no analytics: the demo has to work offline.

    This held for the hand-written page it replaced and nearly did not
    survive the rebuild -- a Google Fonts link is the obvious way to get a
    typeface and it costs a third-party round trip on exactly the cold
    free-tier start where the page is slowest. The faces are served from
    this origin instead, so the check is on the *built* bundle rather than
    on the source, which is where a stray CDN reference would actually
    reach a visitor.
    """
    built = ROOT / "web" / "dist" / "index.html"
    if not built.is_file():
        pytest.skip("console not built; run npm run build in web/")
    html = built.read_text(encoding="utf-8")
    for origin in ("http://", "https://"):
        # The inline SVG favicon carries an xmlns URL, which is a namespace
        # identifier and not a fetch.
        offenders = [
            line for line in html.splitlines()
            if origin in line and "www.w3.org/2000/svg" not in line
        ]
        assert not offenders, offenders

    assets = ROOT / "web" / "dist" / "assets"
    for sheet in assets.glob("*.css"):
        text = sheet.read_text(encoding="utf-8")
        assert "fonts.googleapis" not in text
        assert "fonts.gstatic" not in text
        assert "@font-face" in text, "the typefaces must be declared locally"

    fonts = sorted((ROOT / "web" / "dist" / "fonts").glob("*.woff2"))
    assert fonts, "no self-hosted fonts were emitted"
    total = sum(f.stat().st_size for f in fonts)
    # Latin subsets only. A full set would be several hundred kB on a page
    # whose whole bundle is under 250.
    assert total < 200_000, f"{total} bytes of fonts is more than latin needs"


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm is not installed")
def test_the_console_logic_is_tested_on_its_own():
    """The console's arithmetic has its own suite; run it from here too.

    The page this replaced was checked by a bespoke node harness that drove
    its DOM. The rebuilt console is a component tree, so the part worth
    testing -- reading a band, choosing axis ticks, the colour ramp -- was
    pulled into plain functions and tested directly. Running it from the
    Python suite keeps one command that checks the whole project.
    """
    if not (ROOT / "web" / "node_modules").is_dir():
        pytest.skip("console dependencies not installed; run npm install in web/")
    result = subprocess.run(
        [shutil.which("npm"), "test", "--silent"],
        capture_output=True, text=True, cwd=ROOT / "web", timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@needs_data
def test_the_forecast_response_is_shaped_the_way_the_console_reads_it(forecast):
    """A response the page cannot render is a broken page.

    The console reads metrics from a nested object and the day kind from
    `day_kind`; an earlier version of it guessed both wrong and rendered a
    52% skill figure as 1% and the error as an em dash. Nothing in the
    Python tests would have caught either, because the service was right.
    """
    assert {"date", "day_kind", "blocks", "times", "actual", "point", "intervals",
            "metrics", "alpha"} <= set(forecast)
    assert {"mape", "mae", "peak_error"} <= set(forecast["metrics"])
    # The method the README argues for has to be among them, or the console
    # can show every interval except the one the project is about.
    assert "localized conformal" in forecast["intervals"]
    n = len(forecast["point"])
    for name, band in forecast["intervals"].items():
        assert len(band["lower"]) == n, name
        assert len(band["upper"]) == n, name


@needs_data
def test_the_calibration_window_fills_even_where_the_archive_is_thin():
    """Measured in days, this window collapsed wherever the data was sparse.

    The August-September 2026 gap left a 45-day window holding 1,397 of its
    4,320 blocks -- 14 per block group against a Mondrian minimum of 20 --
    so every group fell back to the pooled quantile and the per-block
    method silently became split conformal. Nothing raised; the bands just
    stopped varying. Counting rows instead makes a thin stretch reach
    further back in time rather than hold less data.
    """
    from gridcast.api.app import ForecastService

    service = ForecastService(CACHE)
    service.warm()
    if not service.ready:
        pytest.skip(service.error or "service not ready")
    assert service.cal_n >= 4000
    assert service.methods["mondrian conformal"].small_groups_ == []
    assert service.cal_from < service.serve_from
