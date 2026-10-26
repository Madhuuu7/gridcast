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


def test_the_page_is_self_contained():
    """No CDN, no font host, no analytics: the demo has to work offline."""
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert "http://" not in html
    assert "https://" not in html
    assert "<link" not in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_console_harness_passes_on_its_fixtures():
    result = subprocess.run(
        [shutil.which("node"), str(HARNESS)],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@needs_data
@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_console_handles_a_live_forecast(client, forecast, tmp_path):
    """The shape check that the Python tests cannot do: a response the page
    cannot render is a broken page, and nothing else would catch it."""
    path = tmp_path / "forecast.json"
    path.write_text(json.dumps(forecast), encoding="utf-8")
    result = subprocess.run(
        [shutil.which("node"), str(HARNESS), "--forecast", str(path)],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
