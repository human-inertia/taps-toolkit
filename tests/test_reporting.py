"""Tests for TAPS reporting/export module."""

import os
import json
import csv
import tempfile
from dataclasses import dataclass, field
from taps.reporting.export import export_json, export_csv, export_research_bundle
from taps.assessment.parameters import TAPSParameters


@dataclass
class MockTapEvent:
    channel: str = "INDEX"
    onset_ms: float = 100.0
    offset_ms: float = 180.0
    duration_ms: float = 80.0
    peak_magnitude: float = 1.5
    secondary_channels: dict = field(default_factory=dict)


def make_sample_data():
    params = TAPSParameters(
        tap_count=40,
        iti_mean_ms=250.0,
        iti_sd_ms=25.0,
        iti_cv=0.1,
        tap_duration_mean_ms=80.0,
        freezing_count=1,
        coactivation_index=0.15,
        bilateral_asymmetry=0.05,
        fatigue_slope=2.3,
        rhythm_entropy=1.8,
        epoch_start_ms=0.0,
        epoch_end_ms=10000.0,
        epoch_duration_s=10.0,
        channel="ALL",
    )
    composites = {
        "taps_motor_index": 72.5,
        "taps_variability_index": 68.0,
        "taps_coordination_index": 85.0,
        "taps_composite_score": 74.2,
    }
    taps = [MockTapEvent(onset_ms=i * 250, offset_ms=i * 250 + 80) for i in range(5)]
    return params, composites, taps


class TestJSONExport:
    def test_creates_file(self):
        params, composites, taps = make_sample_data()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.json")
            export_json(params, composites, taps, path)
            assert os.path.exists(path)

    def test_valid_json(self):
        params, composites, taps = make_sample_data()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.json")
            export_json(params, composites, taps, path)
            with open(path) as f:
                data = json.load(f)
            assert data["taps_version"] == "0.1"
            assert data["composite_scores"]["taps_composite_score"] == 74.2
            assert len(data["tap_events"]) == 5


class TestCSVExport:
    def test_creates_csv(self):
        params, composites, taps = make_sample_data()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.csv")
            export_csv([(params, composites)], path)
            assert os.path.exists(path)

    def test_csv_has_header_and_row(self):
        params, composites, taps = make_sample_data()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.csv")
            export_csv([(params, composites)], path)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["tap_count"] == "40"


class TestResearchBundle:
    def test_creates_all_files(self):
        params, composites, taps = make_sample_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bundle")
            export_research_bundle(params, composites, taps, out)
            assert os.path.exists(os.path.join(out, "assessment.json"))
            assert os.path.exists(os.path.join(out, "tap_events.csv"))
            assert os.path.exists(os.path.join(out, "parameters_summary.csv"))
            assert os.path.exists(os.path.join(out, "metadata.json"))
