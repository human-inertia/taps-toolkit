"""Tests for TAPS parameter computation."""

import numpy as np
import pytest
from dataclasses import dataclass, field
from taps.assessment.parameters import (
    TAPSParameters, compute_parameters, compute_composite_scores
)


@dataclass
class MockTapEvent:
    channel: str = "INDEX"
    onset_ms: float = 0.0
    offset_ms: float = 0.0
    duration_ms: float = 0.0
    peak_magnitude: float = 1.0
    secondary_channels: dict = field(default_factory=dict)


def make_taps(count=40, iti_mean=250, iti_sd=20, duration=80, channel="INDEX"):
    """Generate synthetic tap events with controlled parameters."""
    taps = []
    t = 0.0
    rng = np.random.RandomState(42)
    for _ in range(count):
        interval = max(50, rng.normal(iti_mean, iti_sd))
        t += interval
        taps.append(MockTapEvent(
            channel=channel,
            onset_ms=t,
            offset_ms=t + duration,
            duration_ms=duration,
            peak_magnitude=rng.uniform(0.5, 2.0),
        ))
    return taps


class TestBasicParameters:
    def test_empty_input(self):
        params = compute_parameters([])
        assert params.tap_count == 0

    def test_single_tap(self):
        tap = MockTapEvent(onset_ms=100, offset_ms=180, duration_ms=80)
        params = compute_parameters([tap])
        assert params.tap_count == 1
        assert params.iti_mean_ms == 0.0

    def test_tap_count(self):
        taps = make_taps(count=40)
        params = compute_parameters(taps)
        assert params.tap_count == 40

    def test_iti_mean_reasonable(self):
        taps = make_taps(count=50, iti_mean=250, iti_sd=10)
        params = compute_parameters(taps)
        assert 200 < params.iti_mean_ms < 300

    def test_iti_sd_lower_than_mean(self):
        taps = make_taps(count=50, iti_mean=250, iti_sd=20)
        params = compute_parameters(taps)
        assert params.iti_sd_ms < params.iti_mean_ms

    def test_cv_is_sd_over_mean(self):
        taps = make_taps(count=50)
        params = compute_parameters(taps)
        expected_cv = params.iti_sd_ms / params.iti_mean_ms
        assert abs(params.iti_cv - expected_cv) < 0.001

    def test_tap_duration(self):
        taps = make_taps(count=30, duration=80)
        params = compute_parameters(taps)
        assert params.tap_duration_mean_ms == 80.0

    def test_freezing_detection(self):
        """Inject a long pause — should be detected as freezing."""
        taps = make_taps(count=20, iti_mean=200, iti_sd=5)
        # Insert a freeze: gap of 600ms (>2x 200ms mean)
        taps.append(MockTapEvent(
            channel="INDEX",
            onset_ms=taps[-1].onset_ms + 600,
            offset_ms=taps[-1].onset_ms + 680,
            duration_ms=80,
        ))
        params = compute_parameters(taps)
        assert params.freezing_count >= 1

    def test_channel_filter(self):
        index_taps = make_taps(count=20, channel="INDEX")
        middle_taps = make_taps(count=10, channel="MIDDLE")
        all_taps = index_taps + middle_taps
        params = compute_parameters(all_taps, channel_filter="INDEX")
        assert params.tap_count == 20
        assert params.channel == "INDEX"


class TestCoActivation:
    def test_no_coactivation(self):
        taps = make_taps(count=20)
        params = compute_parameters(taps)
        assert params.coactivation_index == 0.0

    def test_full_coactivation(self):
        taps = make_taps(count=20)
        for t in taps:
            t.secondary_channels = {"MIDDLE": 0.8}
        params = compute_parameters(taps)
        assert params.coactivation_index == 1.0

    def test_partial_coactivation(self):
        taps = make_taps(count=20)
        for t in taps[:10]:
            t.secondary_channels = {"RING": 0.5}
        params = compute_parameters(taps)
        assert 0.4 < params.coactivation_index < 0.6


class TestCompositeScores:
    def test_scores_in_range(self):
        taps = make_taps(count=40, iti_mean=250, iti_sd=20)
        params = compute_parameters(taps)
        composites = compute_composite_scores(params)
        for key in ["taps_motor_index", "taps_variability_index",
                     "taps_coordination_index", "taps_composite_score"]:
            assert 0 <= composites[key] <= 100, f"{key} = {composites[key]}"

    def test_healthy_scores_higher(self):
        """Healthy-like tapping should score higher than impaired-like."""
        healthy = make_taps(count=48, iti_mean=200, iti_sd=15, duration=70)
        impaired = make_taps(count=28, iti_mean=400, iti_sd=80, duration=120)

        h_params = compute_parameters(healthy)
        i_params = compute_parameters(impaired)
        h_comp = compute_composite_scores(h_params)
        i_comp = compute_composite_scores(i_params)

        assert h_comp["taps_composite_score"] > i_comp["taps_composite_score"]


class TestFatigueSlope:
    def test_increasing_iti_positive_slope(self):
        """ITIs that get longer over time should produce positive fatigue slope."""
        taps = []
        t = 0.0
        for i in range(30):
            interval = 200 + i * 5  # Gets progressively slower
            t += interval
            taps.append(MockTapEvent(
                channel="INDEX", onset_ms=t, offset_ms=t + 80, duration_ms=80
            ))
        params = compute_parameters(taps)
        assert params.fatigue_slope > 0

    def test_stable_iti_near_zero_slope(self):
        taps = make_taps(count=30, iti_mean=250, iti_sd=2)
        params = compute_parameters(taps)
        assert abs(params.fatigue_slope) < 50  # Near zero


class TestRhythmEntropy:
    def test_entropy_nonnegative(self):
        taps = make_taps(count=30)
        params = compute_parameters(taps)
        assert params.rhythm_entropy >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
