"""Tests for TAPS tap detection pipeline."""

import numpy as np
import pytest
from taps.processing.tap_detection import detect_taps_single_channel, detect_taps_multichannel
from taps.processing.filters import bandpass_filter, highpass_filter, compute_magnitude, estimate_sample_rate


def make_synthetic_signal(n_taps=10, fs=200, tap_amplitude=5.0, noise_level=0.1):
    """Generate a synthetic accelerometer magnitude signal with known taps.

    Returns:
        timestamps_ms, magnitude, expected_tap_times
    """
    duration_s = n_taps * 0.3  # ~300ms per tap cycle
    n_samples = int(duration_s * fs)
    dt_ms = 1000.0 / fs

    timestamps = np.arange(n_samples) * dt_ms
    signal = np.random.normal(0, noise_level, n_samples)

    tap_times = []
    tap_duration_samples = int(0.05 * fs)  # 50ms taps

    for i in range(n_taps):
        center = int((i + 0.5) * 0.3 * fs)
        start = max(0, center - tap_duration_samples // 2)
        end = min(n_samples, center + tap_duration_samples // 2)
        signal[start:end] += tap_amplitude
        tap_times.append(timestamps[center])

    return timestamps, np.abs(signal), tap_times


class TestSingleChannelDetection:
    def test_detects_known_taps(self):
        ts, mag, expected = make_synthetic_signal(n_taps=8, tap_amplitude=5.0)
        taps = detect_taps_single_channel(ts, mag, "INDEX", noise_floor=0.2)
        assert len(taps) >= 5, f"Expected ~8 taps, got {len(taps)}"

    def test_no_taps_in_noise(self):
        ts = np.arange(1000) * 5.0
        mag = np.random.normal(0, 0.05, 1000)
        mag = np.abs(mag)
        taps = detect_taps_single_channel(ts, mag, "INDEX", noise_floor=0.1)
        assert len(taps) < 3  # Noise shouldn't produce many taps

    def test_channel_name_preserved(self):
        ts, mag, _ = make_synthetic_signal(n_taps=5)
        taps = detect_taps_single_channel(ts, mag, "PINKY", noise_floor=0.2)
        for t in taps:
            assert t.channel == "PINKY"

    def test_duration_within_bounds(self):
        ts, mag, _ = make_synthetic_signal(n_taps=5)
        taps = detect_taps_single_channel(
            ts, mag, "INDEX", noise_floor=0.2,
            min_duration_ms=20, max_duration_ms=300
        )
        for t in taps:
            assert 20 <= t.duration_ms <= 300

    def test_short_signal_returns_empty(self):
        taps = detect_taps_single_channel(
            np.array([0, 5]), np.array([0.1, 0.1]), "INDEX"
        )
        assert taps == []


class TestMultiChannelDetection:
    def test_multichannel_combines(self):
        ts1, mag1, _ = make_synthetic_signal(n_taps=5)
        ts2, mag2, _ = make_synthetic_signal(n_taps=3)
        channel_data = {
            "INDEX": {"timestamps": ts1, "magnitude": mag1},
            "MIDDLE": {"timestamps": ts2, "magnitude": mag2},
        }
        taps = detect_taps_multichannel(channel_data)
        assert len(taps) >= 5  # At least the taps from both channels


class TestFilters:
    def test_bandpass_preserves_length(self):
        sig = np.random.randn(500)
        filtered = bandpass_filter(sig, fs=200)
        assert len(filtered) == len(sig)

    def test_highpass_removes_dc(self):
        sig = np.ones(500) * 9.8 + np.random.randn(500) * 0.1
        filtered = highpass_filter(sig, fs=200)
        assert abs(np.mean(filtered)) < 1.0  # DC largely removed

    def test_magnitude_calculation(self):
        x = np.array([3.0])
        y = np.array([4.0])
        z = np.array([0.0])
        mag = compute_magnitude(x, y, z)
        assert abs(mag[0] - 5.0) < 0.001

    def test_sample_rate_estimation(self):
        ts = np.arange(0, 1000, 5)  # 5ms intervals = 200Hz
        fs = estimate_sample_rate(ts)
        assert abs(fs - 200.0) < 5.0

    def test_short_signal_not_filtered(self):
        sig = np.array([1.0, 2.0, 3.0])
        out = bandpass_filter(sig, fs=200)
        np.testing.assert_array_equal(sig, out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
