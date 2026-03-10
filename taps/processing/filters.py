"""TAPS Signal Processing — Filtering and preprocessing."""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs=200, low=0.5, high=25.0, order=4):
    """4th order Butterworth bandpass filter, zero-phase.
    
    Args:
        signal: 1D numpy array
        fs: sampling frequency in Hz
        low: low cutoff frequency
        high: high cutoff frequency
        order: filter order
    
    Returns:
        Filtered signal (same length)
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    if len(signal) < 3 * max(len(b), len(a)):
        return signal  # Too short to filter
    return filtfilt(b, a, signal)


def highpass_filter(signal, fs=200, cutoff=0.5, order=4):
    """High-pass filter to remove gravity component.
    
    Args:
        signal: 1D numpy array
        fs: sampling frequency
        cutoff: cutoff frequency (0.5 Hz removes gravity drift)
        order: filter order
    
    Returns:
        Dynamic acceleration only (gravity removed)
    """
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='high')
    if len(signal) < 3 * max(len(b), len(a)):
        return signal
    return filtfilt(b, a, signal)


def compute_magnitude(x, y, z):
    """Compute acceleration magnitude from 3-axis data."""
    return np.sqrt(x**2 + y**2 + z**2)


def estimate_sample_rate(timestamps_ms):
    """Estimate actual sample rate from timestamps."""
    if len(timestamps_ms) < 2:
        return 200.0  # Default
    diffs = np.diff(timestamps_ms)
    diffs = diffs[diffs > 0]  # Remove zeros
    if len(diffs) == 0:
        return 200.0
    median_interval_ms = np.median(diffs)
    if median_interval_ms == 0:
        return 200.0
    return 1000.0 / median_interval_ms
