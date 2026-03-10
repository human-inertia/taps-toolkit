"""TAPS Tap Detection — Detect discrete tap events from continuous accelerometer signal."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .filters import bandpass_filter, highpass_filter, compute_magnitude, estimate_sample_rate


@dataclass
class TapEvent:
    """A single detected tap."""
    channel: str               # THUMB, INDEX, MIDDLE, RING, PINKY
    onset_ms: float            # Device timestamp of tap start
    offset_ms: float           # Device timestamp of tap end
    duration_ms: float         # offset - onset
    peak_magnitude: float      # Peak acceleration during tap
    secondary_channels: dict = field(default_factory=dict)  # {channel: peak_mag} for co-activation


def detect_taps_single_channel(
    timestamps_ms: np.ndarray,
    magnitude: np.ndarray,
    channel_name: str,
    noise_floor: float = None,
    threshold_multiplier: float = 1.5,
    min_duration_ms: float = 20.0,
    max_duration_ms: float = 300.0,
    min_iti_ms: float = 50.0,
) -> List[TapEvent]:
    """Detect tap events from a single finger channel's acceleration magnitude.
    
    Algorithm (from TAPS spec Section 5.4):
        1. Compute adaptive threshold: 1.5x noise floor
        2. Find contiguous regions above threshold
        3. Filter by duration (20-300ms)
        4. Debounce (merge within 50ms)
    
    Args:
        timestamps_ms: Device timestamps
        magnitude: Acceleration magnitude (gravity-removed, filtered)
        channel_name: Which finger
        noise_floor: RMS of rest calibration. If None, estimated from signal.
        threshold_multiplier: Multiple of noise floor for detection
        min_duration_ms: Minimum tap duration
        max_duration_ms: Maximum tap duration
        min_iti_ms: Minimum inter-tap interval (debounce)
    
    Returns:
        List of TapEvent objects
    """
    if len(magnitude) < 10:
        return []

    # Estimate noise floor if not provided
    if noise_floor is None:
        # Use 25th percentile of magnitude as rough noise estimate
        noise_floor = np.percentile(magnitude, 25)
        if noise_floor < 0.01:
            noise_floor = np.std(magnitude) * 0.5

    threshold = noise_floor * threshold_multiplier

    # Find above-threshold regions
    above = magnitude > threshold
    candidates = []
    in_event = False
    start_idx = 0

    for i in range(len(above)):
        if above[i] and not in_event:
            in_event = True
            start_idx = i
        elif not above[i] and in_event:
            in_event = False
            candidates.append((start_idx, i - 1))

    # Close final event if signal ends above threshold
    if in_event:
        candidates.append((start_idx, len(above) - 1))

    # Filter by duration and build TapEvents
    taps = []
    for start_idx, end_idx in candidates:
        onset = timestamps_ms[start_idx]
        offset = timestamps_ms[end_idx]
        duration = offset - onset

        if duration < min_duration_ms or duration > max_duration_ms:
            continue

        peak_idx = start_idx + np.argmax(magnitude[start_idx:end_idx + 1])
        peak_mag = magnitude[peak_idx]

        taps.append(TapEvent(
            channel=channel_name,
            onset_ms=onset,
            offset_ms=offset,
            duration_ms=duration,
            peak_magnitude=float(peak_mag),
        ))

    # Debounce: merge taps closer than min_iti_ms
    if len(taps) < 2:
        return taps

    debounced = [taps[0]]
    for tap in taps[1:]:
        prev = debounced[-1]
        if tap.onset_ms - prev.offset_ms < min_iti_ms:
            # Merge: extend previous tap
            prev.offset_ms = max(prev.offset_ms, tap.offset_ms)
            prev.duration_ms = prev.offset_ms - prev.onset_ms
            prev.peak_magnitude = max(prev.peak_magnitude, tap.peak_magnitude)
        else:
            debounced.append(tap)

    return debounced


def detect_taps_multichannel(channel_data: dict, noise_floors: dict = None) -> List[TapEvent]:
    """Detect taps across all finger channels and compute co-activation.
    
    Args:
        channel_data: {channel_name: {"timestamps": np.array, "magnitude": np.array}}
        noise_floors: {channel_name: float} from calibration. Optional.
    
    Returns:
        List of TapEvent with primary channel and co-activation info
    """
    all_taps = {}
    for ch_name, data in channel_data.items():
        nf = noise_floors.get(ch_name) if noise_floors else None
        taps = detect_taps_single_channel(
            data["timestamps"], data["magnitude"], ch_name, noise_floor=nf
        )
        all_taps[ch_name] = taps

    # Flatten and sort by onset time
    flat_taps = []
    for ch_taps in all_taps.values():
        flat_taps.extend(ch_taps)
    flat_taps.sort(key=lambda t: t.onset_ms)

    # Compute co-activation: for each tap, check if other channels
    # had significant activation during the same time window
    for tap in flat_taps:
        for ch_name, data in channel_data.items():
            if ch_name == tap.channel:
                continue
            # Find magnitude during this tap's time window
            mask = (data["timestamps"] >= tap.onset_ms) & \
                   (data["timestamps"] <= tap.offset_ms)
            if np.any(mask):
                secondary_peak = np.max(data["magnitude"][mask])
                if secondary_peak > tap.peak_magnitude * 0.5:
                    tap.secondary_channels[ch_name] = float(secondary_peak)

    return flat_taps


def load_and_detect(csv_path: str) -> List[TapEvent]:
    """Load a session CSV and detect all tap events.
    
    This is the main entry point for offline tap detection.
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Filter to finger accelerometer data only
    finger_df = df[df["sample_type"] == "ACCEL_FINGER"]
    
    if finger_df.empty:
        print("[DETECT] No finger accelerometer data found in file.")
        return []

    channels = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
    channel_data = {}
    
    for ch in channels:
        ch_df = finger_df[finger_df["channel"] == ch].copy()
        if ch_df.empty:
            continue
        
        ts = ch_df["device_ts_ms"].values.astype(float)
        x = ch_df["x"].values.astype(float)
        y = ch_df["y"].values.astype(float)
        z = ch_df["z"].values.astype(float)
        
        # Estimate sample rate
        fs = estimate_sample_rate(ts)
        
        # Remove gravity via high-pass filter
        if len(x) > 20:
            x_filt = highpass_filter(x, fs=fs)
            y_filt = highpass_filter(y, fs=fs)
            z_filt = highpass_filter(z, fs=fs)
        else:
            x_filt, y_filt, z_filt = x, y, z
        
        mag = compute_magnitude(x_filt, y_filt, z_filt)
        
        # Apply bandpass
        if len(mag) > 20:
            mag = bandpass_filter(mag, fs=fs)
            mag = np.abs(mag)  # Rectify after bandpass
        
        channel_data[ch] = {"timestamps": ts, "magnitude": mag}
    
    if not channel_data:
        print("[DETECT] Could not process any channels.")
        return []
    
    print(f"[DETECT] Processing {len(channel_data)} channels...")
    fs = estimate_sample_rate(list(channel_data.values())[0]["timestamps"])
    print(f"[DETECT] Estimated sample rate: {fs:.1f} Hz")
    
    taps = detect_taps_multichannel(channel_data)
    print(f"[DETECT] Found {len(taps)} tap events")
    
    return taps
