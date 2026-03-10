"""TAPS Assessment — 10 Clinical Parameters + Composite Scores.

Each parameter has published clinical validation in the MCI/dementia literature.
See TAPS Systems Design v0.1, Section 6.2 for full references.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TAPSParameters:
    """All 10 TAPS parameters computed for a single epoch."""
    # P1: Tap Count
    tap_count: int = 0
    # P2: Mean Inter-Tap Interval (ms)
    iti_mean_ms: float = 0.0
    # P3: Standard Deviation of ITI (ms) — most sensitive parameter
    iti_sd_ms: float = 0.0
    # P4: Coefficient of Variation of ITI
    iti_cv: float = 0.0
    # P5: Mean Tap Duration (ms)
    tap_duration_mean_ms: float = 0.0
    # P6: Freezing Count
    freezing_count: int = 0
    # P7: Co-Activation Index (NOVEL — requires multi-channel wearable)
    coactivation_index: float = 0.0
    # P8: Bilateral Asymmetry Index (-1 to 1)
    bilateral_asymmetry: float = 0.0
    # P9: Fatigue Slope (ms/second for ITI trend)
    fatigue_slope: float = 0.0
    # P10: Rhythm Entropy (bits)
    rhythm_entropy: float = 0.0

    # Per-finger breakdown
    taps_per_finger: Dict[str, int] = None

    # Metadata
    epoch_start_ms: float = 0.0
    epoch_end_ms: float = 0.0
    epoch_duration_s: float = 0.0
    channel: str = "ALL"

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


def compute_parameters(taps, channel_filter=None) -> TAPSParameters:
    """Compute all 10 TAPS parameters from a list of TapEvent objects.
    
    Args:
        taps: List of TapEvent objects (from tap_detection)
        channel_filter: If set, only count taps from this channel
    
    Returns:
        TAPSParameters dataclass
    """
    if channel_filter:
        taps = [t for t in taps if t.channel == channel_filter]

    params = TAPSParameters()

    if len(taps) < 2:
        params.tap_count = len(taps)
        return params

    # Sort by onset
    taps = sorted(taps, key=lambda t: t.onset_ms)

    onsets = np.array([t.onset_ms for t in taps])
    durations = np.array([t.duration_ms for t in taps])

    params.epoch_start_ms = float(onsets[0])
    params.epoch_end_ms = float(onsets[-1])
    params.epoch_duration_s = (onsets[-1] - onsets[0]) / 1000.0
    params.channel = channel_filter or "ALL"

    # P1: Tap Count
    params.tap_count = len(taps)

    # Inter-tap intervals
    itis = np.diff(onsets)
    itis = itis[itis > 0]  # Remove any zero intervals

    if len(itis) == 0:
        return params

    # P2: Mean ITI
    params.iti_mean_ms = float(np.mean(itis))

    # P3: ITI Standard Deviation
    params.iti_sd_ms = float(np.std(itis))

    # P4: Coefficient of Variation
    if params.iti_mean_ms > 0:
        params.iti_cv = params.iti_sd_ms / params.iti_mean_ms
    
    # P5: Mean Tap Duration
    params.tap_duration_mean_ms = float(np.mean(durations))

    # P6: Freezing Count — ITI > 2x mean
    if params.iti_mean_ms > 0:
        freeze_threshold = 2.0 * params.iti_mean_ms
        params.freezing_count = int(np.sum(itis > freeze_threshold))

    # P7: Co-Activation Index
    coactivated = sum(1 for t in taps if len(t.secondary_channels) > 0)
    params.coactivation_index = coactivated / len(taps)

    # P8: Bilateral Asymmetry (computed separately — needs both hands)
    # Placeholder: 0.0 unless computed externally
    params.bilateral_asymmetry = 0.0

    # P9: Fatigue Slope — linear regression of ITI over time
    if len(itis) >= 5:
        x = np.arange(len(itis))
        try:
            slope, _ = np.polyfit(x, itis, 1)
            # Convert to ms per second
            samples_per_sec = 1000.0 / params.iti_mean_ms if params.iti_mean_ms > 0 else 1
            params.fatigue_slope = float(slope * samples_per_sec)
        except Exception:
            params.fatigue_slope = 0.0

    # P10: Rhythm Entropy — Shannon entropy of ITI distribution
    if len(itis) >= 5:
        # Bin ITIs into histogram
        n_bins = min(20, len(itis) // 3)
        if n_bins >= 2:
            hist, _ = np.histogram(itis, bins=n_bins, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            bin_width = (np.max(itis) - np.min(itis)) / n_bins
            if bin_width > 0:
                probs = hist * bin_width  # Normalize to probabilities
                probs = probs[probs > 0]
                params.rhythm_entropy = float(-np.sum(probs * np.log2(probs)))

    # Per-finger counts
    finger_counts = {}
    for t in taps:
        finger_counts[t.channel] = finger_counts.get(t.channel, 0) + 1
    params.taps_per_finger = finger_counts

    return params


def compute_composite_scores(params: TAPSParameters) -> Dict[str, float]:
    """Compute TMI, TVI, TCI, TCS composite scores.
    
    Note: Full composite scoring requires normative data for z-score
    conversion. This implementation uses raw parameter ratios as
    preliminary scores until normative database is built.
    """
    # TMI: Motor Index — speed and force dimension
    # Higher tap count + lower ITI + lower tap duration = better
    tmi = 0.0
    if params.iti_mean_ms > 0:
        tmi = (params.tap_count / 50.0) * 40 + \
              (1000.0 / params.iti_mean_ms) * 30 + \
              (200.0 / max(params.tap_duration_mean_ms, 50)) * 30
    tmi = min(100, max(0, tmi))

    # TVI: Variability Index — rhythm and consistency
    # Lower variability = better
    tvi = 100.0
    if params.iti_cv > 0:
        tvi = max(0, 100 - params.iti_cv * 300 - params.freezing_count * 5 - params.rhythm_entropy * 10)

    # TCI: Coordination Index
    # Lower co-activation = better finger independence
    tci = max(0, 100 - params.coactivation_index * 100)

    # TCS: Composite Score
    tcs = 0.4 * tmi + 0.35 * tvi + 0.25 * tci

    return {
        "taps_motor_index": round(tmi, 1),
        "taps_variability_index": round(tvi, 1),
        "taps_coordination_index": round(tci, 1),
        "taps_composite_score": round(tcs, 1),
    }


def print_report(params: TAPSParameters, composites: Dict[str, float] = None):
    """Print a human-readable parameter report."""
    print()
    print("=" * 55)
    print("  TAPS Assessment Report")
    print("=" * 55)
    print(f"  Channel:       {params.channel}")
    print(f"  Duration:      {params.epoch_duration_s:.1f}s")
    print(f"  Tap Count:     {params.tap_count}")
    print("-" * 55)
    print(f"  P1  Tap Count          {params.tap_count:>8}")
    print(f"  P2  Mean ITI           {params.iti_mean_ms:>8.1f} ms")
    print(f"  P3  ITI Std Dev        {params.iti_sd_ms:>8.1f} ms  {'*' if params.iti_sd_ms > 110 else ''}")
    print(f"  P4  ITI Coeff Var      {params.iti_cv:>8.3f}")
    print(f"  P5  Tap Duration       {params.tap_duration_mean_ms:>8.1f} ms")
    print(f"  P6  Freezing Count     {params.freezing_count:>8}")
    print(f"  P7  Co-Activation Idx  {params.coactivation_index:>8.3f}")
    print(f"  P8  Bilateral Asym     {params.bilateral_asymmetry:>8.3f}")
    print(f"  P9  Fatigue Slope      {params.fatigue_slope:>8.3f} ms/s")
    print(f"  P10 Rhythm Entropy     {params.rhythm_entropy:>8.3f} bits")
    print("-" * 55)

    if params.taps_per_finger:
        print("  Per-finger:")
        for finger, count in sorted(params.taps_per_finger.items()):
            bar = "#" * min(count, 40)
            print(f"    {finger:<8} {count:>4}  {bar}")

    if composites:
        print("-" * 55)
        print(f"  TMI  Motor Index       {composites['taps_motor_index']:>8.1f}")
        print(f"  TVI  Variability Index {composites['taps_variability_index']:>8.1f}")
        print(f"  TCI  Coordination Idx  {composites['taps_coordination_index']:>8.1f}")
        print(f"  TCS  COMPOSITE SCORE   {composites['taps_composite_score']:>8.1f} / 100")

    print("=" * 55)
    print()
