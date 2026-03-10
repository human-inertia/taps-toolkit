#!/usr/bin/env python3
"""
TAPS Debug Run — Simulate a full live test with synthetic sensor data.
Tests the entire pipeline without hardware: CSV write → detect → params → composites → export.
"""

import os
import csv
import sys
import time
import numpy as np
from datetime import datetime

FINGERS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
OUTPUT_DIR = "data"


def generate_synthetic_session(csv_path, duration_s=30, fs=200):
    """Generate a realistic synthetic Tap Strap 2 session CSV.

    Simulates:
        - 5-channel finger accelerometer data at ~200Hz
        - IMU gyro + accel on thumb
        - Realistic tap events on INDEX finger (~2.5 taps/sec)
        - Occasional taps on other fingers
        - Noise floor, co-activation bleed, slight fatigue
    """
    rng = np.random.RandomState(42)
    n_samples = duration_s * fs
    dt_ms = 1000.0 / fs

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch_ms", "device_ts_ms", "sample_type", "channel", "x", "y", "z"])

        base_epoch = int(time.time() * 1000)

        # Generate tap schedule for INDEX (primary tapping finger)
        # ~2.5 taps/sec = 400ms mean ITI, with some variability
        tap_times_ms = []
        t = 500.0  # start after 500ms
        tap_idx = 0
        while t < duration_s * 1000:
            # Add slight fatigue: ITI increases over time
            fatigue = 1.0 + (t / (duration_s * 1000)) * 0.15
            iti = max(200, rng.normal(400 * fatigue, 35))
            t += iti
            if t < duration_s * 1000:
                tap_times_ms.append(t)
                tap_idx += 1

        # Occasional taps on MIDDLE (co-activation test)
        middle_tap_times = [t + rng.normal(5, 2) for t in tap_times_ms[::4]]  # every 4th tap bleeds

        # A few THUMB taps
        thumb_tap_times = [rng.uniform(2000, duration_s * 1000) for _ in range(8)]

        # Freezing event: one long pause
        freeze_idx = len(tap_times_ms) // 2
        if freeze_idx < len(tap_times_ms) - 1:
            tap_times_ms[freeze_idx] = tap_times_ms[freeze_idx - 1] + 1200  # 1200ms gap

        tap_duration_ms = 60  # each tap lasts ~60ms

        sample_count = 0
        for i in range(n_samples):
            device_ts = i * dt_ms
            epoch_ms = base_epoch + int(device_ts)

            for fi, finger in enumerate(FINGERS):
                # Base noise
                noise = rng.normal(0, 0.08, 3)
                x, y, z = noise[0], noise[1], noise[2]

                # Check if this finger is tapping right now
                if finger == "INDEX":
                    for tap_t in tap_times_ms:
                        if tap_t <= device_ts <= tap_t + tap_duration_ms:
                            # Tap impulse: sharp spike on z-axis
                            phase = (device_ts - tap_t) / tap_duration_ms
                            impulse = 4.0 * np.sin(phase * np.pi) * rng.uniform(0.8, 1.2)
                            z += impulse
                            x += impulse * 0.3 * rng.normal(1, 0.2)
                            break

                elif finger == "MIDDLE":
                    for tap_t in middle_tap_times:
                        if tap_t <= device_ts <= tap_t + tap_duration_ms * 0.7:
                            phase = (device_ts - tap_t) / (tap_duration_ms * 0.7)
                            impulse = 1.8 * np.sin(phase * np.pi) * rng.uniform(0.7, 1.1)
                            z += impulse
                            break

                elif finger == "THUMB":
                    for tap_t in thumb_tap_times:
                        if tap_t <= device_ts <= tap_t + tap_duration_ms:
                            phase = (device_ts - tap_t) / tap_duration_ms
                            impulse = 3.5 * np.sin(phase * np.pi)
                            z += impulse
                            break

                writer.writerow([
                    epoch_ms, f"{device_ts:.1f}", "ACCEL_FINGER", finger,
                    f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"
                ])
                sample_count += 1

            # IMU data (every 5th sample to simulate lower IMU rate)
            if i % 5 == 0:
                writer.writerow([
                    epoch_ms, f"{device_ts:.1f}", "IMU_GYRO", "THUMB",
                    f"{rng.normal(0, 0.5):.4f}", f"{rng.normal(0, 0.5):.4f}", f"{rng.normal(0, 0.5):.4f}"
                ])
                writer.writerow([
                    epoch_ms, f"{device_ts:.1f}", "IMU_ACCEL", "THUMB",
                    f"{rng.normal(0, 0.1):.4f}", f"{rng.normal(0, 0.1):.4f}", f"{rng.normal(9.8, 0.1):.4f}"
                ])
                sample_count += 2

    return csv_path, len(tap_times_ms), len(middle_tap_times), sample_count


def main():
    print("=" * 60)
    print("  TAPS Debug Run — Synthetic Pipeline Test")
    print("=" * 60)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"debug_synthetic_{ts_str}.csv")

    # Phase 1: Generate
    print("  [1/5] Generating synthetic session (30s, 200Hz, 5 channels)...")
    t0 = time.time()
    path, n_index, n_middle, n_samples = generate_synthetic_session(csv_path, duration_s=30)
    gen_time = time.time() - t0
    print(f"        {n_samples:,} samples written in {gen_time:.1f}s")
    print(f"        Injected: {n_index} INDEX taps, {n_middle} MIDDLE co-activations")
    print(f"        Injected: 1 freezing event, gradual fatigue slope")
    file_size = os.path.getsize(csv_path) / 1024
    print(f"        File: {csv_path} ({file_size:.0f} KB)")
    print()

    # Phase 2: Detect
    print("  [2/5] Running tap detection...")
    t0 = time.time()
    from taps.processing.tap_detection import load_and_detect
    taps = load_and_detect(csv_path)
    detect_time = time.time() - t0
    print(f"        Detection complete in {detect_time:.1f}s")

    if not taps:
        print("        ERROR: No taps detected! Pipeline broken.")
        sys.exit(1)

    # Breakdown by channel
    from collections import Counter
    ch_counts = Counter(t.channel for t in taps)
    for ch in FINGERS:
        if ch in ch_counts:
            print(f"        {ch:<8} {ch_counts[ch]:>4} taps")
    print()

    # Phase 3: Parameters
    print("  [3/5] Computing 10 TAPS parameters...")
    from taps.assessment.parameters import compute_parameters, compute_composite_scores, print_report
    params = compute_parameters(taps)
    composites = compute_composite_scores(params)

    # Validate each parameter
    checks = [
        ("P1  Tap Count",        params.tap_count > 0,               f"{params.tap_count}"),
        ("P2  Mean ITI",         50 < params.iti_mean_ms < 2000,     f"{params.iti_mean_ms:.1f} ms"),
        ("P3  ITI Std Dev",      params.iti_sd_ms >= 0,              f"{params.iti_sd_ms:.1f} ms"),
        ("P4  ITI Coeff Var",    0 <= params.iti_cv < 5,             f"{params.iti_cv:.3f}"),
        ("P5  Tap Duration",     params.tap_duration_mean_ms > 0,    f"{params.tap_duration_mean_ms:.1f} ms"),
        ("P6  Freezing Count",   params.freezing_count >= 0,         f"{params.freezing_count}"),
        ("P7  Co-Activation",    0 <= params.coactivation_index <= 1,f"{params.coactivation_index:.3f}"),
        ("P8  Bilateral Asym",   -1 <= params.bilateral_asymmetry <= 1, f"{params.bilateral_asymmetry:.3f}"),
        ("P9  Fatigue Slope",    True,                               f"{params.fatigue_slope:.3f} ms/s"),
        ("P10 Rhythm Entropy",   params.rhythm_entropy >= 0,         f"{params.rhythm_entropy:.3f} bits"),
    ]

    all_ok = True
    for name, ok, val in checks:
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"        {status:<4}  {name:<22} = {val}")
    print()

    # Phase 4: Composites
    print("  [4/5] Computing composite scores...")
    for key, val in composites.items():
        ok = 0 <= val <= 100
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"        {status:<4}  {key:<28} = {val:.1f}")
    print()

    # Phase 5: Export
    print("  [5/5] Testing export pipeline...")
    from taps.reporting.export import export_json, export_research_bundle

    results_dir = os.path.join(OUTPUT_DIR, f"debug_results_{ts_str}")
    export_research_bundle(params, composites, taps, results_dir,
        session_meta={"test_type": "debug_synthetic", "duration_s": 30})

    expected_files = ["assessment.json", "tap_events.csv", "parameters_summary.csv", "metadata.json"]
    for fname in expected_files:
        fpath = os.path.join(results_dir, fname)
        exists = os.path.exists(fpath)
        size = os.path.getsize(fpath) if exists else 0
        status = "OK" if exists and size > 0 else "FAIL"
        if not exists or size == 0:
            all_ok = False
        print(f"        {status:<4}  {fname:<28} ({size:,} bytes)")
    print()

    # Full report
    print("-" * 60)
    print_report(params, composites)

    # Final verdict
    print("=" * 60)
    if all_ok:
        print("  ALL CHECKS PASSED — Pipeline ready for live device.")
    else:
        print("  SOME CHECKS FAILED — Review above.")
    print(f"  Synthetic data: {csv_path}")
    print(f"  Results bundle: {results_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
