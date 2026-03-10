#!/usr/bin/env python3
"""
TAPS Calibration — Establish noise floor and per-finger tap baselines.

Run this before your first real data collection session.
Follow the prompts: rest hand, then tap each finger when asked.

Usage:
    python taps_calibrate.py
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime, timezone
from taps.collection.tap_strap_2 import TapStrap2


def main():
    print("=" * 55)
    print("  TAPS Calibration v0.1")
    print("=" * 55)
    print()
    print("  This calibration takes about 2 minutes.")
    print("  You'll need your Tap Strap 2 connected.")
    print()

    tap = TapStrap2(sensitivity=[0, 0, 0])

    # Data buffers
    rest_data = {ch: [] for ch in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]}
    tap_data = {ch: [] for ch in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]}
    current_phase = [None]  # 'rest', 'tap_THUMB', etc.
    connected = [False]

    def on_connection(identifier, name, fw):
        connected[0] = True
        print(f"  Connected: {name}")

    def on_raw_data(identifier, raw_sensor_data):
        data_type = str(raw_sensor_data.type)
        if "Device" not in data_type:
            return

        phase = current_phase[0]
        if phase is None:
            return

        channels = [("THUMB", 0), ("INDEX", 1), ("MIDDLE", 2), ("RING", 3), ("PINKY", 4)]
        for ch_name, idx in channels:
            try:
                point = raw_sensor_data.GetPoint(idx)
                if point is not None:
                    mag = np.sqrt(point.x**2 + point.y**2 + point.z**2)
                    if phase == "rest":
                        rest_data[ch_name].append(mag)
                    elif phase == f"tap_{ch_name}":
                        tap_data[ch_name].append(mag)
            except Exception:
                pass

    tap.on_connection(on_connection)
    tap.on_raw_data(on_raw_data)

    # Start SDK in background thread
    import threading
    thread = threading.Thread(target=tap.run, daemon=True)
    thread.start()

    # Wait for connection
    print("  Waiting for Tap Strap 2...")
    timeout = 30
    start = time.time()
    while not connected[0] and time.time() - start < timeout:
        time.sleep(0.5)

    if not connected[0]:
        print("  ERROR: Could not connect. Is device paired?")
        sys.exit(1)

    time.sleep(1)

    # Phase 1: Rest calibration
    print()
    print("  PHASE 1: REST CALIBRATION")
    print("  Place your hand flat on the table.")
    print("  Do not move for 10 seconds.")
    input("  Press Enter when ready...")

    current_phase[0] = "rest"
    print("  Recording rest data...")
    for i in range(10, 0, -1):
        sys.stdout.write(f"\r  {i}s remaining...")
        sys.stdout.flush()
        time.sleep(1)
    current_phase[0] = None
    print("\r  Rest calibration complete.     ")

    # Phase 2: Per-finger taps
    print()
    print("  PHASE 2: FINGER TAP CALIBRATION")
    print("  Tap each finger 20 times on a hard surface when prompted.")
    print()

    for finger in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]:
        print(f"  TAP {finger} 20 times. Steady rhythm.")
        input(f"  Press Enter, then start tapping {finger}...")
        current_phase[0] = f"tap_{finger}"
        tap.send_haptic([200])  # Signal start
        time.sleep(12)  # Allow ~12 seconds for 20 taps
        current_phase[0] = None
        tap.send_haptic([100, 100, 100])  # Signal done
        count = len(tap_data[finger])
        print(f"  {finger}: captured {count} samples")
        time.sleep(0.5)

    # Compute calibration values
    print()
    print("  Computing calibration...")

    calibration = {
        "taps_version": "0.1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "noise_floors": {},
        "tap_signatures": {},
    }

    for ch in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]:
        if rest_data[ch]:
            arr = np.array(rest_data[ch])
            calibration["noise_floors"][ch] = {
                "rms": float(np.sqrt(np.mean(arr**2))),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "samples": len(arr),
            }
        if tap_data[ch]:
            arr = np.array(tap_data[ch])
            calibration["tap_signatures"][ch] = {
                "peak_mean": float(np.mean(arr[arr > np.percentile(arr, 75)])),
                "peak_max": float(np.max(arr)),
                "samples": len(arr),
            }

    # Save
    cal_dir = os.path.join("data", "calibration")
    os.makedirs(cal_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cal_path = os.path.join(cal_dir, f"calibration_{ts}.json")
    with open(cal_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print()
    print("  Calibration Results:")
    print("  " + "-" * 50)
    print(f"  {'Finger':<10} {'Noise RMS':>10} {'Tap Peak':>10}")
    print("  " + "-" * 50)
    for ch in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]:
        nf = calibration["noise_floors"].get(ch, {})
        ts_data = calibration["tap_signatures"].get(ch, {})
        print(f"  {ch:<10} {nf.get('rms', 0):>10.4f} {ts_data.get('peak_mean', 0):>10.4f}")
    print("  " + "-" * 50)
    print(f"  Saved to: {cal_path}")
    print()

    tap.stop()


if __name__ == "__main__":
    main()
