#!/usr/bin/env python3
"""
TAPS Data Logger — Week 1, Step 1

Connect to Tap Strap 2, stream raw sensor data, save to CSV.
This is the first thing you run. Everything else builds on this data.

Prerequisites:
    1. pip install tapsdk numpy pandas
    2. Tap Strap 2 paired in Windows Bluetooth Settings
    3. Developer Mode enabled in TapManager app on phone

Usage:
    python taps_logger.py
    python taps_logger.py --sensitivity 1 0 0    # ±2G finger accel
    python taps_logger.py --mode A                # Structured assessment
"""

import sys
import time
import argparse
import threading
from taps.collection.tap_strap_2 import TapStrap2
from taps.collection.session import Session


def main():
    parser = argparse.ArgumentParser(description="TAPS Raw Data Logger")
    parser.add_argument(
        "--sensitivity", nargs=3, type=int, default=[0, 0, 0],
        help="Sensor sensitivity [finger_accel, imu_gyro, imu_accel]. Default: 0 0 0"
    )
    parser.add_argument(
        "--mode", choices=["A", "B", "C"], default="C",
        help="Collection mode. A=structured, B=semi-structured, C=passive"
    )
    parser.add_argument(
        "--participant", type=str, default="self",
        help="Participant ID (default: self)"
    )
    parser.add_argument(
        "--output", type=str, default="data",
        help="Output directory (default: data)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  TAPS Data Logger v0.1")
    print("  Tap Assessment Protocol Standard")
    print("=" * 60)
    print()

    # Initialize device
    tap = TapStrap2(sensitivity=args.sensitivity)
    session = Session(
        output_dir=args.output,
        mode=args.mode,
        participant_id=args.participant
    )

    session_started = False
    sample_counter = [0]
    last_print = [time.time()]

    def on_connection(identifier, name, fw):
        nonlocal session_started
        device_info = tap.get_device_info()
        scale_factors = tap.get_scale_factors()
        session.start(device_info=device_info, scale_factors=scale_factors)
        session_started = True
        tap.send_haptic([300, 100, 300])  # Confirm with haptic
        print()
        print("[TAPS] Collecting data. Press Ctrl+C to stop.")
        print()

    def on_raw_data(identifier, raw_sensor_data):
        if session_started:
            session.record_raw_sample(identifier, raw_sensor_data)
            sample_counter[0] += 1
            # Print progress every 2 seconds
            now = time.time()
            if now - last_print[0] > 2.0:
                elapsed = now - session.start_time.timestamp()
                rate = sample_counter[0] / max(elapsed, 0.1)
                sys.stdout.write(
                    f"\r  Samples: {sample_counter[0]:>8,}  |  "
                    f"Rate: {rate:>6.0f}/s  |  "
                    f"Gaps: {session.gap_count}  |  "
                    f"Time: {elapsed:>6.1f}s"
                )
                sys.stdout.flush()
                last_print[0] = now

    tap.on_connection(on_connection)
    tap.on_raw_data(on_raw_data)

    try:
        tap.run()
    except KeyboardInterrupt:
        pass
    finally:
        if session_started:
            meta = session.stop()
            print()
            print("=" * 60)
            print("  Session saved. You now have raw sensor data.")
            print(f"  Next: python taps_detect.py {session.filepath}")
            print("=" * 60)


if __name__ == "__main__":
    main()
