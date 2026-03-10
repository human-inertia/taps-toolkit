#!/usr/bin/env python3
"""
TAPS Live Test — Connect to Tap Strap 2, capture 30s of data, run full pipeline.

No tapsdk dependency — uses bleak directly for BLE.

Usage:
    python taps_live_test.py
    python taps_live_test.py --duration 60
    python taps_live_test.py --sensitivity 1 0 0
    python taps_live_test.py --scan-only

Prerequisites:
    pip install bleak numpy pandas scipy
    Tap Strap 2 on hand, Bluetooth on, Developer Mode enabled in TapManager app
"""

import asyncio
import argparse
import struct
import sys
import time
import os
import csv
import json
import numpy as np
from datetime import datetime, timezone

# BLE UUIDs (from Tap SDK)
NUS_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_CHAR = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # Write commands here
NUS_TX_CHAR = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # Raw data notifications
TAP_SERVICE = "c3ff0001-1d8b-40fd-a56f-c7bd5d0f3370"

MSG_TYPE_BIT = 2**31
FINGERS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]


def build_raw_mode_command(sensitivity=None):
    """Build the BLE command to switch Tap Strap 2 into raw sensor mode."""
    sens = sensitivity or [0, 0, 0]
    cmd = bytearray([0x03, 0x0c, 0x00, 0x0a]) + bytearray(sens)
    return cmd


def parse_raw_packet(data: bytearray):
    """Parse raw sensor BLE notification into structured data.

    Returns list of dicts with keys: type, ts, payload
        type='accl': payload = 15 int16 values (5 fingers x 3 axes)
        type='imu':  payload = 6 int16 values (gyro xyz + accel xyz)
    """
    messages = []
    ptr = 0
    L = len(data)

    while ptr + 4 <= L:
        ts_raw = int.from_bytes(data[ptr:ptr+4], "little", signed=False)
        if ts_raw == 0:
            break
        ptr += 4

        if ts_raw > MSG_TYPE_BIT:
            msg_type = "accl"
            ts = ts_raw - MSG_TYPE_BIT
            n_samples = 15  # 5 fingers x 3 axes
        else:
            msg_type = "imu"
            ts = ts_raw
            n_samples = 6   # gyro xyz + accel xyz

        if ptr + n_samples * 2 > L:
            break

        payload = []
        for _ in range(n_samples):
            val = int.from_bytes(data[ptr:ptr+2], "little", signed=True)
            payload.append(val)
            ptr += 2

        messages.append({"type": msg_type, "ts": ts, "payload": payload})

    return messages


async def scan_for_tap(timeout=10):
    """Scan for Tap Strap 2 devices."""
    from bleak import BleakScanner

    print(f"  Scanning for Tap Strap 2 ({timeout}s)...")
    devices = await BleakScanner.discover(timeout=timeout, return_adv=True)

    tap_devices = []
    for d, adv in devices.values():
        name = d.name or adv.local_name or ""
        rssi = adv.rssi if adv else -100
        if "tap" in name.lower():
            tap_devices.append(d)
            print(f"  FOUND: {name}  [{d.address}]  RSSI={rssi}")

    if not tap_devices:
        print("  No Tap devices found. Showing all nearby BLE devices:")
        sorted_devs = sorted(devices.values(), key=lambda x: x[1].rssi if x[1] else -100, reverse=True)
        for d, adv in sorted_devs[:15]:
            name = d.name or (adv.local_name if adv else None) or "(unnamed)"
            rssi = adv.rssi if adv else "?"
            print(f"    {name:<30} [{d.address}]  RSSI={rssi}")

    return tap_devices


async def run_live_test(duration=30, sensitivity=None, output_dir="data"):
    """Full live test: connect → raw mode → capture → detect → score."""
    from bleak import BleakClient, BleakScanner

    print("=" * 60)
    print("  TAPS Live Test v0.1")
    print("  Tap Assessment Protocol Standard")
    print("=" * 60)
    print()

    # --- Phase 1: Find device (fast scan loop) ---
    from bleak import BleakScanner
    device = None
    for attempt in range(6):
        print(f"  Scanning (attempt {attempt+1}/6)...")
        devs = await BleakScanner.discover(timeout=5, return_adv=True)
        for addr, (d, adv) in devs.items():
            name = d.name or (adv.local_name if adv else "") or ""
            if "tap" in name.lower():
                rssi = adv.rssi if adv else "?"
                print(f"  FOUND: {name} [{d.address}] RSSI={rssi}")
                device = d
                break
        if device:
            break
        print("  Not found yet — keep tapping to wake device...")

    if not device:
        print("\n  No Tap Strap 2 found. Make sure:")
        print("    1. Device is on your hand and awake (tap fingers)")
        print("    2. Bluetooth is ON in Windows Settings")
        print("    3. Developer Mode is ON in TapManager phone app")
        print("    4. Device is NOT connected to another app")
        return False

    print(f"  Connecting immediately to {device.name}...")

    # --- Phase 2: Connect and start raw mode (FAST — no delays) ---
    raw_samples = []
    sample_count = [0]
    start_time = [None]

    def on_raw_notification(sender, data: bytearray):
        """Handle incoming raw sensor BLE notifications."""
        # Try parsing as raw sensor packet
        messages = parse_raw_packet(data)
        epoch_ms = int(time.time() * 1000)

        if messages:
            for msg in messages:
                raw_samples.append({
                    "epoch_ms": epoch_ms,
                    "device_ts_ms": msg["ts"],
                    "type": msg["type"],
                    "payload": msg["payload"],
                })
                sample_count[0] += 1
        else:
            # Store unparsed notification too (tap events etc)
            sample_count[0] += 1

    async with BleakClient(device.address, timeout=15.0) as client:
        if not client.is_connected:
            print("  ERROR: Failed to connect.")
            return False

        print(f"  Connected!")

        # Subscribe to ALL notify chars FAST — no enumeration printing
        for service in client.services:
            for char in service.characteristics:
                if "notify" in char.properties:
                    try:
                        await client.start_notify(char.uuid, on_raw_notification)
                    except:
                        pass

        # Send controller mode first (wakes up data stream)
        await client.write_gatt_char(NUS_RX_CHAR, bytearray([0x03, 0x0c, 0x00, 0x01]), response=False)
        await asyncio.sleep(0.5)

        # Now switch to raw mode
        cmd = build_raw_mode_command(sensitivity)
        await client.write_gatt_char(NUS_RX_CHAR, cmd, response=False)
        print(f"  Raw sensor mode active. Sensitivity: {sensitivity or [0,0,0]}")

        # --- Phase 3: Capture data ---
        print(f"\n  CAPTURING {duration} seconds of data.")
        print("  Tap your fingers on a hard surface!")
        print("  Try: index finger tapping steadily for best results.")
        print()

        start_time[0] = time.time()
        for remaining in range(duration, 0, -1):
            elapsed = time.time() - start_time[0]
            rate = sample_count[0] / max(elapsed, 0.1)
            sys.stdout.write(
                f"\r  [{remaining:>3}s]  Samples: {sample_count[0]:>6,}  "
                f"Rate: {rate:>5.0f}/s  "
                f"Packets: {len(raw_samples):>6,}"
            )
            sys.stdout.flush()
            await asyncio.sleep(1)

        # Stop notifications
        await client.stop_notify(NUS_TX_CHAR)

    elapsed = time.time() - start_time[0]
    print(f"\n\n  Capture complete: {sample_count[0]:,} samples in {elapsed:.1f}s")

    if sample_count[0] == 0:
        print("  ERROR: No data received. Check device connection.")
        return False

    # --- Phase 4: Save raw data to CSV ---
    os.makedirs(output_dir, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"live_test_{ts_str}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch_ms", "device_ts_ms", "sample_type", "channel", "x", "y", "z"])

        for s in raw_samples:
            if s["type"] == "accl" and len(s["payload"]) == 15:
                for i, finger in enumerate(FINGERS):
                    x = s["payload"][i * 3]
                    y = s["payload"][i * 3 + 1]
                    z = s["payload"][i * 3 + 2]
                    writer.writerow([
                        s["epoch_ms"], s["device_ts_ms"],
                        "ACCEL_FINGER", finger, x, y, z
                    ])
            elif s["type"] == "imu" and len(s["payload"]) == 6:
                gx, gy, gz = s["payload"][0:3]
                ax, ay, az = s["payload"][3:6]
                writer.writerow([
                    s["epoch_ms"], s["device_ts_ms"],
                    "IMU_GYRO", "THUMB", gx, gy, gz
                ])
                writer.writerow([
                    s["epoch_ms"], s["device_ts_ms"],
                    "IMU_ACCEL", "THUMB", ax, ay, az
                ])

    print(f"  Saved to: {csv_path}")

    # --- Phase 5: Run TAPS pipeline ---
    print("\n  Running TAPS analysis pipeline...")
    print("-" * 60)

    from taps.processing.tap_detection import load_and_detect
    from taps.assessment.parameters import compute_parameters, compute_composite_scores, print_report
    from taps.reporting.export import export_json, export_research_bundle

    taps = load_and_detect(csv_path)

    if not taps:
        print("  No taps detected in capture.")
        print("  This may mean:")
        print("    - You didn't tap during capture")
        print("    - Tap force was too light")
        print("    - Noise floor needs calibration")
        print(f"\n  Raw data still saved at: {csv_path}")
        return False

    params = compute_parameters(taps)
    composites = compute_composite_scores(params)
    print_report(params, composites)

    # Per-finger breakdown
    print("  Per-finger breakdown:")
    for finger in FINGERS:
        finger_taps = [t for t in taps if t.channel == finger]
        if len(finger_taps) >= 3:
            fp = compute_parameters(finger_taps)
            print(f"    {finger:<8}  TC={fp.tap_count:>3}  "
                  f"ITI={fp.iti_mean_ms:>7.1f}ms  "
                  f"SD={fp.iti_sd_ms:>6.1f}ms  "
                  f"CV={fp.iti_cv:>.3f}")
    print()

    # Export results
    results_dir = os.path.join(output_dir, f"results_{ts_str}")
    export_research_bundle(
        params, composites, taps, results_dir,
        session_meta={
            "test_type": "live_test",
            "device": device.name,
            "device_address": device.address,
            "duration_s": duration,
            "sensitivity": sensitivity or [0, 0, 0],
            "total_raw_samples": sample_count[0],
        }
    )
    print(f"  Research bundle exported to: {results_dir}/")

    # --- Summary ---
    print()
    print("=" * 60)
    print("  LIVE TEST COMPLETE")
    print(f"  TCS (Composite Score): {composites['taps_composite_score']:.1f} / 100")
    print(f"  Taps detected: {len(taps)}")
    print(f"  Raw data: {csv_path}")
    print(f"  Results:  {results_dir}/")
    print("=" * 60)

    return True


async def main():
    parser = argparse.ArgumentParser(description="TAPS Live Test — Tap Strap 2")
    parser.add_argument("--duration", type=int, default=30, help="Capture duration in seconds (default: 30)")
    parser.add_argument("--sensitivity", nargs=3, type=int, default=None,
                        help="Sensor sensitivity [finger imu_gyro imu_accel] (default: 0 0 0)")
    parser.add_argument("--scan-only", action="store_true", help="Just scan for devices, don't capture")
    parser.add_argument("--output", type=str, default="data", help="Output directory (default: data)")
    args = parser.parse_args()

    if args.scan_only:
        print("=" * 60)
        print("  TAPS Device Scanner")
        print("=" * 60)
        print()
        await scan_for_tap(timeout=15)
        return

    success = await run_live_test(
        duration=args.duration,
        sensitivity=args.sensitivity,
        output_dir=args.output,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
