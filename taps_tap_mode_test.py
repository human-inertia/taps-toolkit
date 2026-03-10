#!/usr/bin/env python3
"""
TAPS Tap Mode Test — Capture tap events (controller mode) from Tap Strap 2.

This works WITHOUT Developer Mode. Uses the standard tap event characteristic
(c3ff0005) which reports which fingers tapped as a 5-bit bitmask.

Raw sensor mode (200Hz per-finger accelerometry) requires Developer Mode
enabled in the TapManager phone app. This test uses what's available.

Usage:
    python taps_tap_mode_test.py
    python taps_tap_mode_test.py --duration 60
"""

import asyncio
import argparse
import sys
import os
import csv
import json
import time
import numpy as np
from datetime import datetime, timezone
from collections import Counter

NUS_RX_CHAR = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
TAP_DATA_CHAR = "c3ff0005-1d8b-40fd-a56f-c7bd5d0f3370"
MOUSE_DATA_CHAR = "c3ff0006-1d8b-40fd-a56f-c7bd5d0f3370"
FINGERS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]


def decode_tap_event(data: bytearray):
    """Decode a tap event. Byte 0 is a 5-bit bitmask of which fingers tapped.

    Bit 0 = THUMB, Bit 1 = INDEX, Bit 2 = MIDDLE, Bit 3 = RING, Bit 4 = PINKY
    """
    if len(data) < 1:
        return []
    code = data[0]
    fingers = []
    for i, name in enumerate(FINGERS):
        if code & (1 << i):
            fingers.append(name)
    return fingers


async def scan_and_connect():
    from bleak import BleakScanner, BleakClient

    print("  Scanning for Tap Strap 2...")
    for attempt in range(6):
        devs = await BleakScanner.discover(timeout=5, return_adv=True)
        for addr, (d, adv) in devs.items():
            name = d.name or (adv.local_name if adv else "") or ""
            if "tap" in name.lower():
                rssi = adv.rssi if adv else "?"
                print(f"  Found: {name} [{d.address}] RSSI={rssi}")
                return d
        print(f"  Attempt {attempt+1}/6 — keep tapping to wake device...")
    return None


async def run_test(duration=30):
    from bleak import BleakClient

    print("=" * 60)
    print("  TAPS Tap Mode Test v0.1")
    print("  Controller mode — no Developer Mode required")
    print("=" * 60)
    print()

    device = await scan_and_connect()
    if not device:
        print("  Tap Strap 2 not found.")
        return False

    tap_events = []  # list of (timestamp_s, fingers_list)
    all_notifications = []

    def on_tap(sender, data: bytearray):
        sender_str = str(sender)
        ts = time.time()

        # Tap data characteristic
        if TAP_DATA_CHAR in sender_str or "c3ff0005" in sender_str:
            fingers = decode_tap_event(data)
            if fingers:
                tap_events.append((ts, fingers))
                finger_str = "+".join(fingers)
                print(f"    TAP: {finger_str:<30} [{len(tap_events):>4}]")

        all_notifications.append((ts, sender_str[-8:], data.hex()))

    print(f"  Connecting to {device.name}...")
    async with BleakClient(device.address, timeout=15.0) as client:
        print("  Connected!")

        # Subscribe to tap-related chars
        for svc in client.services:
            for ch in svc.characteristics:
                if "notify" in ch.properties:
                    try:
                        await client.start_notify(ch.uuid, on_tap)
                    except:
                        pass

        # Set controller mode
        await client.write_gatt_char(NUS_RX_CHAR,
            bytearray([0x03, 0x0c, 0x00, 0x01]), response=False)
        print("  Controller mode active.")
        print(f"\n  TAP YOUR FINGERS for {duration} seconds!")
        print("  Each tap will show which fingers were detected.")
        print()

        start = time.time()
        while time.time() - start < duration:
            remaining = duration - (time.time() - start)
            sys.stdout.write(f"\r  [{remaining:>5.1f}s left]  Taps: {len(tap_events):>4}  "
                           f"Total notifications: {len(all_notifications):>5}")
            sys.stdout.flush()
            await asyncio.sleep(0.2)

    elapsed = time.time() - start
    print(f"\n\n  Capture complete: {len(tap_events)} taps in {elapsed:.1f}s")
    print(f"  Total BLE notifications: {len(all_notifications)}")

    if not tap_events:
        print("  No taps detected. Were you tapping?")
        if all_notifications:
            print(f"  (Got {len(all_notifications)} other notifications though)")
            print("  First 5 notifications:")
            for ts, sender, hex_data in all_notifications[:5]:
                print(f"    [{sender}] {hex_data}")
        return False

    # --- Analysis ---
    print()
    print("=" * 60)
    print("  TAP EVENT ANALYSIS")
    print("=" * 60)

    # Timing
    timestamps = [t for t, _ in tap_events]
    itis = np.diff(timestamps) * 1000  # Convert to ms
    itis = itis[itis > 30]  # Filter debounce artifacts

    # Finger counts
    finger_counter = Counter()
    combo_counter = Counter()
    for _, fingers in tap_events:
        for f in fingers:
            finger_counter[f] += 1
        combo_counter["+".join(sorted(fingers))] += 1

    print(f"\n  Total taps: {len(tap_events)}")
    print(f"  Duration:   {elapsed:.1f}s")
    print(f"  Tap rate:   {len(tap_events)/elapsed:.1f} taps/sec")

    if len(itis) > 1:
        print(f"\n  Inter-Tap Intervals:")
        print(f"    Mean ITI:     {np.mean(itis):>8.1f} ms")
        print(f"    ITI Std Dev:  {np.std(itis):>8.1f} ms")
        print(f"    ITI CV:       {np.std(itis)/np.mean(itis):>8.3f}")
        print(f"    Min ITI:      {np.min(itis):>8.1f} ms")
        print(f"    Max ITI:      {np.max(itis):>8.1f} ms")

        # Freezing
        if np.mean(itis) > 0:
            freeze_threshold = 2.0 * np.mean(itis)
            freeze_count = int(np.sum(itis > freeze_threshold))
            print(f"    Freezing:     {freeze_count:>8} (ITI > {freeze_threshold:.0f}ms)")

        # Fatigue slope
        if len(itis) >= 5:
            x = np.arange(len(itis))
            slope, _ = np.polyfit(x, itis, 1)
            print(f"    Fatigue slope: {slope:>7.2f} ms/tap")

        # Rhythm entropy
        if len(itis) >= 5:
            n_bins = min(15, len(itis) // 3)
            if n_bins >= 2:
                hist, _ = np.histogram(itis, bins=n_bins, density=True)
                hist = hist[hist > 0]
                bw = (np.max(itis) - np.min(itis)) / n_bins
                if bw > 0:
                    probs = hist * bw
                    probs = probs[probs > 0]
                    entropy = -np.sum(probs * np.log2(probs))
                    print(f"    Rhythm entropy: {entropy:>6.3f} bits")

    print(f"\n  Finger Breakdown:")
    for finger in FINGERS:
        count = finger_counter.get(finger, 0)
        bar = "#" * min(count, 40)
        print(f"    {finger:<8} {count:>4}  {bar}")

    if len(combo_counter) > 1:
        print(f"\n  Tap Combos:")
        for combo, count in combo_counter.most_common(10):
            print(f"    {combo:<25} {count:>4}")

    # Save results
    os.makedirs("data", exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV of tap events
    csv_path = os.path.join("data", f"tap_events_{ts_str}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "fingers", "finger_count"])
        for ts, fingers in tap_events:
            w.writerow([f"{ts:.3f}", "+".join(fingers), len(fingers)])

    # JSON results
    results = {
        "taps_version": "0.1",
        "test_type": "tap_mode",
        "device": device.name,
        "duration_s": round(elapsed, 1),
        "total_taps": len(tap_events),
        "taps_per_second": round(len(tap_events) / elapsed, 2),
        "finger_counts": dict(finger_counter),
        "combo_counts": dict(combo_counter),
    }
    if len(itis) > 1:
        results["iti_mean_ms"] = round(float(np.mean(itis)), 1)
        results["iti_sd_ms"] = round(float(np.std(itis)), 1)
        results["iti_cv"] = round(float(np.std(itis) / np.mean(itis)), 3)

    json_path = os.path.join("data", f"tap_results_{ts_str}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {json_path}")
    print("=" * 60)
    return True


async def main():
    parser = argparse.ArgumentParser(description="TAPS Tap Mode Test")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    args = parser.parse_args()
    success = await run_test(duration=args.duration)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
