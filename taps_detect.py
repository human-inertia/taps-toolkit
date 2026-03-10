#!/usr/bin/env python3
"""
TAPS Tap Detector + Parameter Extractor

Run this on a captured session CSV to detect taps and compute all 10 parameters.

Usage:
    python taps_detect.py data/session_XXXX.csv
    python taps_detect.py data/session_XXXX.csv --channel INDEX
    python taps_detect.py data/session_XXXX.csv --export results.json
"""

import sys
import json
import argparse
from taps.processing.tap_detection import load_and_detect
from taps.assessment.parameters import compute_parameters, compute_composite_scores, print_report


def main():
    parser = argparse.ArgumentParser(description="TAPS Tap Detector + Analysis")
    parser.add_argument("csv_file", help="Path to session CSV file")
    parser.add_argument("--channel", type=str, default=None,
                        help="Filter to single channel (THUMB, INDEX, MIDDLE, RING, PINKY)")
    parser.add_argument("--export", type=str, default=None,
                        help="Export results to JSON file")
    args = parser.parse_args()

    print("=" * 55)
    print("  TAPS Detector v0.1")
    print("=" * 55)
    print(f"  Input: {args.csv_file}")
    print()

    # Detect taps
    taps = load_and_detect(args.csv_file)

    if not taps:
        print("No taps detected. Check your data file.")
        sys.exit(1)

    # Compute parameters
    params = compute_parameters(taps, channel_filter=args.channel)
    composites = compute_composite_scores(params)

    # Print report
    print_report(params, composites)

    # Export if requested
    if args.export:
        output = {
            "taps_version": "0.1",
            "source_file": args.csv_file,
            "tap_count": len(taps),
            "parameters": params.to_dict(),
            "composite_scores": composites,
            "taps": [
                {
                    "channel": t.channel,
                    "onset_ms": t.onset_ms,
                    "offset_ms": t.offset_ms,
                    "duration_ms": t.duration_ms,
                    "peak_magnitude": t.peak_magnitude,
                    "coactivation": t.secondary_channels,
                }
                for t in taps
            ],
        }
        with open(args.export, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Exported to: {args.export}")

    # Per-finger analysis
    if not args.channel:
        print("\n  Per-finger parameter breakdown:")
        for finger in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]:
            finger_taps = [t for t in taps if t.channel == finger]
            if len(finger_taps) >= 3:
                fp = compute_parameters(finger_taps, channel_filter=finger)
                print(f"    {finger:<8}  TC={fp.tap_count:>3}  "
                      f"ITI={fp.iti_mean_ms:>7.1f}ms  "
                      f"SD={fp.iti_sd_ms:>6.1f}ms  "
                      f"CV={fp.iti_cv:>.3f}  "
                      f"FC={fp.freezing_count}")
        print()


if __name__ == "__main__":
    main()
