"""TAPS Reporting — Export assessment results to CSV, JSON, and research formats."""

import csv
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional


def export_json(params, composites, taps, output_path, session_meta=None):
    """Export full assessment to JSON.

    Args:
        params: TAPSParameters dataclass
        composites: Dict from compute_composite_scores
        taps: List of TapEvent objects
        output_path: Where to save
        session_meta: Optional session metadata dict
    """
    output = {
        "taps_version": "0.1",
        "export_time": datetime.now(timezone.utc).isoformat(),
        "session": session_meta or {},
        "parameters": params.to_dict(),
        "composite_scores": composites,
        "summary": {
            "total_taps": params.tap_count,
            "duration_s": params.epoch_duration_s,
            "taps_per_second": round(
                params.tap_count / max(params.epoch_duration_s, 0.1), 2
            ),
        },
        "tap_events": [
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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    return output_path


def export_csv(params_list, output_path):
    """Export one row per epoch to CSV — suitable for research datasets.

    Args:
        params_list: List of (TAPSParameters, composites_dict) tuples
        output_path: Where to save
    """
    fieldnames = [
        "epoch_start_ms", "epoch_end_ms", "epoch_duration_s", "channel",
        "tap_count", "iti_mean_ms", "iti_sd_ms", "iti_cv",
        "tap_duration_mean_ms", "freezing_count", "coactivation_index",
        "bilateral_asymmetry", "fatigue_slope", "rhythm_entropy",
        "taps_motor_index", "taps_variability_index",
        "taps_coordination_index", "taps_composite_score",
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for params, composites in params_list:
            row = {
                "epoch_start_ms": params.epoch_start_ms,
                "epoch_end_ms": params.epoch_end_ms,
                "epoch_duration_s": round(params.epoch_duration_s, 2),
                "channel": params.channel,
                "tap_count": params.tap_count,
                "iti_mean_ms": round(params.iti_mean_ms, 2),
                "iti_sd_ms": round(params.iti_sd_ms, 2),
                "iti_cv": round(params.iti_cv, 4),
                "tap_duration_mean_ms": round(params.tap_duration_mean_ms, 2),
                "freezing_count": params.freezing_count,
                "coactivation_index": round(params.coactivation_index, 4),
                "bilateral_asymmetry": round(params.bilateral_asymmetry, 4),
                "fatigue_slope": round(params.fatigue_slope, 4),
                "rhythm_entropy": round(params.rhythm_entropy, 4),
            }
            row.update(composites)
            writer.writerow(row)
    return output_path


def export_research_bundle(params, composites, taps, output_dir,
                           session_meta=None, calibration=None):
    """Export a complete research-ready data bundle.

    Creates:
        <output_dir>/
            assessment.json       — Full parameters + composites
            tap_events.csv        — One row per tap event
            parameters_summary.csv — One row per epoch
            metadata.json         — Session + calibration info
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Full JSON assessment
    export_json(params, composites, taps,
                os.path.join(output_dir, "assessment.json"), session_meta)

    # 2. Tap events CSV
    tap_csv_path = os.path.join(output_dir, "tap_events.csv")
    with open(tap_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "channel", "onset_ms", "offset_ms", "duration_ms",
            "peak_magnitude", "coactivation_channels", "coactivation_count"
        ])
        for t in taps:
            writer.writerow([
                t.channel, t.onset_ms, t.offset_ms, t.duration_ms,
                round(t.peak_magnitude, 4),
                ";".join(t.secondary_channels.keys()),
                len(t.secondary_channels),
            ])

    # 3. Parameters summary CSV
    export_csv([(params, composites)],
               os.path.join(output_dir, "parameters_summary.csv"))

    # 4. Metadata
    meta = {
        "taps_version": "0.1",
        "export_time": datetime.now(timezone.utc).isoformat(),
        "session": session_meta or {},
        "calibration": calibration or {},
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return output_dir
