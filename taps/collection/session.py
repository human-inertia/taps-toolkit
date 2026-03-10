"""TAPS Session Management — Records raw sensor data to structured files."""

import os
import csv
import json
import uuid
import time
from datetime import datetime, timezone


class Session:
    """Manages a single data collection session."""

    def __init__(self, output_dir="data", mode="C", participant_id=None):
        self.session_id = str(uuid.uuid4())[:8]
        self.participant_id = participant_id or "self"
        self.mode = mode
        self.output_dir = output_dir
        self.start_time = None
        self.sample_count = 0
        self.gap_count = 0
        self._csv_writer = None
        self._csv_file = None
        self._last_device_ts = None

        os.makedirs(output_dir, exist_ok=True)

    def start(self, device_info=None, scale_factors=None):
        """Begin recording session."""
        self.start_time = datetime.now(timezone.utc)
        ts = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.filename = f"session_{ts}_{self.session_id}.csv"
        self.filepath = os.path.join(self.output_dir, self.filename)

        # Write metadata sidecar
        meta = {
            "taps_version": "0.1",
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "collection_mode": self.mode,
            "start_time": self.start_time.isoformat(),
            "device": device_info or {},
            "scale_factors": scale_factors or {},
        }
        meta_path = self.filepath.replace(".csv", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Open CSV
        self._csv_file = open(self.filepath, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "epoch_ms", "device_ts_ms", "sample_type", "channel",
            "x", "y", "z"
        ])
        print(f"[SESSION] Recording to {self.filepath}")
        return self

    def record_raw_sample(self, identifier, raw_sensor_data):
        """Write a raw sensor data packet to CSV.
        
        Each packet contains either finger accelerometer data (5 channels)
        or IMU data (gyro + accelerometer on thumb).
        """
        if self._csv_writer is None:
            return

        epoch_ms = int(time.time() * 1000)
        device_ts = raw_sensor_data.timestamp

        # Check for gaps
        if self._last_device_ts is not None:
            expected_gap = 5  # ~200Hz = 5ms between samples
            actual_gap = device_ts - self._last_device_ts
            if actual_gap > expected_gap * 3:
                self.gap_count += 1
        self._last_device_ts = device_ts

        data_type = str(raw_sensor_data.type)

        if "Device" in data_type:
            # Finger accelerometer data — 5 channels
            channels = [
                ("THUMB", 0), ("INDEX", 1), ("MIDDLE", 2),
                ("RING", 3), ("PINKY", 4)
            ]
            for ch_name, idx in channels:
                try:
                    point = raw_sensor_data.GetPoint(idx)
                    if point is not None:
                        self._csv_writer.writerow([
                            epoch_ms, device_ts, "ACCEL_FINGER", ch_name,
                            f"{point.x:.4f}", f"{point.y:.4f}", f"{point.z:.4f}"
                        ])
                        self.sample_count += 1
                except Exception:
                    pass

        elif "IMU" in data_type:
            # IMU data — gyro (index 0) + accelerometer (index 1)
            try:
                gyro = raw_sensor_data.GetPoint(0)
                if gyro is not None:
                    self._csv_writer.writerow([
                        epoch_ms, device_ts, "IMU_GYRO", "THUMB",
                        f"{gyro.x:.4f}", f"{gyro.y:.4f}", f"{gyro.z:.4f}"
                    ])
                    self.sample_count += 1
            except Exception:
                pass
            try:
                accel = raw_sensor_data.GetPoint(1)
                if accel is not None:
                    self._csv_writer.writerow([
                        epoch_ms, device_ts, "IMU_ACCEL", "THUMB",
                        f"{accel.x:.4f}", f"{accel.y:.4f}", f"{accel.z:.4f}"
                    ])
                    self.sample_count += 1
            except Exception:
                pass

        # Flush periodically
        if self.sample_count % 1000 == 0:
            self._csv_file.flush()

    def stop(self):
        """End session and close files."""
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None

        duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        # Update metadata with session stats
        meta_path = self.filepath.replace(".csv", "_meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["end_time"] = datetime.now(timezone.utc).isoformat()
        meta["duration_seconds"] = round(duration, 1)
        meta["total_samples"] = self.sample_count
        meta["gap_events"] = self.gap_count
        meta["integrity_pct"] = round(
            (1 - self.gap_count / max(self.sample_count / 5, 1)) * 100, 1
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n[SESSION] Complete.")
        print(f"  Duration:  {duration:.1f}s")
        print(f"  Samples:   {self.sample_count:,}")
        print(f"  Gaps:      {self.gap_count}")
        print(f"  Saved to:  {self.filepath}")

        return meta
