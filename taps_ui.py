#!/usr/bin/env python3
"""TAPS Connect — Native Qt6 UI for Tap Strap 2 live test."""

import asyncio
import threading
import sys
import os
import csv
import time
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QTextEdit, QLabel, QFrame, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor

# BLE
NUS_RX_CHAR = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX_CHAR = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
MSG_TYPE_BIT = 2**31
FINGERS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]


def parse_raw_packet(data: bytearray):
    messages = []
    ptr = 0
    L = len(data)
    while ptr + 4 <= L:
        ts_raw = int.from_bytes(data[ptr:ptr+4], "little", signed=False)
        if ts_raw == 0:
            break
        ptr += 4
        if ts_raw > MSG_TYPE_BIT:
            msg_type, ts, n = "accl", ts_raw - MSG_TYPE_BIT, 15
        else:
            msg_type, ts, n = "imu", ts_raw, 6
        if ptr + n * 2 > L:
            break
        payload = []
        for _ in range(n):
            payload.append(int.from_bytes(data[ptr:ptr+2], "little", signed=True))
            ptr += 2
        messages.append({"type": msg_type, "ts": ts, "payload": payload})
    return messages


class Signals(QObject):
    log = pyqtSignal(str)
    status = pyqtSignal(str)
    devices_found = pyqtSignal(list)
    connected = pyqtSignal()
    capture_done = pyqtSignal(str, int)
    analysis_done = pyqtSignal(str)
    counter_update = pyqtSignal(int, float, float)


class TAPSWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TAPS — Tap Strap 2 Live Test")
        self.setMinimumSize(650, 750)
        self.setStyleSheet("""
            QMainWindow { background: #0a0e14; }
            QLabel { color: #e6edf3; }
            QListWidget { background: #111820; color: #e6edf3; border: 1px solid #1e2a3a;
                          font-family: Consolas; font-size: 11px; }
            QListWidget::item:selected { background: #58a6ff; color: #0a0e14; }
            QTextEdit { background: #111820; color: #8b949e; border: 1px solid #1e2a3a;
                        font-family: Consolas; font-size: 10px; }
        """)

        self.sig = Signals()
        self.sig.log.connect(self._append_log)
        self.sig.status.connect(self._set_status)
        self.sig.devices_found.connect(self._show_devices)
        self.sig.connected.connect(self._on_connected)
        self.sig.capture_done.connect(self._on_capture_done)
        self.sig.analysis_done.connect(self._on_analysis_done)
        self.sig.counter_update.connect(self._update_counters)

        self.client = None
        self.device_list = []
        self.raw_samples = []
        self.sample_count = 0
        self.capturing = False
        self.capture_start = None
        self.csv_path = None

        self._build_ui()

        # Timer for live counters during capture
        self.counter_timer = QTimer()
        self.counter_timer.timeout.connect(self._poll_counters)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header
        title = QLabel("TAPS Live Test")
        title.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #58a6ff;")
        layout.addWidget(title)

        sub = QLabel("Tap Assessment Protocol Standard")
        sub.setFont(QFont("Consolas", 10))
        sub.setStyleSheet("color: #484f58;")
        layout.addWidget(sub)

        # Buttons
        btn_row = QHBoxLayout()
        btn_style = """
            QPushButton { background: #1a2130; border: 1px solid #1e2a3a; border-radius: 6px;
                          padding: 8px 18px; font-family: Consolas; font-size: 12px; font-weight: bold; }
            QPushButton:hover { background: #222d40; }
            QPushButton:disabled { color: #484f58; }
        """
        self.scan_btn = QPushButton("SCAN")
        self.scan_btn.setStyleSheet(btn_style + "QPushButton { color: #58a6ff; }")
        self.scan_btn.clicked.connect(self.do_scan)

        self.connect_btn = QPushButton("CONNECT")
        self.connect_btn.setStyleSheet(btn_style + "QPushButton { color: #3fb950; }")
        self.connect_btn.setEnabled(False)
        self.connect_btn.clicked.connect(self.do_connect)

        self.capture_btn = QPushButton("CAPTURE 30s")
        self.capture_btn.setStyleSheet(btn_style + "QPushButton { color: #f0c75e; }")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self.do_capture)

        self.analyze_btn = QPushButton("ANALYZE")
        self.analyze_btn.setStyleSheet(btn_style + "QPushButton { color: #d2a8ff; }")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.do_analyze)

        self.load_btn = QPushButton("LOAD FILE")
        self.load_btn.setStyleSheet(btn_style + "QPushButton { color: #f78166; }")
        self.load_btn.clicked.connect(self.do_load_file)

        for b in [self.scan_btn, self.connect_btn, self.capture_btn, self.analyze_btn, self.load_btn]:
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        # Device list
        self.dev_list = QListWidget()
        self.dev_list.setMaximumHeight(130)
        layout.addWidget(self.dev_list)

        # Status
        self.status_label = QLabel("Ready. Put Tap Strap 2 on hand and click SCAN.")
        self.status_label.setFont(QFont("Consolas", 10))
        self.status_label.setStyleSheet("background: #1a2130; color: #3fb950; padding: 6px; border-radius: 4px;")
        layout.addWidget(self.status_label)

        # Counters
        counter_row = QHBoxLayout()
        self.count_label = QLabel("Samples: 0")
        self.rate_label = QLabel("Rate: 0/s")
        self.time_label = QLabel("Time: --")
        for lbl, color in [(self.count_label, "#58a6ff"), (self.rate_label, "#3fb950"), (self.time_label, "#f0c75e")]:
            lbl.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
            lbl.setStyleSheet(f"background: #111820; color: {color}; padding: 8px; border-radius: 4px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            counter_row.addWidget(lbl)
        layout.addLayout(counter_row)

        # Log
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_box)

    def _append_log(self, msg):
        self.log_box.append(msg)

    def _set_status(self, msg):
        self.status_label.setText(msg)

    def _show_devices(self, devices):
        self.dev_list.clear()
        self.device_list = devices
        for dev, adv, name, rssi in devices:
            label = f"{name or '(unnamed)':<25} [{dev.address}]  RSSI={rssi}"
            self.dev_list.addItem(label)
        # Auto-select Tap device
        for i, (_, _, name, _) in enumerate(devices):
            if "tap" in name.lower():
                self.dev_list.setCurrentRow(i)
        self.scan_btn.setEnabled(True)
        self.connect_btn.setEnabled(True)

    def _on_connected(self):
        self.capture_btn.setEnabled(True)
        self.connect_btn.setEnabled(False)

    def _on_capture_done(self, csv_path, count):
        self.csv_path = csv_path
        self.counter_timer.stop()
        self.count_label.setText(f"Samples: {count:,}")
        self.rate_label.setText("Done")
        self.time_label.setText("0s")
        if count > 0:
            self.analyze_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)

    def _on_analysis_done(self, summary):
        self.log_box.append(summary)
        self.analyze_btn.setEnabled(True)

    def _update_counters(self, count, rate, remaining):
        self.count_label.setText(f"Samples: {count:,}")
        self.rate_label.setText(f"Rate: {rate:.0f}/s")
        self.time_label.setText(f"{remaining:.0f}s left")

    def _poll_counters(self):
        if self.capturing and self.capture_start:
            elapsed = time.time() - self.capture_start
            rate = self.sample_count / max(elapsed, 0.1)
            remaining = max(0, 30 - elapsed)
            self.sig.counter_update.emit(self.sample_count, rate, remaining)

    def _run_bg(self, coro):
        def runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(coro)
            except Exception as e:
                self.sig.log.emit(f"ERROR: {e}")
            finally:
                loop.close()
        threading.Thread(target=runner, daemon=True).start()

    # --- SCAN ---
    def do_scan(self):
        self.scan_btn.setEnabled(False)
        self.sig.status.emit("Scanning (10s)...")
        self.sig.log.emit("Scanning for BLE devices...")
        self._run_bg(self._scan())

    async def _scan(self):
        from bleak import BleakScanner
        devices = await BleakScanner.discover(timeout=10, return_adv=True)
        result = []
        for addr, (dev, adv) in devices.items():
            name = dev.name or (adv.local_name if adv else None) or ""
            rssi = adv.rssi if adv else -100
            result.append((dev, adv, name, rssi))
        result.sort(key=lambda x: (0 if "tap" in x[2].lower() else 1, -x[3]))

        tap_found = any("tap" in x[2].lower() for x in result)
        if tap_found:
            self.sig.status.emit("Tap found! Select it and click CONNECT.")
            self.sig.log.emit("Tap Strap detected!")
        else:
            self.sig.status.emit(f"{len(result)} devices found. Select one.")
            self.sig.log.emit(f"No 'Tap' name found. {len(result)} devices.")
        self.sig.devices_found.emit(result)

    # --- CONNECT ---
    def do_connect(self):
        sel = self.dev_list.currentRow()
        if sel < 0:
            self.sig.status.emit("Select a device first!")
            return
        dev, adv, name, rssi = self.device_list[sel]
        self.connect_btn.setEnabled(False)
        self.sig.status.emit(f"Connecting to {name or dev.address}...")
        self.sig.log.emit(f"Connecting to {name} [{dev.address}]...")
        self._run_bg(self._connect(dev))

    async def _connect(self, device):
        from bleak import BleakClient
        try:
            self.client = BleakClient(device.address, timeout=15.0)
            await self.client.connect()
        except Exception as e:
            self.sig.log.emit(f"Connection failed: {e}")
            self.sig.status.emit("Connection failed.")
            return

        self.sig.log.emit("Connected!")
        for svc in self.client.services:
            self.sig.log.emit(f"  {svc.uuid}")
            for ch in svc.characteristics:
                self.sig.log.emit(f"    {ch.uuid} [{','.join(ch.properties)}]")
        self.sig.status.emit("Connected! Click CAPTURE.")
        self.sig.connected.emit()

    # --- CAPTURE ---
    def do_capture(self):
        self.capture_btn.setEnabled(False)
        self.raw_samples = []
        self.sample_count = 0
        self.capturing = True
        self.sig.status.emit("CAPTURING — TAP YOUR FINGERS!")
        self.sig.log.emit("\nCapturing 30s... TAP NOW!")
        self.counter_timer.start(500)
        self._run_bg(self._capture(30))

    async def _capture(self, duration):
        if not self.client or not self.client.is_connected:
            self.sig.log.emit("Not connected!")
            return

        def on_notify(sender, data: bytearray):
            msgs = parse_raw_packet(data)
            epoch_ms = int(time.time() * 1000)
            for m in msgs:
                self.raw_samples.append({"epoch_ms": epoch_ms, "device_ts_ms": m["ts"],
                                         "type": m["type"], "payload": m["payload"]})
                self.sample_count += 1

        # Subscribe all notify chars
        for svc in self.client.services:
            for ch in svc.characteristics:
                if "notify" in ch.properties:
                    try:
                        await self.client.start_notify(ch.uuid, on_notify)
                        self.sig.log.emit(f"Subscribed: {ch.uuid}")
                    except Exception as e:
                        self.sig.log.emit(f"Skip {ch.uuid}: {e}")

        # Send raw mode
        cmd = bytearray([0x03, 0x0c, 0x00, 0x0a, 0x00, 0x00, 0x00])
        try:
            await self.client.write_gatt_char(NUS_RX_CHAR, cmd, response=False)
            self.sig.log.emit("Raw mode command sent")
        except Exception as e:
            self.sig.log.emit(f"NUS RX write failed: {e}")
            try:
                await self.client.write_gatt_char("c3ff0009-1d8b-40fd-a56f-c7bd5d0f3370", cmd, response=False)
                self.sig.log.emit("Sent via UI CMD char")
            except Exception as e2:
                self.sig.log.emit(f"UI CMD failed: {e2}")

        self.capture_start = time.time()
        await asyncio.sleep(duration)
        self.capturing = False

        # Stop
        for svc in self.client.services:
            for ch in svc.characteristics:
                if "notify" in ch.properties:
                    try:
                        await self.client.stop_notify(ch.uuid)
                    except:
                        pass

        # Save CSV
        os.makedirs("data", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join("data", f"live_{ts}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch_ms", "device_ts_ms", "sample_type", "channel", "x", "y", "z"])
            for s in self.raw_samples:
                if s["type"] == "accl" and len(s["payload"]) == 15:
                    for i, finger in enumerate(FINGERS):
                        w.writerow([s["epoch_ms"], s["device_ts_ms"], "ACCEL_FINGER", finger,
                                    s["payload"][i*3], s["payload"][i*3+1], s["payload"][i*3+2]])
                elif s["type"] == "imu" and len(s["payload"]) == 6:
                    w.writerow([s["epoch_ms"], s["device_ts_ms"], "IMU_GYRO", "THUMB",
                                s["payload"][0], s["payload"][1], s["payload"][2]])
                    w.writerow([s["epoch_ms"], s["device_ts_ms"], "IMU_ACCEL", "THUMB",
                                s["payload"][3], s["payload"][4], s["payload"][5]])

        self.sig.log.emit(f"Saved: {csv_path} ({self.sample_count:,} samples)")
        self.sig.capture_done.emit(csv_path, self.sample_count)

    # --- ANALYZE ---
    def do_analyze(self):
        if not self.csv_path:
            return
        self.analyze_btn.setEnabled(False)
        self.sig.status.emit("Analyzing...")
        # Route to correct analyzer based on file type
        threading.Thread(target=self._analyze_csv, args=(self.csv_path,), daemon=True).start()

    def _analyze(self):
        try:
            from taps.processing.tap_detection import load_and_detect
            from taps.assessment.parameters import compute_parameters, compute_composite_scores
            from taps.reporting.export import export_research_bundle

            taps = load_and_detect(self.csv_path)
            if not taps:
                self.sig.log.emit("No taps detected.")
                self.sig.status.emit("No taps found. Tap harder next time.")
                self.sig.analysis_done.emit("")
                return

            params = compute_parameters(taps)
            comp = compute_composite_scores(params)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            rdir = os.path.join("data", f"results_{ts}")
            export_research_bundle(params, comp, taps, rdir)

            lines = [
                "", "=" * 55, "  TAPS ASSESSMENT RESULTS", "=" * 55,
                f"  Taps: {params.tap_count}  |  Duration: {params.epoch_duration_s:.1f}s",
                "-" * 55,
                f"  P1  Tap Count        {params.tap_count:>8}",
                f"  P2  Mean ITI         {params.iti_mean_ms:>8.1f} ms",
                f"  P3  ITI Std Dev      {params.iti_sd_ms:>8.1f} ms",
                f"  P4  ITI CV           {params.iti_cv:>8.3f}",
                f"  P5  Tap Duration     {params.tap_duration_mean_ms:>8.1f} ms",
                f"  P6  Freezing         {params.freezing_count:>8}",
                f"  P7  Co-Activation    {params.coactivation_index:>8.3f}",
                f"  P8  Bilateral Asym   {params.bilateral_asymmetry:>8.3f}",
                f"  P9  Fatigue Slope    {params.fatigue_slope:>8.3f} ms/s",
                f"  P10 Rhythm Entropy   {params.rhythm_entropy:>8.3f} bits",
                "-" * 55,
                f"  TMI  Motor           {comp['taps_motor_index']:>8.1f}",
                f"  TVI  Variability     {comp['taps_variability_index']:>8.1f}",
                f"  TCI  Coordination    {comp['taps_coordination_index']:>8.1f}",
                f"  TCS  COMPOSITE       {comp['taps_composite_score']:>8.1f} / 100",
                "=" * 55, "",
            ]

            for finger in FINGERS:
                ft = [t for t in taps if t.channel == finger]
                if len(ft) >= 3:
                    fp = compute_parameters(ft)
                    lines.append(f"  {finger:<8} TC={fp.tap_count:>3}  ITI={fp.iti_mean_ms:>6.1f}ms  SD={fp.iti_sd_ms:>5.1f}ms")

            lines.append(f"\n  Results: {rdir}/")

            self.sig.status.emit(f"TCS = {comp['taps_composite_score']:.1f}/100  |  {params.tap_count} taps")
            self.sig.analysis_done.emit("\n".join(lines))

        except Exception as e:
            self.sig.log.emit(f"Analysis error: {e}")
            self.sig.analysis_done.emit("")

    # --- LOAD FILE ---
    def do_load_file(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load TAPS Data", data_dir,
            "Data files (*.csv *.json);;All files (*)")
        if not path:
            return

        self.sig.log.emit(f"\nLoading: {path}")

        if path.endswith(".json"):
            # Load tap_results JSON from tap mode test
            threading.Thread(target=self._analyze_json, args=(path,), daemon=True).start()
        elif path.endswith(".csv"):
            # Could be raw sensor CSV or tap_events CSV
            self.csv_path = path
            threading.Thread(target=self._analyze_csv, args=(path,), daemon=True).start()

    def _analyze_json(self, path):
        """Display results from a tap_results JSON file."""
        try:
            import json
            with open(path) as f:
                data = json.load(f)

            lines = [
                "", "=" * 55,
                "  TAPS TAP MODE RESULTS (loaded from file)",
                "=" * 55,
                f"  Device:      {data.get('device', '?')}",
                f"  Duration:    {data.get('duration_s', '?')}s",
                f"  Total taps:  {data.get('total_taps', '?')}",
                f"  Tap rate:    {data.get('taps_per_second', '?')}/sec",
                "-" * 55,
            ]

            if "iti_mean_ms" in data:
                lines.append(f"  Mean ITI:      {data['iti_mean_ms']:>8.1f} ms")
                lines.append(f"  ITI Std Dev:   {data['iti_sd_ms']:>8.1f} ms")
                lines.append(f"  ITI CV:        {data['iti_cv']:>8.3f}")

            fc = data.get("finger_counts", {})
            if fc:
                lines.append("\n  Finger Breakdown:")
                for finger in FINGERS:
                    count = fc.get(finger, 0)
                    bar = "#" * min(count, 40)
                    lines.append(f"    {finger:<8} {count:>4}  {bar}")

            cc = data.get("combo_counts", {})
            if cc:
                lines.append("\n  Top Combos:")
                for combo, count in sorted(cc.items(), key=lambda x: -x[1])[:8]:
                    lines.append(f"    {combo:<25} {count:>4}")

            lines.append("=" * 55)

            self.sig.status.emit(f"Loaded: {data.get('total_taps', '?')} taps from {os.path.basename(path)}")
            self.sig.analysis_done.emit("\n".join(lines))

        except Exception as e:
            self.sig.log.emit(f"Error loading JSON: {e}")

    def _analyze_csv(self, path):
        """Analyze a CSV file — detect if it's raw sensor or tap events format."""
        try:
            with open(path, encoding="utf-8-sig") as f:
                header = f.readline().strip().lower()

            self.sig.log.emit(f"CSV header: {header}")

            if "timestamp_s" in header or "fingers" in header:
                # Tap events CSV from tap_mode_test
                self.sig.log.emit("Tap events CSV detected.")
                self._analyze_tap_events_csv(path)
            elif "sample_type" in header:
                # Raw sensor CSV — use full pipeline
                self.sig.log.emit("Raw sensor CSV detected. Running full pipeline...")
                self._analyze()
            else:
                self.sig.log.emit(f"Unknown CSV format: {header[:80]}")

        except Exception as e:
            self.sig.log.emit(f"Error: {e}")

    def _analyze_tap_events_csv(self, path):
        """Analyze tap_events CSV from controller mode capture."""
        try:
            timestamps = []
            finger_events = []
            from collections import Counter

            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamps.append(float(row["timestamp_s"]))
                    finger_events.append(row["fingers"].split("+"))

            if not timestamps:
                self.sig.log.emit("No data in CSV.")
                return

            itis = np.diff(timestamps) * 1000
            itis = itis[itis > 30]

            finger_counter = Counter()
            for fingers in finger_events:
                for f in fingers:
                    finger_counter[f] += 1

            duration = timestamps[-1] - timestamps[0]

            lines = [
                "", "=" * 55,
                "  TAPS TAP EVENT ANALYSIS (from CSV)",
                "=" * 55,
                f"  Total taps:  {len(timestamps)}",
                f"  Duration:    {duration:.1f}s",
                f"  Tap rate:    {len(timestamps)/max(duration, 0.1):.1f}/sec",
                "-" * 55,
            ]

            if len(itis) > 1:
                lines.append(f"  Mean ITI:       {np.mean(itis):>8.1f} ms")
                lines.append(f"  ITI Std Dev:    {np.std(itis):>8.1f} ms")
                lines.append(f"  ITI CV:         {np.std(itis)/np.mean(itis):>8.3f}")
                lines.append(f"  Min ITI:        {np.min(itis):>8.1f} ms")
                lines.append(f"  Max ITI:        {np.max(itis):>8.1f} ms")

                if np.mean(itis) > 0:
                    freeze_count = int(np.sum(itis > 2 * np.mean(itis)))
                    lines.append(f"  Freezing:       {freeze_count:>8}")

                if len(itis) >= 5:
                    slope, _ = np.polyfit(np.arange(len(itis)), itis, 1)
                    lines.append(f"  Fatigue slope:  {slope:>8.2f} ms/tap")

                    n_bins = min(15, len(itis) // 3)
                    if n_bins >= 2:
                        hist, _ = np.histogram(itis, bins=n_bins, density=True)
                        hist = hist[hist > 0]
                        bw = (np.max(itis) - np.min(itis)) / n_bins
                        if bw > 0:
                            probs = hist * bw
                            probs = probs[probs > 0]
                            entropy = -np.sum(probs * np.log2(probs))
                            lines.append(f"  Rhythm entropy: {entropy:>8.3f} bits")

            lines.append("\n  Finger Breakdown:")
            for finger in FINGERS:
                count = finger_counter.get(finger, 0)
                bar = "#" * min(count, 40)
                lines.append(f"    {finger:<8} {count:>4}  {bar}")

            lines.append("=" * 55)

            self.sig.status.emit(f"{len(timestamps)} taps analyzed from {os.path.basename(path)}")
            self.sig.analysis_done.emit("\n".join(lines))

        except Exception as e:
            self.sig.log.emit(f"Error analyzing tap CSV: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TAPSWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
