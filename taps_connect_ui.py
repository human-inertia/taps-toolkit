#!/usr/bin/env python3
"""
TAPS Connect UI — Simple GUI to scan, connect, and capture from Tap Strap 2.

Usage:
    python taps_connect_ui.py
"""

import asyncio
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import time
import os
import csv
import sys
import numpy as np
from datetime import datetime

# BLE protocol constants
NUS_RX_CHAR = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX_CHAR = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
TAP_SERVICE = "c3ff0001-1d8b-40fd-a56f-c7bd5d0f3370"
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
            msg_type = "accl"
            ts = ts_raw - MSG_TYPE_BIT
            n_samples = 15
        else:
            msg_type = "imu"
            ts = ts_raw
            n_samples = 6
        if ptr + n_samples * 2 > L:
            break
        payload = []
        for _ in range(n_samples):
            val = int.from_bytes(data[ptr:ptr+2], "little", signed=True)
            payload.append(val)
            ptr += 2
        messages.append({"type": msg_type, "ts": ts, "payload": payload})
    return messages


class TAPSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TAPS - Tap Strap 2 Live Test")
        self.root.geometry("600x700")
        self.root.configure(bg="#0a0e14")

        self.client = None
        self.raw_samples = []
        self.sample_count = 0
        self.capturing = False
        self.capture_start = None
        self.device_list = []
        self.loop = None
        self.ble_thread = None

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TButton", padding=6, font=("Consolas", 10))
        style.configure("TLabel", background="#0a0e14", foreground="#e6edf3", font=("Consolas", 10))
        style.configure("Header.TLabel", font=("Consolas", 14, "bold"), foreground="#58a6ff")

        # Header
        ttk.Label(self.root, text="TAPS Live Test", style="Header.TLabel").pack(pady=(10, 5))
        ttk.Label(self.root, text="Tap Assessment Protocol Standard", style="TLabel").pack()

        # Control frame
        ctrl = tk.Frame(self.root, bg="#0a0e14")
        ctrl.pack(fill="x", padx=10, pady=10)

        self.scan_btn = tk.Button(ctrl, text="SCAN", command=self.do_scan,
                                  bg="#1a2130", fg="#58a6ff", font=("Consolas", 11, "bold"),
                                  relief="flat", padx=15, pady=5)
        self.scan_btn.pack(side="left", padx=5)

        self.connect_btn = tk.Button(ctrl, text="CONNECT", command=self.do_connect,
                                     bg="#1a2130", fg="#3fb950", font=("Consolas", 11, "bold"),
                                     relief="flat", padx=15, pady=5, state="disabled")
        self.connect_btn.pack(side="left", padx=5)

        self.capture_btn = tk.Button(ctrl, text="CAPTURE 30s", command=self.do_capture,
                                     bg="#1a2130", fg="#f0c75e", font=("Consolas", 11, "bold"),
                                     relief="flat", padx=15, pady=5, state="disabled")
        self.capture_btn.pack(side="left", padx=5)

        self.analyze_btn = tk.Button(ctrl, text="ANALYZE", command=self.do_analyze,
                                     bg="#1a2130", fg="#d2a8ff", font=("Consolas", 11, "bold"),
                                     relief="flat", padx=15, pady=5, state="disabled")
        self.analyze_btn.pack(side="left", padx=5)

        # Device list
        self.device_listbox = tk.Listbox(self.root, bg="#111820", fg="#e6edf3",
                                          font=("Consolas", 10), height=5,
                                          selectbackground="#58a6ff", selectforeground="#0a0e14",
                                          relief="flat")
        self.device_listbox.pack(fill="x", padx=10, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready. Put Tap Strap 2 on hand and click SCAN.")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              bg="#1a2130", fg="#3fb950", font=("Consolas", 10),
                              anchor="w", padx=10, pady=5)
        status_bar.pack(fill="x", padx=10)

        # Live counters
        counter_frame = tk.Frame(self.root, bg="#0a0e14")
        counter_frame.pack(fill="x", padx=10, pady=5)

        self.count_var = tk.StringVar(value="Samples: 0")
        self.rate_var = tk.StringVar(value="Rate: 0/s")
        self.time_var = tk.StringVar(value="Time: 0s")

        for var, col in [(self.count_var, "#58a6ff"), (self.rate_var, "#3fb950"), (self.time_var, "#f0c75e")]:
            tk.Label(counter_frame, textvariable=var, bg="#111820", fg=col,
                     font=("Consolas", 12, "bold"), padx=15, pady=5).pack(side="left", padx=5, expand=True, fill="x")

        # Log output
        self.log = scrolledtext.ScrolledText(self.root, bg="#111820", fg="#8b949e",
                                              font=("Consolas", 9), height=20,
                                              relief="flat", insertbackground="#e6edf3")
        self.log.pack(fill="both", expand=True, padx=10, pady=5)

        self.csv_path = None

    def log_msg(self, msg):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def set_status(self, msg):
        self.status_var.set(msg)

    def _run_async(self, coro):
        """Run an async coroutine in a background thread with its own event loop."""
        def runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            try:
                loop.run_until_complete(coro)
            except Exception as e:
                self.root.after(0, lambda: self.log_msg(f"ERROR: {e}"))
            finally:
                loop.close()
        t = threading.Thread(target=runner, daemon=True)
        t.start()

    # --- SCAN ---
    def do_scan(self):
        self.scan_btn.config(state="disabled")
        self.device_listbox.delete(0, "end")
        self.device_list = []
        self.set_status("Scanning for BLE devices (10s)...")
        self.log_msg("Scanning...")
        self._run_async(self._scan())

    async def _scan(self):
        from bleak import BleakScanner
        devices = await BleakScanner.discover(timeout=10, return_adv=True)

        self.device_list = []
        for addr, (dev, adv) in devices.items():
            name = dev.name or (adv.local_name if adv else None) or ""
            rssi = adv.rssi if adv else -100
            self.device_list.append((dev, adv, name, rssi))

        # Sort: Tap devices first, then by RSSI
        self.device_list.sort(key=lambda x: (0 if "tap" in x[2].lower() else 1, -x[3]))

        def update_ui():
            self.device_listbox.delete(0, "end")
            for dev, adv, name, rssi in self.device_list:
                label = f"{name or '(unnamed)':<25} [{dev.address}]  RSSI={rssi}"
                self.device_listbox.insert("end", label)
                if "tap" in name.lower():
                    idx = self.device_listbox.size() - 1
                    self.device_listbox.itemconfig(idx, fg="#3fb950")
                    self.device_listbox.selection_set(idx)

            tap_found = any("tap" in x[2].lower() for x in self.device_list)
            self.scan_btn.config(state="normal")
            self.connect_btn.config(state="normal")
            if tap_found:
                self.set_status("Tap device found! Select and click CONNECT.")
                self.log_msg("Tap Strap detected!")
            else:
                self.set_status(f"Found {len(self.device_list)} devices. Select one and CONNECT.")
                self.log_msg(f"No 'Tap' device found. {len(self.device_list)} devices visible.")

        self.root.after(0, update_ui)

    # --- CONNECT ---
    def do_connect(self):
        sel = self.device_listbox.curselection()
        if not sel:
            self.set_status("Select a device first!")
            return
        idx = sel[0]
        dev, adv, name, rssi = self.device_list[idx]
        self.connect_btn.config(state="disabled")
        self.set_status(f"Connecting to {name or dev.address}...")
        self.log_msg(f"Connecting to {name} [{dev.address}]...")
        self._run_async(self._connect(dev))

    async def _connect(self, device):
        from bleak import BleakClient

        try:
            self.client = BleakClient(device.address, timeout=15.0)
            await self.client.connect()
        except Exception as e:
            self.root.after(0, lambda: self.log_msg(f"Connection failed: {e}"))
            self.root.after(0, lambda: self.set_status("Connection failed. Try again."))
            self.root.after(0, lambda: self.connect_btn.config(state="normal"))
            return

        def update_ui():
            self.log_msg("Connected!")
            for svc in self.client.services:
                self.log_msg(f"  Service: {svc.uuid}")
                for ch in svc.characteristics:
                    props = ",".join(ch.properties)
                    self.log_msg(f"    {ch.uuid} [{props}]")
            self.set_status("Connected! Click CAPTURE to start recording.")
            self.capture_btn.config(state="normal")
            self.connect_btn.config(state="disabled")

        self.root.after(0, update_ui)

    # --- CAPTURE ---
    def do_capture(self):
        self.capture_btn.config(state="disabled")
        self.raw_samples = []
        self.sample_count = 0
        self.capturing = True
        self.set_status("CAPTURING — tap your fingers on a hard surface!")
        self.log_msg("Starting 30s capture... TAP YOUR FINGERS!")
        self._run_async(self._capture(30))

    async def _capture(self, duration):
        if not self.client or not self.client.is_connected:
            self.root.after(0, lambda: self.log_msg("Not connected!"))
            return

        def on_notify(sender, data: bytearray):
            messages = parse_raw_packet(data)
            epoch_ms = int(time.time() * 1000)
            for msg in messages:
                self.raw_samples.append({
                    "epoch_ms": epoch_ms,
                    "device_ts_ms": msg["ts"],
                    "type": msg["type"],
                    "payload": msg["payload"],
                })
                self.sample_count += 1

        # Subscribe to all notify characteristics
        for svc in self.client.services:
            for ch in svc.characteristics:
                if "notify" in ch.properties:
                    try:
                        await self.client.start_notify(ch.uuid, on_notify)
                        self.root.after(0, lambda u=ch.uuid: self.log_msg(f"Subscribed: {u}"))
                    except Exception as e:
                        self.root.after(0, lambda u=ch.uuid, err=e: self.log_msg(f"Skip {u}: {err}"))

        # Send raw mode command
        cmd = bytearray([0x03, 0x0c, 0x00, 0x0a, 0x00, 0x00, 0x00])
        try:
            await self.client.write_gatt_char(NUS_RX_CHAR, cmd, response=False)
            self.root.after(0, lambda: self.log_msg("Raw mode command sent"))
        except Exception as e:
            self.root.after(0, lambda: self.log_msg(f"Write NUS RX failed: {e}"))
            # Try UI CMD char
            try:
                ui_cmd = "c3ff0009-1d8b-40fd-a56f-c7bd5d0f3370"
                await self.client.write_gatt_char(ui_cmd, cmd, response=False)
                self.root.after(0, lambda: self.log_msg("Raw mode sent via UI CMD"))
            except Exception as e2:
                self.root.after(0, lambda: self.log_msg(f"UI CMD also failed: {e2}"))

        self.capture_start = time.time()

        # Update counters in UI
        def update_counters():
            if not self.capturing:
                return
            elapsed = time.time() - self.capture_start
            remaining = max(0, duration - elapsed)
            rate = self.sample_count / max(elapsed, 0.1)
            self.count_var.set(f"Samples: {self.sample_count:,}")
            self.rate_var.set(f"Rate: {rate:.0f}/s")
            self.time_var.set(f"Time: {remaining:.0f}s left")
            if remaining > 0:
                self.root.after(500, update_counters)

        self.root.after(100, update_counters)

        # Wait for duration
        await asyncio.sleep(duration)
        self.capturing = False

        # Stop notifications
        for svc in self.client.services:
            for ch in svc.characteristics:
                if "notify" in ch.properties:
                    try:
                        await self.client.stop_notify(ch.uuid)
                    except:
                        pass

        # Save CSV
        os.makedirs("data", exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join("data", f"live_test_{ts_str}.csv")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch_ms", "device_ts_ms", "sample_type", "channel", "x", "y", "z"])
            for s in self.raw_samples:
                if s["type"] == "accl" and len(s["payload"]) == 15:
                    for i, finger in enumerate(FINGERS):
                        x = s["payload"][i * 3]
                        y = s["payload"][i * 3 + 1]
                        z = s["payload"][i * 3 + 2]
                        writer.writerow([s["epoch_ms"], s["device_ts_ms"], "ACCEL_FINGER", finger, x, y, z])
                elif s["type"] == "imu" and len(s["payload"]) == 6:
                    gx, gy, gz = s["payload"][0:3]
                    ax, ay, az = s["payload"][3:6]
                    writer.writerow([s["epoch_ms"], s["device_ts_ms"], "IMU_GYRO", "THUMB", gx, gy, gz])
                    writer.writerow([s["epoch_ms"], s["device_ts_ms"], "IMU_ACCEL", "THUMB", ax, ay, az])

        def done():
            self.log_msg(f"\nCapture complete: {self.sample_count:,} samples")
            self.log_msg(f"Saved to: {self.csv_path}")
            self.count_var.set(f"Samples: {self.sample_count:,}")
            self.rate_var.set("Done")
            self.time_var.set("0s left")
            if self.sample_count > 0:
                self.set_status("Capture saved! Click ANALYZE to run pipeline.")
                self.analyze_btn.config(state="normal")
            else:
                self.set_status("No data received. Try reconnecting.")
            self.capture_btn.config(state="normal")

        self.root.after(0, done)

    # --- ANALYZE ---
    def do_analyze(self):
        if not self.csv_path:
            return
        self.analyze_btn.config(state="disabled")
        self.set_status("Running TAPS analysis pipeline...")
        self.log_msg(f"\nAnalyzing: {self.csv_path}")

        def run_analysis():
            try:
                from taps.processing.tap_detection import load_and_detect
                from taps.assessment.parameters import compute_parameters, compute_composite_scores
                from taps.reporting.export import export_research_bundle

                taps = load_and_detect(self.csv_path)

                if not taps:
                    self.root.after(0, lambda: self.log_msg("No taps detected in data."))
                    self.root.after(0, lambda: self.set_status("No taps found. Try tapping harder."))
                    self.root.after(0, lambda: self.analyze_btn.config(state="normal"))
                    return

                params = compute_parameters(taps)
                composites = compute_composite_scores(params)

                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_dir = os.path.join("data", f"results_{ts_str}")
                export_research_bundle(params, composites, taps, results_dir)

                def show_results():
                    self.log_msg("\n" + "=" * 55)
                    self.log_msg("  TAPS ASSESSMENT RESULTS")
                    self.log_msg("=" * 55)
                    self.log_msg(f"  Taps detected:     {params.tap_count}")
                    self.log_msg(f"  Duration:          {params.epoch_duration_s:.1f}s")
                    self.log_msg("-" * 55)
                    self.log_msg(f"  P1  Tap Count          {params.tap_count:>8}")
                    self.log_msg(f"  P2  Mean ITI           {params.iti_mean_ms:>8.1f} ms")
                    self.log_msg(f"  P3  ITI Std Dev        {params.iti_sd_ms:>8.1f} ms")
                    self.log_msg(f"  P4  ITI Coeff Var      {params.iti_cv:>8.3f}")
                    self.log_msg(f"  P5  Tap Duration       {params.tap_duration_mean_ms:>8.1f} ms")
                    self.log_msg(f"  P6  Freezing Count     {params.freezing_count:>8}")
                    self.log_msg(f"  P7  Co-Activation      {params.coactivation_index:>8.3f}")
                    self.log_msg(f"  P8  Bilateral Asym     {params.bilateral_asymmetry:>8.3f}")
                    self.log_msg(f"  P9  Fatigue Slope      {params.fatigue_slope:>8.3f} ms/s")
                    self.log_msg(f"  P10 Rhythm Entropy     {params.rhythm_entropy:>8.3f} bits")
                    self.log_msg("-" * 55)
                    self.log_msg(f"  TMI  Motor Index       {composites['taps_motor_index']:>8.1f}")
                    self.log_msg(f"  TVI  Variability       {composites['taps_variability_index']:>8.1f}")
                    self.log_msg(f"  TCI  Coordination      {composites['taps_coordination_index']:>8.1f}")
                    self.log_msg(f"  TCS  COMPOSITE         {composites['taps_composite_score']:>8.1f} / 100")
                    self.log_msg("=" * 55)

                    # Per finger
                    self.log_msg("\n  Per-finger:")
                    for finger in FINGERS:
                        ft = [t for t in taps if t.channel == finger]
                        if len(ft) >= 3:
                            fp = compute_parameters(ft)
                            self.log_msg(f"    {finger:<8}  TC={fp.tap_count:>3}  ITI={fp.iti_mean_ms:>7.1f}ms  SD={fp.iti_sd_ms:>6.1f}ms")

                    self.log_msg(f"\n  Results saved to: {results_dir}/")
                    self.set_status(f"TCS = {composites['taps_composite_score']:.1f}/100  |  {params.tap_count} taps detected")
                    self.analyze_btn.config(state="normal")

                self.root.after(0, show_results)
            except Exception as e:
                self.root.after(0, lambda: self.log_msg(f"Analysis error: {e}"))
                self.root.after(0, lambda: self.analyze_btn.config(state="normal"))

        threading.Thread(target=run_analysis, daemon=True).start()


def main():
    root = tk.Tk()
    app = TAPSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
