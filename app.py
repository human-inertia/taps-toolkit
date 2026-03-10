#!/usr/bin/env python3
"""TAPS Web — Hosted Streamlit app with Web Bluetooth + full analysis pipeline."""

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import time
from datetime import datetime, timezone
from io import StringIO

# TAPS pipeline imports
from taps.processing.tap_detection import (
    TapEvent, detect_taps_single_channel, detect_taps_multichannel
)
from taps.processing.filters import (
    bandpass_filter, highpass_filter, compute_magnitude, estimate_sample_rate
)
from taps.assessment.parameters import (
    TAPSParameters, compute_parameters, compute_composite_scores
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TAPS — Tap Assessment Protocol",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

    .stApp { background: #0a0e14; }

    .taps-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .taps-header h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        color: #58a6ff;
        margin: 0;
        letter-spacing: -1px;
    }
    .taps-header p {
        font-family: 'JetBrains Mono', monospace;
        color: #484f58;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }

    .mode-card {
        background: #111820;
        border: 1px solid #1e2a3a;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
    }
    .mode-card:hover {
        border-color: #58a6ff;
        background: #141d28;
    }
    .mode-card h3 {
        font-family: 'JetBrains Mono', monospace;
        color: #e6edf3;
        margin: 0.5rem 0 0.3rem;
    }
    .mode-card p {
        color: #8b949e;
        font-size: 0.85rem;
        margin: 0;
    }

    .metric-box {
        background: #111820;
        border: 1px solid #1e2a3a;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-box .label {
        font-family: 'JetBrains Mono', monospace;
        color: #8b949e;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-box .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.2rem 0;
    }
    .metric-box .unit {
        color: #484f58;
        font-size: 0.75rem;
    }

    .param-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0.8rem;
        border-bottom: 1px solid #1e2a3a;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    .param-row .name { color: #8b949e; }
    .param-row .val { color: #e6edf3; font-weight: 600; }

    .finger-bar-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.3rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    .finger-bar-container .fname { color: #8b949e; width: 70px; text-align: right; }
    .finger-bar-container .fcount { color: #e6edf3; width: 35px; text-align: right; font-weight: 600; }
    .finger-bar {
        height: 20px;
        border-radius: 4px;
        transition: width 0.5s;
    }

    .score-ring {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        margin: 0 auto;
    }
    .score-ring .score-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1;
    }
    .score-ring .score-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .section-title {
        font-family: 'JetBrains Mono', monospace;
        color: #58a6ff;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #1e2a3a;
    }

    .bt-status {
        font-family: 'JetBrains Mono', monospace;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        text-align: center;
        font-size: 0.9rem;
    }
    .bt-connected { background: #0d2818; border: 1px solid #3fb950; color: #3fb950; }
    .bt-disconnected { background: #1a1520; border: 1px solid #8b949e; color: #8b949e; }
    .bt-capturing { background: #2a1a00; border: 1px solid #f0c75e; color: #f0c75e; }

    /* Hide streamlit defaults */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    div[data-testid="stHorizontalBlock"] { gap: 0.8rem; }

    .stButton > button {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# State init
# ---------------------------------------------------------------------------
if "mode" not in st.session_state:
    st.session_state.mode = None  # None, "bluetooth", "upload", "demo"
if "results" not in st.session_state:
    st.session_state.results = None
if "bt_tap_data" not in st.session_state:
    st.session_state.bt_tap_data = None

# ---------------------------------------------------------------------------
# DEMO DATA — real capture from Tap_D185
# ---------------------------------------------------------------------------
DEMO_TAP_EVENTS_CSV = """timestamp_s,fingers,finger_count
1773118026.507,THUMB,1
1773118026.836,THUMB,1
1773118027.197,INDEX,1
1773118027.512,MIDDLE,1
1773118027.707,MIDDLE+RING,2
1773118028.172,PINKY,1
1773118028.981,INDEX+MIDDLE,2
1773118029.162,INDEX+MIDDLE,2
1773118029.281,INDEX,1
1773118029.566,THUMB,1
1773118029.911,THUMB,1
1773118030.181,INDEX,1
1773118030.437,MIDDLE,1
1773118030.661,MIDDLE,1
1773118030.916,MIDDLE+RING,2
1773118031.276,THUMB,1
1773118031.621,THUMB,1
1773118031.906,THUMB,1
1773118032.207,INDEX,1
1773118032.426,MIDDLE,1
1773118032.706,PINKY+THUMB,1
1773118032.986,RING+THUMB,2
1773118033.226,INDEX+MIDDLE,2
1773118033.550,THUMB,1
1773118033.836,INDEX,1
1773118034.131,MIDDLE+THUMB,2
1773118034.376,THUMB,1
1773118034.656,INDEX,1
1773118034.941,INDEX+THUMB,2
1773118035.236,MIDDLE,1
1773118035.556,THUMB,1
1773118035.826,INDEX,1
1773118036.151,INDEX,1
1773118036.392,MIDDLE,1
1773118036.711,PINKY+THUMB,2
1773118036.966,RING+THUMB,2
1773118037.291,INDEX+MIDDLE,2
1773118037.512,THUMB,1
1773118037.826,MIDDLE+THUMB,2
1773118038.131,INDEX+THUMB,2
1773118038.416,MIDDLE,1
1773118038.721,PINKY,1
1773118039.096,RING+THUMB,2
1773118039.406,MIDDLE+PINKY,2
1773118039.706,THUMB,1
1773118040.006,INDEX+MIDDLE,2
1773118040.251,MIDDLE+RING,2
1773118040.566,THUMB,1
1773118040.871,THUMB,1
1773118041.186,INDEX,1
1773118041.476,MIDDLE+THUMB,2
1773118041.766,PINKY+THUMB,2
1773118042.071,INDEX,1
1773118042.356,INDEX+MIDDLE,2
1773118042.696,MIDDLE,1
1773118042.931,RING+THUMB,2
1773118043.246,INDEX,1
1773118043.581,PINKY+THUMB,2
1773118043.926,THUMB,1
1773118044.191,MIDDLE+THUMB,2
1773118044.531,PINKY+THUMB,2
1773118044.821,INDEX+PINKY+RING,3
1773118045.146,MIDDLE+PINKY+RING+THUMB,4
1773118045.566,INDEX+MIDDLE+THUMB,3
1773118046.066,INDEX+MIDDLE+PINKY+THUMB,4
1773118046.706,INDEX+THUMB,2
"""

DEMO_RESULTS_JSON = {
    "taps_version": "0.1",
    "test_type": "tap_mode",
    "device": "Tap_D185",
    "duration_s": 30.1,
    "total_taps": 63,
    "taps_per_second": 2.09,
    "finger_counts": {"THUMB": 35, "INDEX": 20, "MIDDLE": 26, "RING": 10, "PINKY": 11},
    "combo_counts": {
        "THUMB": 15, "INDEX": 7, "MIDDLE": 6, "MIDDLE+RING": 4, "PINKY": 2,
        "INDEX+MIDDLE": 7, "MIDDLE+PINKY": 1, "PINKY+THUMB": 5, "RING+THUMB": 4,
        "MIDDLE+THUMB": 5, "INDEX+THUMB": 3, "INDEX+PINKY+RING": 1,
        "MIDDLE+PINKY+RING+THUMB": 1, "INDEX+MIDDLE+THUMB": 1,
        "INDEX+MIDDLE+PINKY+THUMB": 1,
    },
    "iti_mean_ms": 385.9,
    "iti_sd_ms": 197.9,
    "iti_cv": 0.513,
}

FINGERS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
FINGER_COLORS = {
    "THUMB": "#58a6ff",
    "INDEX": "#3fb950",
    "MIDDLE": "#f0c75e",
    "RING": "#f78166",
    "PINKY": "#d2a8ff",
}


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------
def analyze_tap_events_csv(csv_text: str) -> dict:
    """Analyze tap events CSV (controller mode format)."""
    df = pd.read_csv(StringIO(csv_text))

    timestamps = df["timestamp_s"].values
    finger_lists = [f.split("+") for f in df["fingers"].values]

    if len(timestamps) < 2:
        return None

    itis = np.diff(timestamps) * 1000
    itis = itis[itis > 30]

    # Finger counts
    finger_counts = {}
    combo_counts = {}
    for fingers in finger_lists:
        combo_key = "+".join(sorted(fingers))
        combo_counts[combo_key] = combo_counts.get(combo_key, 0) + 1
        for f in fingers:
            finger_counts[f] = finger_counts.get(f, 0) + 1

    duration_s = timestamps[-1] - timestamps[0]

    # Build TapEvent objects for the parameter engine
    tap_events = []
    for i, (ts, fingers) in enumerate(zip(timestamps, finger_lists)):
        primary = fingers[0]
        secondaries = {f: 1.0 for f in fingers[1:]}
        tap_events.append(TapEvent(
            channel=primary,
            onset_ms=ts * 1000,
            offset_ms=ts * 1000 + 50,  # Estimated 50ms duration
            duration_ms=50.0,
            peak_magnitude=1.0,
            secondary_channels=secondaries,
        ))

    params = compute_parameters(tap_events)
    composites = compute_composite_scores(params)

    # Fatigue slope in ms/tap (more intuitive for tap events)
    fatigue_slope_per_tap = 0.0
    if len(itis) >= 5:
        slope, _ = np.polyfit(np.arange(len(itis)), itis, 1)
        fatigue_slope_per_tap = float(slope)

    # Rhythm entropy
    rhythm_entropy = 0.0
    if len(itis) >= 5:
        n_bins = min(15, len(itis) // 3)
        if n_bins >= 2:
            hist, _ = np.histogram(itis, bins=n_bins, density=True)
            hist = hist[hist > 0]
            bw = (np.max(itis) - np.min(itis)) / n_bins
            if bw > 0:
                probs = hist * bw
                probs = probs[probs > 0]
                rhythm_entropy = float(-np.sum(probs * np.log2(probs)))

    return {
        "tap_count": len(timestamps),
        "duration_s": round(duration_s, 1),
        "taps_per_second": round(len(timestamps) / max(duration_s, 0.1), 2),
        "iti_mean_ms": round(float(np.mean(itis)), 1) if len(itis) > 0 else 0,
        "iti_sd_ms": round(float(np.std(itis)), 1) if len(itis) > 0 else 0,
        "iti_cv": round(float(np.std(itis) / np.mean(itis)), 3) if len(itis) > 0 and np.mean(itis) > 0 else 0,
        "iti_min_ms": round(float(np.min(itis)), 1) if len(itis) > 0 else 0,
        "iti_max_ms": round(float(np.max(itis)), 1) if len(itis) > 0 else 0,
        "freezing_count": params.freezing_count,
        "coactivation_index": round(params.coactivation_index, 3),
        "fatigue_slope": round(fatigue_slope_per_tap, 2),
        "rhythm_entropy": round(rhythm_entropy, 3),
        "finger_counts": finger_counts,
        "combo_counts": combo_counts,
        "composites": composites,
        "params": params,
        "timestamps": timestamps.tolist(),
        "itis": itis.tolist(),
    }


def analyze_raw_sensor_csv(csv_text: str) -> dict:
    """Analyze raw sensor CSV (200Hz accelerometer format)."""
    df = pd.read_csv(StringIO(csv_text))

    finger_df = df[df["sample_type"] == "ACCEL_FINGER"]
    if finger_df.empty:
        return None

    channels = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
    channel_data = {}

    for ch in channels:
        ch_df = finger_df[finger_df["channel"] == ch].copy()
        if ch_df.empty:
            continue
        ts = ch_df["device_ts_ms"].values.astype(float)
        x = ch_df["x"].values.astype(float)
        y = ch_df["y"].values.astype(float)
        z = ch_df["z"].values.astype(float)

        fs = estimate_sample_rate(ts)
        if len(x) > 20:
            x_f = highpass_filter(x, fs=fs)
            y_f = highpass_filter(y, fs=fs)
            z_f = highpass_filter(z, fs=fs)
        else:
            x_f, y_f, z_f = x, y, z

        mag = compute_magnitude(x_f, y_f, z_f)
        if len(mag) > 20:
            mag = np.abs(bandpass_filter(mag, fs=fs))

        channel_data[ch] = {"timestamps": ts, "magnitude": mag}

    if not channel_data:
        return None

    taps = detect_taps_multichannel(channel_data)
    if not taps:
        return None

    params = compute_parameters(taps)
    composites = compute_composite_scores(params)

    timestamps = [t.onset_ms / 1000.0 for t in taps]
    itis = np.diff([t.onset_ms for t in taps])
    itis = itis[itis > 30]

    finger_counts = params.taps_per_finger or {}

    return {
        "tap_count": params.tap_count,
        "duration_s": round(params.epoch_duration_s, 1),
        "taps_per_second": round(params.tap_count / max(params.epoch_duration_s, 0.1), 2),
        "iti_mean_ms": round(params.iti_mean_ms, 1),
        "iti_sd_ms": round(params.iti_sd_ms, 1),
        "iti_cv": round(params.iti_cv, 3),
        "iti_min_ms": round(float(np.min(itis)), 1) if len(itis) > 0 else 0,
        "iti_max_ms": round(float(np.max(itis)), 1) if len(itis) > 0 else 0,
        "freezing_count": params.freezing_count,
        "coactivation_index": round(params.coactivation_index, 3),
        "fatigue_slope": round(params.fatigue_slope, 2),
        "rhythm_entropy": round(params.rhythm_entropy, 3),
        "tap_duration_mean_ms": round(params.tap_duration_mean_ms, 1),
        "finger_counts": finger_counts,
        "combo_counts": {},
        "composites": composites,
        "params": params,
        "timestamps": timestamps,
        "itis": itis.tolist(),
    }


def detect_csv_format(csv_text: str) -> str:
    """Detect if CSV is tap_events or raw sensor format."""
    first_line = csv_text.split("\n")[0].lower()
    if "timestamp_s" in first_line or "fingers" in first_line:
        return "tap_events"
    elif "sample_type" in first_line:
        return "raw_sensor"
    return "unknown"


# ---------------------------------------------------------------------------
# Web Bluetooth component
# ---------------------------------------------------------------------------
WEB_BLUETOOTH_HTML = """
<div id="bt-app" style="font-family: 'JetBrains Mono', 'Consolas', monospace;">

    <div id="bt-status" style="
        background: #111820; border: 1px solid #1e2a3a; border-radius: 10px;
        padding: 1rem; margin-bottom: 1rem; text-align: center;
    ">
        <div id="status-text" style="color: #8b949e; font-size: 0.9rem;">
            Ready to connect
        </div>
    </div>

    <div style="display: flex; gap: 0.8rem; margin-bottom: 1rem; flex-wrap: wrap;">
        <button id="btn-scan" onclick="doScan()" style="
            flex: 1; min-width: 120px; padding: 0.8rem 1.2rem; border: 1px solid #1e2a3a;
            border-radius: 8px; background: #1a2130; color: #58a6ff;
            font-family: inherit; font-weight: 600; font-size: 0.95rem; cursor: pointer;
        ">CONNECT</button>
        <button id="btn-capture" onclick="doCapture()" disabled style="
            flex: 1; min-width: 120px; padding: 0.8rem 1.2rem; border: 1px solid #1e2a3a;
            border-radius: 8px; background: #1a2130; color: #f0c75e;
            font-family: inherit; font-weight: 600; font-size: 0.95rem; cursor: pointer;
            opacity: 0.4;
        ">CAPTURE 30s</button>
    </div>

    <!-- Live counters -->
    <div id="counters" style="display: none; margin-bottom: 1rem;">
        <div style="display: flex; gap: 0.8rem;">
            <div style="flex: 1; background: #111820; border: 1px solid #1e2a3a; border-radius: 8px; padding: 0.8rem; text-align: center;">
                <div style="color: #484f58; font-size: 0.7rem; text-transform: uppercase;">Taps</div>
                <div id="tap-count" style="color: #58a6ff; font-size: 1.8rem; font-weight: 700;">0</div>
            </div>
            <div style="flex: 1; background: #111820; border: 1px solid #1e2a3a; border-radius: 8px; padding: 0.8rem; text-align: center;">
                <div style="color: #484f58; font-size: 0.7rem; text-transform: uppercase;">Rate</div>
                <div id="tap-rate" style="color: #3fb950; font-size: 1.8rem; font-weight: 700;">0/s</div>
            </div>
            <div style="flex: 1; background: #111820; border: 1px solid #1e2a3a; border-radius: 8px; padding: 0.8rem; text-align: center;">
                <div style="color: #484f58; font-size: 0.7rem; text-transform: uppercase;">Time</div>
                <div id="time-left" style="color: #f0c75e; font-size: 1.8rem; font-weight: 700;">30s</div>
            </div>
        </div>
    </div>

    <!-- Live tap feed -->
    <div id="tap-feed" style="
        display: none; background: #111820; border: 1px solid #1e2a3a;
        border-radius: 8px; padding: 0.8rem; max-height: 180px; overflow-y: auto;
        font-size: 0.8rem; color: #8b949e;
    "></div>

    <!-- Done banner (hidden until capture completes) -->
    <div id="done-banner" style="display: none; margin-top: 1rem;">
        <div style="background: #0d2818; border: 1px solid #3fb950; border-radius: 10px; padding: 1.2rem; text-align: center;">
            <div style="color: #3fb950; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;">
                Capture Complete!
            </div>
            <div id="done-summary" style="color: #8b949e; font-size: 0.85rem; margin-bottom: 0.8rem;"></div>
            <div style="color: #3fb950; font-size: 0.85rem; font-weight: 600;">
                CSV copied to clipboard + downloaded
            </div>
            <div style="color: #484f58; font-size: 0.8rem; margin-top: 0.5rem;">
                Paste below with <kbd style="background:#1a2130;padding:2px 6px;border-radius:3px;border:1px solid #1e2a3a;">Ctrl+V</kbd> and click <strong style="color:#d2a8ff;">Analyze</strong>
            </div>
        </div>
    </div>
</div>

<script>
const NUS_SERVICE = '6e400001-b5a3-f393-e0a9-e50e24dcca9e';
const NUS_RX = '6e400002-b5a3-f393-e0a9-e50e24dcca9e';
const TAP_DATA_CHAR = 'c3ff0005-1d8b-40fd-a56f-c7bd5d0f3370';
const TAP_SERVICE = 'c3ff0001-1d8b-40fd-a56f-c7bd5d0f3370';
const FINGERS = ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY'];

let device = null;
let server = null;
let tapEvents = [];
let capturing = false;
let captureStart = 0;
let counterInterval = null;

function setStatus(msg, type) {
    const el = document.getElementById('status-text');
    el.textContent = msg;
    const box = document.getElementById('bt-status');
    const colors = {
        connected: '#3fb950', capturing: '#f0c75e',
        done: '#d2a8ff', error: '#f78166'
    };
    const c = colors[type] || '#8b949e';
    box.style.borderColor = c;
    el.style.color = c;
}

function addTapToFeed(fingers) {
    const feed = document.getElementById('tap-feed');
    const line = document.createElement('div');
    const colors = { THUMB: '#58a6ff', INDEX: '#3fb950', MIDDLE: '#f0c75e', RING: '#f78166', PINKY: '#d2a8ff' };
    let html = fingers.map(f => '<span style="color:' + (colors[f] || '#e6edf3') + '">' + f + '</span>').join(' + ');
    line.innerHTML = '<span style="color:#484f58">' + tapEvents.length + '</span> ' + html;
    feed.insertBefore(line, feed.firstChild);
    if (feed.children.length > 50) feed.removeChild(feed.lastChild);
}

function decodeTap(dataView) {
    const code = dataView.getUint8(0);
    const fingers = [];
    for (let i = 0; i < 5; i++) {
        if (code & (1 << i)) fingers.push(FINGERS[i]);
    }
    return fingers;
}

function buildCSV() {
    let csv = 'timestamp_s,fingers,finger_count\\n';
    for (const evt of tapEvents) {
        csv += evt.ts.toFixed(3) + ',' + evt.fingers.join('+') + ',' + evt.fingers.length + '\\n';
    }
    return csv;
}

function downloadCSV(csv) {
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const now = new Date();
    const ts = now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') +
        String(now.getDate()).padStart(2,'0') + '_' +
        String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0') +
        String(now.getSeconds()).padStart(2,'0');
    a.download = 'taps_capture_' + ts + '.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

async function doScan() {
    if (!navigator.bluetooth) {
        setStatus('Web Bluetooth not supported. Use Chrome on desktop.', 'error');
        return;
    }

    try {
        setStatus('Requesting device...', 'info');
        document.getElementById('btn-scan').disabled = true;

        device = await navigator.bluetooth.requestDevice({
            filters: [{ namePrefix: 'Tap' }],
            optionalServices: [NUS_SERVICE, TAP_SERVICE],
        });

        setStatus('Connecting to ' + device.name + '...', 'info');
        server = await device.gatt.connect();
        setStatus('Connected to ' + device.name, 'connected');

        const btn = document.getElementById('btn-capture');
        btn.disabled = false;
        btn.style.opacity = '1';
        document.getElementById('btn-scan').textContent = 'RECONNECT';
        document.getElementById('btn-scan').disabled = false;

    } catch (err) {
        setStatus('Connection failed: ' + err.message, 'error');
        document.getElementById('btn-scan').disabled = false;
    }
}

async function doCapture() {
    if (!server || !server.connected) {
        setStatus('Not connected', 'error');
        return;
    }

    document.getElementById('btn-capture').disabled = true;
    document.getElementById('btn-capture').style.opacity = '0.4';
    document.getElementById('counters').style.display = 'block';
    document.getElementById('tap-feed').style.display = 'block';
    document.getElementById('tap-feed').innerHTML = '';
    document.getElementById('done-banner').style.display = 'none';
    tapEvents = [];
    capturing = true;
    captureStart = performance.now() / 1000;

    setStatus('CAPTURING \u2014 TAP YOUR FINGERS!', 'capturing');

    try {
        let tapChar = null;
        try {
            const tapService = await server.getPrimaryService(TAP_SERVICE);
            tapChar = await tapService.getCharacteristic(TAP_DATA_CHAR);
        } catch (e) {
            console.log('Tap service not found, trying NUS...');
        }

        if (tapChar) {
            await tapChar.startNotifications();
            tapChar.addEventListener('characteristicvaluechanged', (event) => {
                if (!capturing) return;
                const fingers = decodeTap(event.target.value);
                if (fingers.length > 0) {
                    const ts = performance.now() / 1000;
                    tapEvents.push({ ts, fingers });
                    addTapToFeed(fingers);
                }
            });

            try {
                const nusService = await server.getPrimaryService(NUS_SERVICE);
                const rxChar = await nusService.getCharacteristic(NUS_RX);
                await rxChar.writeValueWithoutResponse(new Uint8Array([0x03, 0x0c, 0x00, 0x01]));
            } catch (e) {
                console.log('NUS write skipped:', e);
            }
        }

    } catch (err) {
        setStatus('Capture error: ' + err.message, 'error');
        capturing = false;
        return;
    }

    counterInterval = setInterval(() => {
        const elapsed = performance.now() / 1000 - captureStart;
        const remaining = Math.max(0, 30 - elapsed);
        document.getElementById('tap-count').textContent = tapEvents.length;
        const rate = tapEvents.length / Math.max(elapsed, 0.1);
        document.getElementById('tap-rate').textContent = rate.toFixed(1) + '/s';
        document.getElementById('time-left').textContent = remaining.toFixed(0) + 's';

        if (remaining <= 0) {
            clearInterval(counterInterval);
            finishCapture();
        }
    }, 200);
}

function finishCapture() {
    capturing = false;
    const csv = buildCSV();
    const elapsed = (performance.now() / 1000 - captureStart).toFixed(1);

    // 1. Copy to clipboard
    navigator.clipboard.writeText(csv).catch(() => {});

    // 2. Auto-download the CSV file
    downloadCSV(csv);

    // 3. Show done banner
    document.getElementById('done-summary').textContent =
        tapEvents.length + ' taps captured in ' + elapsed + 's';
    document.getElementById('done-banner').style.display = 'block';

    setStatus('Done! ' + tapEvents.length + ' taps — CSV copied & downloaded', 'done');

    document.getElementById('btn-capture').disabled = false;
    document.getElementById('btn-capture').style.opacity = '1';
}
</script>
"""


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------
def render_header():
    st.markdown("""
    <div class="taps-header">
        <h1>TAPS</h1>
        <p>Tap Assessment Protocol Standard</p>
    </div>
    """, unsafe_allow_html=True)


def render_score_ring(score, label, color):
    """Render a circular score indicator."""
    bg_alpha = "33"  # 20% opacity
    return f"""
    <div class="score-ring" style="
        background: {color}{bg_alpha};
        border: 3px solid {color};
    ">
        <div class="score-val" style="color: {color};">{score:.0f}</div>
        <div class="score-label">{label}</div>
    </div>
    """


def render_metric(label, value, unit="", color="#e6edf3"):
    return f"""
    <div class="metric-box">
        <div class="label">{label}</div>
        <div class="value" style="color: {color};">{value}</div>
        <div class="unit">{unit}</div>
    </div>
    """


def render_finger_bars(finger_counts):
    max_count = max(finger_counts.values()) if finger_counts else 1
    html = ""
    for finger in FINGERS:
        count = finger_counts.get(finger, 0)
        pct = (count / max_count * 100) if max_count > 0 else 0
        color = FINGER_COLORS.get(finger, "#8b949e")
        html += f"""
        <div class="finger-bar-container">
            <span class="fname">{finger}</span>
            <span class="fcount">{count}</span>
            <div style="flex: 1; background: #1a2130; border-radius: 4px; height: 20px;">
                <div class="finger-bar" style="width: {pct}%; background: {color};"></div>
            </div>
        </div>
        """
    return html


def render_results(results):
    """Render full results dashboard."""
    composites = results["composites"]
    tcs = composites["taps_composite_score"]

    # Composite score color
    if tcs >= 70:
        score_color = "#3fb950"
    elif tcs >= 40:
        score_color = "#f0c75e"
    else:
        score_color = "#f78166"

    # --- Top metrics row ---
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    metrics = [
        ("Taps", str(results["tap_count"]), "", "#58a6ff"),
        ("Duration", f"{results['duration_s']}", "seconds", "#3fb950"),
        ("Rate", f"{results['taps_per_second']}", "taps/sec", "#f0c75e"),
        ("Mean ITI", f"{results['iti_mean_ms']}", "ms", "#d2a8ff"),
        ("ITI CV", f"{results['iti_cv']}", "", "#f78166"),
    ]
    for col, (label, value, unit, color) in zip(cols, metrics):
        with col:
            st.markdown(render_metric(label, value, unit, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Composite scores ---
    left_col, right_col = st.columns([2, 3])

    with left_col:
        st.markdown('<div class="section-title">Composite Score</div>', unsafe_allow_html=True)
        st.markdown(render_score_ring(tcs, "TCS", score_color), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        score_cols = st.columns(3)
        score_data = [
            ("TMI", composites["taps_motor_index"], "#58a6ff"),
            ("TVI", composites["taps_variability_index"], "#3fb950"),
            ("TCI", composites["taps_coordination_index"], "#d2a8ff"),
        ]
        for col, (label, val, color) in zip(score_cols, score_data):
            with col:
                st.markdown(render_metric(label, f"{val:.0f}", "/100", color), unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-title">Finger Breakdown</div>', unsafe_allow_html=True)
        st.markdown(render_finger_bars(results["finger_counts"]), unsafe_allow_html=True)

    # --- Clinical parameters ---
    st.markdown('<div class="section-title">Clinical Parameters (P1–P10)</div>', unsafe_allow_html=True)

    param_cols = st.columns(2)
    left_params = [
        ("P1  Tap Count", str(results["tap_count"]), ""),
        ("P2  Mean ITI", f"{results['iti_mean_ms']}", "ms"),
        ("P3  ITI Std Dev", f"{results['iti_sd_ms']}", "ms"),
        ("P4  ITI Coeff Var", f"{results['iti_cv']}", ""),
        ("P5  Tap Duration", f"{results.get('tap_duration_mean_ms', '~50')}", "ms"),
    ]
    right_params = [
        ("P6  Freezing Count", str(results["freezing_count"]), ""),
        ("P7  Co-Activation", f"{results['coactivation_index']}", ""),
        ("P8  Bilateral Asym", "0.000", "(single hand)"),
        ("P9  Fatigue Slope", f"{results['fatigue_slope']}", "ms/tap"),
        ("P10 Rhythm Entropy", f"{results['rhythm_entropy']}", "bits"),
    ]

    with param_cols[0]:
        for name, val, unit in left_params:
            st.markdown(f"""
            <div class="param-row">
                <span class="name">{name}</span>
                <span class="val">{val} <span style="color:#484f58;font-weight:400;font-size:0.75rem;">{unit}</span></span>
            </div>
            """, unsafe_allow_html=True)

    with param_cols[1]:
        for name, val, unit in right_params:
            st.markdown(f"""
            <div class="param-row">
                <span class="name">{name}</span>
                <span class="val">{val} <span style="color:#484f58;font-weight:400;font-size:0.75rem;">{unit}</span></span>
            </div>
            """, unsafe_allow_html=True)

    # --- ITI Distribution chart ---
    if results.get("itis") and len(results["itis"]) > 3:
        st.markdown('<div class="section-title">Inter-Tap Interval Distribution</div>', unsafe_allow_html=True)

        iti_data = pd.DataFrame({"ITI (ms)": results["itis"]})
        st.bar_chart(
            iti_data["ITI (ms)"].value_counts(bins=15).sort_index(),
            color="#58a6ff",
        )

    # --- Tap timeline ---
    if results.get("timestamps") and len(results["timestamps"]) > 1:
        st.markdown('<div class="section-title">Tap Timeline</div>', unsafe_allow_html=True)
        ts = np.array(results["timestamps"])
        ts_rel = ts - ts[0]
        itis_ms = np.diff(ts) * 1000
        timeline_df = pd.DataFrame({
            "Time (s)": ts_rel[1:],
            "ITI (ms)": itis_ms,
        })
        st.line_chart(timeline_df.set_index("Time (s)"), color="#d2a8ff")

    # --- Combo breakdown (if available) ---
    combos = results.get("combo_counts", {})
    if combos and len(combos) > 1:
        st.markdown('<div class="section-title">Tap Combinations</div>', unsafe_allow_html=True)
        sorted_combos = sorted(combos.items(), key=lambda x: -x[1])
        combo_df = pd.DataFrame(sorted_combos[:10], columns=["Combo", "Count"])
        st.bar_chart(combo_df.set_index("Combo"), color="#f0c75e")

    # --- Download ---
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    export_data = {
        "taps_version": "0.1",
        "export_time": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "tap_count": results["tap_count"],
            "iti_mean_ms": results["iti_mean_ms"],
            "iti_sd_ms": results["iti_sd_ms"],
            "iti_cv": results["iti_cv"],
            "freezing_count": results["freezing_count"],
            "coactivation_index": results["coactivation_index"],
            "fatigue_slope": results["fatigue_slope"],
            "rhythm_entropy": results["rhythm_entropy"],
        },
        "composite_scores": results["composites"],
        "finger_counts": results["finger_counts"],
    }

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Results (JSON)",
            json.dumps(export_data, indent=2),
            file_name="taps_results.json",
            mime="application/json",
        )
    with col2:
        # CSV export
        csv_lines = ["parameter,value"]
        for k, v in export_data["parameters"].items():
            csv_lines.append(f"{k},{v}")
        for k, v in export_data["composite_scores"].items():
            csv_lines.append(f"{k},{v}")
        st.download_button(
            "Download Results (CSV)",
            "\n".join(csv_lines),
            file_name="taps_results.csv",
            mime="text/csv",
        )


def render_interpretation(results):
    """Clinical interpretation guidance."""
    composites = results["composites"]
    tcs = composites["taps_composite_score"]
    cv = results["iti_cv"]
    freeze = results["freezing_count"]
    entropy = results["rhythm_entropy"]

    st.markdown('<div class="section-title">Interpretation Guide</div>', unsafe_allow_html=True)

    findings = []

    if tcs >= 70:
        findings.append(("Composite Score", f"TCS = {tcs:.0f}/100 — within normal range", "#3fb950"))
    elif tcs >= 40:
        findings.append(("Composite Score", f"TCS = {tcs:.0f}/100 — mild impairment suggested", "#f0c75e"))
    else:
        findings.append(("Composite Score", f"TCS = {tcs:.0f}/100 — moderate-to-severe impairment suggested", "#f78166"))

    if cv < 0.15:
        findings.append(("Rhythm Consistency", f"CV = {cv:.3f} — very consistent rhythm (normal)", "#3fb950"))
    elif cv < 0.3:
        findings.append(("Rhythm Consistency", f"CV = {cv:.3f} — moderate variability", "#f0c75e"))
    else:
        findings.append(("Rhythm Consistency", f"CV = {cv:.3f} — high variability (may indicate motor timing deficit)", "#f78166"))

    if freeze == 0:
        findings.append(("Motor Freezing", "No freezing events detected", "#3fb950"))
    elif freeze <= 2:
        findings.append(("Motor Freezing", f"{freeze} freezing event(s) — occasional hesitation", "#f0c75e"))
    else:
        findings.append(("Motor Freezing", f"{freeze} freezing events — frequent motor blocks", "#f78166"))

    if results["fatigue_slope"] < 1.0:
        findings.append(("Fatigue", "Minimal fatigue effect", "#3fb950"))
    elif results["fatigue_slope"] < 5.0:
        findings.append(("Fatigue", f"Moderate fatigue (slope = {results['fatigue_slope']:.1f} ms/tap)", "#f0c75e"))
    else:
        findings.append(("Fatigue", f"Significant fatigue (slope = {results['fatigue_slope']:.1f} ms/tap)", "#f78166"))

    for title, desc, color in findings:
        st.markdown(f"""
        <div style="background: #111820; border-left: 3px solid {color}; padding: 0.7rem 1rem; margin: 0.4rem 0; border-radius: 0 6px 6px 0;">
            <div style="font-family: 'JetBrains Mono', monospace; color: {color}; font-size: 0.8rem; font-weight: 600;">{title}</div>
            <div style="font-family: 'JetBrains Mono', monospace; color: #e6edf3; font-size: 0.85rem; margin-top: 0.2rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #111820; border: 1px solid #1e2a3a; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
        <div style="font-family: 'JetBrains Mono', monospace; color: #484f58; font-size: 0.75rem;">
            <strong style="color: #8b949e;">Disclaimer:</strong> TAPS is a research tool. Results are not diagnostic.
            Composite scores use preliminary normalization and require clinical validation against
            established instruments (MMSE, MoCA, UPDRS) for clinical use.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
render_header()

# Mode selection
if st.session_state.results is None:
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(3)

    with cols[0]:
        st.markdown("""
        <div class="mode-card">
            <div style="font-size: 2rem;">📡</div>
            <h3>Bluetooth</h3>
            <p>Connect Tap Strap 2 via Web Bluetooth and capture live</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Connect Device", key="bt_btn", use_container_width=True):
            st.session_state.mode = "bluetooth"
            st.rerun()

    with cols[1]:
        st.markdown("""
        <div class="mode-card">
            <div style="font-size: 2rem;">📂</div>
            <h3>Upload</h3>
            <p>Upload a CSV file from a previous capture session</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Upload File", key="up_btn", use_container_width=True):
            st.session_state.mode = "upload"
            st.rerun()

    with cols[2]:
        st.markdown("""
        <div class="mode-card">
            <div style="font-size: 2rem;">🧪</div>
            <h3>Demo</h3>
            <p>See results from a real 30-second capture session</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Try Demo", key="demo_btn", use_container_width=True):
            st.session_state.mode = "demo"
            st.rerun()

    # Info section
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.expander("What is TAPS?"):
        st.markdown("""
        **TAPS (Tap Assessment Protocol Standard)** is a clinical motor assessment tool
        that uses finger tapping patterns to evaluate motor control, timing, and coordination.

        It measures **10 clinical parameters** including tap speed, rhythm consistency,
        motor freezing, finger co-activation, fatigue, and rhythm entropy — producing a
        single **Composite Score (TCS)** from 0–100.

        TAPS uses the **Tap Strap 2** wearable device to capture finger tapping data
        via Bluetooth. The assessment takes just **30 seconds**.

        **Relevant to:** MCI screening, Parkinson's motor assessment, stroke recovery
        monitoring, concussion evaluation, and general motor function tracking.
        """)

# --- BLUETOOTH MODE ---
if st.session_state.mode == "bluetooth" and st.session_state.results is None:
    st.markdown('<div class="section-title">Bluetooth Capture</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #111820; border: 1px solid #1e2a3a; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 1rem;
        font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #8b949e;">
        Requires <strong style="color: #58a6ff;">Chrome</strong> or <strong style="color: #58a6ff;">Edge</strong> on desktop.
        Wake your Tap Strap 2 with a tap, then click <strong>Connect</strong>.
    </div>
    """, unsafe_allow_html=True)

    import streamlit.components.v1 as components
    components.html(WEB_BLUETOOTH_HTML, height=520)

    # Step 2: Paste + Analyze
    st.markdown("""
    <div class="section-title" style="margin-top: 0.5rem;">Step 2: Analyze</div>
    <div style="font-family: 'JetBrains Mono', monospace; color: #8b949e; font-size: 0.85rem; margin-bottom: 0.5rem;">
        When capture finishes, CSV is <strong style="color: #3fb950;">auto-copied to clipboard</strong>
        + <strong style="color: #3fb950;">downloaded</strong>. Paste here with Ctrl+V:
    </div>
    """, unsafe_allow_html=True)

    bt_csv = st.text_area(
        "Paste CSV",
        height=120,
        placeholder="Ctrl+V to paste capture data here...",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_disabled = not (bt_csv and "timestamp_s" in bt_csv)
        if st.button(
            "Analyze Capture",
            use_container_width=True,
            type="primary",
            disabled=analyze_disabled,
        ):
            with st.spinner("Analyzing..."):
                results = analyze_tap_events_csv(bt_csv)
            if results:
                st.session_state.results = results
                st.rerun()
            else:
                st.error("Not enough data. Try tapping more during capture.")
    with col2:
        if st.button("← Back", use_container_width=True):
            st.session_state.mode = None
            st.rerun()

    # Also accept the downloaded CSV via upload as fallback
    with st.expander("Or upload the downloaded CSV"):
        bt_upload = st.file_uploader("Upload capture CSV", type=["csv"], key="bt_upload")
        if bt_upload:
            csv_text = bt_upload.read().decode("utf-8-sig")
            if "timestamp_s" in csv_text:
                if st.button("Analyze uploaded file", type="primary"):
                    with st.spinner("Analyzing..."):
                        results = analyze_tap_events_csv(csv_text)
                    if results:
                        st.session_state.results = results
                        st.rerun()

# --- UPLOAD MODE ---
if st.session_state.mode == "upload" and st.session_state.results is None:
    st.markdown('<div class="section-title">Upload Capture Data</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a TAPS CSV file",
        type=["csv"],
        help="Supports both tap events CSV (controller mode) and raw sensor CSV (200Hz accelerometer)",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if uploaded:
            csv_text = uploaded.read().decode("utf-8-sig")
            fmt = detect_csv_format(csv_text)

            if fmt == "tap_events":
                st.info(f"Detected: **tap events** format ({csv_text.count(chr(10))} rows)")
                if st.button("Analyze", use_container_width=True, type="primary"):
                    with st.spinner("Analyzing..."):
                        results = analyze_tap_events_csv(csv_text)
                    if results:
                        st.session_state.results = results
                        st.rerun()
                    else:
                        st.error("Not enough tap data to analyze.")
            elif fmt == "raw_sensor":
                st.info("Detected: **raw sensor** format (200Hz accelerometer)")
                if st.button("Analyze", use_container_width=True, type="primary"):
                    with st.spinner("Processing signal data..."):
                        results = analyze_raw_sensor_csv(csv_text)
                    if results:
                        st.session_state.results = results
                        st.rerun()
                    else:
                        st.error("Could not detect taps in sensor data.")
            else:
                st.error("Unknown CSV format. Expected TAPS tap events or raw sensor data.")

    with col2:
        if st.button("← Back", use_container_width=True):
            st.session_state.mode = None
            st.rerun()

# --- DEMO MODE ---
if st.session_state.mode == "demo" and st.session_state.results is None:
    with st.spinner("Loading demo data..."):
        results = analyze_tap_events_csv(DEMO_TAP_EVENTS_CSV)
    if results:
        st.session_state.results = results
        st.rerun()

# --- RESULTS ---
if st.session_state.results is not None:
    render_results(st.session_state.results)
    st.markdown("<br>", unsafe_allow_html=True)
    render_interpretation(st.session_state.results)

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("← New Assessment", use_container_width=True):
        st.session_state.results = None
        st.session_state.mode = None
        st.rerun()
