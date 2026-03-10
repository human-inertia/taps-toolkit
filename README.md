# TAPS — Tap Assessment Protocol Standard

A 30-second finger tapping motor assessment. Connect a Tap Strap 2 via Bluetooth, tap your fingers, get clinical-grade motor analysis.

**10 clinical parameters** | **Composite scoring** | **Web Bluetooth** | **Instant results**

## Try It

Open the hosted app (Streamlit Cloud link here) or run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Three modes:
- **Bluetooth** — Connect Tap Strap 2 from your browser (Chrome/Edge)
- **Upload** — Analyze a CSV from a previous capture
- **Demo** — See real results from a 63-tap session

## What It Measures

| Parameter | Description |
|-----------|-------------|
| P1 Tap Count | Total taps in 30s |
| P2 Mean ITI | Average inter-tap interval (ms) |
| P3 ITI Std Dev | Timing variability — most sensitive marker |
| P4 ITI CV | Coefficient of variation |
| P5 Tap Duration | Mean tap contact time (ms) |
| P6 Freezing Count | Motor blocks (ITI > 2x mean) |
| P7 Co-Activation | Finger independence index |
| P8 Bilateral Asymmetry | Left vs right hand difference |
| P9 Fatigue Slope | ITI trend over time |
| P10 Rhythm Entropy | Timing regularity (bits) |

**Composite scores:** TMI (Motor), TVI (Variability), TCI (Coordination) → **TCS (0–100)**

## Hardware

**Tap Strap 2** — wearable finger controller with per-finger accelerometry.
- Controller mode (tap events) works out of the box
- Raw 200Hz mode requires Developer Mode in TapManager app

## Architecture

```
app.py                  ← Streamlit web app
taps/
├── processing/
│   ├── filters.py      ← Bandpass, gravity removal, magnitude
│   └── tap_detection.py← Adaptive threshold detection + co-activation
├── assessment/
│   └── parameters.py   ← 10 parameters + composite scoring
└── reporting/
    └── export.py       ← JSON/CSV/research bundle export
```

## Relevant To

MCI screening, Parkinson's motor assessment, stroke recovery, concussion evaluation, general motor function tracking.

## Disclaimer

TAPS is a research tool. Results are not diagnostic. Composite scores use preliminary normalization pending clinical validation.
