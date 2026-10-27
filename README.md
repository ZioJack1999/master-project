# Real-Time Deep Learning BCI Studio & 3D Virtual Hand Controller

High-performance, closed-loop Brain-Computer Interface (BCI) decoding system for motor imagery (right-hand grasp vs. rest) with sub-10ms neural network inference and zero-lag 60 FPS telemetry streaming.

---

## System Architecture

```
┌────────────────────────────────┐         ┌──────────────────────────────┐         ┌───────────────────────────────┐
│     EEG Amplifier (LSL)        │         │   BCI Studio Console (:8050) │         │     Virtual Hand (External)   │
│  B-Alert X24 / OpenBCI / RDA   │────────►│  • EEG Oscilloscope (C3/Cz/C4│────────►│  Listening on UDP Port 5005  │
│  24 Channels @ 256 Hz          │         │  • Dynamic Sensitivity Slider│         │  {"grasp_command": 0/1,       │
│  C3, Cz, C4 Motor Cortex       │         │  • 8s Dynamic Calibration    │         │   "continuous_flex": 0.0-1.0} │
└────────────────────────────────┘         └──────────────────────────────┘         └───────────────────────────────┘
```

---

## Repository Contents

* **`Launch_BCI_3D_Studio.bat`**: One-click Windows batch launcher for the complete studio.
* **`bci_studio_gui.py`**: Web-based neuro-engineering console featuring:
  - Real-time multi-channel oscilloscope for motor cortex channels ($C_3, C_z, C_4$ @ 8.0–30.0 Hz).
  - Classification probability strip-chart $p(\text{action})$ plotted against calibrated decision boundary.
  - Interactive sensitivity threshold slider.
  - UDP Port 5005 JSON telemetry broadcaster for virtual hand control.
  - Animated 8-second SVG circular calibration countdown with sub-second precision.
  - One-click **"Save Session & Stop"** auto-exporting publication-ready PNG plots and time-indexed CSV datasets.
* **`run_calibration.py`**: Standalone 8-second trial calibration runner (3 pairs of 8s Rest vs. 8s Motor Imagery).
* **`realtime_bci_decoder.py`**: Core real-time decoder engine with streaming ring buffers, causal IIR bandpass filtering, and PyTorch EEGNet inference.
* **`decoder_and_models/`**: Pre-trained EEGNet master live weights (`eegnet_master_live.pt`).
* **`README_LAB_GUIDE.md`**: Step-by-step laboratory hardware and testing checklist.
* **`requirements.txt`**: Minimal, verified dependencies.

---

## Lab Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/ZioJack1999/master-project.git
cd master-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Studio
* **Option A**: Double-click **`Launch_BCI_3D_Studio.bat`**.
* **Option B**: Run from terminal:
  ```bash
  python bci_studio_gui.py
  ```

Open your browser to `http://127.0.0.1:8050`. The console will automatically detect any active LabStreamingLayer (LSL) EEG stream (e.g., from the B-Alert X24 system) and turn **`🟢 HARDWARE CONNECTED`**!
