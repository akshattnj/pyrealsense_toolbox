# pyrealsense_toolbox

A modular Python framework for real-time spatial computing, bio-sensing, edge vision-language reasoning, and 3D gesture tracking using Intel RealSense depth sensors, MediaPipe Tasks, YOLO, and Vision-Language Models (VLMs).

---

## 🛠️ Module Overview

### 1. Physiological Sensing & Bio-Signals (`rPPG` & Breathing)

* **`rppg_interactive`**: Real-time continuous remote photoplethysmography using MediaPipe Face Landmarker Tasks, depth-aware 3D spatial-temporal correction, multi-point facial hue sampling, and sliding-window spectral estimation for pulse rate (BPM).
* **`rppg-tt.py`**: Lightweight, terminal-based rPPG execution pipeline for rapid pulse extraction.
* **`respiratory_volume_monitoring.py`**: Depth-based chest displacement analysis to estimate respiration rate and relative tidal volume changes.

### 2. Vision-Language Models (VLM) & Object Detection

* **`florence_realsense.py`**: Integrates Microsoft Florence VLM with live RealSense depth streams for spatial grounding and visual query answering.
* **`moondreamplusyolo.py` / `moondream_yolo_opt.py**`: Hybrid perception pipeline combining Moondream VLM with optimized YOLO models (`yolo26n.pt`) for real-time edge detection and open-vocabulary understanding.
* **`moondreamplusyoloplusinput.py`**: Interactive mode accepting user text prompts to direct VLM attention to specific objects localized by YOLO bounding boxes.
* **`yolo26realsense.py`**: RealSense RGB-D pipeline running YOLO object detection with real-time 3D coordinate estimation ($X, Y, Z$).

### 3. Spatial Tracking, Gesture HCI, & Scanning

* **`realsense_screen_control.py`**: Hand-gesture-based desktop control using MediaPipe Hand Landmarker and RealSense depth mapping.
* **`realsense_track_hands.py` / `realsense_track_multiple_hands.py**`: Single and multi-hand 3D landmark tracking with depth coordinate resolution.
* **`realsense_track_particular_pixel.py`**: Interactive utility to track real-world 3D coordinates ($X, Y, Z$) and depth distance of any clicked pixel in the live feed.
* **`realsense_track.py`**: Baseline pipeline for color-depth alignment and spatial tracking.
* **`realsense_scan.py`**: 3D environment surface and point-cloud capture utility.

---

## 📦 Setup & Environment

### 1. Virtual Environment Setup

Ensure Python 3.10 to 3.12 is installed. Pin `numpy<2` to avoid C-extension ABI conflicts with MediaPipe and OpenCV.

```bash
git clone https://github.com/your-username/pyrealsense_toolbox.git
cd pyrealsense_toolbox

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install "numpy<2" opencv-python pyrealsense2 mediapipe scipy matplotlib joblib torch torchvision ultralytics

```

### 2. Model Assets & Downloads

The toolbox relies on pretrained MediaPipe Tasks binaries and YOLO weights. Download required assets into the repository root:

```bash
# MediaPipe Tasks Bundles
wget -O face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
wget -O pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

```

---

## 🚀 Quick Start Examples

### Launch Interactive rPPG

```bash
python rppg_interactive

```

### Run 3D YOLO Object Detection

```bash
python yolo26realsense.py

```

### Run VLM + YOLO Hybrid Pipeline

```bash
python moondream_yolo_opt.py

```

### Track 3D Pixel Distance

```bash
python realsense_track_particular_pixel.py

```

---

## 🔒 Hardware Requirements

* **Camera**: Intel RealSense D400-series (D435, D435i, D455) connected via **USB 3.0+** (blue port, 5 Gbps bandwidth required for simultaneous BGR8 and Z16 streams).
* **OS**: Linux (Ubuntu 20.04/22.04/24.04 recommended) or Windows 11.