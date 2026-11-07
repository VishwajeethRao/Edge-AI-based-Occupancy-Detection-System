# EdgeVision-Lite
### CPU-First Edge AI Person Detection, Tracking & Unique Counting

**EdgeVision-Lite** is a lightweight Edge AI pipeline designed for real-time person detection and anonymous unique-person counting using CPU-based inference.

The system combines YOLOv8n, INT8 quantization, ONNX Runtime, ByteTrack-style tracking, and OSNet Person Re-Identification to detect people, track their movement, and recognize returning individuals across a session.

**Developed by Vishwajeeth Rao**

---

## Overview

Traditional object detection can count people in individual frames but cannot inherently recognize someone who leaves the camera view and returns later.

EdgeVision-Lite addresses this by combining object detection, multi-object tracking, and appearance-based Person Re-ID.

It maintains two separate counts:

- **Current People:** Number of people visible in the current frame.
- **Unique People:** Number of distinct individuals detected during the session.

## Key Features

- Real-time person detection using YOLOv8n.
- FP32 to INT8 post-training quantization.
- CPU-based inference using ONNX Runtime.
- Lightweight ByteTrack-style multi-object tracking.
- OSNet x0.25 appearance-based Person Re-ID.
- Anonymous identity memory using cosine similarity.
- Dynamic identity generation (P1, P2, P3, etc.).
- Optional SQLite-based persistent identity storage.
- FP32 vs INT8 performance benchmarking.
- Optimized Re-ID scheduling to reduce CPU workload.

## System Architecture

```text
Camera / Video Input
        |
        v
    YOLOv8n INT8
        |
        v
   ONNX Runtime
        |
        v
   Person Detection
        |
        v
   ByteTrack-style Tracker
        |
        v
   Temporary Track IDs
        |
        v
    Person Cropping
        |
        v
     OSNet Re-ID
        |
        v
  512-D Feature Embeddings
        |
        v
   Identity Memory
        |
        v
   Cosine Similarity
        |
        v
  Identity Matching
        |
        v
  Unique Person Count
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| YOLOv8n | Person detection |
| ONNX | Model deployment format |
| ONNX Runtime | CPU-based inference |
| INT8 Quantization | Model size and inference optimization |
| NumPy | Tracking and numerical operations |
| OpenCV | Image processing and video input |
| ByteTrack-style Tracker | Multi-object tracking |
| OSNet x0.25 | Person Re-ID |
| SQLite | Optional identity persistence |

## Model Optimization

The detection model follows this pipeline:

```text
YOLOv8n FP32
     |
     v
ONNX Export
     |
     v
FP32 ONNX Model
     |
     v
Dynamic INT8 Quantization
     |
     v
INT8 ONNX Model
```

The project uses ONNX Runtime's post-training dynamic quantization to reduce model size and potentially improve CPU inference performance.

The actual speed improvement depends on the target hardware and should be evaluated through benchmarking.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/VishwajeethRao/Edge-AI-based-Occupancy-Detection-System.git
cd Edge-AI-based-Occupancy-Detection-System
```

### 2. Create a Virtual Environment

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Run Person Detection

```bash
python detect.py --model yolov8n_int8.onnx --source 0
```

### Run Complete Person Counting System

```bash
python people_counting-1.py --reid-model models/osnet_x0_25_msmt17.onnx
```

### Run with a Custom Re-ID Threshold

```bash
python people_counting-1.py --reid-model models/osnet_x0_25_msmt17.onnx --reid-threshold 0.6
```

### Enable Persistent Identity Storage

```bash
python people_counting-1.py --reid-model models/osnet_x0_25_msmt17.onnx --persist-db identities.db
```

**Note:** Webcam input is used by default. Refer to the command-line options for additional configuration.

## Project Structure

```text
EdgeVision-Lite/
│
├── models/
│   └── osnet_x0_25_msmt17.onnx
│
├── yolov8n.onnx
├── yolov8n_int8.onnx
│
├── detect.py
├── quantization.py
├── benchmarking.py
├── tracking.py
├── reid.py
├── identity_memory.py
├── people_counting-1.py
│
├── requirements.txt
└── README.md
```

## Configuration

Important command-line parameters:

| Parameter | Description | Default |
|---|---|---|
| `--model` | YOLO ONNX model | yolov8n_int8.onnx |
| `--reid-model` | OSNet model | Configurable |
| `--source` | Webcam/video source | 0 |
| `--input-size` | YOLO input resolution | 640 |
| `--conf` | Detection confidence | 0.35 |
| `--track-high-thresh` | High tracking threshold | 0.5 |
| `--track-low-thresh` | Low tracking threshold | 0.1 |
| `--match-iou` | IoU matching threshold | 0.3 |
| `--track-buffer` | Lost-track buffer | 30 |
| `--reid-threshold` | Re-ID similarity threshold | 0.6 |
| `--reid-interval` | Re-ID refresh interval | 30 |
| `--capacity` | Optional room capacity | 0 |
| `--persist-db` | SQLite identity database | None |

## Performance Benchmarking

The project includes benchmarking for evaluating:

- FP32 and INT8 model sizes.
- Detection inference latency.
- Re-ID inference latency.
- Tracking overhead.
- Total pipeline latency.
- Effective FPS.

Example:

```powershell
python benchmark.py --fp32 yolov8n.onnx --int8 yolov8n_int8.onnx --reid models\osnet_x0_25_msmt17.onnx --runs 50
```

Use `python benchmarking.py --help` if your script is named `benchmarking.py`.

Actual performance depends on the CPU, input resolution, camera, and workload.

## Identity Recognition

The system uses OSNet x0.25 to generate 512-dimensional feature embeddings for detected people.

These embeddings are compared against stored identity representations using cosine similarity.

- Similarity ≥ 0.6: Match an existing identity.
- Similarity < 0.6: Create a new anonymous identity.

The threshold is a starting value and may require calibration for different environments.

Identity storage supports two modes:

- **Session Mode:** Identity memory is maintained only while the application is running.
- **Persistent Mode:** Identity embeddings can optionally be stored locally using SQLite.

## Limitations

- Similar clothing can cause identity mismatches.
- Occlusion and motion blur can affect detection and tracking.
- Lighting and camera-angle changes can affect Re-ID accuracy.
- Appearance-based recognition does not guarantee perfect identity matching.
- CPU performance varies depending on hardware and workload.

## Future Improvements

- Multiple embeddings per identity.
- Improved identity matching and threshold calibration.
- Better crop-quality filtering.
- Multi-camera identity recognition.
- OpenVINO CPU acceleration.
- Raspberry Pi deployment.
- Static INT8 quantization with calibration.
- Automated performance testing.

## Applications

Potential applications include:

- Classroom occupancy monitoring.
- Office and workspace occupancy.
- Laboratory monitoring.
- Room-level people counting.
- Anonymous visitor counting.

## Author

**Vishwajeeth Rao**

Interested in Edge AI, Computer Vision, Robotics, and Embedded Systems.

