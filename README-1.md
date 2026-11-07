# EdgeVision-Lite

**A CPU-first edge AI pipeline: YOLOv8n → INT8 quantization → ONNX Runtime detection → ByteTrack → OSNet Person Re-ID → anonymous unique-person counting.**

Built by **Vishwajeeth Rao** as a hands-on edge AI project focused on lightweight inference and real-time deployment on CPU hardware.

---

## 1. Project Overview

EdgeVision-Lite has two main parts:

### 1. Detection Core

The detection pipeline uses:

* YOLOv8n
* FP32 → INT8 post-training quantization
* ONNX Runtime
* CPU inference
* Real-time webcam/video detection
* Benchmarking of FP32 vs INT8 inference

### 2. Unique Person Counting

The detection pipeline is extended with:

* A lightweight ByteTrack-style tracker
* OSNet Person Re-ID
* Anonymous identity memory
* Cosine-similarity matching
* Unique-person counting

The goal is to distinguish between:

**Current people**

> How many people are visible right now?

and:

**Unique people seen**

> How many different people have appeared during the session?

For example:

```text
Person A enters       → Unique = 1
Person B enters       → Unique = 2
Person C enters       → Unique = 3

Person A leaves       → Unique = 3

Person A returns      → Unique = 3

Person D enters       → Unique = 4
```

There is **no hard-coded maximum number of identities**. The identity memory can dynamically create P1, P2, P3, ..., P100, etc.

---

# 2. Problem Statement

A normal object detector can determine how many people are present in an individual frame.

However, a detector has no memory of people across time.

If someone leaves the camera view and later comes back, the detector cannot inherently know that the returning person is the same person.

Therefore, a useful room/classroom counter needs two separate concepts:

* **Current people** — people visible in the current frame.
* **Unique people seen** — different people observed during the entire session.

ByteTrack handles continuous tracking while a person remains visible.

Person Re-ID provides an appearance-based mechanism for recognizing a returning person after their previous track has disappeared.

---

# 3. Why Edge Computing?

The system is designed to run locally on CPU hardware rather than depending on a cloud GPU or API.

Advantages include:

* No network connection required during inference
* No cloud API cost
* Low infrastructure requirements
* Local processing
* Suitable for classrooms, rooms, laboratories, and other edge environments
* Easier deployment on embedded/edge computers

These requirements motivated the use of:

* YOLOv8n rather than a large detector
* INT8 quantization
* ONNX Runtime
* A lightweight NumPy-based tracker
* Small OSNet x0.25 for Re-ID

---

# 4. YOLOv8n Detection

The detector is **YOLOv8n**, the nano variant of YOLOv8.

The model uses pretrained COCO weights.

No custom dataset or detector training is performed in this project.

Only the COCO `person` class is passed to the tracking and Re-ID stages.

The model is exported to ONNX and then quantized to INT8.

---

# 5. FP32 → INT8 Quantization

The project uses ONNX Runtime post-training dynamic quantization.

The general pipeline is:

```text
YOLOv8n FP32
      ↓
ONNX export
      ↓
yolov8n.onnx
      ↓
Dynamic INT8 quantization
      ↓
yolov8n_int8.onnx
```

The purpose is to reduce model size and potentially improve CPU inference performance.

The trade-off is that quantization can introduce some accuracy differences.

Actual performance should always be measured on the target machine rather than assuming a particular FPS or latency.

---

# 6. ONNX Runtime

Both detection and Re-ID use:

```text
onnxruntime.InferenceSession
```

with:

```text
CPUExecutionProvider
```

Therefore, the running application does not require PyTorch or TensorFlow for inference.

The development environment may contain additional packages used for model conversion/export, but the actual detection and Re-ID inference is performed through ONNX Runtime.

---

# 7. Benchmarking

The project measures:

* FP32 detection latency
* INT8 detection latency
* Model sizes
* Optional Re-ID inference latency
* Full pipeline latency
* YOLO latency
* Tracking latency
* Re-ID latency
* Effective FPS

The values printed by the program are measured on the machine where the program is executed.

Do not use example numbers as benchmark results.

---

# 8. ByteTrack

`tracking.py` contains a lightweight ByteTrack-style tracker implemented using NumPy.

The tracker uses:

* Kalman filtering
* Motion prediction
* IoU-based association
* High-confidence detection association
* Low-confidence detection association
* Track buffering

The tracker generates **temporary track IDs**.

For example:

```text
Track 1
Track 2
Track 3
```

These IDs describe continuous trajectories.

They are **not permanent identities**.

If a person leaves the camera for long enough and later returns, the tracker may create a new track:

```text
Person A

First appearance:
Track 1

Leaves frame

Returns:

Track 4
```

ByteTrack itself cannot know that Track 1 and Track 4 belong to the same physical person.

This is why Person Re-ID is required.

---

# 9. Person Re-ID

The project uses **OSNet x0.25** for appearance-based Person Re-ID.

The model is pretrained and is not trained as part of this project.

### Model

```text
OSNet x0.25
```

### Input

```text
128 × 256
RGB
ImageNet normalization
```

### Output

```text
512-dimensional embedding
```

The embedding is L2-normalized before similarity matching.

Conceptually:

```text
Person crop
     ↓
OSNet
     ↓
512-D embedding
     ↓
cosine similarity
     ↓
Identity Memory
```

The model used in the current project is:

```text
models\osnet_x0_25_msmt17.onnx
```

The model was exported using the current BoxMOT command-line exporter.

---

# 10. Anonymous Identity Memory

`identity_memory.py` maintains anonymous identities.

An identity looks conceptually like:

```text
P1 → embedding
P2 → embedding
P3 → embedding
...
```

The system does not assign names to people.

It stores appearance embeddings used for similarity matching.

The memory can operate in two modes.

### Session-only mode

This is the default.

Identities exist only while the Python program is running.

When the program exits:

```text
P1
P2
P3
...
```

are removed from memory.

### Persistent mode

An optional SQLite database can be enabled with:

```powershell
--persist-db identities.db
```

This allows the identity embeddings to be saved locally and loaded again when the application is restarted.

Persistence does not guarantee recognition across long periods because appearance-based Re-ID can be affected by clothing, lighting, camera angle, and other factors.

---

# 11. Unique Person Counting

For a person who needs Re-ID, the system:

1. Crops the tracked person.
2. Generates a 512-D OSNet embedding.
3. Compares it with stored identities.
4. Finds the most similar existing identity.
5. Compares that similarity with the configured threshold.

The current starting threshold is:

```text
0.6
```

Conceptually:

```text
Similarity >= 0.6
        ↓
Existing identity
        ↓
Do not increase unique count
```

while:

```text
Similarity < 0.6
        ↓
Create new P#
        ↓
Increase unique count
```

The value `0.6` is a **starting value**, not a guarantee of perfect identity separation.

The correct threshold depends on the camera, scene, lighting, clothing, camera angle, and quality of the person crops.

---

# 12. Complete Architecture

```text
Camera / Video
      ↓
YOLOv8n INT8
      ↓
ONNX Runtime CPU
      ↓
Person Detection
      ↓
ByteTrackLite
      ↓
Temporary Track ID
      ↓
Person Crop
      ↓
OSNet Re-ID
      ↓
512-D Embedding
      ↓
Identity Memory
      ↓
Cosine Similarity
      ↓
 ┌───────────────┐
 │ Existing P# ? │
 └───────┬───────┘
         │
     ┌───┴───┐
     │       │
    YES      NO
     │       │
     ↓       ↓
 Existing   Create
 identity   new P#
     │       │
     └───┬───┘
         ↓
   Unique Count
```

---

# 13. Project Structure

Current important files:

```text
mdm project/
│
├── yolov8n.onnx
├── yolov8n_int8.onnx
│
├── models/
│   └── osnet_x0_25_msmt17.onnx
│
├── detect.py
├── quantization.py / quantize.py
├── benchmarking.py / benchmark.py
├── tracking.py
├── reid.py
├── identity_memory.py
├── people_counting-1.py
│
├── requirements.txt
│
└── venv/
```

The exact quantization/benchmark script filename depends on the current project version.

---

# 14. Installation — First Time Only

You only need to perform the full installation when setting up the project on a new machine.

On Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If BoxMOT is required for exporting the OSNet model:

```powershell
pip install boxmot
```

---

# 15. OSNet Model Export

The current BoxMOT version uses the `boxmot export` command.

The old command:

```powershell
python -m boxmot.appearance.reid_export ...
```

is **not used with the current BoxMOT version**.

Current export command:

```powershell
boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --dynamic
```

The exported model was then copied into:

```text
models\osnet_x0_25_msmt17.onnx
```

The current ONNX model has:

```text
Input:
images [batch, 3, 256, 128]

Output:
output0 [batch, 512]
```

The exported model was verified using ONNX Runtime.

---

# 16. Running the Project After Closing VS Code / Terminal

You **do not need to reinstall everything**.

Your virtual environment and installed packages remain on your disk.

Whenever you open a new PowerShell terminal:

### Step 1 — Go to the project

```powershell
cd "C:\Users\Vishwajeet\Downloads\mdm project"
```

### Step 2 — Activate the existing virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

You should see:

```text
(venv) PS C:\Users\Vishwajeet\Downloads\mdm project>
```

### Step 3 — Run the application

```powershell
python people_counting-1.py --reid-model models\osnet_x0_25_msmt17.onnx
```

That's it.

You do **not** need to run:

```text
python -m venv venv
pip install ...
pip install boxmot
```

again.

---

# 17. Quick Start — The Commands You Actually Need

After restarting the computer or closing everything:

```powershell
cd "C:\Users\Vishwajeet\Downloads\mdm project"
```

```powershell
.\venv\Scripts\Activate.ps1
```

Then:

```powershell
python people_counting-1.py --reid-model models\osnet_x0_25_msmt17.onnx
```

### Explicitly using the current starting Re-ID threshold

```powershell
python people_counting-1.py --reid-model models\osnet_x0_25_msmt17.onnx --reid-threshold 0.6
```

### With persistent identity memory

```powershell
python people_counting-1.py --reid-model models\osnet_x0_25_msmt17.onnx --persist-db identities.db
```

---

# 18. Running the Detector Alone

To test YOLO detection without tracking or Re-ID:

```powershell
python detect.py --model yolov8n_int8.onnx --source 0
```

---

# 19. Configuration

Important command-line options:

| Flag                  | Meaning                            |                  Default |
| --------------------- | ---------------------------------- | -----------------------: |
| `--model`             | YOLO ONNX model                    |      `yolov8n_int8.onnx` |
| `--reid-model`        | OSNet ONNX model                   | `models/reid_model.onnx` |
| `--source`            | Webcam/video source                |                      `0` |
| `--input-size`        | YOLO input resolution              |                    `640` |
| `--conf`              | Detection confidence               |                   `0.35` |
| `--track-high-thresh` | High-confidence tracking threshold |                    `0.5` |
| `--track-low-thresh`  | Low-confidence tracking threshold  |                    `0.1` |
| `--match-iou`         | Track/detection IoU threshold      |                    `0.3` |
| `--track-buffer`      | Lost-track buffer                  |                     `30` |
| `--reid-threshold`    | Re-ID similarity threshold         |                    `0.6` |
| `--reid-interval`     | Re-ID refresh interval             |                     `30` |
| `--capacity`          | Optional room capacity             |                      `0` |
| `--persist-db`        | SQLite identity database           |                     None |
| `--no-display`        | Disable OpenCV window              |                      Off |
| `--max-frames`        | Stop after N frames                |                      `0` |

Because the current model is named:

```text
models\osnet_x0_25_msmt17.onnx
```

use that path explicitly when running the current project.

---

# 20. Re-ID Scheduling

Re-ID is deliberately not run on every person in every frame.

The application performs Re-ID when:

1. A track is new.
2. A track has not yet been assigned an identity.
3. The periodic Re-ID interval has been reached.

The default interval is:

```text
30 frames
```

This reduces unnecessary CPU computation.

For example:

```text
Frame 1
Track 1 → Re-ID → P1

Frames 2–29
Track 1 → reuse P1

Frame 30+
Track 1 → Re-ID refresh
```

This is an important optimization because OSNet does not need to process the same person on every frame.

---

# 21. Example Output

The following is an **illustrative example only**.

```text
Running EdgeVision-Lite people counting. Press 'q' to quit.

...

Processed 800 frames
Final current people in frame: 2
Final unique people seen this session: 3

Per-stage average latency:
  YOLO INT8 inference : XX.XX ms
  Tracking (ByteTrack): X.XX ms
  Re-ID inference     : X.XX ms/frame
  Total pipeline      : XX.XX ms
  Effective FPS       : XX.XX
```

Actual values depend on the computer, camera, resolution, scene, and number of Re-ID calls.

---

# 22. Testing Scenarios

The desired behavior is:

| Scenario           | Current People | Unique People |
| ------------------ | -------------: | ------------: |
| Person A enters    |              1 |             1 |
| A continues moving |              1 |             1 |
| Person B enters    |              2 |             2 |
| Person C enters    |              3 |             3 |
| A leaves           |              2 |             3 |
| A returns          |              3 |             3 |
| Person D enters    |              4 |             4 |

The important requirement is:

```text
A leaves
    ↓
A returns
    ↓
Do NOT create another identity
```

At the same time:

```text
New person enters
    ↓
Create P#
    ↓
Unique count increases
```

There is no hard-coded limit such as:

```text
max_people = 3
```

The identity IDs are generated dynamically.

---

# 23. Important Limitation

This system performs **anonymous appearance-based Re-ID**.

It is not guaranteed to identify every person correctly.

Failure can occur because of:

* Similar clothing
* Identical uniforms
* Occlusion
* Poor lighting
* Motion blur
* Camera angle changes
* People crossing each other
* Very small person crops
* Major appearance changes
* Long gaps between appearances

The Re-ID threshold is also a single global threshold.

Therefore:

```text
0.6
```

is currently a starting point for experimentation rather than a mathematically guaranteed separation between different people.

A false match can cause two different people to be treated as one identity.

A false non-match can cause one returning person to receive a new identity.

This is an inherent limitation of appearance-based Re-ID.

---

# 24. Current Identity-Memory Design

The current baseline uses one representative embedding per anonymous identity.

Conceptually:

```text
P1 → representative embedding
P2 → representative embedding
P3 → representative embedding
```

When an existing identity is matched, its embedding is slowly updated using an exponential moving average.

This allows the representation to adapt gradually to changes in appearance.

A possible future improvement is a **gallery containing multiple embeddings per identity**, rather than a single representative embedding.

For example:

```text
P1
 ├── embedding from front view
 ├── embedding from side view
 ├── embedding under different lighting
 └── embedding from later appearance
```

This could improve long-gap and viewpoint robustness.

---

# 25. Future Improvements

Possible future improvements include:

* Multiple embeddings per identity
* More robust identity-gallery matching
* Better crop-quality filtering
* Track-level embedding aggregation
* Improved Re-ID threshold calibration
* Multi-camera identity matching
* OpenVINO CPU acceleration
* Static INT8 quantization with calibration
* Raspberry Pi deployment
* Real-world FPS measurement on edge hardware
* More extensive automated testing

---

# 26. Benchmarking

Run the benchmark script using the actual filename in the project.

For example, if the project contains `benchmark.py`:

```powershell
python benchmark.py --fp32 yolov8n.onnx --int8 yolov8n_int8.onnx --reid models\osnet_x0_25_msmt17.onnx --runs 50
```

If the current file is named `benchmarking.py`, use:

```powershell
python benchmarking.py --help
```

to see its available arguments.

---

# 27. Troubleshooting

### Virtual environment is not active

If the terminal does not begin with:

```text
(venv)
```

run:

```powershell
.\venv\Scripts\Activate.ps1
```

---

### Python cannot find a package

First check:

```powershell
python --version
```

Then:

```powershell
pip list
```

You should be using the Python executable inside:

```text
...\mdm project\venv\
```

Check with:

```powershell
python -c "import sys; print(sys.executable)"
```

---

### Check OSNet model

Run:

```powershell
python -c "import onnxruntime as ort; s=ort.InferenceSession('models/osnet_x0_25_msmt17.onnx', providers=['CPUExecutionProvider']); print('Loaded successfully'); print('Input:', s.get_inputs()[0].name, s.get_inputs()[0].shape); print('Output:', s.get_outputs()[0].name, s.get_outputs()[0].shape)"
```

Expected:

```text
Loaded successfully
Input: images ['batch', 3, 256, 128]
Output: output0 ['batch', 512]
```

---

### Check Python syntax

```powershell
python -m py_compile identity_memory.py
```

```powershell
python -m py_compile people_counting-1.py
```

No output means the files compiled successfully.

---

# 28. Important: Do Not Reinstall Everything After Closing the Project

Closing:

* VS Code
* PowerShell
* the Python program
* the webcam window
* Windows itself

does **not** delete your Python environment.

Your project remains:

```text
C:\Users\Vishwajeet\Downloads\mdm project
```

and your installed packages remain inside:

```text
C:\Users\Vishwajeet\Downloads\mdm project\venv
```

After reopening the computer, the normal workflow is simply:

```powershell
cd "C:\Users\Vishwajeet\Downloads\mdm project"
```

```powershell
.\venv\Scripts\Activate.ps1
```

```powershell
python people_counting-1.py --reid-model models\osnet_x0_25_msmt17.onnx --reid-threshold 0.6
```

No new installation is necessary.

---

# 29. Files

| File                               | Purpose                                                  |
| ---------------------------------- | -------------------------------------------------------- |
| `detect.py`                        | YOLO object detection                                    |
| `quantization.py` / `quantize.py`  | FP32 → INT8 quantization                                 |
| `benchmarking.py` / `benchmark.py` | Model/pipeline benchmarking                              |
| `tracking.py`                      | Lightweight ByteTrack-style tracker                      |
| `reid.py`                          | OSNet Re-ID embedding extraction                         |
| `identity_memory.py`               | Anonymous identity storage and cosine matching           |
| `people_counting-1.py`             | Main detection → tracking → Re-ID → counting application |
| `models/osnet_x0_25_msmt17.onnx`   | OSNet x0.25 Re-ID model                                  |
| `yolov8n.onnx`                     | YOLOv8n FP32 ONNX model                                  |
| `yolov8n_int8.onnx`                | YOLOv8n INT8 ONNX model                                  |
| `requirements.txt`                 | Python dependencies                                      |

---

# 30. Resume / Interview Description

A concise description of the project:

> **EdgeVision-Lite — CPU-First Edge AI Person Counting:** Built a lightweight edge AI pipeline combining YOLOv8n INT8 with ONNX Runtime for CPU-based person detection, a NumPy-based ByteTrack-style tracker for temporal association, and OSNet Person Re-ID for anonymous unique-person counting. Implemented identity memory with cosine-similarity matching and optimized Re-ID scheduling to avoid running the model on every person in every frame. Benchmarked per-stage CPU latency and end-to-end pipeline performance.

The project demonstrates:

* Computer vision
* ONNX Runtime
* Model quantization
* Edge AI
* Object detection
* Multi-object tracking
* Person Re-ID
* Embedding similarity
* CPU optimization
* Real-time pipeline design
* Performance benchmarking
* Python/NumPy/OpenCV
