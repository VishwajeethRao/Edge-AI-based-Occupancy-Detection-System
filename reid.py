"""
EdgeVision-Lite / reid.py

Person Re-Identification (Re-ID) embedding extractor.

WHY RE-ID IS NEEDED ON TOP OF BYTETRACK
-----------------------------------------
ByteTrack (tracking.py) gives every visible person a temporary track
ID, but that ID is only a trajectory label -- it means nothing once
the person leaves the frame. Re-ID solves a different problem:
"does this new crop look like a person I've already counted?" -- by
turning a person crop into a fixed-length appearance embedding that
can be compared against embeddings stored in identity_memory.py.

MODEL CHOICE: OSNet (osnet_x0_25), pretrained on Market-1501
-----------------------------------------------------------------
- What it is: Omni-Scale Network, a CNN designed specifically for
  person re-ID (Zhou et al., "Omni-Scale Feature Learning for Person
  Re-Identification", ICCV 2019 / TPAMI 2021).
- Why this model: OSNet is the standard lightweight Re-ID backbone
  used by widely-adopted CPU tracking stacks (e.g. BoxMOT/DeepOCSORT,
  StrongSORT). The x0_25 width variant is the smallest in the family
  (~2 MB as a PyTorch checkpoint, low-single-digit MB as ONNX),
  specifically built for real-time / embedded use, which matches this
  project's CPU / edge-deployment goal.
- Input size: 128 (width) x 256 (height), RGB, ImageNet-normalized.
- Embedding dimension: 512-d float vector, L2-normalized so cosine
  similarity == dot product.
- CPU-friendly: yes -- it's a few million parameters, runs comfortably
  on CPU through ONNX Runtime; that's the entire point of choosing the
  x0_25 (quarter-width) variant over the full-size OSNet or heavier
  backbones like ResNet50-based Re-ID models.
- Where the model file comes from: it is NOT trained here, and this
  project does not ship the weights or invent a download URL. Obtain
  it via the open-source `boxmot` package's official export tool,
  which downloads the pretrained PyTorch checkpoint from its model zoo
  and converts it to ONNX for you:

      pip install boxmot
      python3 -m boxmot.appearance.reid_export \
          --weights osnet_x0_25_msmt17.pt \
          --include onnx --device cpu

  This produces an .onnx file (the script auto-downloads
  `osnet_x0_25_msmt17.pt` from the boxmot model zoo on first run).
  Move/rename the resulting file to `models/reid_model.onnx` (or point
  `--reid-model` at wherever you saved it). See README.md for the full
  step-by-step.
- How it's loaded: like the YOLO model, through a plain
  `onnxruntime.InferenceSession` with the CPU execution provider --
  no new inference framework introduced.
"""

import cv2
import numpy as np
import onnxruntime as ort

# OSNet's expected input: (height, width) = (256, 128)
REID_INPUT_HEIGHT = 256
REID_INPUT_WIDTH = 128

# Standard ImageNet normalization used by torchreid/OSNet training
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PersonReID:
    """Loads the OSNet ONNX model and turns person crops into embeddings."""

    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, crop_bgr):
        img = cv2.resize(crop_bgr, (REID_INPUT_WIDTH, REID_INPUT_HEIGHT))
        img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0,1]
        img = (img - _MEAN) / _STD
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return np.ascontiguousarray(img)

    def get_embedding(self, crop_bgr):
        """
        crop_bgr: a person crop in OpenCV BGR format (H, W, 3), any size.
        Returns: L2-normalized np.ndarray of shape (512,), or None if the
                 crop is empty/degenerate (e.g. a box clipped to zero area).
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
            return None
        inp = self._preprocess(crop_bgr)
        out = self.session.run(None, {self.input_name: inp})[0]
        emb = out.reshape(-1).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
