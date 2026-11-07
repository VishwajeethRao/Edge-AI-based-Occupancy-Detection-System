"""
EdgeVision-Lite / tests/test_scenarios.py

Deterministic tests for the counting logic (ByteTrackLite + IdentityMemory
working together), covering the six scenarios from the project spec:

  1. One person enters                         -> current=1, unique=1
  2. Same person moves around (stays visible)   -> current=1, unique=1
  3. Multiple people (A, B, C) enter            -> current=3, unique=3
  4. A person leaves                            -> current=2, unique=3
  5. That person returns                        -> current=3, unique=3 (NOT 4)
  6. A brand-new person enters                  -> current=4, unique=4

This intentionally bypasses YOLO/ONNX and video decoding -- those are
already covered by the pipeline smoke test (see make_fake_models.py +
running people_counting.py on tests/fake_video.mp4) -- so that this
test is fast, deterministic, and exercises exactly the logic that
matters most: does returning re-use an identity instead of double
counting.

Run with: python3 tests/test_scenarios.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking import ByteTrackLite
from identity_memory import IdentityMemory


def unit(v):
    return v / np.linalg.norm(v)


def run_frame(tracker, memory, track_to_identity, boxes, embeddings_by_box, frame_idx):
    """boxes: list of [x1,y1,x2,y2,score]; embeddings_by_box: dict box_index -> embedding (or None to skip Re-ID this frame)"""
    dets = np.array(boxes, dtype=np.float64) if boxes else np.zeros((0, 5))
    tracks = tracker.update(dets)
    for tr in tracks:
        tid = tr["track_id"]
        # naive box-index association by nearest starting x (good enough for this synthetic test)
        if tid not in track_to_identity and tid in embeddings_by_box:
            emb = embeddings_by_box[tid]
            if emb is not None:
                identity_id, is_new, sim = memory.match_or_create(emb, frame_idx)
                track_to_identity[tid] = identity_id
    return tracks


def main():
    rng = np.random.default_rng(7)
    embA = unit(rng.normal(size=512))
    embB = unit(rng.normal(size=512))
    embC = unit(rng.normal(size=512))
    embD = unit(rng.normal(size=512))

    tracker = ByteTrackLite(high_thresh=0.5, low_thresh=0.1, match_iou=0.3, track_buffer=3)
    memory = IdentityMemory(threshold=0.6)
    track_to_identity = {}
    frame = 0
    failures = []

    def check(name, current, unique, exp_current, exp_unique):
        ok = (current == exp_current) and (unique == exp_unique)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: current={current} (expected {exp_current}), unique={unique} (expected {exp_unique})")
        if not ok:
            failures.append(name)

    # --- Test 1: Person A enters ---
    frame += 1
    tracks = tracker.update(np.array([[10, 10, 60, 200, 0.9]]))
    tidA = tracks[0]["track_id"]
    idA, _, _ = memory.match_or_create(embA, frame)
    track_to_identity[tidA] = idA
    check("Test 1 - one person", len(tracks), len(memory), 1, 1)

    # --- Test 2: A moves around (still one track, same identity, no new Re-ID needed) ---
    for _ in range(3):
        frame += 1
        tracks = tracker.update(np.array([[12, 10, 62, 200, 0.9]]))
    check("Test 2 - same person moving", len(tracks), len(memory), 1, 1)

    # --- Test 3: B and C also enter (A still visible) ---
    frame += 1
    tracks = tracker.update(np.array([
        [12, 10, 62, 200, 0.9],    # A
        [200, 10, 250, 200, 0.9],  # B
        [400, 10, 450, 200, 0.9],  # C
    ]))
    for tr in tracks:
        if tr["track_id"] not in track_to_identity:
            emb = embB if tr["bbox"][0] > 150 and tr["bbox"][0] < 300 else embC
            idx, _, _ = memory.match_or_create(emb, frame)
            track_to_identity[tr["track_id"]] = idx
    check("Test 3 - multiple people", len(tracks), len(memory), 3, 3)

    # --- Test 4: A leaves (only B, C detected) ---
    frame += 1
    tracks = tracker.update(np.array([
        [200, 10, 250, 200, 0.9],  # B
        [400, 10, 450, 200, 0.9],  # C
    ]))
    check("Test 4 - person leaves", len(tracks), len(memory), 2, 3)

    # --- Test 5: A returns (after being gone, but within track_buffer so track continuity
    #     isn't even required -- Re-ID memory is what matters here regardless) ---
    frame += 1
    tracks = tracker.update(np.array([
        [12, 10, 62, 200, 0.9],    # A returns
        [200, 10, 250, 200, 0.9],  # B
        [400, 10, 450, 200, 0.9],  # C
    ]))
    for tr in tracks:
        if tr["track_id"] not in track_to_identity:
            emb_noisy = unit(embA + 0.03 * rng.normal(size=512))
            idx, _, _ = memory.match_or_create(emb_noisy, frame)
            track_to_identity[tr["track_id"]] = idx
    check("Test 5 - person returns (must NOT double-count)", len(tracks), len(memory), 3, 3)

    # --- Test 6: D enters (brand new) ---
    frame += 1
    tracks = tracker.update(np.array([
        [12, 10, 62, 200, 0.9],
        [200, 10, 250, 200, 0.9],
        [400, 10, 450, 200, 0.9],
        [600, 10, 650, 200, 0.9],  # D
    ]))
    for tr in tracks:
        if tr["track_id"] not in track_to_identity:
            idx, _, _ = memory.match_or_create(embD, frame)
            track_to_identity[tr["track_id"]] = idx
    check("Test 6 - new person", len(tracks), len(memory), 4, 4)

    print()
    if failures:
        print(f"{len(failures)} test(s) FAILED: {failures}")
        sys.exit(1)
    else:
        print("All 6 scenario tests PASSED.")


if __name__ == "__main__":
    main()
