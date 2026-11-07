"""
EdgeVision-Lite / people_counting.py

Anonymous unique-person-counting application.

Pipeline
--------
Camera/Video -> YOLOv8n INT8 (ONNX Runtime) -> filter to "person" class
             -> ByteTrack (tracking.py, temporary track IDs)
             -> crop each tracked person
             -> Person Re-ID (reid.py) -> 512-d embedding
             -> IdentityMemory (identity_memory.py) -> cosine similarity match
             -> already seen?  yes -> keep identity, no increment
                                no  -> new anonymous identity, unique_count += 1

This script reuses detect.py's existing letterbox/preprocess/postprocess/nms
functions and COCO_CLASSES list unchanged -- no preprocessing or ONNX
session logic is duplicated here.

IMPORTANT, read before trusting the numbers on screen: this is a
prototype demonstrating ANONYMOUS, SIMILARITY-BASED person
re-identification. It is not a biometric identification system and it
is not guaranteed to recognize every returning person correctly --
see README.md "Limitations" for known failure modes (similar
clothing, occlusion, lighting change, camera angle, long gaps between
appearances, etc.).
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

from detect import COCO_CLASSES, preprocess, postprocess, nms
from tracking import ByteTrackLite
from reid import PersonReID
from identity_memory import IdentityMemory

PERSON_CLASS_ID = COCO_CLASSES.index("person")


def parse_args():
    p = argparse.ArgumentParser(description="EdgeVision-Lite: anonymous unique person counting")
    p.add_argument("--model", default="yolov8n_int8.onnx", help="Path to YOLOv8n ONNX model (INT8 recommended)")
    p.add_argument("--reid-model", default="models/reid_model.onnx", help="Path to OSNet Re-ID ONNX model")
    p.add_argument("--source", default=0, help="Webcam index or video file path")
    p.add_argument("--input-size", type=int, default=640, help="YOLO input resolution")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold (final display)")

    # ByteTrack parameters
    p.add_argument("--track-high-thresh", type=float, default=0.5, help="ByteTrack stage-1 high-confidence threshold")
    p.add_argument("--track-low-thresh", type=float, default=0.1, help="ByteTrack stage-2 low-confidence threshold")
    p.add_argument("--match-iou", type=float, default=0.3, help="Minimum IOU to associate a detection with a track")
    p.add_argument("--track-buffer", type=int, default=30, help="Frames a lost track is kept before removal")

    # Re-ID / identity memory parameters
    p.add_argument("--reid-threshold", type=float, default=0.6,
                    help="Cosine similarity threshold to treat a person as 'already seen' (STARTING VALUE, tune on your own video)")
    p.add_argument("--reid-interval", type=int, default=30,
                    help="Re-run Re-ID on an already-identified track every N frames (periodic refresh)")

    p.add_argument("--capacity", type=int, default=0, help="Optional room/class capacity; 0 disables the 'remaining' display")
    p.add_argument("--persist-db", default=None,
                    help="Optional path to a SQLite file (e.g. identities.db). If given, identities "
                         "survive across runs -- loaded on startup, written through on every update. "
                         "If omitted (default), identity memory is session-only, exactly as before.")
    p.add_argument("--no-display", action="store_true", help="Run headless (useful for testing/CI, no cv2 window)")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = run until source ends / 'q')")
    return p.parse_args()


def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return x1, y1, x2, y2


def main():
    args = parse_args()

    yolo_session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    yolo_input_name = yolo_session.get_inputs()[0].name

    reid = PersonReID(args.reid_model)

    tracker = ByteTrackLite(
        high_thresh=args.track_high_thresh,
        low_thresh=args.track_low_thresh,
        match_iou=args.match_iou,
        track_buffer=args.track_buffer,
    )
    memory = IdentityMemory(threshold=args.reid_threshold, db_path=args.persist_db)
    if args.persist_db:
        print(f"Persistent identity memory: {args.persist_db} ({len(memory)} identities loaded from previous sessions)")

    # track_id -> identity_id, so we don't re-run Re-ID on a track we already resolved
    track_to_identity = {}
    # track_id -> frame index of last Re-ID run on that track (for periodic refresh)
    last_reid_frame = {}

    try:
        source = int(args.source)
    except (ValueError, TypeError):
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    # per-stage timing (ms), rolling window for the on-screen overlay + full history for the final report
    yolo_times, track_times, reid_times, total_times = [], [], [], []
    reid_calls_per_frame = deque(maxlen=30)

    frame_idx = 0
    print("Running EdgeVision-Lite people counting. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        h, w = frame.shape[:2]

        t_total0 = time.perf_counter()

        # --- 1. YOLOv8n INT8 detection (reusing detect.py's preprocessing) ---
        inp, scale, pad_x, pad_y = preprocess(frame, args.input_size)
        t0 = time.perf_counter()
        outputs = yolo_session.run(None, {yolo_input_name: inp})
        t1 = time.perf_counter()
        yolo_times.append((t1 - t0) * 1000)

        boxes, class_ids, confidences = postprocess(outputs[0], scale, pad_x, pad_y, conf_thres=args.conf)
        keep = nms(boxes, confidences) if len(boxes) else []

        # keep only "person" detections, converted to [x1,y1,x2,y2,score] for the tracker
        person_dets = []
        for i in keep:
            if class_ids[i] != PERSON_CLASS_ID:
                continue
            x, y, bw, bh = boxes[i]
            person_dets.append([x, y, x + bw, y + bh, float(confidences[i])])
        person_dets = np.array(person_dets, dtype=np.float64) if person_dets else np.zeros((0, 5))

        # --- 2. ByteTrack: temporary track IDs, handles entering/leaving/occlusion ---
        t0 = time.perf_counter()
        tracks = tracker.update(person_dets)
        t1 = time.perf_counter()
        track_times.append((t1 - t0) * 1000)

        # --- 3. Re-ID: run only when needed (new track, unresolved track, or periodic refresh) ---
        reid_calls = 0
        t_reid_total = 0.0
        for tr in tracks:
            tid = tr["track_id"]
            need_reid = (
                tid not in track_to_identity
                or (frame_idx - last_reid_frame.get(tid, -10**9)) >= args.reid_interval
            )
            if not need_reid:
                continue

            x1, y1, x2, y2 = clip_box(*tr["bbox"], w, h)
            crop = frame[y1:y2, x1:x2]

            t0 = time.perf_counter()
            embedding = reid.get_embedding(crop)
            t1 = time.perf_counter()
            t_reid_total += (t1 - t0) * 1000
            reid_calls += 1

            if embedding is None:
                continue  # degenerate crop, try again next frame

            identity_id, is_new, similarity = memory.match_or_create(embedding, frame_idx)
            track_to_identity[tid] = identity_id
            last_reid_frame[tid] = frame_idx

        reid_times.append(t_reid_total)
        reid_calls_per_frame.append(reid_calls)

        t_total1 = time.perf_counter()
        total_times.append((t_total1 - t_total0) * 1000)

        # --- 4. Counting ---
        current_people = len(tracks)
        unique_people = len(memory)

        # --- 5. Visualization ---
        if not args.no_display:
            for tr in tracks:
                tid = tr["track_id"]
                x1, y1, x2, y2 = clip_box(*tr["bbox"], w, h)
                identity_id = track_to_identity.get(tid)
                identity_label = f"P{identity_id}" if identity_id is not None else "..."

                cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 94, 170), 2)
                label = f"Track:{tid}  ID:{identity_label}  {tr['score']:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (46, 94, 170), 2)

            overlay = f"Current people: {current_people}   Unique people seen: {unique_people}"
            cv2.putText(frame, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            if args.capacity > 0:
                remaining = max(args.capacity - unique_people, 0)
                cap_line = f"Class capacity: {args.capacity}   Remaining: {remaining}"
                cv2.putText(frame, cap_line, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            avg_total = sum(total_times[-30:]) / len(total_times[-30:])
            perf_line = f"Total: {avg_total:.1f} ms  |  YOLO: {yolo_times[-1]:.1f} ms  |  Track: {track_times[-1]:.1f} ms  |  Re-ID: {reid_times[-1]:.1f} ms ({reid_calls} calls)"
            cv2.putText(frame, perf_line, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            cv2.imshow("EdgeVision-Lite: Unique Person Counting (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print(f"[frame {frame_idx}] current={current_people} unique={unique_people}")

        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    memory.close()

    # --- Final report (measured, not fabricated) ---
    print(f"\nProcessed {frame_idx} frames")
    print(f"Final current people in frame: {current_people if frame_idx else 0}")
    print(f"Final unique people seen this session: {len(memory)}")
    if total_times:
        print("\nPer-stage average latency:")
        print(f"  YOLO INT8 inference : {sum(yolo_times)/len(yolo_times):.2f} ms")
        print(f"  Tracking (ByteTrack): {sum(track_times)/len(track_times):.2f} ms")
        avg_reid = sum(reid_times) / len(reid_times)
        avg_reid_calls = sum(reid_calls_per_frame) / len(reid_calls_per_frame) if reid_calls_per_frame else 0
        print(f"  Re-ID inference     : {avg_reid:.2f} ms/frame (avg {avg_reid_calls:.1f} calls/frame)")
        print(f"  Total pipeline      : {sum(total_times)/len(total_times):.2f} ms")
        print(f"  Effective FPS       : {1000/(sum(total_times)/len(total_times)):.2f}")


if __name__ == "__main__":
    main()
