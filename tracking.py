"""
EdgeVision-Lite / tracking.py

A compact, dependency-free implementation of the ByteTrack algorithm
(Zhang et al., 2022) for multi-person tracking.

Why a hand-rolled version instead of the official `bytetrack` /
`boxmot` packages?
    The official implementations pull in extra compiled dependencies
    (cython_bbox, lap, sometimes torch) purely for association speed.
    For an edge project that already runs everything through
    ONNX Runtime + NumPy, that's a heavy price for a few % faster
    matching. This module implements the same *algorithm* --
    Kalman-filter motion prediction + two-stage
    high-confidence/low-confidence IOU association -- using only
    NumPy, so it drops into the existing pipeline with zero new
    dependencies.

WHY TRACKING IDS ARE NOT IDENTITIES
------------------------------------
A ByteTrack ID is only a *label for a continuous trajectory*. It is
assigned when a track is created and is retired (never reused) once
the track is lost for too many frames. If a person walks out of frame
and comes back 10 seconds later, ByteTrack has no memory of them --
it will simply start a brand-new track with a brand-new ID. That's
why counting unique people by "number of ByteTrack IDs ever created"
would over-count everyone who left and returned even once.

This is exactly the gap that Person Re-ID (see reid.py) and the
identity memory (see identity_memory.py) are designed to close: they
recognize a *returning* person by appearance, independent of whatever
temporary track ID the tracker happens to assign this time around.
"""

import numpy as np

# ----------------------------------------------------------------------
# Minimal constant-velocity Kalman filter (SORT-style state layout)
# State: [cx, cy, s, r, vcx, vcy, vs]
#   cx, cy = box center | s = box area (scale) | r = aspect ratio (w/h)
#   vcx, vcy, vs = velocities of cx, cy, s (r is assumed constant)
# ----------------------------------------------------------------------


def _xyxy_to_z(bbox):
    """[x1,y1,x2,y2] -> [cx,cy,s,r] measurement vector."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    s = max(w * h, 1e-6)
    r = w / max(h, 1e-6)
    return np.array([cx, cy, s, r], dtype=np.float64)


def _x_to_xyxy(x):
    """[cx,cy,s,r,...] state vector -> [x1,y1,x2,y2] bbox."""
    cx, cy, s, r = x[0], x[1], max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


class _KalmanBoxTracker:
    """Tracks a single box's motion across frames with a linear Kalman filter."""

    count = 0

    def __init__(self, bbox, score, cls_id=0):
        # State transition matrix F (constant velocity model)
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H (we observe cx, cy, s, r directly)
        self.H = np.zeros((4, 7))
        for i in range(4):
            self.H[i, i] = 1.0

        # Process / measurement noise -- standard SORT defaults
        self.Q = np.eye(7) * 1.0
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4) * 1.0

        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # high uncertainty in initial velocity

        self.x = np.zeros(7)
        self.x[:4] = _xyxy_to_z(bbox)

        _KalmanBoxTracker.count += 1
        self.track_id = _KalmanBoxTracker.count

        self.score = score
        self.cls_id = cls_id
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.state = "tracked"  # "tracked" | "lost"

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return _x_to_xyxy(self.x)

    def update(self, bbox, score):
        z = _xyxy_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.score = score
        self.state = "tracked"

    @property
    def bbox(self):
        return _x_to_xyxy(self.x)


def _iou_matrix(boxes_a, boxes_b):
    """Vectorized IOU between two sets of [x1,y1,x2,y2] boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a = np.asarray(boxes_a)[:, None, :]
    b = np.asarray(boxes_b)[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, a[..., 2] - a[..., 0]) * np.maximum(0, a[..., 3] - a[..., 1])
    area_b = np.maximum(0, b[..., 2] - b[..., 0]) * np.maximum(0, b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def _greedy_iou_match(tracks_boxes, det_boxes, iou_thresh):
    """
    Greedy bipartite matching by IOU (no scipy dependency).
    Good enough at the scale of a single-camera person-counting scene
    (tens of simultaneous boxes, not hundreds), and keeps the whole
    tracker to plain NumPy.
    Returns: matches [(track_idx, det_idx), ...], unmatched_tracks, unmatched_dets
    """
    if len(tracks_boxes) == 0 or len(det_boxes) == 0:
        return [], list(range(len(tracks_boxes))), list(range(len(det_boxes)))

    iou = _iou_matrix(tracks_boxes, det_boxes)
    pairs = [(iou[i, j], i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])
             if iou[i, j] >= iou_thresh]
    pairs.sort(key=lambda p: p[0], reverse=True)

    matched_t, matched_d, matches = set(), set(), []
    for score, t, d in pairs:
        if t in matched_t or d in matched_d:
            continue
        matched_t.add(t)
        matched_d.add(d)
        matches.append((t, d))

    unmatched_tracks = [i for i in range(len(tracks_boxes)) if i not in matched_t]
    unmatched_dets = [j for j in range(len(det_boxes)) if j not in matched_d]
    return matches, unmatched_tracks, unmatched_dets


class ByteTrackLite:
    """
    Compact ByteTrack: two-stage association using both high- and
    low-confidence detections, so partially-occluded / motion-blurred
    people (which YOLO still detects, just with lower confidence)
    keep their track instead of being dropped and re-born with a new ID.

    Parameters
    ----------
    high_thresh : detections >= this score are used in stage 1 (primary
                  matching against all active tracks).
    low_thresh  : detections between low_thresh and high_thresh are only
                  used in stage 2, to rescue tracks that stage 1 missed
                  (e.g. a person briefly occluded).
    match_iou   : minimum IOU to accept a match.
    track_buffer: number of consecutive missed frames a track is kept
                  alive ("lost" state) before being removed for good.
    """

    def __init__(self, high_thresh=0.5, low_thresh=0.1, match_iou=0.3, track_buffer=30):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_iou = match_iou
        self.track_buffer = track_buffer
        self.tracks = []  # list[_KalmanBoxTracker]

    def update(self, detections):
        """
        detections: np.ndarray of shape (N, 5) -> [x1, y1, x2, y2, score]
        (caller is responsible for already filtering to the "person" class)

        Returns: list of dicts, one per currently-visible (tracked) track:
            {"track_id": int, "bbox": [x1,y1,x2,y2], "score": float,
             "is_new": bool}  -- is_new True the first frame a track exists
        """
        if len(detections) == 0:
            detections = np.zeros((0, 5))

        high_mask = detections[:, 4] >= self.high_thresh
        low_mask = (detections[:, 4] >= self.low_thresh) & ~high_mask
        high_dets = detections[high_mask]
        low_dets = detections[low_mask]

        # 1. predict motion for every existing track
        pred_boxes = [t.predict() for t in self.tracks]

        # 2. stage 1: match high-score detections against ALL tracks
        matches1, unmatched_tracks1, unmatched_high = _greedy_iou_match(
            pred_boxes, high_dets[:, :4], self.match_iou
        )
        for t_idx, d_idx in matches1:
            self.tracks[t_idx].update(high_dets[d_idx, :4], high_dets[d_idx, 4])

        # 3. stage 2: try to rescue still-unmatched tracks with low-score dets
        remaining_boxes = [pred_boxes[i] for i in unmatched_tracks1]
        matches2, unmatched_tracks2_local, _ = _greedy_iou_match(
            remaining_boxes, low_dets[:, :4], self.match_iou
        )
        for t_local, d_idx in matches2:
            t_idx = unmatched_tracks1[t_local]
            self.tracks[t_idx].update(low_dets[d_idx, :4], low_dets[d_idx, 4])
        unmatched_tracks_final = [unmatched_tracks1[i] for i in unmatched_tracks2_local]

        # 4. mark still-unmatched tracks as lost (kept alive for track_buffer frames)
        for t_idx in unmatched_tracks_final:
            self.tracks[t_idx].state = "lost"

        # 5. spawn new tracks from unmatched high-confidence detections only
        #    (never from low-confidence ones -- too noisy to seed a new identity)
        new_tracks = []
        for d_idx in unmatched_high:
            nt = _KalmanBoxTracker(high_dets[d_idx, :4], high_dets[d_idx, 4])
            new_tracks.append(nt)
        self.tracks.extend(new_tracks)

        # 6. drop tracks that have been lost for too long
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.track_buffer]

        # 7. build output: only currently-visible tracks (matched this frame)
        results = []
        new_ids = {t.track_id for t in new_tracks}
        for t in self.tracks:
            if t.time_since_update == 0:  # updated (matched) this frame
                results.append({
                    "track_id": t.track_id,
                    "bbox": t.bbox.tolist(),
                    "score": t.score,
                    "is_new": t.track_id in new_ids,
                })
        return results
