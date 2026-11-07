"""
EdgeVision-Lite
Real-time object detection on CPU using a quantized YOLOv8n ONNX model
via ONNX Runtime -- a lightweight, cross-platform inference engine
built for edge / embedded deployment (the same runtime used on phones,
Raspberry Pis, and microcontrollers-with-NPUs in production).

Author: Vishwajeeth Rao
"""

import time
import argparse
import cv2
import numpy as np
import onnxruntime as ort

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def letterbox(img, new_shape=640):
    """Resize + pad image to a square while keeping aspect ratio (YOLO-style)."""
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def preprocess(frame, input_size=640):
    img, scale, pad_x, pad_y = letterbox(frame, input_size)
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0  # BGR->RGB, HWC->CHW
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img), scale, pad_x, pad_y


def nms(boxes, scores, iou_threshold=0.45):
    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.25, nms_threshold=iou_threshold)
    return idxs.flatten() if len(idxs) > 0 else []


def postprocess(output, scale, pad_x, pad_y, conf_thres=0.25):
    """output: (1, 84, 8400) for YOLOv8 -> [x,y,w,h, 80 class scores]"""
    preds = output[0].transpose(1, 0)  # (8400, 84)
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    mask = confidences > conf_thres
    boxes_xywh, class_ids, confidences = boxes_xywh[mask], class_ids[mask], confidences[mask]

    boxes_xyxy = []
    for (cx, cy, w, h) in boxes_xywh:
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale
        boxes_xyxy.append([x1, y1, x2 - x1, y2 - y1])  # x,y,w,h for NMSBoxes

    return np.array(boxes_xyxy) if len(boxes_xyxy) else np.zeros((0, 4)), class_ids, confidences


def main():
    parser = argparse.ArgumentParser(description="EdgeVision-Lite: ONNX Runtime real-time object detection")
    parser.add_argument("--model", default="yolov8n.onnx", help="Path to ONNX model")
    parser.add_argument("--source", default=0, help="Webcam index or video file path")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    try:
        source = int(args.source)
    except (ValueError, TypeError):
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    latencies = []
    frame_count = 0

    print("Running EdgeVision-Lite. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        inp, scale, pad_x, pad_y = preprocess(frame, args.input_size)

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: inp})
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)
        frame_count += 1

        boxes, class_ids, confidences = postprocess(outputs[0], scale, pad_x, pad_y, args.conf)
        keep = nms(boxes, confidences) if len(boxes) else []

        for i in keep:
            x, y, w, h = boxes[i].astype(int)
            label = f"{COCO_CLASSES[class_ids[i]]} {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (46, 94, 170), 2)
            cv2.putText(frame, label, (x, max(y - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (46, 94, 170), 2)

        fps = 1000 / latency_ms if latency_ms > 0 else 0
        avg_latency = sum(latencies[-30:]) / len(latencies[-30:])
        cv2.putText(frame, f"FPS: {fps:.1f}  |  Avg inference: {avg_latency:.1f} ms",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("EdgeVision-Lite (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if latencies:
        print(f"\nProcessed {frame_count} frames")
        print(f"Average inference latency: {sum(latencies)/len(latencies):.2f} ms")
        print(f"Average FPS: {1000/(sum(latencies)/len(latencies)):.2f}")


if __name__ == "__main__":
    main()
