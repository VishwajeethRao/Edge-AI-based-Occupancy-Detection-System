"""Builds tiny synthetic ONNX models with the RIGHT input/output shapes
so people_counting.py can be smoke-tested end-to-end without needing
the real (large) YOLOv8n / OSNet weights. NOT used by the real app.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto


def make_yolo_stub(path, n_persons=2):
    """Outputs a fixed (1, 84, 8400) tensor with a few confident 'person' boxes
    (class 0) placed at known locations, regardless of the input image."""
    num_anchors = 8400
    data = np.zeros((1, 84, num_anchors), dtype=np.float32)
    # box format per anchor: [cx, cy, w, h, cls0_score, cls1_score, ...]
    rng = np.random.default_rng(0)
    for i in range(n_persons):
        idx = 100 + i * 500
        cx, cy, w, h = 200 + i * 150, 200, 80, 200
        data[0, 0, idx] = cx
        data[0, 1, idx] = cy
        data[0, 2, idx] = w
        data[0, 3, idx] = h
        data[0, 4, idx] = 0.95  # person class score (class 0)

    const_node = helper.make_node(
        "Constant", [], ["output"],
        value=helper.make_tensor("data", TensorProto.FLOAT, data.shape, data.flatten().tolist())
    )
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 84, num_anchors])
    graph = helper.make_graph([const_node], "yolo_stub", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)


def make_reid_stub(path):
    """Outputs a per-image embedding that is a deterministic (but crop-content-dependent)
    function of the input, so different crops -> different embeddings, same crop -> same embedding.
    We do this with a GlobalAveragePool + Flatten + a fixed linear projection to 512-d."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 256, 128])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 512])

    # GAP over spatial dims -> (1,3,1,1)
    gap = helper.make_node("GlobalAveragePool", ["input"], ["gap"])
    flat = helper.make_node("Flatten", ["gap"], ["flat"], axis=1)  # (1,3)

    rng = np.random.default_rng(42)
    W = rng.normal(size=(3, 512)).astype(np.float32)
    w_init = helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten().tolist())
    matmul = helper.make_node("MatMul", ["flat", "W"], ["output"])

    graph = helper.make_graph([gap, flat, matmul], "reid_stub", [inp], [out], initializer=[w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)


def make_fake_video(path, width=640, height=480, num_frames=40):
    """Two solid-colored blocks standing in for two different-looking people,
    static for the whole clip -- enough to smoke-test the plumbing (ONNX
    sessions, cropping, tracking, Re-ID, counting) end-to-end."""
    import cv2
    import numpy as np
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, 10, (width, height))
    for _ in range(num_frames):
        frame = np.full((height, width, 3), 200, dtype=np.uint8)
        cv2.rectangle(frame, (160, 100), (320, 400), (0, 0, 255), -1)
        cv2.rectangle(frame, (400, 100), (560, 400), (255, 0, 0), -1)
        vw.write(frame)
    vw.release()


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    make_yolo_stub("models/fake_yolo.onnx")
    make_reid_stub("models/fake_reid.onnx")
    make_fake_video("tests/fake_video.mp4")
    print("wrote models/fake_yolo.onnx, models/fake_reid.onnx, tests/fake_video.mp4")
