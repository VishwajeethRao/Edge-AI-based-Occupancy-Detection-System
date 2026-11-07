"""
EdgeVision-Lite / benchmark.py

Compares inference latency of the FP32 model vs. its INT8-quantized
version on CPU -- this is the core "edge AI" experiment: showing that
quantization trades a small amount of accuracy for a real, measurable
speed/size win, which is exactly the trade-off edge deployment cares about.

Run quantize.py first to produce yolov8n_int8.onnx.

Optionally also benchmarks the Re-ID (OSNet) model in isolation with
--reid, so the extra cost of the person-counting pipeline's Re-ID stage
is reported honestly rather than folded silently into "detection is
slow" -- see also the live per-stage breakdown people_counting.py
prints at the end of a run, which measures the *whole* pipeline
(YOLO + tracking + Re-ID) together on real frames, not synthetic input.
"""

import time
import argparse
import numpy as np
import onnxruntime as ort
import os


def benchmark_model(model_path, input_size=640, runs=50, warmup=5, fixed_input_shape=None):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    shape = fixed_input_shape if fixed_input_shape is not None else (1, 3, input_size, input_size)
    dummy = np.random.rand(*shape).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "model": os.path.basename(model_path),
        "size_mb": os.path.getsize(model_path) / (1024 * 1024),
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", default="yolov8n.onnx")
    parser.add_argument("--int8", default="yolov8n_int8.onnx")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--reid", default=None,
                         help="Optional path to the Re-ID ONNX model; if given, also benchmarks it "
                              "(input size fixed at 1x3x256x128, OSNet's expected input) so the extra "
                              "per-crop cost of person re-identification is reported honestly.")
    args = parser.parse_args()

    results = []
    if os.path.exists(args.fp32):
        results.append(benchmark_model(args.fp32, runs=args.runs))
    if os.path.exists(args.int8):
        results.append(benchmark_model(args.int8, runs=args.runs))

    print(f"\n{'Model':<20}{'Size (MB)':<12}{'Avg (ms)':<12}{'Min (ms)':<12}{'Max (ms)':<12}")
    print("-" * 68)
    for r in results:
        print(f"{r['model']:<20}{r['size_mb']:<12.2f}{r['avg_ms']:<12.2f}{r['min_ms']:<12.2f}{r['max_ms']:<12.2f}")

    if len(results) == 2:
        speedup = results[0]["avg_ms"] / results[1]["avg_ms"]
        shrink = results[0]["size_mb"] / results[1]["size_mb"]
        print(f"\nINT8 model is {speedup:.2f}x faster and {shrink:.2f}x smaller than FP32.")

    if args.reid and os.path.exists(args.reid):
        reid_result = benchmark_model(args.reid, input_size=None, runs=args.runs,
                                       fixed_input_shape=(1, 3, 256, 128))
        print(f"\n{'Model':<20}{'Size (MB)':<12}{'Avg (ms)':<12}{'Min (ms)':<12}{'Max (ms)':<12}")
        print("-" * 68)
        print(f"{reid_result['model']:<20}{reid_result['size_mb']:<12.2f}{reid_result['avg_ms']:<12.2f}"
              f"{reid_result['min_ms']:<12.2f}{reid_result['max_ms']:<12.2f}")
        print("(Re-ID runs once PER PERSON CROP, not once per frame -- see people_counting.py's "
              "'Re-ID inference' line for the real, amortized per-frame cost on your video, which "
              "depends on how many people are visible and the --reid-interval setting.)")


if __name__ == "__main__":
    main()
