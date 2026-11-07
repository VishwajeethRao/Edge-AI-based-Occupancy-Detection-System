"""
EdgeVision-Lite / quantize.py

Post-training dynamic quantization: converts the FP32 ONNX model to
INT8. This is the single most common technique for shrinking a model
and speeding up CPU inference for edge/embedded deployment.
"""

import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="yolov8n.onnx")
    parser.add_argument("--output", default="yolov8n_int8.onnx")
    args = parser.parse_args()

    quantize_dynamic(
        model_input=args.input,
        model_output=args.output,
        weight_type=QuantType.QUInt8,
    )
    print(f"Quantized model saved to {args.output}")


if __name__ == "__main__":
    main()
