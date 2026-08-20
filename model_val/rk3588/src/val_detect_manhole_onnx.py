#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation_common import run_validation  # noqa: E402


class OnnxRunner:
    def __init__(self, model_path, device="cpu"):
        import onnxruntime as ort

        self.session = ort.InferenceSession(model_path, providers=resolve_onnx_provider(device))
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0].name

    def __call__(self, tensor):
        return self.session.run([self.output], {self.input.name: tensor})[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--data", default="data_rknn.yaml")
    parser.add_argument("--device", default="cpu",
                        help="inference device: 'cpu' or 'cudaN' (e.g. cuda0, cuda1, cuda2)")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    runner = OnnxRunner(args.onnx, args.device)
    run_validation(runner, runner.input.shape, "onnx", args.data,
                   args.conf_thres, args.iou_thres, 300,
                   "runs/manhole-cover-yolo11s-production_onnx.txt",
                   "runs/manhole-cover-yolo11s-production_onnx_predictions.jsonl",
                   "runs/manhole-cover-yolo11s-production_onnx_images", args.limit)


if __name__ == "__main__":
    main()
