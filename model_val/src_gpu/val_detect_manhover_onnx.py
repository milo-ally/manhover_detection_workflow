#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation_common import run_validation  # noqa: E402


class OnnxRunner:
    def __init__(self, model_path, output_name, provider):
        import onnxruntime as ort

        available = ort.get_available_providers()
        if provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(f"CUDAExecutionProvider unavailable: {available}")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input = self.session.get_inputs()[0]
        output_names = [item.name for item in self.session.get_outputs()]
        if output_name not in output_names:
            raise ValueError(f"output {output_name!r} not found: {output_names}")
        self.output_name = output_name
        print(f"ONNX Runtime providers: {self.session.get_providers()}")

    def __call__(self, tensor):
        return self.session.run([self.output_name], {self.input.name: tensor})[0]


def main():
    parser = argparse.ArgumentParser(description="Validate the manhole-cover ONNX detector")
    parser.add_argument("--onnx_model", required=True)
    parser.add_argument("--data", default="data_gpu.yaml")
    parser.add_argument("--output-name", default="output0")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-path", default="runs/best_sim_onnx.txt")
    args = parser.parse_args()
    runner = OnnxRunner(args.onnx_model, args.output_name, args.provider)
    run_validation(runner, runner.input.shape, "onnx", args.data, args.conf_thres,
                   args.iou_thres, args.max_det, args.save_path, args.limit)


if __name__ == "__main__":
    main()
