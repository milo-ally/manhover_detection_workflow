#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation_common import run_validation  # noqa: E402


class OnnxRunner:
    def __init__(self, model_path, output_name, device):
        import onnxruntime as ort

        self.session = ort.InferenceSession(model_path, providers=resolve_onnx_provider(device))
        self.input = self.session.get_inputs()[0]
        output_names = [item.name for item in self.session.get_outputs()]
        if output_name not in output_names:
            raise ValueError(f"output {output_name!r} not found: {output_names}")
        self.output_name = output_name
        print(f"ONNX Runtime providers: {self.session.get_providers()}")

    def __call__(self, tensor):
        return self.session.run([self.output_name], {self.input.name: tensor})[0]


def parse_val_txt_report(txt_path: Path):
    """解析run_validation输出的txt报告，提取全局指标"""
    lines = [line.rstrip("\n") for line in txt_path.read_text(encoding="utf-8").splitlines()]
    header_idx = None
    for idx, line in enumerate(lines):
        if "Class" in line and ("mAP50-95" in line or "mAP50‑95" in line):
            header_idx = idx
            break
    if header_idx is None:
        raise RuntimeError("Cannot find report header, invalid validation report txt file.")
    all_line = lines[header_idx + 1].strip()
    parts = [p for p in all_line.split() if p]
    return {
        "images": int(parts[1]),
        "total_targets": int(parts[2]),
        "precision": float(parts[3]),
        "recall": float(parts[4]),
        "mAP50": float(parts[5]),
        "mAP50_95": float(parts[6])
    }


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected: true / false")


def main():
    parser = argparse.ArgumentParser(description="Validate the manhole-cover ONNX detector")
    parser.add_argument("--onnx_model", required=True)
    parser.add_argument("--data", default="data_gpu.yaml")
    parser.add_argument("--output-name", default="output0")
    parser.add_argument("--device", default="cpu",
                        help="inference device: 'cpu' or 'cudaN' (e.g. cuda0, cuda1, cuda2)")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-path", default="runs/manhole-cover-yolo11s-production_onnx.txt")
    parser.add_argument("--prediction-path", default="runs/manhole-cover-yolo11s-production_onnx_predictions.jsonl")
    parser.add_argument("--image-dir", default="runs/manhole-cover-yolo11s-production_onnx_images")
    parser.add_argument("--save-images", type=str2bool, default=True, help="whether save drawn result images, true/false")
    parser.add_argument("--metrics-json", help="Optional: output metrics json file path, e.g runs/metrics.json")

    args = parser.parse_args()
    runner = OnnxRunner(args.onnx_model, args.output_name, args.device)

    actual_image_dir = args.image_dir if args.save_images else None

    run_validation(
        runner,
        runner.input.shape,
        "onnx",
        args.data,
        args.conf_thres,
        args.iou_thres,
        args.max_det,
        args.save_path,
        args.prediction_path,
        actual_image_dir,
        args.limit
    )

    if args.metrics_json:
        txt_file = Path(args.save_path)
        metric = parse_val_txt_report(txt_file)
        out_json = {
            "summary": metric
        }
        out_p = Path(args.metrics_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nMetrics json saved: {out_p.resolve()}")


if __name__ == "__main__":
    main()
