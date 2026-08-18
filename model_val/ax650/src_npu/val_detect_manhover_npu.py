#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation_common import run_validation  # noqa: E402


class AxRunner:
    def __init__(self, model_path, output_name):
        import axengine

        if hasattr(axengine, "InferenceSession"):
            self.session = axengine.InferenceSession(model_path)
        elif hasattr(axengine, "Inference"):
            self.session = axengine.Inference(model_path)
        else:
            raise RuntimeError("axengine has no supported inference session")
        self.input = self.session.get_inputs()[0]
        self.outputs = list(self.session.get_outputs())
        output_names = [item.name for item in self.outputs]
        if output_name not in output_names:
            raise ValueError(f"output {output_name!r} not found: {output_names}")
        self.output_name = output_name
        self.output_index = output_names.index(output_name)
        print(f"AXEngine input: {self.input.name} {self.input.shape}; outputs: {output_names}")

    def __call__(self, tensor):
        try:
            outputs = self.session.run(None, {self.input.name: tensor})
        except TypeError:
            outputs = self.session.run([tensor])
        if isinstance(outputs, dict):
            return np.asarray(outputs[self.output_name])
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        return np.asarray(outputs[self.output_index])


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
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected: true / false")


def main():
    parser = argparse.ArgumentParser(description="Validate the manhole-cover AX650 detector")
    parser.add_argument("--axmodel", required=True)
    parser.add_argument("--data", default="data_npu.yaml")
    parser.add_argument("--output-name", default="output0")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-path", default="runs/manhole-cover-yolo11s-production_axmodel.txt")
    parser.add_argument("--prediction-path", default="runs/manhole-cover-yolo11s-production_axmodel_predictions.jsonl")
    parser.add_argument("--image-dir", default="runs/manhole-cover-yolo11s-production_axmodel_images")
    parser.add_argument("--save-images", type=str2bool, default=True, help="whether save drawn result images, true/false")
    parser.add_argument("--metrics-json", help="Optional: output metrics json file path, e.g runs/npu_metrics.json")

    args = parser.parse_args()
    runner = AxRunner(args.axmodel, args.output_name)

    actual_image_dir = args.image_dir if args.save_images else None

    run_validation(
        runner,
        runner.input.shape,
        "axmodel",
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
        out_json = {"summary": metric}
        out_p = Path(args.metrics_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nMetrics json saved: {out_p.resolve()}")


if __name__ == "__main__":
    main()
