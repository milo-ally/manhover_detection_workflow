#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation_common import run_validation  # noqa: E402


class RknnRunner:
    def __init__(self, model_path):
        from rknnlite.api import RKNNLite

        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model_path) != 0:
            raise RuntimeError(f"failed to load RKNN model: {model_path}")
        if self.rknn.init_runtime() != 0:
            raise RuntimeError("failed to initialize RKNNLite runtime")
        self.input_shape = [1, 640, 640, 3]

    def __call__(self, tensor):
        outputs = self.rknn.inference(inputs=[tensor])
        if not outputs:
            raise RuntimeError("RKNN returned no outputs")
        return np.asarray(outputs[0])

    def close(self):
        self.rknn.release()


def main():
    parser = argparse.ArgumentParser(description="Validate the RK3588 RKNN model")
    parser.add_argument("--rknn", required=True)
    parser.add_argument("--data", default="data_rknn.yaml")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-path", default="runs/manhole-cover-yolo11s-production_rknn.txt")
    parser.add_argument("--prediction-path", default="runs/manhole-cover-yolo11s-production_rknn_predictions.jsonl")
    parser.add_argument("--image-dir", default="runs/manhole-cover-yolo11s-production_rknn_images")
    args = parser.parse_args()

    runner = RknnRunner(args.rknn)
    try:
        run_validation(runner, runner.input_shape, "rknn", args.data,
                       args.conf_thres, args.iou_thres, args.max_det,
                       args.save_path, args.prediction_path, args.image_dir,
                       args.limit)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
