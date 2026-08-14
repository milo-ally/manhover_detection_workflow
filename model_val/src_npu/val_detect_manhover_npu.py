#!/usr/bin/env python3
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Validate the manhole-cover AX650 detector")
    parser.add_argument("--axmodel", required=True)
    parser.add_argument("--data", default="data_npu.yaml")
    parser.add_argument("--output-name", default="output0")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-path", default="runs/best_sim_axmodel.txt")
    args = parser.parse_args()
    runner = AxRunner(args.axmodel, args.output_name)
    run_validation(runner, runner.input.shape, "axmodel", args.data, args.conf_thres,
                   args.iou_thres, args.max_det, args.save_path, args.limit)


if __name__ == "__main__":
    main()
