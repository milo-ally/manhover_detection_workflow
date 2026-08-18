#!/usr/bin/env python3
import argparse
from pathlib import Path

from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to RKNN for RK3588")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--dataset", default="dataset/calibration.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("i8", "fp"), default="fp")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=False)
    try:
        if rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                       target_platform="rk3588") != 0:
            raise RuntimeError("RKNN config failed")
        if rknn.load_onnx(model=args.onnx) != 0:
            raise RuntimeError("ONNX load failed")
        quantize = args.dtype != "fp"
        if rknn.build(do_quantization=quantize,
                      dataset=args.dataset if quantize else None) != 0:
            raise RuntimeError("RKNN build failed")
        if rknn.export_rknn(str(output)) != 0:
            raise RuntimeError("RKNN export failed")
    finally:
        rknn.release()
    print(f"RKNN saved: {output.resolve()}")


if __name__ == "__main__":
    main()
