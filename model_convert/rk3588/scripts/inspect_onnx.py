#!/usr/bin/env python3
import argparse
import onnx


parser = argparse.ArgumentParser()
parser.add_argument("onnx_model")
args = parser.parse_args()
model = onnx.load(args.onnx_model)
onnx.checker.check_model(model)
for value in model.graph.input:
    shape = [d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]
    print("input", value.name, shape)
for value in model.graph.output:
    shape = [d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]
    print("output", value.name, shape)
print("onnx checker: OK")
