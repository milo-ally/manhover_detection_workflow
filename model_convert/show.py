#!/usr/bin/env python3
"""Inspect an ONNX model using stable, explicit command-line arguments."""

import argparse
import json
from pathlib import Path

import onnx


def shape_of(value_info):
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
    return dtype, dims


def describe(model):
    initializers = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    return {
        "ir_version": model.ir_version,
        "opset": [{"domain": item.domain or "ai.onnx", "version": item.version}
                   for item in model.opset_import],
        "inputs": [{"name": item.name, "dtype": shape_of(item)[0], "shape": shape_of(item)[1]}
                   for item in inputs],
        "outputs": [{"name": item.name, "dtype": shape_of(item)[0], "shape": shape_of(item)[1]}
                    for item in model.graph.output],
        "nodes": [{"index": index, "name": node.name, "op_type": node.op_type,
                   "inputs": list(node.input), "outputs": list(node.output)}
                  for index, node in enumerate(model.graph.node)],
    }


def render_text(info):
    lines = [f"IR version: {info['ir_version']}", f"Opset: {info['opset']}", "Inputs:"]
    lines.extend(f"  {item['name']} {item['dtype']} {item['shape']}" for item in info["inputs"])
    lines.append("Outputs:")
    lines.extend(f"  {item['name']} {item['dtype']} {item['shape']}" for item in info["outputs"])
    lines.append(f"Nodes: {len(info['nodes'])}")
    lines.extend(f"  [{node['index']}] {node['op_type']} {node['name']} -> {node['outputs']}"
                 for node in info["nodes"])
    return "\n".join(lines) + "\n"


def render_markdown(info):
    lines = ["# ONNX 模型信息", "", f"- IR version: `{info['ir_version']}`",
             f"- Opset: `{info['opset']}`", "", "## 输入", "",
             "| Tensor | 类型 | 形状 |", "|---|---|---|"]
    lines.extend(f"| `{item['name']}` | `{item['dtype']}` | `{item['shape']}` |"
                 for item in info["inputs"])
    lines.extend(["", "## 输出", "", "| Tensor | 类型 | 形状 |", "|---|---|---|"])
    lines.extend(f"| `{item['name']}` | `{item['dtype']}` | `{item['shape']}` |"
                 for item in info["outputs"])
    lines.extend(["", "## 节点", "", "| # | 类型 | 名称 | 输出 Tensor |", "|---:|---|---|---|"])
    lines.extend(f"| {node['index']} | `{node['op_type']}` | `{node['name']}` | "
                 f"`{', '.join(node['outputs'])}` |" for node in info["nodes"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Inspect an ONNX graph")
    parser.add_argument("--onnx_model", required=True, help="Input ONNX model")
    parser.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    parser.add_argument("--output", help="Write the report to this file")
    parser.add_argument("--check", action="store_true", help="Run onnx.checker.check_model")
    args = parser.parse_args()

    model_path = Path(args.onnx_model)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = onnx.load(str(model_path))
    if args.check:
        onnx.checker.check_model(model)
        print("ONNX checker: PASS")
    info = describe(model)
    if args.format == "json":
        report = json.dumps(info, ensure_ascii=False, indent=2) + "\n"
    elif args.format == "markdown":
        report = render_markdown(info)
    else:
        report = render_text(info)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report, encoding="utf-8")
        print(f"Report: {destination}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
