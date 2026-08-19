#!/usr/bin/env python3
"""Generate a Pulsar2 build_config.json from an ONNX model contract."""

import argparse
import copy
import json
from pathlib import Path

import onnx


def graph_contract(model):
    initializers = {item.name for item in model.graph.initializer}
    inputs = [item.name for item in model.graph.input if item.name not in initializers]
    outputs = [item.name for item in model.graph.output]
    if not inputs or not outputs:
        raise ValueError("ONNX graph must have at least one runtime input and one output")
    return inputs, outputs


def default_config(input_name, outputs, dataset, size, npu_mode):
    return {
        "model_type": "ONNX",
        "npu_mode": npu_mode,
        "quant": {
            "input_configs": [{
                "tensor_name": input_name,
                "calibration_dataset": dataset,
                "calibration_size": size,
                "calibration_mean": [0, 0, 0],
                "calibration_std": [255.0, 255.0, 255.0],
            }],
            "layer_configs": [{
                "start_tensor_names": ["DEFAULT"],
                "end_tensor_names": ["DEFAULT"],
                "data_type": "U16",
            }],
            "calibration_method": "MinMax",
            "precision_analysis": True,
            "precision_analysis_method": "EndToEnd",
        },
        "input_processors": [{
            "tensor_name": input_name,
            "tensor_format": "RGB",
            "src_format": "RGB",
            "src_dtype": "U8",
            "src_layout": "NHWC",
        }],
        "output_processors": [{"tensor_name": name} for name in outputs],
        "compiler": {"check": 0},
    }


def update_template(config, input_name, outputs, dataset, size, npu_mode):
    config = copy.deepcopy(config)
    config["model_type"] = "ONNX"
    config["npu_mode"] = npu_mode
    quant = config.setdefault("quant", {})
    input_configs = quant.setdefault("input_configs", [{}])
    if not input_configs:
        input_configs.append({})
    input_configs[0]["tensor_name"] = input_name
    input_configs[0]["calibration_dataset"] = dataset
    input_configs[0]["calibration_size"] = size
    input_configs[0].setdefault("calibration_mean", [0, 0, 0])
    input_configs[0].setdefault("calibration_std", [255.0, 255.0, 255.0])
    processors = config.setdefault("input_processors", [{}])
    if not processors:
        processors.append({})
    processors[0].update({"tensor_name": input_name, "tensor_format": "RGB",
                          "src_format": "RGB", "src_dtype": "U8", "src_layout": "NHWC"})
    config["output_processors"] = [{"tensor_name": name} for name in outputs]
    config.setdefault("compiler", {"check": 0})
    return config


def main():
    parser = argparse.ArgumentParser(description="Generate Pulsar2 build configuration")
    parser.add_argument("--onnx_model", required=True, help="Input ONNX model")
    parser.add_argument("--output_config", default="config/build_config.json")
    parser.add_argument("--calibration_dataset", default="./dataset/manhole_cover.tar")
    parser.add_argument("--calibration_size", type=int, default=32)
    parser.add_argument("--npu_mode", default="NPU1")
    parser.add_argument("--output_tensors", nargs="+", help="Override ONNX graph outputs")
    parser.add_argument("--template", help="Optional existing JSON template")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_path, destination = Path(args.onnx_model), Path(args.output_config)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"{destination} exists; add --overwrite")
    model = onnx.load(str(model_path), load_external_data=False)
    onnx.checker.check_model(model)
    inputs, graph_outputs = graph_contract(model)
    outputs = args.output_tensors or graph_outputs
    all_tensors = {name for node in model.graph.node for name in (*node.input, *node.output) if name}
    all_tensors.update(item.name for item in model.graph.input)
    all_tensors.update(item.name for item in model.graph.output)
    missing = [name for name in outputs if name not in all_tensors]
    if missing:
        raise ValueError(f"output tensor(s) not found in ONNX: {missing}")

    template_path = Path(args.template) if args.template else None
    if template_path and template_path.is_file():
        template = json.loads(template_path.read_text(encoding="utf-8"))
        config = update_template(template, inputs[0], outputs, args.calibration_dataset,
                                 args.calibration_size, args.npu_mode)
    else:
        config = default_config(inputs[0], outputs, args.calibration_dataset,
                                args.calibration_size, args.npu_mode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {destination}")
    print(f"Input tensor: {inputs[0]}")
    print(f"Output tensors: {outputs}")


if __name__ == "__main__":
    main()
