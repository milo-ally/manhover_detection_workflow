#!/usr/bin/env python3
"""Extract an ONNX subgraph by tensor names."""

import argparse
from pathlib import Path

import onnx
from onnx import utils


def tensor_names(model):
    names = set()
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if value.name:
            names.add(value.name)
    for node in model.graph.node:
        names.update(name for name in node.input if name)
        names.update(name for name in node.output if name)
    names.update(item.name for item in model.graph.initializer)
    return names


def print_candidates(names, missing):
    print("Available tensor names:")
    terms = [item.lower() for item in missing]
    preferred = [name for name in sorted(names) if any(term in name.lower() for term in terms)]
    for name in (preferred or sorted(names)):
        print(f"  {name}")


def main():
    parser = argparse.ArgumentParser(description="Cut an ONNX model at named tensors")
    parser.add_argument("--onnx_model", required=True, help="Source ONNX model")
    parser.add_argument("--output", "-o", required=True, help="Destination ONNX model")
    parser.add_argument("--inputs", "-i", nargs="+", required=True, help="Input tensor names")
    parser.add_argument("--outputs", "-O", nargs="+", required=True, help="Output tensor names")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-on-error", action="store_true")
    args = parser.parse_args()

    source, destination = Path(args.onnx_model), Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"{destination} exists; add --overwrite")
    model = onnx.load(str(source), load_external_data=False)
    names = tensor_names(model)
    requested = args.inputs + args.outputs
    missing = [name for name in requested if name not in names]
    if missing:
        print(f"Requested tensor name(s) were not found: {missing}")
        if args.list_on_error:
            print_candidates(names, missing)
        raise SystemExit(2)

    destination.parent.mkdir(parents=True, exist_ok=True)
    utils.extract_model(str(source), str(destination), args.inputs, args.outputs, check_model=True)
    result = onnx.load(str(destination))
    onnx.checker.check_model(result)
    print(f"Saved: {destination}")
    print(f"Inputs: {args.inputs}")
    print(f"Outputs: {args.outputs}")


if __name__ == "__main__":
    main()
