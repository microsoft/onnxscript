#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility for optimizing ONNX models.

Usage:
    python optimize.py model.onnx optimized_model.onnx
"""

import argparse
import os

import onnx_ir

import onnxscript


def _has_external_data(model: onnx_ir.Model) -> bool:
    """Return True if any initializer in the model is stored as external data."""
    return any(
        isinstance(value.const_value, onnx_ir.ExternalTensor)
        for graph in model.graphs()
        for value in graph.initializers.values()
    )


def main(args) -> None:
    path = args.path
    output_path = args.output_path

    model = onnx_ir.load(path)
    # optimize() inlines local functions before it runs the optimization passes,
    # so no separate inliner call is needed.
    model = onnxscript.optimizer.optimize(model)
    # Only externalize if the input already used external data, and name the
    # external data file after the output model so it doesn't collide with input.
    external_data = (
        f"{os.path.basename(output_path)}.data" if _has_external_data(model) else None
    )
    onnx_ir.save(model, output_path, external_data=external_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize an ONNX model.")
    parser.add_argument("path", type=str, help="Path to the ONNX model.")
    parser.add_argument("output_path", type=str, help="Path to save the optimized model.")
    main(parser.parse_args())
