#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility for optimizing ONNX models.

Usage:
    python optimize.py model.onnx optimized_model.onnx
"""

import argparse
import os

import onnx
import onnx.inliner

import onnxscript


def main(args) -> None:
    path = args.path
    output_path = args.output_path

    pwd = os.getcwd()
    model_dir = os.path.dirname(path)

    # Hack: Change the working directory to the model directory so the optimizer
    # can load external data files with relative paths.
    # TODO: Remove this hack by fixing the optimizer to handle external data files properly.
    os.chdir(model_dir)
    model = onnx.load(path, load_external_data=False)
    model = onnxscript.optimizer.optimize(model)
    model = onnx.inliner.inline_local_functions(model)
    # Optimize again in case inlining created new opportunities.
    model = onnxscript.optimizer.optimize(model)

    os.chdir(pwd)
    onnx.save(model, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize an ONNX model.")
    parser.add_argument("path", type=str, help="Path to the ONNX model.")
    parser.add_argument("output_path", type=str, help="Path to save the optimized model.")
    args = parser.parse_args()
    main(args)
