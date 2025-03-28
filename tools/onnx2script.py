# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
onnx2script.py

This module provides a script to convert ONNX model files to Python scripts using the onnxscript library.

Usage:
    python onnx2script.py <input_file> [-o output_file] [-v]

Arguments:
    input_file: The ONNX model file to convert.
    -o, --output: The output file name. If not provided, the output will be named after the input file with a .py extension.
    -v, --verbose: Enables verbose mode. This suppresses the use of overloaded operators and inline constants.

Example:
    python onnx2script.py model.onnx -o model.py -v
"""

import argparse
import os
from typing import Optional

import onnx

import onnxscript


def convert2script(
    input_file_name: str, output_file_name: Optional[str], verbose: bool, initializers: bool
) -> None:
    model = onnx.load(input_file_name, load_external_data=False)
    python_code = onnxscript.proto2python(
        model,
        use_operators=not verbose,
        inline_const=not verbose,
        skip_initializers=not initializers,
    )

    # If output file name is not provided, use the input file name with .py extension
    if output_file_name is None:
        base_name = os.path.splitext(input_file_name)[0]  # Remove extension
        output_file_name = base_name + ".py"

    with open(output_file_name, "w", encoding="utf-8") as f:
        f.write(python_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX model file to onnxscript file")
    parser.add_argument("input", help="ONNX model file to convert")
    parser.add_argument("-o", "--output", help="Output file name")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose mode, suppresses use of overloaded operators and inline constants",
        default=False,
    )
    parser.add_argument(
        "-i",
        "--initializers",
        action="store_true",
        help="Include initializers in the generated script",
        default=False,
    )

    args = parser.parse_args()
    convert2script(args.input, args.output, args.verbose, args.initializers)
