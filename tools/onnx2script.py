# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import onnx
import onnxscript
import os


def convert2script() -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX model file to onnxscript file")
    parser.add_argument("input", help="ONNX model file to convert")
    parser.add_argument("-o", "--output", help="Output file name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode", default=False)

    args = parser.parse_args()

    # If output file name is not provided, use the input file name with .py extension
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]  # Remove extension
        args.output = base_name + ".py"

    model = onnx.load(args.input, load_external_data=False)
    python_code = onnxscript.proto2python(model, use_operators = not args.verbose, inline_constants = not args.verbose)

    with open(args.output, "w") as f:
        f.write(python_code)


if __name__ == "__main__":
    convert2script()