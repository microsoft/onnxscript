# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import onnx
import onnxscript
import os
from typing import Optional


def convert2script(input_file_name: str, output_file_name: Optional[str], verbose: bool) -> None:
    model = onnx.load(input_file_name, load_external_data=False)
    python_code = onnxscript.proto2python(model, use_operators = not verbose, inline_constants = not verbose)

    # If output file name is not provided, use the input file name with .py extension
    if output_file_name is None:
        base_name = os.path.splitext(input_file_name)[0]  # Remove extension
        output_file_name = base_name + ".py"

    with open(output_file_name, "w") as f:
        f.write(python_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX model file to onnxscript file")
    parser.add_argument("input", help="ONNX model file to convert")
    parser.add_argument("-o", "--output", help="Output file name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode", default=False)

    args = parser.parse_args()
    convert2script(args.input, args.output, args.verbose)