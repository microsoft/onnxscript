# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import onnx
import onnx.external_data_helper
import os

# What if it is already in external format? How to check?
# Output: folder?

def convert2external() -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX model file to external data format")
    parser.add_argument("input", help="ONNX model file to convert")

    args = parser.parse_args()
    input_file = args.input
    dir_name = os.path.dirname(input_file)
    base_name, suffix = os.path.splitext(os.path.basename(input_file))
    model = onnx.load(input_file)
    os.makedirs(os.path.join(dir_name, base_name), exist_ok=True)
    onnx.external_data_helper.convert_model_to_external_data(model, location="external_data.onnx")
    onnx.save(model, os.path.join(dir_name, base_name, "model.onnx"))

if __name__ == "__main__":
    convert2external()