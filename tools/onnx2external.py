# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import os

import onnx
import onnx.external_data_helper


def convert2external(input_file_name: str) -> None:
    dir_name = os.path.dirname(input_file_name)
    base_name, suffix = os.path.splitext(os.path.basename(input_file_name))
    model = onnx.load(input_file_name)
    os.makedirs(os.path.join(dir_name, base_name), exist_ok=True)
    onnx.external_data_helper.convert_model_to_external_data(
        model, location="external_data.onnx", size_threshold=128
    )
    onnx.save(model, os.path.join(dir_name, base_name, "model.onnx"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ONNX model file to external data format"
    )
    parser.add_argument("input", help="ONNX model file to convert")
    args = parser.parse_args()

    convert2external(args.input)
