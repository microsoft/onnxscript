# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path
from typing import BinaryIO, Protocol

from onnxscript.codeanalysis import onnx_to_onnxscript


class ConvertCommandArgs(Protocol):
    onnx_model_reader: BinaryIO
    onnxscript_writer: BinaryIO


def convert_command(args: ConvertCommandArgs):
    args.onnxscript_writer.write(
        onnx_to_onnxscript.Driver(args.onnx_model_reader).to_python_code(
            None
            if args.onnxscript_writer.name == "<stdout>"
            else Path(args.onnxscript_writer.name)
        )
    )


def main():
    parser = argparse.ArgumentParser(prog="onnxscript")
    subparsers = parser.add_subparsers(required=True)

    parser_convert = subparsers.add_parser(
        "convert",
        help="Convert an ONNX model to ONNX Script Python code",
        description="Convert an ONNX model to ONNX Script Python code",
    )
    parser_convert.set_defaults(func=convert_command)
    parser_convert.add_argument(
        "onnx_model_reader",
        metavar="ONNX_MODEL_FILE",
        type=argparse.FileType("rb"),
    )
    parser_convert.add_argument(
        "--output",
        dest="onnxscript_writer",
        metavar="OUTPUT_FILE",
        type=argparse.FileType("wb"),
        help="file path for writing generated ONNX Script code",
        default="-",
        required=False,
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
