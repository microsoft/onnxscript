# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from typing import Optional

import click
import onnx
from onnx import helper

from onnxscript import converter
from onnxscript.backend import onnx_export


@click.group()
def cli():
    pass


def convert_file(script):
    convert = converter.Converter()
    return convert.convert_file(script)


def to_single_model_proto(model, input_py_file: str, output_onnx_file: Optional[str] = None):
    if not output_onnx_file:
        prefix, _ = os.path.splitext(input_py_file)
        output_onnx_file = f"{prefix}.onnx"

    fnlist = convert_file(input_py_file)

    if not fnlist:
        print("No functions in input.")
        return

    if model:
        # treat last function as main graph for model
        main = fnlist.pop(-1)
        graph = main.to_graph_proto()
    else:
        # For now, we use a ModelProto with an empty graph to represent a library
        graph = onnx.GraphProto()

    # For now, hard-code opset imports.
    # TODO: extract opset imports from translated IR

    model = onnx.helper.make_model(
        graph,
        functions=[f.to_function_proto() for f in fnlist],
        producer_name="p2o",
        opset_imports=[onnx.helper.make_opsetid("", 15)],
    )

    # TODO: add options for user to specify whether to check generated model
    # model = onnx.shape_inference.infer_shapes(model)
    # onnx.checker.check_model(model)

    onnx.save(model, output_onnx_file)


def to_text(input_py_file: str):
    def print_ir_function(f):
        print(str(f))
        for s in f.stmts:
            for attr in s.attrs:
                if attr.attr_proto.HasField("g"):
                    print(helper.printable_graph(attr.attr_proto.g))

    fnlist = convert_file(input_py_file)
    for f in fnlist:
        print_ir_function(f)


@cli.command()
@click.option(
    "--fmt",
    type=click.Choice(["text", "model", "lib"], case_sensitive=False),
    help="Translate input to a single ModelProto ('model'), "
    "into a LibProto ('lib'), "
    "or into text 'text').",
)
@click.option(
    "name",
    "--name",
    envvar="PATHS",
    multiple=True,
    type=click.Path(),
    help="File or files to convert.",
)
def translate(fmt="text", names=None):
    """Translate a file or many files into a ModelProto, a LibProto or text."""
    if fmt == "text":
        for name in names:
            to_text(name)
    else:
        for name in names:
            to_single_model_proto(fmt == "model", name)


@cli.command()
@click.option(
    "name",
    "--name",
    envvar="PATHS",
    multiple=False,
    type=click.Path(),
    help="filename to convert",
)
@click.option(
    "--op",
    is_flag=True,
    default=False,
    help="converts a numerical operator into op.Add (False) or keep it (True)",
)
@click.option("--rename", is_flag=True, default=False, help="to use shorter variable name")
def onnx2script(name, op=False, rename=False):
    """Exports an onnx graph to a script in following onnx-script syntax.
    The result is printed on the standard output.
    """
    code = onnx_export.export2python(name, use_operators=op, rename=rename)
    print(code)


if __name__ == "__main__":
    cli()
