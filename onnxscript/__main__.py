# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import os
from .converter import Converter
import onnx
import onnx.helper as helper

# command-line utility to invoke converter on a python file


def convert_file(script):
    converter = Converter()
    return converter.convert_file(script)


def to_single_model_proto(args, input_py_file: str, output_onnx_file: Optional[str] = None):
    if (not output_onnx_file):
        prefix, ext = os.path.splitext(input_py_file)
        output_onnx_file = prefix + ".onnx"

    fnlist = convert_file(input_py_file)

    if (not fnlist):
        print("No functions in input.")
        return

    if args.model:
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
        producer_name='p2o',
        opset_imports=[onnx.helper.make_opsetid("", 15)])

    # TODO: add options for user to specify whether to check generated model
    # model = onnx.shape_inference.infer_shapes(model)
    # onnx.checker.check_model(model)

    onnx.save(model, output_onnx_file)


def to_text(input_py_file: str):
    def print_ir_function (f):
        print(str(f))
        for s in f.stmts:
            for attr in s.attrs:
                if attr.attr_proto.HasField("g"):
                    print(helper.printable_graph(attr.attr_proto.g))

    fnlist = convert_file(input_py_file)
    for f in fnlist:
        print_ir_function(f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-m', '--model', help='Translate input to a single ModelProto', action='store_true')
    group.add_argument('-l', '--lib', help='Translate input to a LibProto',
                       action='store_true')
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    for input_file in args.rest:
        if args.model or args.lib:
            to_single_model_proto(args, input_file)
        else:
            to_text(input_file)
