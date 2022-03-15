# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import os
from .converter import convert
import onnx

# command-line utility to invoke converter on a python file


def to_single_model_proto(input_py_file: str, output_onnx_file: Optional[str] = None):
    if (not output_onnx_file):
        prefix, ext = os.path.splitext(input_py_file)
        output_onnx_file = prefix + ".onnx"

    fnlist = convert(input_py_file)

    # for now, treat the last function in input-file as the "main graph"
    # TODO: let user specify main function via an option
    if (not fnlist):
        print("No functions in input.")
        return
    main = fnlist.pop(-1)

    # For now, hard-code opset imports.
    # TODO: extract opset imports from translated IR
    graph = main.to_graph_proto()
    model = onnx.helper.make_model(
        graph,
        functions=[f.to_function_proto() for f in fnlist],
        producer_name='p2o',
        opset_imports=[onnx.helper.make_opsetid("", 15)])

    # TODO: add options for user to specify whether to check generated model
    # model = onnx.shape_inference.infer_shapes(model)
    # onnx.checker.check_model(model)

    onnx.save(model, output_onnx_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', help='Translate input to a single ModelProto', action='store_true')
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    for input_file in args.rest:
        if args.model:
            to_single_model_proto(input_file)
        else:
            convert(input_file)
