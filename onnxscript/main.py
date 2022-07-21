# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import ast
import inspect
import textwrap
import onnx.helper
from .converter import Converter
from . import values
from .values import OnnxFunction


def script_check(f: ast.FunctionDef, opset, global_names, source,
                 default_opset=None):
    '''
    Check that a function falls into the ONNXScript subset of Python.
    '''
    # See if conversion succeeds.
    # TODO: cleanup Converter interface/API, separating checker from
    # converter
    converter = Converter(opset=opset, global_names=global_names, source=source,
                          default_opset=default_opset)
    return converter.top_level_stmt(f)


def script(opset=None, default_opset=None, **kwargs):
    """
    Main decorator. Declares a function as an onnx function.

    :param opset: opset the function belongs to (see :ref:`l-api-opsets`)
    :return: an instance of :class:`onnxscript.values.OnnxFunction`

    Example:

    ::

        @script()
        def log2(x):
            one = op.Constant(value=make_tensor('one', TensorProto.FLOAT, [1], [1]))
            return op.Div(op.Log(x), op.CastLike(op.Log(cst), x))

    Or:

    ::

        from onnxscript.onnx import opset16

        @script(opset16)
        def log2(x):
            one = op.Constant(value=make_tensor('one', TensorProto.FLOAT, [1], [1]))
            return op.Div(op.Log(x), op.CastLike(op.Log(cst), x))
    """
    if (opset is None):
        opset = values.Opset('this', 1)
    if not isinstance(opset, values.Opset):
        raise TypeError(
            "Script parameter must be an opset. Did you use @script instead of @script()?")

    def transform(f):
        if inspect.isfunction(f):
            src = inspect.getsource(f)
            src = textwrap.dedent(src)
            module = inspect.getmodule(f)
            top_level_ast = ast.parse(src)
            assert type(top_level_ast) == ast.Module
            assert len(top_level_ast.body) == 1
            f_ast = top_level_ast.body[0]
            assert type(f_ast) == ast.FunctionDef
            result = script_check(f_ast, opset, module.__dict__.copy(), src,
                                  default_opset=default_opset)
            # TODO: add transformations.
            return OnnxFunction(opset, f, result, src, kwargs)
        else:
            raise TypeError(
                "The ONNXScript decorator should be applied to functions only.")

    return transform

# For now, a decorator for models is the same as the decorator for functions.
# Will add model-specific checkers later on.
model = script

def is_converted_fun(f):
    '''
    Return True if f is a function converted by onnx-script decorator.
    '''
    return isinstance(f, OnnxFunction)


def export_onnx_lib(functions, filename: str) -> None:
    # Since we don't yet have LibProto defined, we use a ModelProto as a temporary
    # container for the list of functions exported as a library, with an empty graph
    # and dummy opset_imports.
    model = onnx.helper.make_model(
        onnx.GraphProto(),
        functions=[f.to_function_proto() for f in functions],
        producer_name='p2o',
        opset_imports=[onnx.helper.make_opsetid("", 15)])

    onnx.save(model, filename)
