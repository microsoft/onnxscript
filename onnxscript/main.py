# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import ast
import inspect
import textwrap

import onnx.helper

from onnxscript import converter, values
import onnxscript

def get_src_and_ast(f):
    try:
        src = inspect.getsource(f)
    except OSError as e:
        raise RuntimeError(
            f"Decorator script does not work on dynamically "
            f"compiled function {f.__name__}."
        ) from e
    src = textwrap.dedent(src)
    top_level_ast = ast.parse(src)
    assert isinstance(top_level_ast, ast.Module)
    assert len(top_level_ast.body) == 1
    f_ast = top_level_ast.body[0]
    assert isinstance(f_ast, ast.FunctionDef)
    return src, f_ast


def get_ast(f):
    _, ast = get_src_and_ast(f)  # pylint: disable=redefined-outer-name
    return ast


def script_check(f: ast.FunctionDef, opset, global_names, source, default_opset=None):
    """
    Check that a function falls into the ONNXScript subset of Python.
    """
    # See if conversion succeeds.
    # TODO: cleanup Converter interface/API, separating checker from
    # converter
    convert = converter.Converter(
        opset=opset,
        global_names=global_names,
        source=source,
        default_opset=default_opset,
    )
    return convert.top_level_stmt(f)


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

        from onnxscript.onnx_opset import opset16

        @script(opset16)
        def log2(x):
            one = op.Constant(value=make_tensor('one', TensorProto.FLOAT, [1], [1]))
            return op.Div(op.Log(x), op.CastLike(op.Log(cst), x))
    """
    if opset is None:
        opset = values.Opset("this", 1)
    if not isinstance(opset, values.Opset):
        raise TypeError(
            "Script parameter must be an opset. Did you use @script instead of @script()?"
        )

    def transform(f):
        if inspect.isfunction(f):
            src, ast = get_src_and_ast(f)  # pylint: disable=redefined-outer-name
            module = inspect.getmodule(f)
            result = script_check(
                ast, opset, module.__dict__.copy(), src, default_opset=default_opset
            )
            # TODO: add transformations.
            return onnxscript.OnnxFunction(opset, f, result, src, kwargs)
        raise TypeError("The ONNXScript decorator should be applied to functions only.")

    return transform


def is_converted_fun(f):
    """
    Return True if f is a function converted by onnx-script decorator.
    """
    return isinstance(f, onnxscript.OnnxFunction)


def export_onnx_lib(functions, filename: str) -> None:
    # Since we don't yet have LibProto defined, we use a ModelProto as a temporary
    # container for the list of functions exported as a library, with an empty graph
    # and dummy opset_imports.
    model = onnx.helper.make_model(
        onnx.GraphProto(),
        functions=[f.to_function_proto() for f in functions],
        producer_name="p2o",
        opset_imports=[onnx.helper.make_opsetid("", 15)],
    )

    onnx.save(model, filename)
