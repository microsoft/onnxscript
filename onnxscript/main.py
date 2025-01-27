# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pylint disable: protected-access
from __future__ import annotations

import ast
import inspect
import sys
from typing import Any, Callable, Optional, Sequence

import onnx.helper

import onnxscript
from onnxscript import converter, irbuilder, values
from onnxscript._internal import ast_utils


def script_check(
    f: ast.FunctionDef,
    opset: values.Opset,
    global_names: dict[str, Any],
    source: str,
    default_opset: Optional[values.Opset] = None,
) -> irbuilder.IRFunction:
    """Check that a function falls into the ONNXScript subset of Python."""
    # See if conversion succeeds.
    # TODO: cleanup Converter interface/API, separating checker from
    # converter
    convert = converter.Converter(
        opset=opset,
        global_names=global_names,
        source=source,
        default_opset=default_opset,
    )
    return convert.translate_function_def(f)


def script(
    opset: Optional[values.Opset] = None,
    default_opset: Optional[values.Opset] = None,
    **kwargs: Any,
) -> Callable[[Callable], onnxscript.OnnxFunction]:
    """Main decorator. Declares a function as an onnx function.

    Args:
        opset: Opset the function belongs to (see :ref:`l-api-opsets`).
        default_opset: Opset to use for operators not in the function's opset.
        kwargs: Additional keyword arguments.

    Returns:
        an instance of :class:`onnxscript.values.OnnxFunction`

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
    opset = opset or values.Opset("this", 1)
    if not isinstance(opset, values.Opset):
        raise TypeError(
            "Script parameter must be an opset. Did you use @script instead of @script()?"
        )

    def transform(f: Callable) -> onnxscript.OnnxFunction:
        if not inspect.isfunction(f):
            raise TypeError("The ONNXScript decorator should be applied to functions only.")

        src, f_ast = ast_utils.get_src_and_ast(f)
        # The script should be compiled using the globals/locals at the definition site.
        # This allows the script to reference names defined outside the script,
        # which is used for a few different purposes.
        # The following is an approximate solution that works for normal use.
        module = inspect.getmodule(f)
        closure = inspect.getclosurevars(f)
        env = module.__dict__.copy()
        env.update(closure.nonlocals)
        result = script_check(f_ast, opset, env, src, default_opset=default_opset)
        # TODO: add transformations.
        return onnxscript.OnnxFunction(opset, f, result, src, kwargs)

    return transform


def graph() -> Callable[[Callable], values.OnnxClosure]:
    """A parametric decorator used to annotate nested-functions that are used
    as graph-attributes.

    Returns:
        A decorator that returns its input function, but attaches a graph_proto
        attribute representing the input function. The translation is not
        done at this time, but previously when the outer-level function
        was translated to an OnnxFunction. The decorator just looks up
        and retrieves the GraphProto representation previously generated.

    Example:
    ::

        @script()
        def cumulative_sum(X: INT64['N']):

            # Translation of cumulative_sum by @script will also translate Sum
            # into a GraphProto, which will be stored in the OnnxFunction generated
            # for cumulative_sum. At run-time (in eager-mode), the @graph decorator
            # retrieves the pre-computed GraphProto and attaches it to the Sum function.
            @graph()
            def Sum(sum_in, next):
                sum_out = sum_in + next
                scan_out = op.Identity(sum_out)
                return sum_out, scan_out
            zero = op.Constant(value_int=0)
            # The call to higher-order operator Scan below uses the above function
            # Sum as a graph-attribute.
            all_sum, result = op.Scan (zero, X, body=Sum, num_scan_inputs=1)
            return result

    """
    # This is a bit fragile. We want to get the ONNXFunction object representing
    # the outer-scope ONNXScript function from the execution stack. The caller of
    # @graph is the original script function (cumulative_sum in the above example),
    # and the caller of that function is the wrapper function/method in the
    # corresponding OnnxFunction object.
    # Currently, there is no support for eager-mode execution of nested functions,
    # so we don't need to handle doubly nested functions (e.g., a function defined
    # inside Sum in the above example).

    function_frame = sys._getframe(1)  # pylint: disable=protected-access
    wrapper_frame = sys._getframe(3)  # pylint: disable=protected-access
    onnx_function = wrapper_frame.f_locals["self"]
    nested_functions = onnx_function.function_ir.nested_functions

    def transform(f: Callable) -> values.OnnxClosure:
        return values.OnnxClosure(nested_functions[f.__name__], function_frame, f)

    return transform


def is_converted_fun(f: Any) -> bool:
    """Return True if f is a function converted by onnxscript decorator."""
    return isinstance(f, onnxscript.OnnxFunction)


def export_onnx_lib(functions: Sequence[values.OnnxFunction], filename: str) -> None:
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
