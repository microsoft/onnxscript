# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pylint disable: protected-access
from __future__ import annotations

import ast
import inspect
import sys
from typing import Any, Callable, Optional, Sequence, TypeVar

import onnx_ir as ir
from typing_extensions import ParamSpec

import onnxscript
from onnxscript._internal import ast_utils, converter, irbuilder, values

_R = TypeVar("_R")
_P = ParamSpec("_P")


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
) -> Callable[[Callable[_P, _R]], onnxscript.OnnxFunction[_P, _R]]:
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

    def transform(f: Callable[_P, _R]) -> onnxscript.OnnxFunction[_P, _R]:
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

    Eager execution supports graph functions defined directly inside the active
    scripted function, not graphs nested inside graph bodies. Call the decorated
    scripted function so evaluator dispatch establishes its function context.

    Returns:
        A decorator that returns an OnnxClosure containing the compiled graph,
        the Python function, and its enclosing invocation's frame. Translation
        was already performed when the outer-level function was converted to an
        OnnxFunction.

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
    from onnxscript._internal import evaluator  # pylint: disable=import-outside-toplevel

    # The frame is needed only for closure values, not for discovering the OnnxFunction.
    function_frame = sys._getframe(1)  # pylint: disable=protected-access
    onnx_function = evaluator._current_function.get()
    if onnx_function is None:
        raise RuntimeError(
            "@graph() requires an active ONNXScript function. "
            "Call the @script() function rather than its raw Python function."
        )
    if function_frame.f_code is not onnx_function.function.__code__:
        raise RuntimeError(
            "@graph() supports only graph functions defined directly inside the active "
            "@script() function; graphs nested inside graph bodies are not supported."
        )
    nested_functions = onnx_function.function_ir.nested_functions

    def transform(f: Callable) -> values.OnnxClosure:
        if f.__name__ not in nested_functions:
            raise RuntimeError(
                f"@graph() cannot find compiled graph {f.__name__!r} in "
                f"ONNXScript function {onnx_function.name!r}."
            )
        return values.OnnxClosure(nested_functions[f.__name__], function_frame, f)

    return transform


def is_converted_fun(f: Any) -> bool:
    """Return True if f is a function converted by onnxscript decorator."""
    return isinstance(f, onnxscript.OnnxFunction)


def export_onnx_lib(functions: Sequence[values.OnnxFunction], filename: str) -> None:
    # Since we don't yet have LibProto defined, we use a ModelProto as a temporary
    # container for the list of functions exported as a library, with an empty graph
    # and dummy opset_imports.

    # TODO(justinchuby): This function is not well supported. We should consider removing it
    model = ir.Model(
        ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": 15},
        ),
        functions=[ir.serde.deserialize_function(f.to_function_proto()) for f in functions],
        ir_version=10,
        producer_name="p2o",
    )
    ir.save(model, filename)
