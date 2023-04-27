"""Module for trace_only functions."""
from __future__ import annotations

import ast
import inspect
import textwrap
import types
from typing import Optional

import onnx

import onnxscript
from onnxscript import converter as ons_converter
from onnxscript._internal import version_utils

_ONNX_OP_SCHEMA_WRITABLE = not version_utils.onnx_older_than("1.14")


def _get_src_and_ast(f: types.FunctionType) -> tuple[str, ast.FunctionDef]:
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


class TraceOnlyFunction:
    """TraceOnlyFunction.

    Attributes:
        name: Name of the op. E.g. "aten::add".
        func: Function.
    """

    def __init__(self, opset: onnxscript.values.Opset, func: types.FunctionType):
        self._opset = opset
        self._func = func
        self._opschema: Optional[onnx.defs.OpSchema] = None
        # Set the signature of the class to function's
        self.__signature__ = inspect.signature(func)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __repr__(self):
        return f"TraceOnlyFunction({self!r})"

    @property
    def name(self) -> str:
        """Return the name of the op."""
        return self._func.__name__

    @property
    def source(self) -> str:
        """Return the source of the op."""
        return inspect.getsource(self._func)

    @property
    def opset(self) -> onnxscript.values.Opset:
        """Return the opset."""
        return self._opset

    @property
    def opschema(self) -> Optional[onnx.defs.OpSchema]:
        """Return the opschema."""

        if self._opschema is not None:
            return self._opschema

        if not _ONNX_OP_SCHEMA_WRITABLE:
            return None

        src, func_ast = _get_src_and_ast(self._func)
        module = inspect.getmodule(self._func)
        closure = inspect.getclosurevars(self._func)
        global_names = module.__dict__.copy()
        global_names.update(closure.nonlocals)
        converter = ons_converter.Converter(
            opset=self._opset,
            global_names=global_names,
            source=src,
        )

        function_ir = converter.translate_function_signature(func_ast)

        # FIXME(justinchuby): outputs are empty. Need to fix.
        self._opschema = onnxscript.values.op_schema_from_function_ir(function_ir, self._opset)

        return self._opschema
