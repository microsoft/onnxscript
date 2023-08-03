"""Utilities for working with Python ASTs."""
from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import types

PY_VERSION_GE_39 = sys.version_info >= (3, 9)


def get_src_and_ast(f: types.FunctionType) -> tuple[str, ast.FunctionDef]:
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


def normalize_subscript_expr(expr: ast.Subscript):
    # Normalizes the representation of a subscripted expression, handling python version
    # differences as well as variations between A[x] (single-index) and A[x, y] (multiple indices)
    # Returns a list of expressions, denoting the indices, after stripping the extraneous "Index"
    # wrapper present in python versions before 3.9
    index_expr = expr.slice
    if PY_VERSION_GE_39:
        if isinstance(index_expr, ast.Tuple):
            return index_expr.elts  # multiple indices
        else:
            return [index_expr]  # single index
    else:
        if isinstance(index_expr, ast.ExtSlice):
            indices = index_expr.dims  # type: ignore[attr-defined]
        else:
            indices = [index_expr]  # single slice-index
        return [x.value if isinstance(x, ast.Index) else x for x in indices]  # type: ignore[attr-defined]


def is_print_call(stmt: ast.stmt) -> bool:
    """Return True if the statement is a call to the print function."""
    if isinstance(stmt, ast.Expr):
        if isinstance(stmt.value, ast.Call):
            if isinstance(stmt.value.func, ast.Name):
                return stmt.value.func.id == "print"
    return False


def is_doc_string(stmt: ast.stmt) -> bool:
    """Return True if the statement is a docstring."""
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return isinstance(stmt.value.value, str)
    return False
