"""Utilities for working with Python ASTs."""
from __future__ import annotations

import ast
import inspect
import textwrap
import types


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
