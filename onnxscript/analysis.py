# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import ast
from typing import Any, Optional, Sequence, Set

from onnxscript import sourceinfo
from onnxscript._internal import ast_utils


def get_loop_var(for_stmt: ast.For, formatter: sourceinfo.Formatter) -> str:
    if not isinstance(for_stmt.target, ast.Name):
        raise ValueError(formatter(for_stmt, "For loop target must be a single variable."))
    return for_stmt.target.id


def used_vars(expr: Optional[ast.expr]) -> Set[str]:
    """Return set of all variables used, including function names, in an expression."""
    if expr is None:
        return set()
    if isinstance(expr, ast.Name):
        return {expr.id}
    result = set()
    if isinstance(expr, ast.Call):
        # The callee-expression is not visited
        children = expr.args
        for keyword in expr.keywords:
            if isinstance(keyword.value, ast.Name):
                result.add(keyword.value.id)
    else:
        children = ast.iter_child_nodes(expr)  # type: ignore[assignment]
    for c in children:
        result = result | used_vars(c)
    return result


def local_defs(lhs: ast.expr) -> Set[str]:
    """Utility function to return set of assigned/defined
    variables in the lhs of an assignment statement.
    """

    def get_id(e):
        assert isinstance(e, ast.Name), "Only simple assignments supported."
        return e.id

    if isinstance(lhs, ast.Tuple):
        return {get_id(x) for x in lhs.elts}
    return {get_id(lhs)}


def defs(stmt: ast.stmt) -> Set[str]:
    """Return the set of all variables that may be defined (assigned to) in an
    execution of input stmt.
    """

    def block_defs(block: Sequence[ast.stmt]) -> Set[str]:
        result: set[Any] = set()
        for s in block:
            result = result | defs(s)
        return result

    if isinstance(stmt, ast.Assign):
        return local_defs(stmt.targets[0])
    if isinstance(stmt, ast.AnnAssign):
        return local_defs(stmt.target)
    if isinstance(stmt, ast.Return):
        return set()
    if isinstance(stmt, ast.If):
        return block_defs(stmt.body) | block_defs(stmt.orelse)
    if isinstance(stmt, list):
        return block_defs(stmt)
    if isinstance(stmt, ast.Break):
        return set()
    if ast_utils.is_print_call(stmt):
        return set()
    raise ValueError(f"Unsupported statement type {type(stmt)!r}.")


def do_liveness_analysis(fun: ast.FunctionDef, formatter: sourceinfo.Formatter):
    """Perform liveness analysis of the given function-ast. The results of the
    analysis are stored directly with each statement-ast `s` as attributes `s.live_in`
    and `s.live_out`.
    """

    def visit(stmt: ast.stmt, live_out: Set[str]) -> Set[str]:
        stmt.live_out = live_out  # type: ignore[attr-defined]
        live = do_visit(stmt, live_out)
        stmt.live_in = live  # type: ignore[attr-defined]
        return live

    def do_visit(stmt: ast.stmt, live_out: Set[str]) -> Set[str]:
        def visitBlock(block: Sequence[ast.stmt], live_out: Set[str]) -> Set[str]:
            for s in reversed(block):
                live_out = visit(s, live_out)
            return live_out

        if isinstance(stmt, ast.Assign):
            return live_out.difference(local_defs(stmt.targets[0])) | used_vars(stmt.value)
        if isinstance(stmt, ast.AnnAssign):
            return live_out.difference(local_defs(stmt.target)) | used_vars(stmt.value)
        if isinstance(stmt, ast.Return):
            return used_vars(stmt.value)
        if isinstance(stmt, ast.If):
            live1 = visitBlock(stmt.body, live_out)
            live2 = visitBlock(stmt.orelse, live_out)
            return live1 | live2 | used_vars(stmt.test)
        if isinstance(stmt, ast.For):
            p_loop_var = get_loop_var(stmt, formatter)
            prev = None
            curr = live_out
            while curr != prev:
                prev = curr
                curr = visitBlock(stmt.body, prev).difference({p_loop_var})
            return curr
        if isinstance(stmt, ast.While):
            cond_vars = used_vars(stmt.test)
            prev = None
            curr = live_out | cond_vars
            while curr != prev:
                prev = curr
                curr = visitBlock(stmt.body, prev) | cond_vars
            return curr
        if isinstance(stmt, ast.Break):
            # The following is sufficient for the current restricted usage, where
            # a (conditional) break is allowed only as the last statement of a loop.
            # Break statements in the middle of the loop, however, will require
            # a generalization.
            return live_out
        if isinstance(stmt, ast.Expr) and hasattr(stmt, "value"):
            # docstring
            if hasattr(stmt.value, "value") and isinstance(stmt.value.value, str):
                # python 3.8+
                return live_out
        if isinstance(stmt, ast.FunctionDef):
            return live_out
        if ast_utils.is_print_call(stmt):
            return live_out
        raise ValueError(formatter(stmt, f"Unsupported statement type {type(stmt)!r}."))

    assert isinstance(fun, ast.FunctionDef)
    live: set[Any] = set()
    for s in reversed(fun.body):
        live = visit(s, live)


def exposed_uses(stmts: Sequence[ast.stmt], formatter: sourceinfo.Formatter):
    """Return the set of variables that are used before being defined by given block.
    In essence, this identifies the "inputs" to a given code-block.
    For example, consider the following code-block:
    ::

       x = x + 10
       y = 20
       z = x + y
       x = 30

    The exposed_uses of this code-block is { x }. The value of z is not used within
    the block. Even though the value of y is used within the block, it is assigned
    a value before it is used. However, in contrast, the incoming value of x is used
    (in the first statement). Hence x is included in the exposed_uses.
    """

    def visitBlock(block: Sequence[ast.stmt], live_out: Set[str]) -> Set[str]:
        for stmt in reversed(block):
            live_out = visit(stmt, live_out)
        return live_out

    def visit(stmt: ast.stmt, live_out: Set[str]) -> Set[str]:
        if isinstance(stmt, ast.Assign):
            return live_out.difference(local_defs(stmt.targets[0])) | used_vars(stmt.value)
        if isinstance(stmt, ast.AnnAssign):
            return live_out.difference(local_defs(stmt.target)) | used_vars(stmt.value)
        if isinstance(stmt, ast.Return):
            return used_vars(stmt.value)
        if isinstance(stmt, ast.If):
            live1 = visitBlock(stmt.body, live_out)
            live2 = visitBlock(stmt.orelse, live_out)
            return (live1 | live2) | used_vars(stmt.test)
        if ast_utils.is_print_call(stmt):
            return live_out
        if isinstance(stmt, ast.For):
            # Analysis assumes loop may execute zero times. Results can be improved
            # for loops that execute at least once.
            loop_var_set = {get_loop_var(stmt, formatter)}
            used_after_loop = live_out.difference(loop_var_set)
            used_inside_loop = visitBlock(stmt.body, set()).difference(loop_var_set)
            used_in_loop_header = used_vars(stmt.iter)
            return used_inside_loop | used_in_loop_header | used_after_loop
        if isinstance(stmt, ast.While):
            # Analysis assumes loop may execute zero times. Results can be improved
            # for loops that execute at least once.
            used_inside_loop = visitBlock(stmt.body, set())
            used_in_loop_header = used_vars(stmt.test)
            return used_inside_loop | used_in_loop_header | live_out
        if isinstance(stmt, ast.Break):
            # Currently, we assume that break statements are only allowed as the last
            # statement in a loop, as "if cond: break".
            return live_out
        if isinstance(stmt, ast.FunctionDef):
            if stmt.name in live_out:
                live_out.remove(stmt.name)
                live_out = live_out | outer_scope_variables(stmt, formatter)
            return live_out
        raise ValueError(formatter(stmt, f"Unsupported statement type {type(stmt)!r}."))

    return visitBlock(stmts, set())


def outer_scope_variables(fun: ast.FunctionDef, formatter: sourceinfo.Formatter):
    """Return the set of outer-scope variables used in a nested function.

    Args:
        fun: The function-ast to analyze.
        formatter: The formatter object.

    Returns:
        A set of variable names (strings).
    """
    assert isinstance(fun, ast.FunctionDef)
    used_vars_ = exposed_uses(fun.body, formatter)
    inputs = [x.arg for x in fun.args.args]
    return used_vars_.difference(inputs)
