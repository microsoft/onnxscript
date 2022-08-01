# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import ast
from locale import currency
from .values import DebugInfo


def used_vars(expr):
    ''' Return set of all variables used in an expression.'''
    if isinstance(expr, ast.Name):
        return set([expr.id])
    result = set()
    for c in ast.iter_child_nodes(expr):
        result = result | used_vars(c)
    return result


def local_defs(lhs):
    '''Utility function to return set of assigned/defined
    variables in the lhs of an assignment statement.'''
    def get_id(e):
        assert isinstance(e, ast.Name), "Only simple assignments supported."
        return e.id

    if (isinstance(lhs, ast.Tuple)):
        return set([get_id(x) for x in lhs.elts])
    return set([get_id(lhs)])


def defs(stmt):
    '''
    Return the set of all variables that may be defined (assigned to) in an
    execution of input stmt.
    '''
    def block_defs(block):
        result = set()
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
    try:
        if stmt.value.func.id == 'print':
            # Any call to print function are ignored.
            return set()
    except (TypeError, AttributeError):
        pass
    raise ValueError(f"Unsupported statement type {type(stmt)!r}.")


def do_liveness_analysis(fun, converter):
    '''
    Perform liveness analysis of the given function-ast. The results of the
    analysis are stored directly with each statement-ast `s` as attributes `s.live_in`
    and `s.live_out`.
    '''
    def visit(stmt, live_out):
        stmt.live_out = live_out
        live = do_visit(stmt, live_out)
        stmt.live_in = live
        return live

    def do_visit(stmt, live_out):
        def visitBlock(block, live_out):
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
            if not isinstance(stmt.target, ast.Name):
                raise ValueError(DebugInfo(stmt, converter).msg(
                    "For loop target must be a single variable."))
            p_loop_var = stmt.target.id
            prev = None
            curr = live_out
            while curr != prev:
                prev = curr
                curr = visitBlock(stmt.body, prev).difference (set([p_loop_var]))
            return curr
        if isinstance(stmt, (ast.For, ast.While)):
            return live_out  # TODO
        if isinstance(stmt, ast.Expr) and hasattr(stmt, 'value'):
            # docstring
            if hasattr(stmt.value, 'value') and isinstance(stmt.value.value, str):
                # python 3.8+
                return live_out
            if hasattr(stmt.value, 's') and isinstance(stmt.value.s, str):
                # python 3.7
                return live_out
        try:
            if stmt.value.func.id == 'print':
                # Any call to print function are ignored.
                return live_out
        except (TypeError, AttributeError):
            pass
        raise ValueError(DebugInfo(stmt, converter).msg(
            f"Unsupported statement type {type(stmt)!r}."))

    assert isinstance(fun, ast.FunctionDef)
    live = set()
    for s in reversed(fun.body):
        live = visit(s, live)


def exposed_uses(stmts, converter):
    '''
    Return the set of variables that are used before being defined by given block.
    '''
    def visitBlock(block, live_out):
        for stmt in reversed(block):
            live_out = visit(stmt, live_out)
        return live_out

    def visit(stmt, live_out):
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
        if (isinstance(stmt, ast.Expr) and hasattr(stmt, 'value') and
                isinstance(stmt.value, ast.Call)):
            f = stmt.value.func
            if f.id == 'print':
                return live_out
        if isinstance(stmt, ast.Break):
            # TODO: liveness analysis does not handle breaks in the middle of a loop yet.
            return live_out
        raise ValueError(DebugInfo(stmt, converter).msg(
            f"Unsupported statement type {type(stmt)!r}."))

    return visitBlock(stmts, set())
