# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import ast
from .values import DebugInfo


def used_vars(expr):
    ''' Return set of all variables used with an expression.'''
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
    if isinstance(stmt, ast.AugAssign):
        # See comment in function exposed_uses.
        return local_defs(stmt.target)
    if isinstance(stmt, ast.AnnAssign):
        return local_defs(stmt.target)
    if isinstance(stmt, ast.Return):
        return set()
    if isinstance(stmt, ast.If):
        return block_defs(stmt.body) | block_defs(stmt.orelse)
    if isinstance(stmt, list):
        return block_defs(stmt)
    try:
        if stmt.value.func.id == 'print':
            # Any call to print function are ignored.
            return set()
    except (TypeError, AttributeError):
        pass
    raise ValueError(f"Unsupported statement type: {type(stmt).__name__}.")


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
            f"Unsupported statement type: {type(stmt).__name__}."))

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
        if isinstance(stmt, ast.AugAssign):
            # This is used in loops to update the condition:
            # cond &= new_value. eager mode does not work without that trick.
            # For python it is equivalent to ast.Assign, from python,
            # it enables a variable to be updated and not replaced.
            return live_out.difference(local_defs(stmt.target)) | used_vars(stmt.value)
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
            if getattr(f, 'id', '') == 'print':
                return live_out
            if getattr(f, 'attr', '') == 'append':
                raise NotImplementedError(DebugInfo(stmt, converter).msg(
                    "A loop includes an instruction <name>.append(<tensor>). "
                    "This is not supported yet."))
            raise ValueError(DebugInfo(stmt, converter).msg(
                f"Unsupported statement type {type(stmt).__name__!r}"
                f", type(stmt.value)={type(stmt.value)!r}"
                f", stmt.value.__dict__={stmt.value.__dict__!r}"
                f", type(stmt.value.func)={type(f)!r}"
                f", stmt.value.func.__dict__={f.__dict__!r}"
                f"."))
        raise ValueError(DebugInfo(stmt, converter).msg(
            f"Unsupported statement type {type(stmt).__name__!r}, "
            f"stmt.__dict__={stmt.__dict__!r}."))

    return visitBlock(stmts, set())
