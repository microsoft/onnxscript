# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import ast
from typing import Any, Optional, Sequence, Set

from onnxscript import sourceinfo
from onnxscript._internal import ast_utils


def _get_loop_var(for_stmt: ast.For, formatter: sourceinfo.Formatter) -> str:
    if not isinstance(for_stmt.target, ast.Name):
        raise TypeError(formatter(for_stmt, "For loop target must be a single variable."))
    return for_stmt.target.id


def _used_vars(expr: Optional[ast.expr]) -> Set[str]:
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
        result = result | _used_vars(c)
    return result


def _lhs_vars(lhs: ast.expr) -> Set[str]:
    """Return set of assigned variables in the lhs of an assignment statement."""

    def get_id(e):
        assert isinstance(e, ast.Name), "Only simple assignments supported."
        return e.id

    if isinstance(lhs, ast.Tuple):
        return {get_id(x) for x in lhs.elts}
    return {get_id(lhs)}


class AstAnalyzer:
    def __init__(
        self,
        fun: ast.FunctionDef,
        formatter: sourceinfo.Formatter,
        globals: dict[str, Any] | None = None,
    ) -> None:
        self._formatter = formatter
        self._constant_if_condition: dict[ast.If, bool] = {}
        self._live_in: dict[ast.stmt, Set[str]] = {}
        self._live_out: dict[ast.stmt, Set[str]] = {}
        if globals:
            self._compute_constant_if_conditions(fun, globals)
        self.do_liveness_analysis(fun)

    def live_in(self, stmt: ast.stmt) -> Set[str] | None:
        """Get the set of variables that are live at the entry of the given statement."""
        return self._live_in.get(stmt)

    def live_out(self, stmt: ast.stmt) -> Set[str] | None:
        """Get the set of variables that are live at the exit of the given statement."""
        return self._live_out.get(stmt)

    def _compute_constant_if_conditions(
        self, fun: ast.FunctionDef, globals: dict[str, Any]
    ) -> None:
        """Identify if-statements with constant conditions.

        If-statements of the form `if name:` where `name` is an outer-scope variable
        and name is not assigned to within the function body, are treated as constant
        conditions. The value of such conditions is determined from the outer-scope.
        """

        assigned_vars = self.assigned_vars(fun.body)
        for node in ast.walk(fun):
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Name):
                    python_var = node.test.id
                    if python_var not in assigned_vars and python_var in globals:
                        # Condition depends on an outer-scope variable.
                        self._constant_if_condition[node] = bool(globals[python_var])

    def constant_if_condition(self, if_stmt: ast.If) -> Optional[bool]:
        """Return the constant value of the if-statement condition, if it is constant.

        Args:
            if_stmt: The if-statement-ast to analyze.

        Returns:
            The constant boolean value of the if-statement condition, or None if not constant.
        """
        return self._constant_if_condition.get(if_stmt, None)

    def assigned_vars(self, stmt: ast.stmt | list[ast.stmt]) -> Set[str]:
        """Return the set of all variables that may be assigned to in an execution of input stmt
        or sequence of statements.
        """

        def assigned_in_block(block: Sequence[ast.stmt]) -> Set[str]:
            result: set[Any] = set()
            for s in block:
                result = result | self.assigned_vars(s)
            return result

        if isinstance(stmt, ast.Assign):
            return _lhs_vars(stmt.targets[0])
        if isinstance(stmt, ast.AnnAssign):
            return _lhs_vars(stmt.target)
        if isinstance(stmt, ast.Return):
            return set()
        if isinstance(stmt, ast.If):
            constant_cond = self.constant_if_condition(stmt)
            if constant_cond is None:
                return assigned_in_block(stmt.body) | assigned_in_block(stmt.orelse)
            elif constant_cond:
                return assigned_in_block(stmt.body)
            else:
                return assigned_in_block(stmt.orelse)
        if isinstance(stmt, ast.For):
            return assigned_in_block(stmt.body) | {_get_loop_var(stmt, self._formatter)}
        if isinstance(stmt, ast.While):
            return assigned_in_block(stmt.body)
        if isinstance(stmt, list):
            return assigned_in_block(stmt)
        if isinstance(stmt, ast.Break):
            return set()
        if isinstance(stmt, ast.FunctionDef):
            # Supported function-definitions (used for higher order ops like Scan)
            # do not assign to any variable in the outer scope.
            return set()
        if ast_utils.is_print_call(stmt):
            return set()
        if ast_utils.is_doc_string(stmt):
            return set()
        error_message = self._formatter(stmt, f"Unsupported statement type {type(stmt)!r}.")
        raise ValueError(error_message)

    def do_liveness_analysis(self, fun: ast.FunctionDef) -> None:
        """Perform liveness analysis of the given function-ast."""

        def visit(stmt: ast.stmt, live_out: Set[str]) -> Set[str]:
            self._live_out[stmt] = live_out
            live = do_visit(stmt, live_out)
            self._live_in[stmt] = live
            return live

        def do_visit(stmt: ast.stmt, live_out: Set[str]) -> Set[str]:
            def visitBlock(block: Sequence[ast.stmt], live_out: Set[str]) -> Set[str]:
                for s in reversed(block):
                    live_out = visit(s, live_out)
                return live_out

            if isinstance(stmt, ast.Assign):
                return live_out.difference(_lhs_vars(stmt.targets[0])) | _used_vars(stmt.value)
            if isinstance(stmt, ast.AnnAssign):
                return live_out.difference(_lhs_vars(stmt.target)) | _used_vars(stmt.value)
            if isinstance(stmt, ast.Return):
                return _used_vars(stmt.value)
            if isinstance(stmt, ast.If):
                constant_cond = self.constant_if_condition(stmt)
                if constant_cond is None:
                    live1 = visitBlock(stmt.body, live_out)
                    live2 = visitBlock(stmt.orelse, live_out)
                    return live1 | live2 | _used_vars(stmt.test)
                elif constant_cond:
                    return visitBlock(stmt.body, live_out)
                else:
                    return visitBlock(stmt.orelse, live_out)
            if isinstance(stmt, ast.For):
                p_loop_var = _get_loop_var(stmt, self._formatter)
                prev = None
                curr = live_out
                while curr != prev:
                    prev = curr
                    curr = visitBlock(stmt.body, prev).difference({p_loop_var})
                return curr
            if isinstance(stmt, ast.While):
                cond_vars = _used_vars(stmt.test)
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
            if ast_utils.is_doc_string(stmt):
                return live_out
            if isinstance(stmt, ast.FunctionDef):
                return live_out
            if ast_utils.is_print_call(stmt):
                return live_out
            raise ValueError(
                self._formatter(stmt, f"Unsupported statement type {type(stmt)!r}.")
            )

        assert isinstance(fun, ast.FunctionDef)
        live: set[Any] = set()
        for s in reversed(fun.body):
            live = visit(s, live)

    def exposed_uses(self, stmts: Sequence[ast.stmt]) -> set[str]:
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
                return live_out.difference(_lhs_vars(stmt.targets[0])) | _used_vars(stmt.value)
            if isinstance(stmt, ast.AnnAssign):
                return live_out.difference(_lhs_vars(stmt.target)) | _used_vars(stmt.value)
            if isinstance(stmt, ast.Return):
                return _used_vars(stmt.value)
            if isinstance(stmt, ast.If):
                constant_cond = self.constant_if_condition(stmt)
                if constant_cond is None:
                    live1 = visitBlock(stmt.body, live_out)
                    live2 = visitBlock(stmt.orelse, live_out)
                    return (live1 | live2) | _used_vars(stmt.test)
                elif constant_cond:
                    return visitBlock(stmt.body, live_out)
                else:
                    return visitBlock(stmt.orelse, live_out)
            if ast_utils.is_print_call(stmt):
                return live_out
            if ast_utils.is_doc_string(stmt):
                return live_out
            if isinstance(stmt, ast.For):
                # Analysis assumes loop may execute zero times. Results can be improved
                # for loops that execute at least once.
                loop_var_set = {_get_loop_var(stmt, self._formatter)}
                used_after_loop = live_out.difference(loop_var_set)
                used_inside_loop = visitBlock(stmt.body, set()).difference(loop_var_set)
                used_in_loop_header = _used_vars(stmt.iter)
                return used_inside_loop | used_in_loop_header | used_after_loop
            if isinstance(stmt, ast.While):
                # Analysis assumes loop may execute zero times. Results can be improved
                # for loops that execute at least once.
                used_inside_loop = visitBlock(stmt.body, set())
                used_in_loop_header = _used_vars(stmt.test)
                return used_inside_loop | used_in_loop_header | live_out
            if isinstance(stmt, ast.Break):
                # Currently, we assume that break statements are only allowed as the last
                # statement in a loop, as "if cond: break".
                return live_out
            if isinstance(stmt, ast.FunctionDef):
                if stmt.name in live_out:
                    live_out.remove(stmt.name)
                    live_out = live_out | self.outer_scope_variables(stmt)
                return live_out
            raise ValueError(
                self._formatter(stmt, f"Unsupported statement type {type(stmt)!r}.")
            )

        return visitBlock(stmts, set())

    def outer_scope_variables(self, fun: ast.FunctionDef) -> set[str]:
        """Return the set of outer-scope variables used in a nested function.

        Args:
            fun: The function-ast to analyze.
            formatter: The formatter object.

        Returns:
            A set of variable names (strings).
        """
        assert isinstance(fun, ast.FunctionDef)
        used_vars_ = self.exposed_uses(fun.body)
        inputs = [x.arg for x in fun.args.args]
        return used_vars_.difference(inputs)
