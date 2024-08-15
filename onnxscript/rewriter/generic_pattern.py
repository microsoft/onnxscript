# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import collections
import inspect
import os
import textwrap
import warnings
from typing import Any, Callable, Iterator, Sequence

import onnxscript.rewriter.pattern as orp
from onnxscript import ir


class PatternMatchResult:
    """Stores information about a match if a match was successful.

    * pattern: the GraphPattern which found this result
    * model_nodes: the graph nodes that matched the pattern
    * matched_pattern_to_model_value: a mapping from ValuePattern to ir.Value
    * kwargs: additional attributes the user may add through the method
        :meth:`PatternMatchResult.add_kwargs`
    """

    def __init__(
        self,
        pattern: orp.GraphPattern,
        model_nodes: Sequence[ir.Node],
    ):
        pattern_nodes: list[orp.NodePattern] = list(pattern)
        assert len(model_nodes) == len(pattern_nodes)
        self.pattern = pattern
        self.model_nodes = model_nodes
        self.kwargs: dict[str, Any] = {}
        self.matched_pattern_to_model_value: dict[orp.ValuePattern, ir.Value] = {}

        for graph_node, pattern_node in zip(model_nodes, pattern_nodes):
            assert (
                graph_node.op_identifier() == pattern_node.op_identifier()
            ), f"Unexpected type mismatch {graph_node.op_identifier()!r} != {pattern_node.op_identifier()!r}"
            assert len(graph_node.inputs) == len(
                pattern_node.inputs
            ), f"Unexpected number of inputs for type {graph_node.op_identifier()}"
            for a, b in zip(graph_node.inputs, pattern_node.inputs):
                if b is None:
                    # optional input or not an interesting input
                    continue
                self._bind(b, a)

            assert len(graph_node.outputs) == len(
                pattern_node.outputs
            ), f"Unexpected number of outputs for type {graph_node.op_identifier()}"
            for a, b in zip(graph_node.outputs, pattern_node.outputs):
                self._bind(b, a)

    def _bind(self, value_pattern: orp.ValuePattern, value: ir.Value) -> None:
        map = self.matched_pattern_to_model_value
        if value_pattern in map:
            assert map[value_pattern] == value, (
                f"Ambiguities, pattern output {value_pattern!r} means "
                f"{value!r} or {map[value_pattern]}"
            )
        else:
            map[value_pattern] = value

    def add_kwargs(self, name: str, value: Any):
        """Adds an attribute, it can be done when the match is being validated,
        this attribute can be used when building the replacement nodes.
        """
        self.kwargs[name] = value

    def __repr__(self) -> str:
        return (
            f"PatternMatchResult: {len(self.model_nodes)} nodes ..., {self.pattern.inputs}, "
            f"{self.pattern.outputs})"
        )


def _to_match_result(pmr: PatternMatchResult) -> orp.MatchResult:
    """Converts a PatternMatchResult into a MatchResult.

    TODO: This is a temporary hack until MatchResult and PatternMatchResult are unified.
    """
    result = orp.MatchResult()
    result.nodes.extend(pmr.model_nodes)
    for var, val in pmr.matched_pattern_to_model_value.items():
        if var.name is not None:
            result.bind(var.name, val)
    result.outputs.extend([pmr.matched_pattern_to_model_value[v] for v in pmr.pattern.outputs])
    return result


def _value_to_str(value: ir.Value | orp.ValuePattern) -> str:
    return value.name if value.name is not None else "anonymous:" + str(id(value))


def _opt_value_to_str(value: ir.Value | orp.ValuePattern | None) -> str:
    return _value_to_str(value) if value is not None else "None"


def _node_to_str(node: ir.Node | orp.NodePattern) -> str:
    inputs = ", ".join(_opt_value_to_str(input) for input in node.inputs)
    outputs = ", ".join(_opt_value_to_str(output) for output in node.outputs)
    op_type = node.op_type
    domain = str(node.domain)
    qualified_op = f"{domain}.{op_type}" if domain else op_type
    return f"{outputs} = {qualified_op}({inputs})"


# def _pattern_node_to_str(node: orp.NodePattern) -> str:
#     inputs = ", ".join(_opt_value_to_str(input) for input in node.inputs)
#     outputs = ", ".join(_opt_value_to_str(output) for output in node.outputs)
#     return f"{outputs} = {node.op_type}({inputs})"


class GenericPatternMatcher(orp.PatternMatcher):
    """
    Implements a pattern optimization for quick experimentation.

    Current limitation:

    * The current implementation does match on domain name (easy fix).
    * It does not compares attributes either (easy fix as well).
    """

    def __init__(self, pattern: orp.GraphPattern) -> None:
        super().__init__(pattern)

    def enumerate_matches(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        node: ir.Node | None = None,
        verbose: int = 0,
    ) -> Iterator:
        """Enumerates all the matches."""
        if node is None:
            matched = []
            for node in graph_or_function:
                res = self.match(model, graph_or_function, node, verbose=verbose)
                if res:
                    matched.append(res)
                    yield res
        else:
            res = self.match(model, graph_or_function, node, verbose=verbose)
            if res:
                yield res

    def none(
        self,
        node: ir.Node | None = None,
        lineno: int | None = None,
        msg: str = "",
    ) -> None:
        """Must be called every time a match fails to trace it.

        It may be useful which reason made a pattern matching fail.
        Instead of returning None, method *match* can return the following
        expression:

        ::

            return self.none(node, inspect.currentframe().f_lineno)

        By setting the verbosity (see next Section), the user may then know
        which lines in the code returned None and which condition failed.
        If logs are fully enabled, it shows information about matched none
        and the line deciding the matched failed.
        For example, this tells the matching failed at line 601 in ``generic_pattern.py``.
        It happens when propagating the match in the backward directions.
        The unmatched types are Mul, MatMul and below,
        it shows the matched nodes. The first one was Cast.
        And the failure happened at iteration 5.
        ``139774002356544-139774000632672`` is the pair of ids used in container ``matched``.
        ``id(node)`` is used as a unique identifiers of the nodes.

        ::

            [RotaryEmbeddingPattern.match] NONE - line: 601:__main__, op_type=Cast
                --hint--: BACKWARD: different node types
                --pattern
                Mul(pos_ids, cast) -> (mul)
                -- model
                MatMul(/_original_modu...Expand_output_0, /_original_modu...b/Cast_output_0) -> (/_original_modu...MatMul_output_0)
                iteration=5
                --matched-- #6
                Cast(/_original_modu...mb/Cos_output_0) ~ Cast(cos) [139774002356544-139774000632672]
                Cos(/_original_modu...ncat_1_output_0) ~ Cos(concattraining-transpose-0) [139774002356448-139774000632048]
                ConcatTraining(/_original_modu...nspose_output_0,/_original_modu...nspose_output_0) ~ ConcatTraining(transpose,transpose) [139774002356352-139774000631712]
                Transpose(/_original_modu...MatMul_output_0) ~ Transpose(mul) [139774002356256-139774000631184]
                Sin(/_original_modu...ncat_1_output_0) ~ Sin(concattraining-transpose-0) [139774002358512-139774000631568]
                Cast(/_original_modu...mb/Sin_output_0) ~ Cast(sin) [139774002358608-139774000632384]
                len(stack)=0:[]

        'hints' are not added everywhere. More can easily be added with method ``_hint``.
        """
        if node and self.verbose:
            if self.verbose >= 10:
                if hasattr(self, "_debug"):
                    msg2 = self._debug_print()
                    if msg2:
                        msg2 = f"\n{textwrap.indent(msg2, '    ')}"
                else:
                    msg2 = ""
                print(
                    f"[{self.__class__.__name__}.match] Match failed at line: {lineno}:"
                    f"{os.path.split(self.__class__.__module__)[-1]}, "
                    f"op_type={node.op_type}{msg}{msg2}"
                )
        return None

    def print_match(self, graph_node: ir.Node, pattern_node: orp.NodePattern) -> str:
        s1 = _node_to_str(graph_node)
        s2 = _node_to_str(pattern_node)
        return f"match {s1} with pattern: {s2}"

    def _debug_print(self) -> str:
        if not hasattr(self, "_debug"):
            return ""

        def _s(s: str) -> str:
            if len(s) <= 30:
                return s
            return f"{s[:15]}...{s[-15:]}"

        def _p(n: ir.Node, full: bool = False) -> str:
            if full:
                return str(n)
            return _node_to_str(n)

        rows = []
        for k, v in sorted(self._debug.items()):
            if k == "stack":
                rows.append(f"len({k})={len(v)}:{v}")  # type: ignore[arg-type]
                continue
            if k == "iteration":
                rows.append(f"{k}={v}")
                continue
            if k == "matched":
                rows.append(f"--matched-- #{len(v)}")  # type: ignore[arg-type]
                for pattern_node, graph_node in v.items():
                    rows.append(
                        f"  {_p(pattern_node)} ~ {_p(graph_node)} [{id(pattern_node)}-{id(graph_node)}]"
                    )
                continue
            if k == "hint":
                rows.append(f"--hint--: {v[0]}")  # type: ignore[arg-type]
                for i in v[1:]:
                    if isinstance(i, str):
                        rows.append("  " + i)
                    if isinstance(i, ir.Node):
                        rows.append("  " + _p(i, full=True))
                continue
            if k in {"node", "pattern", "pattern_node", "pattern_nodes"}:
                continue
            rows.append(f"-- not shown {k}")

        return "\n".join(rows)

    def _hint(self, *args: Any) -> None:
        """Add debugging information to help users."""
        self._debug["hint"] = args

    def _match_backward(
        self,
        starting_node: ir.Node,
        matched: dict[orp.NodePattern, ir.Node],
        stack: list[orp.NodePattern],
        graph_node: ir.Node,
        pattern_node: orp.NodePattern,
    ) -> int | None:
        """
        Matches backward.

        Args:
            starting_node: root node (the node the matched begain with, used only for debugging)
            matched: nodes of the pattern matched as already matched
            stack: next node to look into
            graph_node: node coming from the graph
            pattern_node: node coming from the pattern

        Returns:
            number of matched nodes, None or False to indicate a failed match
        """
        match_count = 0

        # predecessors
        if len(graph_node.inputs) != len(pattern_node.inputs):
            # not the same number of inputs
            self._hint(
                "BACKWARD: not the same number of inputs",
                "-- pattern",
                pattern_node,
                "-- model",
                graph_node,
            )
            return self.none(starting_node, inspect.currentframe().f_lineno)

        for graph_input, pattern_input in zip(graph_node.inputs, pattern_node.inputs):
            if len(graph_input.uses()) != len(pattern_input.uses()):
                self._hint(
                    "BACKWARD: one input is used outside the pattern",
                    "-- pattern",
                    pattern_node,
                    "-- model",
                    graph_node,
                )
                return self.none(starting_node, inspect.currentframe().f_lineno)

        for graph_value, pattern_value in zip(graph_node.inputs, pattern_node.inputs):
            # TODO(rama): Handle constant-pattern
            pattern_pred = pattern_value.producer()
            if pattern_pred is None:
                # pattern_pred is None means the pattern backward search ends here.
                result = self._match_values_forward(
                    starting_node, matched, stack, graph_value, pattern_value
                )
                if result is None:
                    return result
                match_count += result
                continue
            graph_pred = graph_value.producer()
            if graph_pred is None:
                # No node in the graph.
                return self.none(starting_node, inspect.currentframe().f_lineno)
            if graph_pred.op_identifier() != pattern_pred.op_identifier():
                self._hint(
                    "BACKWARD: different node types",
                    "--pattern",
                    _node_to_str(pattern_pred),
                    "-- model",
                    _node_to_str(graph_pred),
                )
                return self.none(starting_node, inspect.currentframe().f_lineno)
            # matching backward
            if pattern_pred not in matched:
                if self.verbose >= 10:
                    print(
                        f"[GenericPattern._match_backward] {self.print_match(graph_pred, pattern_pred)}"
                    )
                matched[pattern_pred] = graph_pred
                stack.append(pattern_pred)
                match_count += 1
        if self.verbose > 5 and match_count > 0:
            print(f"[GenericPatternMatcher._match_backward] add {match_count} nodes")
        return match_count

    def _match_values_forward(
        self,
        starting_node: ir.Node,
        matched: dict[orp.NodePattern, ir.Node],
        stack: list[orp.NodePattern],
        graph_value: ir.Value,
        pattern_value: orp.ValuePattern,
    ) -> int | None:
        """
        Matches forward.

        Args:
            starting_node: root node (the node the match begins with, used only for debugging)
            matched: nodes of the pattern matched as already matched
            stack: next node to look into
            graph_value: value coming from the graph
            pattern_value: pattern value coming from the pattern

        Returns:
            number of matched nodes to continue, None or False to indicate a failed match
        """
        match_count = 0
        graph_node_users = [user for user, _ in graph_value.uses()]
        pattern_node_users = [user for user, _ in pattern_value.uses()]
        if not pattern_node_users:
            # The pattern has no node forward, the matching stops.
            return match_count
        if len(graph_node_users) < len(pattern_node_users):
            # Not enough node in the graph to match the pattern. A match is not possible
            return self.none(starting_node, inspect.currentframe().f_lineno)

        # Here comes the fun part, there is the same number of successors or more
        # nodes in the graph to match with the pattern.
        # And we have to handle the nodes already matched as found.
        # Hopefully, there is only one option.

        if len(graph_node_users) == len(pattern_node_users) == 1:
            # Let's deal with the simple case
            if graph_node_users[0].op_identifier() != pattern_node_users[0].op_identifier():
                return self.none(starting_node, inspect.currentframe().f_lineno)

            node = pattern_node_users[0]
            if node not in matched:
                if self.verbose >= 10:
                    print(
                        f"[GenericPatternMatcher._match_values_forward]{self.print_match(graph_node_users[0], pattern_node_users[0])}"
                    )
                matched[node] = graph_node_users[0]
                stack.append(node)
                match_count += 1
            return match_count

        # Let's remove the nodes already matched.
        pattern_node_users_not_matched = [
            unmatched_node
            for unmatched_node in pattern_node_users
            if unmatched_node not in matched
        ]
        pattern_node_users_matched = [
            matched[matched_node]
            for matched_node in pattern_node_users
            if matched_node in matched
        ]
        assert len(pattern_node_users_matched) + len(pattern_node_users_not_matched) == len(
            pattern_node_users
        ), (
            f"pattern_node_users_not_matched={pattern_node_users_not_matched}, "
            f"pattern_node_users_matched={pattern_node_users_matched}, "
            f"pattern_node_users={pattern_node_users}, "
            f"matched={matched}"
        )
        free = list(set(graph_node_users) - set(pattern_node_users_matched))
        if not pattern_node_users_not_matched:
            # Everything is already matched.
            return match_count
        if len(free) < len(pattern_node_users_not_matched):
            # Not enough successors to match the remaining patterns.
            return self.none(starting_node, inspect.currentframe().f_lineno)
        if len(pattern_node_users_not_matched) == len(free) == 1:
            # Only one option again.
            graph_node = free[0]
            if pattern_node_users_not_matched[0].op_identifier() != graph_node.op_identifier():
                return self.none(starting_node, inspect.currentframe().f_lineno)

            key = pattern_node_users_not_matched[0]
            if self.verbose >= 10:
                print(
                    f"[GenericPatternMatcher._match_values_forward] {self.print_match(graph_node, pattern_node_users_not_matched[0])}"
                )
            matched[key] = graph_node
            stack.append(key)
            match_count += 1
            return match_count

        # And now another fun part, let's try to handle the case when
        # there is only one option, matching on node type only returns one
        # option.
        expected_op_type = [_.op_identifier() for _ in pattern_node_users_not_matched]
        got_op_type = [_.op_identifier() for _ in free]

        ec = collections.Counter(expected_op_type)
        gc = collections.Counter(got_op_type)
        if len(ec) != len(gc) or set(ec) != set(gc):
            # unique operator types is different.
            self._hint(
                "FORWARD: unique operator types are different",
                "-- pattern",
                ec,
                pattern_value,
                "-- model",
                gc,
                graph_value,
                "-- model-matched",
                pattern_node_users_matched,
            )
            return self.none(starting_node, inspect.currentframe().f_lineno)
        for k, v in ec.items():
            if gc[k] < v:
                # Not enough types to match.
                return self.none(starting_node, inspect.currentframe().f_lineno)

        # At this stage, we know matching the types is possible.
        # We first mark whatever is possible.
        ptype_to_node = {_.op_identifier(): _ for _ in pattern_node_users_not_matched}
        gtype_to_node = {_.op_identifier(): _ for _ in free}
        missing = []
        for k, v in ec.items():
            if gc[k] == v == 1:
                key = id(ptype_to_node[k])
                if key not in matched:
                    if self.verbose >= 10:
                        print(
                            f"[GenericPatternMatcher._match_values_forward] match "
                            f"{self.print_match(gtype_to_node[k], ptype_to_node[k])}"
                        )
                    matched[key] = gtype_to_node[k]
                    stack.append(key)
                    match_count += 1
            else:
                missing.append(k)

        if not missing:
            return match_count

        # At this stage, there are mutiple options for matching. We can:
        # 1. make assumptions and continue
        # 2. mark the node as incomplete matching, we could end up stuck anyway.
        raise NotImplementedError(
            f"There are more than one option, this will be implemented later, "
            f"ec={ec}, gc={gc}"
        )

    def _match_forward(
        self,
        starting_node: ir.Node,
        matched: dict[orp.NodePattern, ir.Node],
        stack: list[orp.NodePattern],
        graph_node: ir.Node,
        pattern_node: orp.NodePattern,
    ) -> int | None:
        """
        Matches forward.

        Args:
            starting_node: root node (the node the match begins with, used only for debugging)
            matched: nodes of the pattern matched as already matched
            stack: next node to look into
            graph_node: node coming from the graph
            pattern_node: node coming from the pattern

        Returns:
            number of matched nodes to continue, None or False to indicate a failed match
        """
        match_count = 0

        # successors
        if len(graph_node.outputs) != len(pattern_node.outputs):
            # not the same number of outputs
            self._hint(
                "FORWARD: not the same number of output_names",
                "-- pattern",
                pattern_node,
                "-- model",
                graph_node,
            )
            return self.none(starting_node, inspect.currentframe().f_lineno)

        for graph_output, pattern_output in zip(graph_node.outputs, pattern_node.outputs):
            result = self._match_values_forward(
                starting_node, matched, stack, graph_output, pattern_output
            )
            if result is None:
                return result
            match_count += result

        if self.verbose > 5 and match_count > 0:
            print(f"[GenericPatternMatcher._match_forward] add {match_count} nodes")
        return match_count

    def match(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        node: ir.Node,
        verbose: int = 0,
    ) -> orp.MatchResult | None:
        del model
        del graph_or_function
        self.verbose = verbose
        self._debug = {}

        # Let's match the last node.
        # Then we need to match successors and predecessors.
        last_pattern_node = self.pattern.node(-1)
        if node.op_identifier() != last_pattern_node.op_identifier():
            # The last node does not have the same op_identifier().
            return self.none()

        if self.verbose > 5:
            print(
                f"[GenericPatternMatcher.match] Matching started at node: {_node_to_str(node)}"
            )
            if self.verbose >= 10:
                print(f"[GenericPatternMatcher.match] match pattern {self}")

        all_pattern_nodes = set(self.pattern)
        matched: dict[orp.NodePattern, ir.Node] = {last_pattern_node: node}
        stack: list[orp.NodePattern] = [last_pattern_node]
        iteration = 0

        if self.verbose > 5:
            self._debug = dict(
                pattern=self.pattern,
                matched=matched,
                stack=stack,
                iteration=iteration,
                node=node,
                pattern_node=last_pattern_node,
                pattern_nodes=self.pattern,
            )

        max_iter = self.pattern.num_nodes() * 2
        while stack and iteration < max_iter:
            nodes_not_in_pattern = set(matched.keys()) - all_pattern_nodes
            assert not nodes_not_in_pattern, (
                f"Some nodes are not part of the pattern: {nodes_not_in_pattern}"
                f"\nall_pattern_nodes={all_pattern_nodes}"
            )

            # TODO(justinchuby): Change to a for loop
            iteration += 1
            if self.verbose > 5:
                print(
                    f"[GenericPatternMatcher.match] iteration={iteration} "
                    f"n_matched={len(matched)}, n_stack={len(stack)}, "
                    f"matched_types={collections.Counter(_.op_identifier() for _ in matched)}"
                )
            next_pattern_node = stack.pop()
            next_graph_node = matched[next_pattern_node]

            result = self._match_backward(
                node, matched, stack, next_graph_node, next_pattern_node
            )
            if result is None:
                if self.verbose > 5:
                    print("[GenericPatternMatcher.match] done. backward failed.")
                return result

            nodes_not_in_pattern = set(matched.keys()) - all_pattern_nodes
            assert (
                not nodes_not_in_pattern
            ), f"Some nodes are not part of the pattern: {nodes_not_in_pattern}"

            result = self._match_forward(
                node, matched, stack, next_graph_node, next_pattern_node
            )
            if result is None:
                if self.verbose > 5:
                    print("[GenericPatternMatcher.match] done. forward failed.")
                return result

            nodes_not_in_pattern = set(matched.keys()) - all_pattern_nodes
            assert (
                not nodes_not_in_pattern
            ), f"Some nodes are not part of the pattern: {nodes_not_in_pattern}"

            if self.verbose > 5:
                self._debug["iteration"] = iteration

        if iteration >= max_iter and stack:
            self._hint(f"reached {iteration}>={max_iter} iterations")
            return self.none(node, inspect.currentframe().f_lineno)

        if self.verbose > 5:
            print(f"[GenericPatternMatcher.match] done. {len(matched)} matched nodes")

        # At this point, the pattern is matched but let's make sure.
        assert len(matched) == self.pattern.num_nodes(), (
            f"Number of matched nodes is different, {len(matched)} matched nodes, "
            f"and {len(self.pattern)} nodes in the pattern, matched is {matched}"
        )
        assert len(stack) == 0, f"There are still {len(stack)} nodes to explore."

        # We order the matched nodes in the same order than the pattern
        # to let next functions to be able to build the matching again.
        matched_nodes = [matched[pattern_node] for pattern_node in self.pattern]
        return _to_match_result(PatternMatchResult(self.pattern, matched_nodes))


def make_pattern_rule(
    match_pattern_function: Callable,
    apply_pattern_function: Callable,
    validate_mapping: Callable | None = None,
    verbose: int = 0,
) -> orp.RewriteRule:
    """
    Creates a rewriting rule from a callable or a function proto.

    Args:
        match_pattern_function: an onnxscript-like function that defines
            the pattern subgraph (nodes) to be replaced
        apply_pattern_function: an onnxscript-like function that constructs
            the replacement subgraph (new nodes replacing the matched nodes)
        validate_mapping: a function that validates the matching subgraph once
            it is found. If it returns False the pattern is not applied.
            If not specified, it is equivalent to a function that always return True
        verbose: verbosity level

    Returns:
        the rewriting rule
    """

    warnings.warn(
        "make_pattern_rule(...) is deprecated, use pattern.RewriteRule(...) instead",
        FutureWarning,
        stacklevel=2,
    )
    pattern = orp._to_graph_pattern(match_pattern_function)
    matcher = GenericPatternMatcher(pattern)
    return orp.RewriteRule(
        pattern,
        apply_pattern_function,
        validate_mapping,
        matcher,
        verbose=verbose,
    )
