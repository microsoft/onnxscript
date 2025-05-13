# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Implementation of the pattern matching algorithm."""

from __future__ import annotations

import abc
import itertools
import math
from typing import (
    Iterable,
    Sequence,
)

import onnxscript.rewriter._basics as _basics
import onnxscript.rewriter._pattern_ir as _pattern_ir
from onnxscript import ir


def _valid_to_replace(
    matched_nodes: Sequence[ir.Node], output_values: Sequence[ir.Value]
) -> bool:
    """Check that values computed by the matched_nodes, except for output_values, are used only by the matched_nodes."""
    # * Must check that all values matched by pattern are used only by pattern,
    # except for the value that is replaced.
    # * Must ensure that replacement subgraph does not use any of the deleted
    # (intermediate) values. (Not necessary for now. Guaranteed.)
    for n in matched_nodes:
        for v in n.outputs:
            if v in output_values:
                continue
            if v.is_graph_output():
                # value is an output-value of the graph/function.
                return False
            for consumer, _ in v.uses():
                if consumer not in matched_nodes:
                    return False
    return True


class PatternMatcher(abc.ABC):
    def __init__(self, pattern: _pattern_ir.GraphPattern) -> None:
        self.pattern = pattern

    @abc.abstractmethod
    def match(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        node: ir.Node,
        *,
        verbose: int = 0,
        remove_nodes: bool = True,
        tracer: _basics.MatchingTracer | None = None,
    ) -> _basics.MatchResult:
        """Match the pattern against the subgraph ending at the given node."""

    def __str__(self) -> str:
        return str(self.pattern)


class SimplePatternMatcher(PatternMatcher):
    def __init__(self, pattern: _pattern_ir.GraphPattern) -> None:
        super().__init__(pattern)
        self._current_node: ir.Node | None = None

    def fail(self, reason: str, node: ir.Node | None = None) -> bool:
        if self._verbose:
            num_matched_nodes = self._match.num_matched_nodes()
            if num_matched_nodes > 0:  # Print only if at least one node successfully matched.
                print(f"Match failed after {num_matched_nodes} nodes: {reason}")
        self._match.fail(reason, node or self._current_node)
        return False

    def _match_constant(self, pattern_constant: _pattern_ir.Constant, value: ir.Value) -> bool:
        """Match a Constant pattern against a value.

        If the constant value is produced by a Constant node, we do not include
        the constant node as part of the matched graph. Thus, it will not be deleted,
        if subgraph replacement happens. But subsequent DCE will remove the constant
        node if it is not used elsewhere.
        """
        constant_value = value.const_value
        if constant_value is None:
            return self.fail(
                f"Value {value.name} is not a constant, expecting {pattern_constant.value}.",
            )

        try:
            constant_value_numpy = constant_value.numpy()
        except FileNotFoundError:
            return self.fail(f"Constant value of {value.name} not available.")

        pattern_constant_value = pattern_constant._value

        if isinstance(pattern_constant_value, list):
            expected_shape = (len(pattern_constant_value),)
            if constant_value_numpy.shape != expected_shape:
                return self.fail(f"Value has mismatching shape, expecting {expected_shape}.")
            if not all(
                math.isclose(
                    constant_value_numpy.item(i),
                    pattern_constant_value[i],
                    rel_tol=pattern_constant._rel_tol,
                    abs_tol=pattern_constant._abs_tol,
                )
                for i in range(len(pattern_constant_value))
            ):
                return self.fail(
                    f"Value mismatch: expected {pattern_constant_value}, got {constant_value_numpy}."
                )
            return True

        # TODO (rama): allow users to specify shape requirement, if desired.
        if constant_value_numpy.size != 1:
            return self.fail(
                f"Value {value.name} is not a scalar, expecting {pattern_constant_value}.",
            )

        if not math.isclose(
            constant_value_numpy.item(),
            pattern_constant_value,
            rel_tol=pattern_constant._rel_tol,
            abs_tol=pattern_constant._abs_tol,
        ):
            return self.fail(
                f"Constant value mismatch: expected {pattern_constant_value}, got {constant_value_numpy.item()}.",
            )

        return True

    def _match_node(self, pattern_node: _pattern_ir.NodePattern, node: ir.Node) -> bool:
        """Matches a pattern subgraph against subgraph rooted at node."""
        self._current_node = node
        # Graph-matching: we do not allow the same pattern node to be matched against
        # different graph nodes.
        matched_node = self._match.lookup_node(pattern_node)
        if matched_node is not None:
            if matched_node is not node:
                return self.fail("Same pattern node is matched against different graph nodes.")
            return True
        match = self._match
        if not pattern_node.matches(node, match):
            return self.fail(match.reason)

        if self._verbose:
            print(f"Matched: {node.op_type}")

        match.bind_node(pattern_node, node)

        # TODO: Revisit this to handle optional trailing inputs better.
        if pattern_node.allow_other_inputs:
            if len(node.inputs) < len(pattern_node.inputs):
                return self.fail(
                    f"Number of inputs ({len(node.inputs)}) is less than expected ({len(pattern_node.inputs)})"
                )
        else:
            if len(node.inputs) != len(pattern_node.inputs):
                return self.fail(
                    f"Input nums mismatch. {len(node.inputs)} vs {len(pattern_node.inputs)}"
                )

        for arg_value, arg_pattern in zip(node.inputs, pattern_node.inputs):
            # arg_pattern could be a Var, if it's the original arg.
            if arg_pattern is None:
                if arg_value is None:
                    continue
                else:
                    return self.fail("(Optional) input is expected to be None but is not.")
            if not self._match_value(arg_pattern, arg_value):
                return False

        for i, output_value_pattern in enumerate(pattern_node.outputs):
            if not self._match.bind_value(output_value_pattern, node.outputs[i]):
                return False

        return True

    def _match_value(
        self, pattern_value: _pattern_ir.ValuePattern, value: ir.Value | None
    ) -> bool:
        """Match an IR value against a ValuePattern instance."""
        if isinstance(pattern_value, _pattern_ir.AnyValue):
            return True

        if not self._match.bind_value(pattern_value, value):
            return False

        if isinstance(pattern_value, _pattern_ir.NodeOutputPattern):
            if value is None:
                return self.fail("Mismatch: Computed node pattern does not match None.")
            return self._match_node_output(pattern_value, value)
        if isinstance(pattern_value, _pattern_ir.Constant):
            if value is None:
                return self.fail("Mismatch: Constant pattern does not match None.")
            return self._match_constant(pattern_value, value)
        if isinstance(pattern_value, _pattern_ir.BacktrackingOr):
            for i, pattern_choice in enumerate(pattern_value._values):
                self._match.enter_new_match()
                if self._match_value(pattern_choice, value):
                    if pattern_value.tag_var is not None:
                        self._match.bind(pattern_value.tag_var, pattern_value._tag_values[i])
                    self._match.merge_current_match()
                    return True
                self._match.abandon_current_match()
            return self.fail("None of the alternatives matched.")
        if isinstance(pattern_value, _pattern_ir.OpIdDispatchOr):
            if value is None:
                return self.fail("Mismatch: OrValue pattern does not match None.")
            alternative = pattern_value.get_pattern(value)
            if alternative is None:
                return self.fail("Mismatch: OrValue pattern does not match value.")
            i, pattern_choice = alternative
            result = self._match_value(pattern_choice, value)
            if result:
                if pattern_value.tag_var is not None:
                    self._match.bind(pattern_value.tag_var, i)
            return result
        return True

    def _match_node_output(
        self, pattern_value: _pattern_ir.NodeOutputPattern, value: ir.Value
    ) -> bool:
        """Match an IR value against a NodeOutputPattern instance."""
        node = value.producer()
        if node is None:
            return self.fail(
                "Mismatch: Computed node pattern does not match uncomputed IR value."
            )
        if value.index() != pattern_value.output_index:
            return self.fail(
                f"Node output index mismatch: expected {pattern_value._output_index}, got {value.index()}."
            )
        return self._match_node(pattern_value.producer(), node)

    def _init_match(self, verbose: int) -> None:
        """Initialize the match state. Invoked before starting a new match."""
        self._verbose = verbose
        self._match: _basics.MatchResult = _basics.MatchResult()
        self._current_node = None

    def _get_output_values(self) -> list[ir.Value] | None:
        """Get values bound to the output variables of the pattern."""
        output_values: list[ir.Value] = []
        unbound_values: list[str] = []
        for j, value_pattern in enumerate(self.pattern.outputs):
            if value_pattern.name is not None:
                if value_pattern.name in self._match.bindings:
                    output_values.append(self._match.bindings[value_pattern.name])
                else:
                    unbound_values.append(value_pattern.name)
            else:
                if value_pattern in self._match.value_bindings:
                    output_values.append(self._match.value_bindings[value_pattern])
                else:
                    unbound_values.append(f"output_{j}")
        if unbound_values:
            self._match.fail(f"Error: Output values not found: {unbound_values}")
            return None
        return output_values

    def _match_single_output_node(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        node: ir.Node,
        check_removable: bool,
    ) -> _basics.MatchResult:
        del model
        del graph_or_function

        pattern = self.pattern
        match = self._match

        if not pattern.has_single_output_node:
            return match.fail(
                "Internal Error: SimplePatternMatcher should not be used for patterns with multiple output nodes."
            )

        if not self._match_node(pattern.output_node, node):
            return match

        output_values = self._get_output_values()
        if output_values is None:
            # TODO(rama): Is this a valid (useful) case?
            return match
        if check_removable and not _valid_to_replace(match.nodes, output_values):
            # TODO(rama): Match status should be updated to reflect failure reason.
            return match.fail("Matched nodes have other uses preventing replacement.")

        match.outputs.extend(output_values)
        return match

    def _multi_match(
        self, candidate: Iterable[ir.Node], check_removable: bool
    ) -> _basics.MatchResult:
        """Find a match for a pattern with multiple output nodes.

        For a pattern with K output nodes, the input candidate should specify K nodes
        in the graph that will be matched against the pattern output nodes.

        Args:
            candidate: An iterable of nodes that will be matched against the pattern output nodes.
            check_removable: If True, check that the matched nodes can be removed (that is, that
                they are not used elsewhere in the graph).
        """
        match = self._match
        for pattern_node, node in zip(self.pattern.output_nodes, candidate):
            if not self._match_node(pattern_node, node):
                return match
        output_values = self._get_output_values()
        if output_values is None:
            return match

        if check_removable and not _valid_to_replace(match.nodes, output_values):
            return match.fail("Matched nodes have other uses preventing replacement.")

        match.outputs.extend(output_values)
        return match

    def match(
        self,
        model: ir.Model,
        graph_or_function: ir.Graph | ir.Function,
        node: ir.Node,
        *,
        verbose: int = 0,
        remove_nodes: bool = True,
        tracer: _basics.MatchingTracer | None = None,
    ) -> _basics.MatchResult:
        """Match the pattern against the subgraph ending at the given node.

        For patterns with multiple output nodes, the given node is matched
        against the first output node in the pattern. For the remaining
        output nodes in the pattern, we use a brute-force algorithm that
        enumerates all possible combinations of nodes from the graph (with
        a filter based on op-type).

        TODO: Consider omitting parameters model and graph_or_function. With
        the new IR, the graph can be obtained from the node, and the model is
        not used. But this is a shared abstract method of the Matcher interface,
        so other matcher implementation also needs to be updated. More importantly,
        matching in the presence of subgraphs (control-flow) can introduce some
        complications which require careful consideration.
        """
        self._tracer = tracer
        if self.pattern.has_single_output_node:
            self._init_match(verbose)
            return self._match_single_output_node(
                model, graph_or_function, node, check_removable=remove_nodes
            )
        else:
            # Note: This is a potentially expensive algorithm for matching patterns with
            # multiple output nodes. For patterns with N output nodes, we try all possible
            # combinations of N nodes from the graph, and check if they match the pattern.
            # The first node is fixed to the node argument in this method call. We do
            # some simple filtering by restricting the candidates for each remaining
            # output nodes to graph nodes with the same op_type as the corresponding pattern
            # node. For now, this is intended to be a simple, but robust, implementation
            # that can be used for debugging and testing. The GenericPatternMatcher is a
            # more sophisticated implementation, but incomplete.
            pattern_output_nodes = self.pattern.output_nodes
            op_to_nodes: dict[tuple[str, str, str], list[ir.Node]] = {}
            for n in graph_or_function:
                op_to_nodes.setdefault(n.op_identifier(), []).append(n)
            all_nodes = iter(graph_or_function)

            def get_nodes(pattern_node):
                id = pattern_node.op_identifier()
                if id is None:
                    return all_nodes
                return op_to_nodes.get(id, [])

            candidates = [iter([node])] + [get_nodes(pn) for pn in pattern_output_nodes[1:]]
            match = None
            for combination in itertools.product(*candidates):
                self._init_match(verbose)
                match = self._multi_match(combination, check_removable=remove_nodes)
                if match:
                    return match
            if match is None:
                return _basics.MatchResult().fail("No match found.")
            return match
