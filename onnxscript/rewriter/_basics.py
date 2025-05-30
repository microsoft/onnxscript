# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Basic types for the pattern matching and rewriter API."""

from __future__ import annotations

import dataclasses
import enum
from collections import defaultdict
from typing import TYPE_CHECKING, Any, MutableSequence, Sequence, Union

from onnxscript import ir

if TYPE_CHECKING:
    import onnxscript.rewriter._pattern_ir as _pattern_ir
    import onnxscript.rewriter._rewrite_rule as _rewrite_rule


class MatchResult:
    """The state object used by the pattern-matching algorithm.

    A match can either succeed or fail.
    If it succeeds, it returns a list of nodes that matched the pattern
    and a set of bindings for the variables in the pattern.

    Example:
    ::
        def pattern(x, shape1, shape2):
            t1 = op.Reshape(x, shape1)
            t2 = op.Reshape(t1, shape2)
            return t2
    The above pattern matches a sequence of two Reshape ops.
    The matched_nodes will contain the two Reshape ops, and the bindings will
    contain the values that are bound to the variables `x`, `shape1`, and `shape2`.
    """

    def __init__(self) -> None:
        # We use a stack of partial matches to handle OR patterns that require backtracking.
        self._partial_matches: list[PartialMatchResult] = [PartialMatchResult()]

    def __repr__(self) -> str:
        """Returns a string representation of the match result."""
        if not self._partial_matches:
            return "MatchResult()"
        return (
            f"MatchResult(success={bool(self)}, reason={self.reason!r}, nodes={self.nodes!r})"
        )

    @property
    def _current_match(self) -> PartialMatchResult:
        """Returns the current match result."""
        return self._partial_matches[-1]

    def enter_new_match(self) -> None:
        """Starts a new sub-match to try out one of multiple alternatives."""
        match = PartialMatchResult()
        self._partial_matches.append(match)

    def abandon_current_match(self) -> PartialMatchResult:
        """Abandons the current alternative due to failure."""
        if len(self._partial_matches) < 2:
            raise ValueError("No match to abandon.")
        return self._partial_matches.pop()

    def merge_current_match(self) -> None:
        """Merges a successful sub-match for an alternative with the parent one."""
        if len(self._partial_matches) < 2:
            raise ValueError("No match to merge.")
        current_match = self._partial_matches.pop()
        previous_match = self._partial_matches[-1]
        if not current_match:
            raise ValueError("Current match is not successful.")
        # Merge the two matches.
        previous_match.merge(current_match)

    def __bool__(self) -> bool:
        """Returns True if the current match is successful."""
        return bool(self._current_match)

    def fail(
        self,
        reason: str = "",
        failure_source: Union[ir.Node, ir.Value, list[Union[ir.Node, ir.Value]]] | None = None,
    ) -> MatchResult:
        self._current_match.fail(reason, failure_source)
        return self

    @property
    def reason(self) -> str:
        """Returns the reason for the failure."""
        return self._current_match.reason

    @property
    def nodes(self) -> Sequence[ir.Node]:
        """Returns the list of nodes that matched the pattern."""
        return self._current_match.nodes

    def bind_node(self, pattern_node: _pattern_ir.NodePattern, node: ir.Node):
        """Binds a pattern node to a matched node."""
        self.add_node(node)
        self._current_match.node_bindings[pattern_node] = node

    def add_node(self, node: ir.Node) -> None:
        """Adds a node to the list of matched nodes."""
        self._current_match.add_node(node)

    def bind_value(self, pattern_value: _pattern_ir.ValuePattern, value: Any) -> bool:
        var_name = pattern_value.name
        # TODO(rama): Simplify the following. We currently bind values to
        # pattern variables in two different ways: via their name, or via the
        # pattern-value itself.
        if var_name is None:
            for match in self._partial_matches:
                if pattern_value in match.value_bindings:
                    # TODO(rama): Use appropriate equality-check here.
                    if match.value_bindings[pattern_value] == value:
                        return True
                    self._current_match.fail(
                        f"Binding failure: {pattern_value} bound to two different values.",
                        [match.value_bindings[pattern_value], value],
                    )
                    return False
            self._current_match.value_bindings[pattern_value] = value
            return True
        return self.bind(var_name, value)

    def bind(self, var: str, value: Any) -> bool:
        for match in self._partial_matches:
            if var in match.bindings:
                # TODO(rama): Use appropriate equality-check here.
                if match.bindings[var] == value:
                    return True
                self._current_match.fail(
                    f"Binding failure: {var} bound to two different values.",
                    [match.bindings[var], value],
                )
                return False
        self._current_match.bindings[var] = value
        return True

    @property
    def bindings(self) -> dict[str, Any]:
        """Returns the bindings for the pattern variables."""
        if len(self._partial_matches) > 1:
            raise ValueError("Bindings can be accessed only at the top-level match.")
        return self._current_match.bindings

    @property
    def value_bindings(self) -> dict[_pattern_ir.ValuePattern, ir.Value]:
        """Returns the bindings for the value variables."""
        if len(self._partial_matches) > 1:
            raise ValueError("Value bindings can be accessed only at the top-level match.")
        return self._current_match.value_bindings

    @property
    def outputs(self) -> MutableSequence[ir.Value]:
        """Returns the list of output values that matched the pattern."""
        if len(self._partial_matches) > 1:
            raise ValueError("Outputs can be accessed only at the top-level match.")
        return self._current_match.outputs

    @property
    def failure_nodes_and_values(self) -> list[Union[ir.Node, ir.Value]]:
        """Returns the nodes and values that caused the failure."""
        return self._current_match._failure_nodes_and_values

    def lookup_node(self, pattern_node: _pattern_ir.NodePattern) -> ir.Node | None:
        """Looks up the node that matched the given pattern node."""
        for match in self._partial_matches:
            if pattern_node in match.node_bindings:
                return match.node_bindings[pattern_node]
        return None

    def num_matched_nodes(self) -> int:
        """Returns the number of nodes matched so far."""
        return sum(len(match.node_bindings) for match in self._partial_matches)


class PartialMatchResult:
    """The state object used by the pattern-matching algorithm for a sub-match."""

    def __init__(self) -> None:
        self._success: bool = True
        # For a successful match, _matched_nodes is a list of values that matched the pattern.
        # These include the internal nodes of the pattern that were matched, but not
        # the leaves (sub-trees) that match against the variables in the pattern.
        # These represent the values that will be replaced by the replacement pattern.
        self._matched_nodes: MutableSequence[ir.Node] = []
        # For a successful match, bindings is a dictionary of mapping pattern-variable-names
        # to values.
        self._bindings: dict[str, Any] = {}
        self._value_bindings: dict[_pattern_ir.ValuePattern, ir.Value] = {}
        self._node_bindings: dict[_pattern_ir.NodePattern, ir.Node] = {}

        self._outputs: list[ir.Value] = []
        # For a failed match, _reason is a string that describes the reason for the failure.
        self._reason: str = ""
        # Track the node(s) or value(s) that caused the failure.
        self._failure_nodes_and_values: list[Union[ir.Node, ir.Value]] = []

    def __bool__(self):
        return self._success

    def fail(
        self,
        reason: str = "",
        failure_source: Union[ir.Node, ir.Value, list[Union[ir.Node, ir.Value]]] | None = None,
    ) -> None:
        self._success = False
        self._reason = reason
        if failure_source is not None:
            if isinstance(failure_source, list):
                self._failure_nodes_and_values.extend(failure_source)
            else:
                self._failure_nodes_and_values.append(failure_source)

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def nodes(self) -> Sequence[ir.Node]:
        return tuple(self._matched_nodes)

    def add_node(self, node: ir.Node) -> None:
        """Adds a node to the list of matched nodes."""
        self._matched_nodes.append(node)

    @property
    def bindings(self) -> dict[str, Any]:
        return self._bindings

    @property
    def value_bindings(self) -> dict[_pattern_ir.ValuePattern, ir.Value]:
        return self._value_bindings

    @property
    def outputs(self) -> MutableSequence[ir.Value]:
        return self._outputs

    @property
    def node_bindings(self) -> dict[_pattern_ir.NodePattern, ir.Node]:
        return self._node_bindings

    def merge(self, other: PartialMatchResult) -> None:
        """Merges a successful sub-match for an alternative with the parent one."""
        if self._success and other._success:
            # Merge the two successful matches. Matching algorithm responsible for ensuring
            # that the two matches are compatible. No need to check for conflicts here.
            self._bindings.update(other._bindings)
            self._matched_nodes.extend(other.nodes)
            # Note: outputs should be set only at end of the (top-level) match. There
            # should be no outputs in the sub-match.
            assert not other._outputs
        else:
            # This should not happen currently.
            raise NotImplementedError("Merging failed matches is not yet supported.")


class MatchStatus(enum.IntEnum):
    """The status of a pattern-matching operation."""

    NO_MATCH = 0  # No successful match found for entire pattern graph
    CONDITION_FAILED = 1  # Subsequent validation check failed
    REPLACEMENT_FAILED = 2  # Replacement subgraph could not be created
    SUCCESS = 3  # A successful match was found


@dataclasses.dataclass
class MatchInfo:
    """The status of a pattern-matching operation. An extension of MatchResult."""

    match_result: MatchResult
    root_node: ir.Node
    container: ir.Graph | ir.Function
    status: MatchStatus

    def score(self) -> int:
        """Return a score for the match."""
        return len(self.match_result.nodes) + int(self.status.value) * 100

    def print(self):
        separator = "-" * 80
        print(separator)
        print(f"Status: {self.status.name}")
        if self.status != MatchStatus.SUCCESS:
            reason = self.match_result.reason
            if reason:
                if self.status == MatchStatus.CONDITION_FAILED:
                    print(f"Graph matching failed due to failing check condition : {reason}")
                else:
                    print(f"Graph matching failed: {reason}")
            else:
                print("Graph matching failed.")
            failure_nodes_and_values = self.match_result.failure_nodes_and_values
            print("Failure at or around nodes/values:")
            if failure_nodes_and_values:
                for failure_cause in failure_nodes_and_values:
                    failure_cause.display()
        print("Matched nodes:")
        import onnxscript.rewriter._ir_utils as ir_utils

        ir_utils.display_nodes(self.match_result.nodes)
        print(separator)


class MatchingTracer:
    """A debugging helper class to trace the matching of a pattern against a graph.

    This is used to track the best matches found for each rule, and to report the
    results at the end of the matching.
    """

    def __init__(self) -> None:
        self._best_matches_map: dict[_rewrite_rule.RewriteRule, list[MatchInfo]] = defaultdict(
            list
        )

    @property
    def best_matches_map(self) -> dict[_rewrite_rule.RewriteRule, list[MatchInfo]]:
        return self._best_matches_map

    def log(
        self,
        rule: _rewrite_rule.RewriteRule,
        container: ir.Graph | ir.Function,
        node: ir.Node,
        match_result: MatchResult,
        status: MatchStatus,
    ) -> None:
        this_match = MatchInfo(match_result, node, container, status)
        this_score = this_match.score()
        if this_score == 0:
            return
        best_matches = self._best_matches_map[rule]
        if best_matches:
            if this_score < best_matches[0].score():
                return
            if this_score > best_matches[0].score():
                best_matches.clear()
        best_matches.append(this_match)

    def report(self) -> None:
        best_score = 0
        for rule, matches in self._best_matches_map.items():
            if not matches:
                continue
            if matches[0].score() > best_score:
                best_score = matches[0].score()
                best_match = matches[0]
                best_rule = rule

        if best_score > 0:
            print(f"Rule: {best_rule}")
            best_match.print()
        else:
            print("No matches found.")
