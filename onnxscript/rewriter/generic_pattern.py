from __future__ import annotations

import collections
import inspect
import os
import textwrap
from typing import Any, Callable, Iterator, Sequence

import onnx

import onnxscript
import onnxscript.rewriter.pattern as orp
from onnxscript import ir
from onnxscript.rewriter import _ir_utils


class PatternMatchResult:
    """Stores information about a match if a match was successful.

    * pattern: the instance of :class:`GenericPattern` which found this result
    * model_nodes: matched nodes coming from the model
    * pattern_nodes: corresponding nodes coming from the pattern
    * pattern_input_names: input names of the pattern
    * pattern_ouptut_names: output names of the pattern
    * kwargs: additional attributes the user may add through the method
        :meth:`PatternMatchResult.add_kwargs`

    The class creates one attributes `matched_pattern_to_model_name`,
    which maps every result name from the pattern to the corresponding
    result name in the model.
    """

    def __init__(
        self,
        pattern: GenericPattern,
        model_nodes: Sequence[ir.Node],
        pattern_nodes: Sequence[ir.Node],
        pattern_inputs: Sequence[ir.Value],
        pattern_outputs: Sequence[ir.Value],
    ):
        assert len(model_nodes) == len(pattern_nodes)
        self.pattern = pattern
        self.model_nodes = model_nodes
        self.pattern_nodes = pattern_nodes
        self.pattern_inputs = pattern_inputs
        self.pattern_outputs = pattern_outputs
        self.kwargs: dict[str, Any] = {}

        matched_pattern_to_model_value: dict[ir.Value, ir.Value] = {}
        for gn, pn in zip(model_nodes, pattern_nodes):
            assert (
                gn.op_type == pn.op_type
            ), f"Unexpected type mismatch {gn.op_type!r} != {pn.op_type!r}"
            assert len(gn.inputs) == len(
                pn.inputs
            ), f"Unexpected number of inputs for type {gn.op_type}"
            for a, b in zip(gn.inputs, pn.inputs):
                if b is None:
                    # optional input or not an interesting input
                    continue
                if b in matched_pattern_to_model_value:
                    assert matched_pattern_to_model_value[b] == a, (
                        f"Ambiguities, pattern input {b!r} means "
                        f"{a!r} or {matched_pattern_to_model_value[b]}"
                    )
                else:
                    assert b is not None
                    assert a is not None
                    matched_pattern_to_model_value[b] = a

            assert len(gn.outputs) == len(
                pn.outputs
            ), f"Unexpected number of outputs for type {gn.op_type}"
            for a, b in zip(gn.outputs, pn.outputs):
                if b in matched_pattern_to_model_value:
                    assert matched_pattern_to_model_value[b] == a, (
                        f"Ambiguities, pattern output {b!r} means "
                        f"{a!r} or {matched_pattern_to_model_value[b]}"
                    )
                else:
                    matched_pattern_to_model_value[b] = a

        self.matched_pattern_to_model_value = matched_pattern_to_model_value

    def add_kwargs(self, name: str, value: Any):
        """Adds an attribute, it can be done when the match is being validated,
        this attribute can be used when building the replacement nodes.
        """
        self.kwargs[name] = value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}([{self.pattern.__class__.__name__}], "
            f"... {len(self.model_nodes)} nodes ..., {self.pattern_inputs}, "
            f"{self.pattern_outputs})"
        )


class GenericRewriteRule(orp.RewriteRule):
    """
    Defines a rewriting rule.

        pattern: a pattern defines by :class:`GenericPattern`.
    """

    def __init__(self, pattern: GenericPattern):
        self.pattern = pattern
        self.verbose: int = 0  # TODO: remove this

    def matches(self, node: ir.Node, model: ir.Model) -> orp.MatchResult:
        del model
        del node
        raise RuntimeError(f"This pattern {self} is meant to replace not to only match.")

    def try_rewrite(
        self, model: ir.Model, graph_or_function: ir.Graph | ir.Function, node: ir.Node
    ) -> tuple[int, list[ir.Node], list[ir.Node]] | None:
        """See :meth:`RewriteRule.try_rewrite`."""

        deleted_nodes: list[ir.Node] = []
        added_nodes: list[ir.Node] = []
        matched: set[ir.Node] = set()
        match_count = 0
        for match_result in self.pattern.enumerate_matches(model.graph, node):
            conflict = False
            for node in match_result.model_nodes:
                if id(node) in matched:
                    conflict = True
                    break
            if conflict:
                # Some nodes are already matched as rewritten.
                continue

            # Let's build the new nodes
            if not self.pattern.validate_mapping(model, match_result):
                match_result._hint(
                    "validate_mapping", "The pattern was rejected by the validation function."
                )
                continue

            new_nodes = self.pattern.apply(model, match_result, verbose=self.verbose)
            assert all(
                isinstance(i, ir.Node) for i in new_nodes
            ), f"Unexpected types {[type(n) for n in new_nodes]}"

            # Everything is good.
            matched |= set(map(id, match_result.model_nodes))
            added_nodes.extend(new_nodes)
            deleted_nodes.extend(match_result.model_nodes)
            match_count += 1

        if match_count > 0:
            return match_count, deleted_nodes, added_nodes
        return None

    def count_matches(self, model: ir.Model, *, commute: bool = False) -> int:
        """See :meth:`RewriteRule.count_matches`."""
        raise NotImplementedError("Not supported yet.")

    def commute(self) -> list[orp.RewriteRule]:
        """See :meth:`RewriteRule.commute`."""
        raise RuntimeError("Not supported (yet?). It could lead to many patterns.")

    def apply_to_model(self, model: ir.Model, *, commute: bool = False) -> int:
        """See :meth:`RewriteRule.apply_to_model`."""
        return orp.RewriteRuleSet([self], commute=commute).apply_to_model(model)


class GenericPattern:
    """
    Implements a pattern optimization for quick experimentation.

    Current limitation:

    * The current implementation does match on domain name (easy fix).
    * It does not compares attributes either (easy fix as well).
    """

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self._cache: dict = {}

    def validate_mapping(self, g: ir.Model, match_result: PatternMatchResult) -> bool:
        """Evaluates the consistency of the replacements."""
        raise NotImplementedError(
            "This method could return True but it is better to let you know "
            "that it exists. You need to overwrite it to return True."
        )

    def enumerate_matches(
        self, graph: ir.Graph | ir.GraphView, node: ir.Node | None = None
    ) -> Iterator:
        """Enumerates all the matches."""
        if node is None:
            matched = []
            for node in graph:
                res = self.match(graph, node)
                if res:
                    matched.append(res)
                    yield res
        else:
            res = self.match(graph, node)
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
                    f"[{self.__class__.__name__}.match] NONE - line: {lineno}:"
                    f"{os.path.split(self.__class__.__module__)[-1]}, "
                    f"op_type={node.op_type}{msg}{msg2}"
                )

    @classmethod
    def match_pattern(
        cls,
        g: ir.Graph,
        *args: str,
        **kwargs: Any,
    ) -> Sequence[ir.Node] | None:
        """Builds the pattern to match."""
        raise NotImplementedError(
            f"Class {cls.__name__!r} must overwrite method match_pattern."
        )

    def _build_pattern(
        self,
        func: Callable | None = None,
        match: bool = True,
        kwargs: dict[str, Any] | None = None,
    ) -> ir.Graph:
        del match
        if func is None:
            raise NotImplementedError(
                f"Not implemented if func is None in class {self.__class__.__name__}"
            )
        if kwargs:
            raise NotImplementedError(
                f"Not implemented when kwargs is not empty but {kwargs} "
                f"in class {self.__class__.__name__}"
            )
        kwargs = {}
        args = []

        # There should be a better way.
        sig = inspect.signature(func)
        for i, p in enumerate(sig.parameters.values()):
            if i == 0:
                continue
            if p.default is not inspect._empty:
                # an attribute
                kwargs[p.name] = p.default
            else:
                args.append(p.name)

        assert len(kwargs) == 0, f"Attributes are not supported yet but kwargs={kwargs}"

        inputs = [ir.Input(name=name) for name in args]
        builder = orp.RewriterContext()
        outputs = func(builder, *inputs, **kwargs)
        if isinstance(outputs, ir.Value):
            outputs = [outputs]
        graph = ir.Graph(inputs=inputs, outputs=outputs, nodes=builder.nodes)
        graph.outputs[:] = outputs
        return graph

    def _get_match_pattern(self, g: ir.Graph | ir.GraphView) -> ir.Graph:
        cache_key = 0, tuple(sorted(g.opset_imports.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        pat = self._build_pattern(self.match_pattern, match=True)
        self._cache[cache_key] = pat
        return pat

    def print_match(self, n1: ir.Node, n2: ir.Node) -> str:
        s1 = f"{n1.op_type}({n1.inputs})"
        s2 = f"{n2.op_type}({n2.inputs})"
        return f"match {s1} with {s2} (pattern)"

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
            return f"{n.op_type}({', '.join([str(input) for input in n.inputs])})"

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
        node: ir.Node,
        matched: dict[ir.Node, ir.Node],
        stack: list[ir.Node],
        graph_node: ir.Node,
        pattern_node: ir.Node,
    ) -> int | None:
        """
        Matches backward.

        Args:
            node: root node (the node the matched begain with, used only for debugging)
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
            return self.none(node, inspect.currentframe().f_lineno)
        for i, pi in zip(graph_node.inputs, pattern_node.inputs):
            ppred = pi.producer()
            if ppred is None:
                # ppred is None means the pattern ends here.
                continue
            pred = i.producer()
            if pred is None:
                # No node in the graph.
                return self.none(node, inspect.currentframe().f_lineno)
            if pred.op_type != ppred.op_type:
                self._hint(
                    "BACKWARD: different node types",
                    "--pattern",
                    ppred,
                    "-- model",
                    pred,
                )
                return self.none(node, inspect.currentframe().f_lineno)
            # matching backward
            if ppred not in matched:
                if self.verbose >= 10:
                    print(f"[GenericPattern._match_backward] {self.print_match(pred, ppred)}")
                matched[ppred] = pred
                stack.append(ppred)
                match_count += 1
        if self.verbose > 5 and match_count > 0:
            print(f"[GenericPattern._match_backward] add {match_count} nodes")
        return match_count

    def _match_forward(
        self,
        root_node: ir.Node,
        matched: dict[ir.Node, ir.Node],
        stack: list[int],
        graph_node: ir.Node,
        pattern_node: ir.Node,
    ) -> int | None:
        """
        Matches forward.

        Args:
            root_node: root node (the node the match begins with, used only for debugging)
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
            return self.none(root_node, inspect.currentframe().f_lineno)

        for o, op in zip(graph_node.outputs, pattern_node.outputs):
            graph_node_users = [user for user, _ in o.uses()]
            pattern_node_users = [user for user, _ in op.uses()]
            if not pattern_node_users:
                # The pattern has no node forward, the matching stops.
                continue
            if len(graph_node_users) < len(pattern_node_users):
                # Not enough node in the graph to match the pattern. A match is not possible
                return self.none(root_node, inspect.currentframe().f_lineno)

            # Here comes the fun part, there is the same number of successors or more
            # nodes in the graph to match with the pattern.
            # And we have to handle the nodes already matched as found.
            # Hopefully, there is only one option.

            if len(graph_node_users) == len(pattern_node_users) == 1:
                # Let's deal with the simple case
                if graph_node_users[0].op_type != pattern_node_users[0].op_type:
                    return self.none(root_node, inspect.currentframe().f_lineno)

                node = pattern_node_users[0]
                if node not in matched:
                    if self.verbose >= 10:
                        print(
                            f"[GenericPattern._match_forward]{self.print_match(graph_node_users[0], pattern_node_users[0])}"
                        )
                    matched[node] = graph_node_users[0]
                    stack.append(node)
                    match_count += 1
                continue

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
            assert len(pattern_node_users_matched) + len(
                pattern_node_users_not_matched
            ) == len(pattern_node_users), (
                f"pattern_node_users_not_matched={pattern_node_users_not_matched}, "
                f"pattern_node_users_matched={pattern_node_users_matched}, "
                f"pattern_node_users={pattern_node_users}, "
                f"matched={matched}"
            )
            free = list(set(graph_node_users) - set(pattern_node_users_matched))
            if not pattern_node_users_not_matched:
                # Everything is already matched.
                continue
            if len(free) < len(pattern_node_users_not_matched):
                # Not enough successors to match the remaining patterns.
                return self.none(node, inspect.currentframe().f_lineno)
            if len(pattern_node_users_not_matched) == len(free) == 1:
                # Only one option again.
                graph_node = free[0]
                if pattern_node_users_not_matched[0].op_type != graph_node.op_type:
                    return self.none(node, inspect.currentframe().f_lineno)

                key = pattern_node_users_not_matched[0]
                if self.verbose >= 10:
                    print(
                        f"[GenericPattern._match_forward] {self.print_match(graph_node, pattern_node_users_not_matched[0])}"
                    )
                matched[key] = graph_node
                stack.append(key)
                match_count += 1
                continue

            # And now another fun part, let's try to handle the case when
            # there is only one option, matching on node type only returns one
            # option.
            expected_op_type = [_.op_type for _ in pattern_node_users_not_matched]
            got_op_type = [_.op_type for _ in free]

            ec = collections.Counter(expected_op_type)
            gc = collections.Counter(got_op_type)
            if len(ec) != len(gc) or set(ec) != set(gc):
                # unique operator types is different.
                self._hint(
                    "FORWARD: unique operator types are different",
                    "-- pattern",
                    ec,
                    pattern_node,
                    "-- model",
                    gc,
                    graph_node,
                    "-- model-matched",
                    pattern_node_users_matched,
                )
                return self.none(node, inspect.currentframe().f_lineno)
            for k, v in ec.items():
                if gc[k] < v:
                    # Not enough types to match.
                    return self.none(node, inspect.currentframe().f_lineno)

            # At this stage, we know matching the types is possible.
            # We first mark whatever is possible.
            ptype_to_node = {_.op_type: _ for _ in pattern_node_users_not_matched}
            gtype_to_node = {_.op_type: _ for _ in free}
            missing = []
            for k, v in ec.items():
                if gc[k] == v == 1:
                    key = id(ptype_to_node[k])
                    if key not in matched:
                        if self.verbose >= 10:
                            print(
                                f"[GenericPattern._match_forward] match "
                                f"{self.print_match(gtype_to_node[k], ptype_to_node[k])}"
                            )
                        matched[key] = gtype_to_node[k]
                        stack.append(key)
                        match_count += 1
                else:
                    missing.append(k)

            if not missing:
                continue

            # At this stage, there are mutiple options for matching. We can:
            # 1. make assumptions and continue
            # 2. mark the node as incomplete matching, we could end up stuck anyway.
            raise NotImplementedError(
                f"There are more than one option, this will be implemented later, "
                f"ec={ec}, gc={gc}"
            )
        if self.verbose > 5 and match_count > 0:
            print(f"[GenericPattern._match_forward] add {match_count} nodes")
        return match_count

    def match(
        self,
        g: ir.Graph | ir.GraphView,
        node: ir.Node,
    ) -> PatternMatchResult | None:
        self._debug = {}

        match_pattern: ir.Graph = self._get_match_pattern(g)

        # Let's match the last node.
        # Then we need to match successors and predecessors.
        last_pattern_node = match_pattern[-1]
        if node.op_type != last_pattern_node.op_type:
            # The last node does not have the same op_type.
            return self.none()

        if self.verbose > 5:
            print(f"[GenericPattern.match] starts with {node}")
            if self.verbose >= 10:
                print(f"[GenericPattern.match] match pattern {self!r}")

        all_pattern_nodes = set(match_pattern)
        matched: dict[ir.Node, ir.Node] = {last_pattern_node: node}
        stack: list[ir.Node] = [last_pattern_node]
        iteration = 0

        if self.verbose > 5:
            self._debug = dict(
                pattern=match_pattern,
                matched=matched,
                stack=stack,
                iteration=iteration,
                node=node,
                pattern_node=last_pattern_node,
                pattern_nodes=match_pattern,
            )

        max_iter = len(match_pattern) * 2
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
                    f"[GenericPattern.match] iteration={iteration} "
                    f"n_matched={len(matched)}, n_stack={len(stack)}, "
                    f"matched_types={collections.Counter(_.op_type for _ in matched)}"
                )
            pattern_node_from_stack = stack.pop()
            pattern_to_graph_node = matched[pattern_node_from_stack]

            result = self._match_backward(
                node, matched, stack, pattern_to_graph_node, pattern_node_from_stack
            )
            if result is None:
                if self.verbose > 5:
                    print("[GenericPattern.match] done. backward failed.")
                return result

            nodes_not_in_pattern = set(matched.keys()) - all_pattern_nodes
            assert (
                not nodes_not_in_pattern
            ), f"Some nodes are not part of the pattern: {nodes_not_in_pattern}"

            result = self._match_forward(
                node, matched, stack, pattern_to_graph_node, pattern_node_from_stack
            )
            if result is None:
                if self.verbose > 5:
                    print("[GenericPattern.match] done. forward failed.")
                return result

            nodes_not_in_pattern = set(matched.keys()) - all_pattern_nodes
            assert (
                not nodes_not_in_pattern
            ), f"Some nodes are not part of the pattern: {nodes_not_in_pattern}"

            if self.verbose > 5:
                self._debug["iteration"] = iteration

        if iteration >= max_iter and stack:
            self._hint("reached {iteration}>={max_iter} iterations")
            return self.none(node, inspect.currentframe().f_lineno)

        if self.verbose > 5:
            print(f"[GenericPattern.match] done. {len(matched)} matched nodes")

        # At this point, the pattern is matched but let's make sure.
        assert len(matched) == len(match_pattern), (
            f"Number of matched nodes is different, {len(matched)} matched nodes, "
            f"and {len(match_pattern)} nodes in the pattern, matched is {matched}"
        )
        assert len(stack) == 0, f"There are still {len(stack)} nodes to explore."

        # We order the matched nodes in the same order than the pattern
        # to let next functions to be able to build the matching again.
        matched_nodes = [matched[pattern_node] for pattern_node in match_pattern]
        return PatternMatchResult(
            self,
            matched_nodes,
            tuple(match_pattern),
            match_pattern.inputs,
            match_pattern.outputs,
        )

    @classmethod
    def apply_pattern(
        cls,
        g: ir.Model,
        *args: Any,
        **kwargs: Any,
    ) -> Sequence[ir.Node]:
        """Applies the replacement."""
        raise NotImplementedError(
            f"Class {cls.__name__!r} must overwrite method 'apply_pattern'."
        )

    def apply(
        self,
        model: ir.Model,
        match_result: PatternMatchResult,
        verbose: int = 0,
    ) -> Sequence[ir.Node]:
        assert isinstance(match_result, PatternMatchResult)
        new_pattern = self._build_pattern(
            self.apply_pattern, kwargs=match_result.kwargs, match=False
        )
        assert len(new_pattern.inputs) == len(match_result.pattern_inputs), (
            f"Not the same number of inputs, "
            f"matched inputs={len(new_pattern.inputs)}, "
            f"got {len(match_result.pattern_inputs)} in the applied pattern."
        )
        assert len(new_pattern.outputs) == len(match_result.pattern_outputs), (
            f"Not the same number of outputs, matched "
            f"outputs={match_result.pattern_outputs}, "
            f"got {new_pattern.outputs} in the applied pattern."
        )

        if verbose > 5:
            print(
                f"[GenericPattern.apply] replace {len(match_result.model_nodes)} nodes, "
                f"applied {(self.apply_pattern)}"
            )

        # TODO: handle initializers here
        # for name, init in pattern.initializers.items():
        #   # We add them to the graph, they will be removed if unused.
        #   new_name = g.make_initializer(name, init)
        #   replacements[new_name] = name

        applied_pattern_to_match_pattern: dict[ir.Value, ir.Value] = {}
        for i, j in zip(match_result.pattern_inputs, new_pattern.inputs):
            applied_pattern_to_match_pattern[j] = i
        for i, j in zip(match_result.pattern_outputs, new_pattern.outputs):
            applied_pattern_to_match_pattern[j] = i

        replacements: dict[ir.Value, ir.Value] = {}
        for k, v in applied_pattern_to_match_pattern.items():
            replacements[k] = match_result.matched_pattern_to_model_value[v]

        # Creation of the new node.
        new_nodes: list[ir.Node] = []
        for node in new_pattern:
            new_inputs: list[ir.Value] = []
            for i in node.inputs:
                assert i in replacements, f"Unable to find {i!r} in {replacements}"
                ni = replacements[i]
                new_inputs.append(ni)
            new_outputs: list[ir.Value] = []
            for o in node.outputs:
                if o in replacements:
                    new_outputs.append(replacements[o])
                else:
                    # We give it a new name.
                    replacements[o] = o
                    new_outputs.append(o)
            new_node = ir.Node(
                domain=node.domain,
                op_type=node.op_type,
                inputs=new_inputs,
                num_outputs=len(new_outputs),
                attributes=node.attributes.values(),
            )

            for old_output, new_output in zip(node.outputs, new_node.outputs):
                for i, graph_output in enumerate(old_output.producer().graph.outputs):
                    if old_output is graph_output:
                        new_output.meta[_ir_utils.GRAPH_OUTPUT_META_KEY] = i

            new_nodes.append(new_node)

        if verbose > 5:
            print(f"[GenericPattern.apply] done with {len(new_nodes)} nodes")

        return new_nodes

    def make_rule(self) -> orp.RewriteRule:
        """Creates the corresponding rule for this pattern."""
        return GenericRewriteRule(self)


class FunctionPattern(GenericPattern):
    """An instance of GenericPattern taking ir.Function.

    It defines the matching pattern and its replacement.

    Args:
        match_pattern: the onnx ir function defining the matching pattern
        apply_pattern: the onnx ir function defining the new pattern
        validate_mapping: the function used to validate a pattern
        verbose: in [0, 10], increase the verbosity to understand why a pattern
            does not match

    """

    def __init__(
        self,
        match_pattern: ir.Function,
        apply_pattern: ir.Function,
        validate_mapping,
        verbose: int = 0,
    ):
        self.match_pattern = match_pattern
        self.apply_pattern = apply_pattern
        self.validate_mapping = validate_mapping
        self.verbose = verbose

    def _build_pattern(self, pattern: Any, **kwargs) -> ir.Graph:
        return pattern

    def _get_match_pattern(self, *_, **__):
        return self.match_pattern


def make_pattern_rule(
    match_pattern: Callable | onnx.FunctionProto,
    apply_pattern: Callable | onnx.FunctionProto,
    validate_mapping: Callable | None = None,
    verbose: int = 0,
    opsets: dict[str, onnxscript.values.Opset] | None = None,
) -> orp.RewriteRule:
    """
    Creates a rewriting rule from a callable or a function proto.

    Args:
        match_pattern: a function interpreted by onnx-script
            and converted into an onnx model, this model defines the
            nodes to be replaced
        apply_pattern: a function interpreted by onnx-script and
            converted into an onnx model, this model defines the new nodes
            replacing the matched nodes
        validate_mapping: a function validating the matching once
            it has happened, it is not valid, the pattern is not applied,
            if not specified, the function always return True
        opsets: opset to consider when converting the function into ONNX,
            if not specified, it is opset 18 for the main opset, and opset 1
            for domain com.microsoft.
        verbose: verbosity level

    Returns:
        the rewriting rule
    """

    if opsets is None:
        opsets = dict(
            op=onnxscript.opset18, msft_op=onnxscript.values.Opset("com.microsoft", 1)
        )

    if not isinstance(apply_pattern, onnx.FunctionProto):
        apply_pattern = onnxscript.script(**opsets)(apply_pattern).to_function_proto()

    if not isinstance(match_pattern, onnx.FunctionProto):
        match_pattern = onnxscript.script(**opsets)(match_pattern).to_function_proto()

    match_function = ir.serde.deserialize_function(match_pattern)
    apply_function = ir.serde.deserialize_function(apply_pattern)

    pat = FunctionPattern(
        match_function,
        apply_function,
        validate_mapping or (lambda *_, **__: True),
        verbose=verbose,
    )
    return pat.make_rule()
