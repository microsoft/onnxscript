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
from onnxscript.ir import serde
from onnxscript.rewriter import _tape


class _SimpleBuilder:
    """temporary adaptor for building 'generic patterns'."""

    # TODO(justinchuby): Merge with the rest of pattern building methods
    def __init__(self):
        self.tape = _tape.Tape()

    def __getattr__(self, op_type: str) -> Any:
        return lambda *args, **kwargs: self._make_node(op_type, args, kwargs)

    def _make_node(self, op_type: int, inputs: Sequence[ir.Value], kwargs: dict[str, Any]):
        domain = kwargs.pop("domain", "")
        output_names = kwargs.pop("output_names", 1)
        if isinstance(output_names, Sequence):
            num_outputs = len(output_names)
        else:
            assert isinstance(output_names, int)
            num_outputs = output_names
        if num_outputs == 1:
            return self.tape.op(op_type, inputs=inputs, attributes=kwargs, domain=domain)
        return self.tape.op_multi_output(
            op_type, inputs=inputs, attributes=kwargs, domain=domain, num_outputs=num_outputs
        )

    @property
    def nodes(self) -> Sequence[ir.Node]:
        return self.tape.nodes


def enumerate_subgraphs(
    node: ir.Node,
) -> Iterator[tuple[Any, ...]]:
    """Returns the subgraphs inside a graph."""
    for att in node.attributes.values():
        # TODO: improve this
        att = serde.serialize_attribute(att)
        if att.type == onnx.AttributeProto.GRAPH and att.g:
            this = node, att.name, att.g
            yield this

            for no in att.g.node:
                for tu in enumerate_subgraphs(no):
                    yield this + tu


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
        pattern_input_names: Sequence[str],
        pattern_output_names: Sequence[str],
    ):
        assert len(model_nodes) == len(pattern_nodes)
        self.pattern = pattern
        self.model_nodes = model_nodes
        self.pattern_nodes = pattern_nodes
        self.pattern_input_names = pattern_input_names
        self.pattern_output_names = pattern_output_names
        self.kwargs = {}

        matched_pattern_to_model_name: dict[str, str] = {}
        for gn, pn in zip(model_nodes, pattern_nodes):
            assert (
                gn.op_type == pn.op_type
            ), f"Unexpected type mismatch {gn.op_type!r} != {pn.op_type!r}"
            assert len(gn.input_names) == len(
                pn.input_names
            ), f"Unexpected number of inputs for type {gn.op_type}"
            for a, b in zip(gn.input_names, pn.input_names):
                if b == "":
                    # optional input or not an interesting input
                    continue
                if b in matched_pattern_to_model_name:
                    assert matched_pattern_to_model_name[b] == a, (
                        f"Ambiguities, pattern name {b!r} means "
                        f"{a!r} or {matched_pattern_to_model_name[b]}"
                    )
                else:
                    matched_pattern_to_model_name[b] = a

            assert len(gn.output_names) == len(
                pn.output_names
            ), f"Unexpected number of outputs for type {gn.op_type}"
            for a, b in zip(gn.output_names, pn.output_names):
                if b == "":
                    # Only final outputs are interesting.
                    continue
                assert a != "", f"{a!r} cannot be optional"
                if b in matched_pattern_to_model_name:
                    assert matched_pattern_to_model_name[b] == a, (
                        f"Ambiguities, pattern name {b!r} means "
                        f"{a!r} or {matched_pattern_to_model_name[b]}"
                    )
                else:
                    matched_pattern_to_model_name[b] = a

        self.matched_pattern_to_model_name = matched_pattern_to_model_name

    def add_kwargs(self, name: str, value: Any):
        """Adds an attribute, it can be done when the match is being validated,
        this attribute can be used when building the replacement nodes.
        """
        self.kwargs[name] = value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}([{self.pattern.__class__.__name__}], "
            f"... {len(self.model_nodes)} nodes ..., {self.pattern_input_names}, "
            f"{self.pattern_output_names})"
        )


class GenericRewriteRule(orp.RewriteRule):
    """
    Defines a rewriting rule.

    :param pattern: a pattern defines by :class:`GenericPattern`.
    """

    def __init__(self, pattern: GenericPattern):
        self.pattern = pattern
        self.verbose: int = 0  # TODO: remove this

    def matches(self, node: ir.Node, model: ir.Model) -> orp.MatchResult:
        del model
        del node
        raise RuntimeError(f"This pattern {self} is meant to replace not to only match.")

    def try_rewrite(
        self, model: ir.Model, node: ir.Node
    ) -> tuple[int, list[ir.Node], list[ir.Node]] | None:
        """See :meth:`RewriteRule.try_rewrite`."""

        deleted_nodes = []
        added_nodes = []
        marked = set()
        matched = 0
        for match_result in self.pattern.enumerate_matches(model, node):
            conflict = False
            for node in match_result.model_nodes:
                if id(node) in marked:
                    conflict = True
                    break
            if conflict:
                # Some nodes are already marked as rewritten.
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
            marked |= set(map(id, match_result.model_nodes))
            added_nodes.extend(new_nodes)
            deleted_nodes.extend(match_result.model_nodes)
            matched += 1

        if matched > 0:
            return matched, deleted_nodes, added_nodes
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

    def enumerate_matches(self, g: ir.Graph, node: ir.Node | None = None) -> Iterator:
        """Enumerates all the matches."""
        if node is None:
            matched = []
            for node in g.nodes:
                res = self.match(g, node)
                if res:
                    matched.append(res)
                    yield res
        else:
            res = self.match(g, node)
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
        If logs are fully enabled, it shows informations about matched none
        and the line deciding the matched failed.
        For example, this tells the matching failed at line 601 in ``generic_pattern.py``.
        It happens when propagating the match in the backward directions.
        The unmatched types are Mul, MatMul and below,
        it shows the matched nodes. The first one was Cast.
        And the failure happened at iteration 5.
        ``139774002356544-139774000632672`` is the pair of ids used in container ``marked``.
        ``id(node)`` is used as a unique identifiers of the nodes.

        ::

            [RotaryEmbeddingPattern.match] NONE - line: 601:__main__, op_type=Cast
                --hint--: BACKWARD: different node types
                --pattern
                Mul(pos_ids, cast) -> (mul)
                -- model
                MatMul(/_original_modu...Expand_output_0, /_original_modu...b/Cast_output_0) -> (/_original_modu...MatMul_output_0)
                iteration=5
                --marked-- #6
                Cast(/_original_modu...mb/Cos_output_0) ~ Cast(cos) [139774002356544-139774000632672]
                Cos(/_original_modu...ncat_1_output_0) ~ Cos(concattraining-transpose-0) [139774002356448-139774000632048]
                ConcatTraining(/_original_modu...nspose_output_0,/_original_modu...nspose_output_0) ~ ConcatTraining(transpose,transpose) [139774002356352-139774000631712]
                Transpose(/_original_modu...MatMul_output_0) ~ Transpose(mul) [139774002356256-139774000631184]
                Sin(/_original_modu...ncat_1_output_0) ~ Sin(concattraining-transpose-0) [139774002358512-139774000631568]
                Cast(/_original_modu...mb/Sin_output_0) ~ Cast(sin) [139774002358608-139774000632384]
                len(stacked)=0:[]

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
    ) -> list[ir.Node] | None:
        """Builds the pattern to match."""
        raise NotImplementedError(
            f"Class {cls.__name__!r} must overwrite method match_pattern."
        )

    def _build_pattern(
        self,
        fct: Callable | None = None,
        match: bool = True,
        kwargs: dict[str, Any] | None = None,
    ) -> ir.Graph:
        del match
        assert fct, f"Not implemented if fct is None in class {self.__class__.__name__}"
        assert not kwargs, (
            f"Not implemented when kwargs is not empty but {kwargs} "
            f"in class {self.__class__.__name__}"
        )
        kwargs = {}
        args = []

        # There should be a better way.
        sig = inspect.signature(fct)
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
        builder = _SimpleBuilder()
        outputs = fct(builder, *inputs, **kwargs)
        if isinstance(outputs, ir.Value):
            outputs = [outputs]
        graph = ir.Graph(inputs=inputs, outputs=outputs, nodes=builder.nodes)
        graph.outputs[:] = outputs
        return graph

    def _get_match_pattern(self, g: ir.Graph) -> ir.Graph:
        cache_key = 0, tuple(sorted(g.opset_imports.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        pat = self._build_pattern(fct=self.match_pattern, match=True)
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
            if isinstance(n, (ir.Node, onnx.NodeProto)):
                if full:
                    return (
                        f"{n.op_type}({', '.join(map(_s, n.inputs))}) "
                        f"-> ({', '.join(map(_s, n.inputs))})"
                    )
                return f"{n.op_type}({','.join(map(_s, n.inputs))})"
            return str(n)

        rows = []
        for k, v in sorted(self._debug.items()):
            if k == "stacked":
                rows.append(f"len({k})={len(v)}:{v}")
                continue
            if k == "iteration":
                rows.append(f"{k}={v}")
                continue
            if k == "marked":
                rows.append(f"--marked-- #{len(v)}")
                for i, tu in v.items():
                    rows.append(f"  {_p(tu[0])} ~ {_p(tu[1])} [{id(tu[0])}-{i}]")
                continue
            if k == "hint":
                rows.append(f"--hint--: {v[0]}")
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
        g: ir.Graph,
        node: ir.Node,
        pat: ir.Graph,
        marked: dict[int, tuple[ir.Node, ir.Node]],
        stacked: list[int],
        n: ir.Node,
        pn: ir.Node,
    ) -> int | None:
        """
        Matches backward.

        :param g: graph
        :param node: root node (the node the matched begain with,
            used only for debugging)
        :param pat: pattern
        :param marked: nodes of the pattern marked as already matched
        :param stacked: next node to look into
        :param n: node coming from the graph
        :param pn: node coming from the pattern
        :return: number of matched nodes, None or False to indicate a failed match
        """
        res = 0

        # predecessors
        if len(n.inputs) != len(pn.inputs):
            # not the same number of inputs
            self._hint(
                "BACKWARD: not the same number of inputs",
                "-- pattern",
                pn,
                "-- model",
                n,
            )
            return self.none(node, inspect.currentframe().f_lineno)
        for i, pi in zip(n.inputs, pn.inputs):
            ppred = pi.def_node()
            if ppred is None:
                # ppred is None means the pattern ends here.
                continue
            pred = i.def_node()
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
            key = id(ppred)
            if key not in marked:
                if self.verbose >= 10:
                    print(f"[GenericPattern._match_backward] {self.print_match(pred, ppred)}")
                marked[key] = pred, ppred
                stacked.append(key)
                res += 1
        if self.verbose > 5 and res > 0:
            print(f"[GenericPattern._match_backward] add {res} nodes")
        return res

    def _match_forward(
        self,
        g: ir.Graph,
        node: ir.Node,
        pat: ir.Graph,
        marked: dict[int, tuple[ir.Node, ir.Node]],
        stacked: list[int],
        n: ir.Node,
        pn: ir.Node,
    ) -> int | None:
        """
        Matches forward.

        :param g: graph
        :param node: root node (the node the matched begain with,
            used only for debugging)
        :param pat: pattern
        :param marked: nodes of the pattern marked as already matched
        :param stacked: next node to look into
        :param n: node coming from the graph
        :param ns: node coming from the pattern
        :return: number of matched nodes to continue, None or False to indicate a failed match
        """
        res = 0

        # successors
        if len(n.outputs) != len(pn.outputs):
            # not the same number of outputs
            self._hint(
                "FORWARD: not the same number of output_names",
                "-- pattern",
                pn,
                "-- model",
                n,
            )
            return self.none(node, inspect.currentframe().f_lineno)

        for o, op in zip(n.outputs, pn.outputs):
            ns = o.users()
            pns = op.users()
            if len(pns) == 0:
                # The pattern has no node forward, the matching stops.
                continue
            if len(ns) < len(pns):
                # Not enough node in the graph to match the pattern,
                # the result is known.
                return self.none(node, inspect.currentframe().f_lineno)

            # Here comes the fun part, there is the same number of successors or more
            # nodes in the graph to match with the pattern.
            # And we have to handle the nodes already marked as found.
            # Hopefully, there is only one option.

            if len(ns) == len(pns) == 1:
                # Let's deal with the simple case
                if ns[0].op_type != pns[0].op_type:
                    return self.none(node, inspect.currentframe().f_lineno)

                key = id(pns[0])
                if key not in marked:
                    if self.verbose >= 10:
                        print(
                            f"[GenericPattern._match_forward]{self.print_match(ns[0], pns[0])}"
                        )
                    marked[key] = ns[0], pns[0]
                    stacked.append(key)
                    res += 1
                continue

            # Let's remove the nodes already marked.
            p_marked = [_ for _ in pns if id(_) not in marked]
            id_marked = [id(marked[id(_)][0]) for _ in pns if id(_) in marked]
            assert len(id_marked) + len(p_marked) == len(pns), (
                f"Unexpected, id_marked={id_marked}, "
                f"id_p_marked={set(map(id, p_marked))}, "
                f"pns_ids={set(map(id, pns))}, "
                f"ns_ids={set(map(id, ns))}, o={o!r}, op={op!r}, "
                f"n.op_type={n.op_type!r}, "
                f"n.output={n.output}, np.output={pn.output}, "
                f"ns_types={ {_.op_type for _ in ns} }, "
                f"pns_types={ {_.op_type for _ in pns} }"
            )
            free = [_ for _ in ns if id(_) not in id_marked]
            if len(p_marked) == 0:
                # Everything is already marked.
                continue
            if len(free) < len(p_marked):
                # Not enough successors to match the remaining patterns.
                return self.none(node, inspect.currentframe().f_lineno)
            if len(p_marked) == len(free) == 1:
                # Only one option again.
                if p_marked[0].op_type != free[0].op_type:
                    return self.none(node, inspect.currentframe().f_lineno)

                key = id(p_marked[0])
                if key not in marked:
                    if self.verbose >= 10:
                        print(
                            f"[GenericPattern._match_forward] {self.print_match(free[0], p_marked[0])}"
                        )
                    marked[key] = free[0], p_marked[0]
                    stacked.append(key)
                    res += 1
                continue

            # And now another fun part, let's try to handle the case when
            # there is only one option, matching on node type only returns one
            # option.
            expected_op_type = [_.op_type for _ in p_marked]
            got_op_type = [_.op_type for _ in free]

            ec = collections.Counter(expected_op_type)
            gc = collections.Counter(got_op_type)
            if len(ec) != len(gc) or set(ec) != set(gc):
                # unique operator types is different.
                self._hint(
                    "FORWARD: unique operator types are different",
                    "-- pattern",
                    ec,
                    pn,
                    "-- model",
                    gc,
                    n,
                    "-- model-marked",
                    id_marked,
                )
                return self.none(node, inspect.currentframe().f_lineno)
            for k, v in ec.items():
                if gc[k] < v:
                    # Not enough types to match.
                    return self.none(node, inspect.currentframe().f_lineno)

            # At this stage, we know matching the types is possible.
            # We first mark whatever is possible.
            ptype_to_node = {_.op_type: _ for _ in p_marked}
            gtype_to_node = {_.op_type: _ for _ in got_op_type}
            missing = []
            for k, v in ec.items():
                if gc[k] == v == 1:
                    key = id(ptype_to_node[k])
                    if key not in marked:
                        if self.verbose >= 10:
                            print(
                                f"[GenericPattern._match_forward] match "
                                f"{self.print_match(gtype_to_node[k], ptype_to_node[k])}"
                            )
                        marked[key] = gtype_to_node[k], ptype_to_node[k]
                        stacked.append(key)
                        res += 1
                else:
                    missing.append(k)

            if not missing:
                continue

            # At this stage, there are mutiple options for matching. We can:
            # 1. make assumptions and continue
            # 2. mark the node as incomplete matching, we could end up stuck anyway.
            raise AssertionError(
                f"There are more than one option, this will be implemented later, "
                f"ec={ec}, gc={gc}"
            )
        if self.verbose > 5 and res > 0:
            print(f"[GenericPattern._match_forward] add {res} nodes")
        return res

    def match(
        self,
        g: ir.Graph,
        node: ir.Node,
    ) -> PatternMatchResult | None:
        self._debug = {}

        pat = self._get_match_pattern(g)

        # Let's match the last node.
        # Then we need to match successors and predecessors.
        p_node = pat.nodes[-1]  # the last one
        if node.op_type != p_node.op_type:
            # The last node does not have the same type.
            return self.none()

        check_ids = {id(n) for n in pat.nodes}
        if self.verbose > 5:
            print(f"[GenericPattern.match] starts with {node.op_type}({node.inputs})")
            if self.verbose >= 10:
                print(f"[GenericPattern.match] match pattern {self!r}")

        marked = {id(p_node): (node, p_node)}
        stacked = [id(p_node)]
        iteration = 0

        if self.verbose > 5:
            self._debug = dict(
                pattern=pat,
                marked=marked,
                stacked=stacked,
                iteration=iteration,
                node=node,
                pattern_node=p_node,
                pattern_nodes=pat.nodes,
            )

        max_iter = len(pat.nodes) * 2
        while stacked and iteration < max_iter:
            assert all(id(b[1]) in check_ids for b in marked.values()), (
                f"At least one id is not part of the pattern ids={check_ids}, "
                f"marked={ {id(b[1]) for b in marked.values()} }"
            )

            iteration += 1
            if self.verbose > 5:
                print(
                    f"[GenericPattern.match] iteration={iteration} "
                    f"n_marked={len(marked)}, n_stacked={len(stacked)}, "
                    f"marked_types={collections.Counter(_[1].op_type for _ in marked.values())}"
                )
            idn = stacked.pop()
            n, pn = marked[idn]

            res = self._match_backward(g, node, pat, marked, stacked, n, pn)
            if res is None:
                if self.verbose > 5:
                    print("[GenericPattern.match] done. backward failed.")
                return res

            assert all(id(b[1]) in check_ids for b in marked.values()), (
                f"At least one id is not part of the pattern ids={check_ids}, "
                f"marked={ {id(b[1]) for b in marked.values()} }"
            )

            res = self._match_forward(g, node, pat, marked, stacked, n, pn)
            if res is None:
                if self.verbose > 5:
                    print("[GenericPattern.match] done. forward failed.")
                return res

            assert all(id(b[1]) in check_ids for b in marked.values()), (
                f"At least one id is not part of the pattern ids={check_ids}, "
                f"marked={ {id(b[1]) for b in marked.values()} }"
            )

            if self.verbose > 5:
                self._debug["iteration"] = iteration

        if iteration >= max_iter and stacked:
            self._hint("reached {iteration}>={max_iter} iterations")
            return self.none(node, inspect.currentframe().f_lineno)

        if self.verbose > 5:
            print(f"[GenericPattern.match] done. {len(marked)} marked nodes")

        # At this point, the pattern is matched but let's make sure.
        assert len(marked) == len(pat.nodes), (
            f"Number of marked nodes is different, {len(marked)} marked nodes, "
            f"and {len(pat.nodes)} nodes in the pattern, marked is {marked}"
        )
        assert len(stacked) == 0, f"There are still {len(stacked)} nodes to explore."

        # We order the matched nodes in the same order than the pattern
        # to let next functions to be able to build the matching again.
        matched_nodes = [marked[id(n)][0] for i, n in enumerate(pat.nodes)]
        return PatternMatchResult(
            self, matched_nodes, pat.nodes, pat.input_names, pat.output_names
        )

    @classmethod
    def apply_pattern(
        cls,
        g: ir.Model,
        *args: Any,
        **kwargs: Any,
    ) -> list[ir.Node]:
        """Applies the replacement."""
        raise NotImplementedError(
            f"Class {cls.__name__!r} must overwrite method 'apply_pattern'."
        )

    def apply(
        self,
        model: ir.Model,
        match_result: PatternMatchResult,
        verbose: int = 0,
    ) -> list[ir.Node]:
        assert isinstance(match_result, PatternMatchResult)
        new_pat = self._build_pattern(
            fct=self.apply_pattern, kwargs=match_result.kwargs, match=False
        )
        assert len(new_pat.input_names) == len(match_result.pattern_input_names), (
            f"Not the same number of inputs, "
            f"matched inputs={len(new_pat.input_names)}, "
            f"got {len(match_result.pattern_input_names)} in the applied pattern."
        )
        assert len(new_pat.output_names) == len(match_result.pattern_output_names), (
            f"Not the same number of outputs, matched "
            f"outputs={match_result.pattern_output_names}, "
            f"got {new_pat.output_names} in the applied pattern."
        )

        if verbose > 5:
            print(
                f"[GenericPattern.apply] replace {len(match_result.model_nodes)} nodes, "
                f"applied {self.display_pattern(model, self.apply_pattern)}"
            )

        # TODO: handle initializers here
        # for name, init in pattern.initializers.items():
        #   # We add them to the graph, they will be removed if unused.
        #   new_name = g.make_initializer(name, init)
        #   replacements[new_name] = name

        applied_pattern_to_match_pattern = {}
        for i, j in zip(match_result.pattern_input_names, new_pat.input_names):
            applied_pattern_to_match_pattern[j] = i
        for i, j in zip(match_result.pattern_output_names, new_pat.output_names):
            applied_pattern_to_match_pattern[j] = i

        replacements = {}
        for k, v in applied_pattern_to_match_pattern.items():
            replacements[k] = match_result.matched_pattern_to_model_name[v]

        # Creation of the new node.
        new_nodes = []
        for node in new_pat.nodes:
            new_inputs = []
            for i in node.inputs:
                assert i in replacements, f"Unable to find {i!r} in {replacements}"
                ni = replacements[i]
                new_inputs.append(ni)
            new_outputs = []
            for o in node.outputs:
                if o in replacements:
                    new_outputs.append(replacements[o])
                else:
                    # We give it a new name.
                    n = model.unique_name(o)
                    replacements[o] = n
                    new_outputs.append(n)
            # TODO: Add a test for attributes.
            new_node = model.make_node(
                node.op_type,
                new_inputs,
                new_outputs,
                attributes=node.attributes,
                domain=node.domain,
            )
            new_nodes.append(new_node)

        if verbose > 5:
            print(f"[GenericPattern.apply] done with {len(new_nodes)} nodes")

        return new_nodes

    def make_rule(self) -> orp.RewriteRule:
        """Creates the corresponding rule for this pattern."""
        return GenericRewriteRule(self)


class FunctionPattern(GenericPattern):
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

    :param match_pattern: a function interpreted by onnx-script
        and converted into an onnx model, this model defines the
        nodes to be replaced
    :param apply_pattern: a function interpreted by onnx-script and
        converted into an onnx model, this model defines the new nodes
        replacing the matched nodes
    :param validate_mapping: a function validating the matching once
        it has happened, it is not valid, the pattern is not applied,
        if not specified, the function always return True
    :param opsets: opset to consider when converting the function into ONNX,
        if not specified, it is opset 18 for the main opset, and opset 1
        for domain com.microsoft.
    :return: the rewriting rule
    """

    if opsets is None:
        opsets = dict(
            op=onnxscript.opset18, msft_op=onnxscript.values.Opset("com.microsoft", 1)
        )

    if not isinstance(apply_pattern, onnx.FunctionProto):
        apply_pattern = onnxscript.script(**opsets)(apply_pattern).to_function_proto()

    if not isinstance(match_pattern, onnx.FunctionProto):
        match_pattern = onnxscript.script(**opsets)(match_pattern).to_function_proto()

    match_function = serde.deserialize_function(match_pattern)
    apply_function = serde.deserialize_function(apply_pattern)

    pat = FunctionPattern(
        match_function,
        apply_function,
        validate_mapping or (lambda *_, **__: True),
        verbose=verbose,
    )
    return pat.make_rule()
