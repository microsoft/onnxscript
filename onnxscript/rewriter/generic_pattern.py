from __future__ import annotations

import collections
import inspect
import os
import textwrap
import typing

import onnx
import onnx.helper as oh

import onnxscript._legacy_ir as oir
import onnxscript.rewriter.pattern as orp


def enumerate_subgraphs(
    node: oir.Node,
) -> typing.Iterator[tuple[typing.Any, ...]]:
    """Returns the subgraphs inside a graph."""
    for att in node.attribute:
        if att.type == onnx.AttributeProto.GRAPH and att.g:
            this = node, att.name, att.g
            yield this

            for no in att.g.node:
                for tu in enumerate_subgraphs(no):
                    yield this + tu


class _GraphStructureAPI:
    """Common accessors to predecessors and successors."""

    def __init__(self):
        self.predecessors_: dict[str, int] = {}
        self.successors_: dict[str, list[int]] = {}
        self.nodes_: dict[int, oir.Node] = {}

    def node_before(self, name: str) -> oir.Node | None:
        """
        Returns the node producing this output.

        Returns None if it is an input or an initializer.
        """
        if name not in self.predecessors_:
            return None
        predecessor = self.predecessors_[name]
        return self.nodes_[predecessor]

    def next_nodes(self, name: str) -> list[oir.Node] | None:
        """Returns the node consuming the given results."""
        if name not in self.successors_:
            return []
        return [self.nodes_[i] for i in self.successors_[name]]


class BuilderWithGraphStructure(_GraphStructureAPI):
    """Very concise graph builder.

    It wraps an ONNX graph
    and builds successors and predecessors on top of it.
    """

    def __init__(self, bridge: ModelWithGraphStructure):
        super().__init__()
        self.bridge: ModelWithGraphStructure = bridge
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.nodes: list[oir.Node] = []

    def _build(self) -> None:
        self.predecessors_: dict[str, int] = {}
        self.successors_: dict[str, list[int]] = {}
        self.nodes_: dict[int, oir.Node] = {}

        self.outputs_ = set(self.output_names)
        for node in self.nodes:
            self.nodes_[id(node)] = node

        for k, v in self.nodes_.items():
            assert isinstance(v, oir.Node), f"Unexpected type {type(v)} for node {k}"
            for o in v.output_names:
                self.predecessors_[o] = k
            for i in v.input_names:
                if i not in self.successors_:
                    self.successors_[i] = []
                self.successors_[i].append(k)

    def make_input(self, name: str) -> None:
        self.input_names.append(name)

    def make_output(self, name: str) -> None:
        self.output_names.append(name)

    def __getattr__(self, name: str) -> typing.Any:
        if name in self.__dict__:
            return self.__dict__[name]

        # unknown name
        assert (
            name[0].upper() == name[0]
        ), f"A node type must starts with an upper letter but it is {name!r}"
        return lambda *args, _name=name, **kwargs: self._make_node(_name, *args, **kwargs)

    def _make_node(
        self,
        op_type: str,
        *args: str,
        output_names: list[str] | int | None = None,
        **kwargs: typing.Any,
    ) -> str | tuple[str]:
        if output_names is None:
            # We assume there is only one outputs, we could also check into the schema.
            output_names = 1
        return self.make_node(op_type, *args, output_names=output_names, **kwargs)

    def make_node_with_proto(self, node_proto: onnx.NodeProto) -> tuple[str] | str:
        node = oir.Node(node_proto, True)
        self.nodes.append(node)
        assert node.output_names, f"No output in node {node}. This can't be true."
        if len(node.output_names) == 1:
            return node.output_names[0]
        return tuple(node.output_names)

    def make_node(
        self,
        op_type: str,
        *input_names: str,
        output_names: int | list[str] | str | None = None,
        domain: str = "",
        name: str | None = None,
        **kwargs: typing.Any,
    ) -> str | tuple[str]:
        node = oir.Node(
            self.bridge.make_node(
                op_type, input_names, output_names, domain=domain, name=name, **kwargs
            ),
            True,
        )
        self.nodes.append(node)
        assert node.output_names, f"No output in node {node}. This can't be true."
        if len(node.output_names) == 1:
            return node.output_names[0]
        return tuple(node.output_names)


class ModelWithGraphStructure(oir.Model, _GraphStructureAPI):
    """Implements all the necessary API it needs to work.

    Wraps a :class:`Model` and builds successors and predecessors on
    top of it.
    """

    def __init__(self, model: oir.Model, verbose: int = 0):
        oir.Model.__init__(self)
        _GraphStructureAPI.__init__(self)
        self.model = model
        if hasattr(self.model, "graph"):
            self.nodes = list(model.graph.nodes)
            self.input_names = list(model.graph.input_names)
            self.output_names = list(model.graph.output_names)
            self._build()
        else:
            # empty graph
            self._unique_names: set = set()
            self._unique_node_names: set = set()
        self.verbose = verbose

    def _build(self) -> None:
        """Builds successor and predecessor."""
        self.nodes_ = {}
        self.outputs_ = set(self.output_names)
        self._unique_node_names = set()
        for node in self.nodes:
            self.nodes_[id(node)] = node
            if node.name:
                self._unique_node_names.add(node.name)

        self.predecessors_: dict = {}
        self.successors_: dict = {}
        # TODO: # initiliazer are missing
        self._unique_names = set(self.input_names) | set(self.output_names)
        for k, v in self.nodes_.items():
            assert isinstance(v, oir.Node), f"Unexpected type {type(v)} for node {k}"
            for o in v.output_names:
                self.predecessors_[o] = k
            for i in v.input_names:
                if i not in self.successors_:
                    self.successors_[i] = []
                self.successors_[i].append(k)

            for sub in enumerate_subgraphs(v):
                g = sub[-1]
                sub_knowns = set()
                for n in g.input:
                    sub_knowns.add(n.name)
                for n in g.initializer:
                    sub_knowns.add(n.name)
                for n in g.sparse_initializer:
                    sub_knowns.add(n.name)
                for n in g.node:
                    for i in n.input:
                        if i not in sub_knowns:
                            # an input coming from the parent
                            self._unique_names.add(i)
                    for i in n.output:
                        sub_knowns.add(i)

    def unique_name(self, prefix: str) -> str:
        """Generates a unique result name.

        That excludes existing names as well.
        """
        if prefix in self._unique_names:
            i = 2
            sug = f"{prefix}2"
            while sug in self._unique_names:
                i += 1
                sug = f"{prefix}{i}"
            self._unique_names.add(sug)
            return sug
        self._unique_names.add(prefix)
        return prefix

    def unique_node_name(self, name: str | None) -> str:
        """Creates a unique node name."""
        name = name or ""
        if name in self._unique_node_names:
            i = 2
            sug = f"{name}2"
            while sug in self._unique_node_names:
                i += 1
                sug = f"{name}{i}"
            self._unique_node_names.add(sug)
            return sug
        self._unique_node_names.add(name)
        return name

    def make_opset(self) -> BuilderWithGraphStructure:
        return BuilderWithGraphStructure(self)

    @property
    def opsets(self) -> dict:
        """Property."""
        return self.model.version_map

    def make_node(
        self,
        op_type: str,
        input_names: str | typing.Sequence[str] | None,
        output_names: int | typing.Sequence[str] | str | None = 1,
        domain: str = "",
        attributes: list[onnx.AttributeProto] | None = None,
        name: str | None = None,
        **kwargs: typing.Any,
    ) -> onnx.NodeProto:
        """
        Creates a node without adding it to the graph.

        :param op_type: operator type
        :param input_names: input names
        :param output_names: outputs names, if one integer, creates n unique names,
            if str, creates one unique names, if a list, use the name
        :param domain: node domain
        :param attributes: list of attributes
        :param name: node name
        :param kwargs: other attributes
        :return: a node
        """
        name = self.unique_node_name(name)
        if isinstance(output_names, int):
            if output_names == 1:
                output_names = [self.unique_name(f"{op_type.lower()}")]
            else:
                output_names = [
                    self.unique_name(f"{op_type.lower()}-{i}") for i in range(output_names)
                ]
        elif isinstance(output_names, str):
            output_names = [self.unique_name(output_names)]

        proto = oh.make_node(
            op_type,
            (
                input_names
                if isinstance(input_names, (list, tuple))
                else ([input_names] if isinstance(input_names, str) else None)
            ),
            output_names,
            domain=domain,
            name=name,
            **kwargs,
        )
        if attributes:
            proto.attribute.extend(attributes)
        return proto


class GenericRewriteRule(orp.RewriteRule):
    """
    Defines a rewriting rule.

    :param pattern: a pattern defines by :class:`GenericPattern`.
    """

    def __init__(self, pattern: GenericPattern):
        self.pattern = pattern

    def matches(self, node: oir.Node, model: oir.Model) -> orp.MatchResult:
        del model
        del node
        raise RuntimeError(f"This pattern {self} is meant to replace not to only match.")

    def try_rewrite(
        self, model: oir.Model, node: oir.Node
    ) -> tuple[int, list[oir.Node], list[oir.Node]] | None:
        """See :meth:`RewriteRule.try_rewrite`."""
        if isinstance(model, ModelWithGraphStructure):
            bridge = model
        else:
            bridge = ModelWithGraphStructure(model)
        deleted_nodes = []
        added_nodes = []
        marked = set()
        matched = 0
        for matched_nodes in self.pattern.enumerate_matches(bridge, node):
            assert all(isinstance(i, oir.Node) for i in matched_nodes)
            conflict = False
            for node in matched_nodes:
                if id(node) in marked:
                    conflict = True
                    break
            if conflict:
                # Some nodes are already marked as rewritten.
                continue

            # Let's build the new nodes
            new_nodes = self.pattern.apply(bridge, *matched_nodes)
            assert all(
                isinstance(i, oir.Node) for i in new_nodes
            ), f"Unexpected types {[type(n) for n in new_nodes]}"

            if not self.pattern.validate_mapping(bridge, matched_nodes, new_nodes):
                continue

            # Everything is good.
            marked |= set(map(id, matched_nodes))
            added_nodes.extend(new_nodes)
            deleted_nodes.extend(matched_nodes)
            matched += 1

        if matched > 0:
            return matched, deleted_nodes, added_nodes
        return None

    def count_matches(self, model: oir.Model, *, commute: bool = False) -> int:
        """See :meth:`RewriteRule.count_matches`."""
        raise NotImplementedError("Not supported yet.")

    def commute(self) -> list[orp.RewriteRule]:
        """See :meth:`RewriteRule.commute`."""
        raise RuntimeError("Not supported (yet?). It could lead to many patterns.")

    def apply_to_model(self, model: oir.Model, *, commute: bool = False) -> int:
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

    def validate_mapping(
        self, g: oir.Model, deleted_nodes: list[oir.Node], added_nodes: list[oir.Node]
    ) -> bool:
        """Evaluates the consistency of the replacements."""
        raise NotImplementedError(
            "This method could return True but it is better to let you know "
            "that it exists. You need to overwrite it to return True."
        )

    def enumerate_matches(
        self, g: ModelWithGraphStructure, node: oir.Node | None = None
    ) -> typing.Iterator:
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
        node: oir.Node | None = None,
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
        g: ModelWithGraphStructure,
        *args: str,
        **kwargs: typing.Any,
    ) -> list[oir.Node] | None:
        """Builds the pattern to match."""
        raise NotImplementedError(
            f"Class {cls.__name__!r} must overwrite method match_pattern."
        )

    @classmethod
    def _build_pattern(
        cls, g: ModelWithGraphStructure, fct: typing.Callable
    ) -> BuilderWithGraphStructure:
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

        g2 = g.make_opset()
        for name in args:
            g2.make_input(name)
        output = fct(g2, *args, **kwargs)
        if isinstance(output, str):
            g2.make_output(output)
        else:
            for name in output:
                g2.make_output(name)
        g2._build()
        return g2

    def _get_match_pattern(self, g: ModelWithGraphStructure) -> BuilderWithGraphStructure:
        cache_key = 0, tuple(sorted(g.opsets.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        pat = self._build_pattern(g, self.match_pattern)
        self._cache[cache_key] = pat
        return pat

    def _get_apply_pattern(self, g: ModelWithGraphStructure) -> BuilderWithGraphStructure:
        cache_key = 1, tuple(sorted(g.opsets.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        pat = self._build_pattern(g, self.apply_pattern)
        self._cache[cache_key] = pat
        return pat

    def display_pattern(self, g: ModelWithGraphStructure, fct: typing.Callable) -> str:
        """Shows the pattern to match or to apply."""
        pat = self._build_pattern(g, fct)
        rows = []
        rows.append(
            f"{fct.__name__}({', '.join(pat.input_names)}) -> {', '.join(pat.output_names)}"
        )
        for node in pat.nodes:
            rows.append(
                f"{node.op_type}({', '.join(node.input_names)}) -> "
                f"{', '.join(node.output_names)}"
            )
        return "\n".join(rows)

    def print_match(self, n1: oir.Node, n2: oir.Node) -> str:
        s1 = f"{n1.op_type}({','.join(n1.input_names)})"
        s2 = f"{n2.op_type}({','.join(n2.input_names)})"
        return f"match {s1} with {s2} (pattern)"

    def _debug_print(self) -> str:
        if not hasattr(self, "_debug"):
            return ""

        def _s(s: str) -> str:
            if len(s) <= 30:
                return s
            return f"{s[:15]}...{s[-15:]}"

        def _p(n: oir.Node, full: bool = False) -> str:
            if isinstance(n, (oir.Node, onnx.NodeProto)):
                if full:
                    return (
                        f"{n.op_type}({', '.join(map(_s, n.input_names))}) "
                        f"-> ({', '.join(map(_s, n.output_names))})"
                    )
                return f"{n.op_type}({','.join(map(_s, n.input_names))})"
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
                    rows.append("  " + _p(i, full=True))
                continue
            if k in {"node", "pattern", "pattern_node", "pattern_nodes"}:
                continue
            rows.append(f"-- not shown {k}")

        return "\n".join(rows)

    def _hint(self, *args: typing.Any) -> None:
        """Add debugging information to help users."""
        self._debug["hint"] = args

    def _match_backward(
        self,
        g: ModelWithGraphStructure,
        node: oir.Node,
        pat: ModelWithGraphStructure,
        marked: dict[int, tuple[oir.Node, oir.Node]],
        stacked: list[int],
        n: oir.Node,
        pn: oir.Node,
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
        if len(n.input_names) != len(pn.input_names):
            # not the same number of inputs
            self._hint(
                "BACKWARD: not the same number of inputs",
                "-- pattern",
                pn,
                "-- model",
                n,
            )
            return self.none(node, inspect.currentframe().f_lineno)
        for i, pi in zip(n.input_names, pn.input_names):
            ppred = pat.node_before(pi)
            if ppred is None:
                # ppred is None means the pattern ends here.
                continue
            pred = g.node_before(i)
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
        g: ModelWithGraphStructure,
        node: oir.Node,
        pat: ModelWithGraphStructure,
        marked: dict[int, tuple[oir.Node, oir.Node]],
        stacked: list[int],
        n: oir.Node,
        pn: oir.Node,
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
        if len(n.output_names) != len(pn.output_names):
            # not the same number of outputs
            self._hint(
                "FORWARD: not the same number of output_names",
                "-- pattern",
                pn,
                "-- model",
                n,
            )
            return self.none(node, inspect.currentframe().f_lineno)

        for o, op in zip(n.output_names, pn.output_names):
            ns = g.next_nodes(o)
            pns = pat.next_nodes(op)
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
        g: ModelWithGraphStructure,
        node: oir.Node,
    ) -> list[oir.Node] | None:
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
            print(
                f"[GenericPattern.match] starts with "
                f"{node.op_type}({', '.join(node.input_names)})"
            )
            if self.verbose >= 10:
                print("[GenericPattern.match] match pattern")
                print(textwrap.indent(self.display_pattern(g, self.match_pattern), "    "))

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
        return matched_nodes

    @classmethod
    def apply_pattern(
        cls,
        g: ModelWithGraphStructure,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> list[oir.Node]:
        """Applies the replacement."""
        raise NotImplementedError(
            f"Class {cls.__name__!r} must overwrite method 'apply_pattern'."
        )

    def apply(
        self,
        g: ModelWithGraphStructure,
        *nodes: typing.Sequence[oir.Node],
    ) -> list[oir.Node]:
        assert all(isinstance(n, oir.Node) for n in nodes)
        pat = self._build_pattern(g, self.match_pattern)
        assert len(nodes) == len(pat.nodes), (
            f"Mismatch matched nodes pattern has {len(pat.nodes)} != {len(nodes)} = "
            f"the number of matched nodes"
        )
        new_pat = self._build_pattern(g, self.apply_pattern)
        assert len(new_pat.input_names) == len(pat.input_names), (
            f"Not the same number of inputs, matched inputs={len(new_pat.input_names)}, "
            f"got {len(pat.input_names)} in the applied pattern."
        )
        assert len(new_pat.output_names) == len(pat.output_names), (
            f"Not the same number of outputs, matched outputs={pat.output_names}, "
            f"got {new_pat.output_names} in the applied pattern."
        )
        assert all(isinstance(n, oir.Node) for n in pat.nodes)

        if g.verbose > 5:
            print(
                f"[GenericPattern.apply] replace {len(nodes)} nodes, "
                f"applied {self.display_pattern(g, self.apply_pattern)}"
            )

        matched_pattern_to_applied_pattern = {}
        for i, j in zip(pat.input_names, new_pat.input_names):
            matched_pattern_to_applied_pattern[i] = j
        for i, j in zip(pat.output_names, new_pat.output_names):
            matched_pattern_to_applied_pattern[i] = j

        matched_pattern_to_graph_name: dict = {}
        input_names = set(pat.input_names)
        output_names = set(pat.output_names)

        matched_pairs = list(zip(nodes, pat.nodes))
        for gn, pn in matched_pairs:
            assert (
                gn.op_type == pn.op_type
            ), f"Unexpected type mismatch {gn.op_type!r} != {pn.op_type!r}"
            assert len(gn.input_names) == len(
                pn.input_names
            ), f"Unexpected number of inputs for type {gn.op_type}"
            for a, b in zip(gn.input_names, pn.input_names):
                if b not in input_names or b == "":
                    # optional input or not an interesting input
                    continue
                if b in matched_pattern_to_graph_name:
                    assert matched_pattern_to_graph_name[b] == a, (
                        f"Ambiguities, pattern name {b!r} means "
                        f"{a!r} or {matched_pattern_to_graph_name[b]}"
                    )
                else:
                    matched_pattern_to_graph_name[b] = a

            assert len(gn.output_names) == len(
                pn.output_names
            ), f"Unexpected number of outputs for type {gn.op_type}"
            for a, b in zip(gn.output_names, pn.output_names):
                if b not in output_names or b == "":
                    # Only final outputs are interesting.
                    continue
                assert a != "", f"{a!r} cannot be optional"
                if b in matched_pattern_to_graph_name:
                    assert matched_pattern_to_graph_name[b] == a, (
                        f"Ambiguities, pattern name {b!r} means "
                        f"{a!r} or {matched_pattern_to_graph_name[b]}"
                    )
                else:
                    matched_pattern_to_graph_name[b] = a

        # TODO: handle initializers here
        # for name, init in pattern.initializers.items():
        #   # We add them to the graph, they will be removed if unused.
        #   new_name = g.make_initializer(name, init)
        #   replacements[new_name] = name

        replacements = {}
        for k, v in matched_pattern_to_graph_name.items():
            replacements[matched_pattern_to_applied_pattern[k]] = v

        # Creation of the new node.
        new_nodes = []
        for node in new_pat.nodes:
            new_inputs = []
            for i in node.input_names:
                assert i in replacements, f"Unable to find {i!r} in {replacements}"
                ni = replacements[i]
                new_inputs.append(ni)
            new_outputs = []
            for o in node.output_names:
                if o in replacements:
                    new_outputs.append(replacements[o])
                else:
                    # We give it a new name.
                    n = g.unique_name(o)
                    replacements[o] = n
                    new_outputs.append(n)
            new_node = g.make_node(node.op_type, new_inputs, new_outputs, domain=node.domain)
            new_node.attribute.extend(node.attribute)
            new_nodes.append(oir.Node(new_node, True))

        if g.verbose > 5:
            print(f"[GenericPattern.apply] done with {len(new_nodes)} nodes")

        return new_nodes

    def make_rule(self) -> orp.RewriteRule:
        """Creates the corresponding rule for this pattern."""
        return GenericRewriteRule(self)


class OnnxGenericPattern(GenericPattern):
    """An instance of GenericPattern taking onnx model.

    It defines the matching pattern and its replacement.

    :param match_proto: the onnx function defining the matching pattern
    :param apply_proto: the onnx function defining the new pattern
    :param validate_mapping: the function used to validate a pattern
    :param verbose: in [0, 10], increase the verbosity to understand why a pattern
        does not match
    """

    def __init__(
        self,
        match_proto: onnx.FunctionProto,
        apply_proto: onnx.FunctionProto,
        validate_mapping: typing.Callable,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.match_proto = match_proto
        self._validate_mapping = validate_mapping
        self.apply_proto = apply_proto
        self._cache = {}

    def validate_mapping(
        self, g: oir.Model, deleted_nodes: list[oir.Node], added_nodes: list[oir.Node]
    ) -> bool:
        """Evaluates the consistency of the replacements."""
        return self._validate_mapping(g, deleted_nodes, added_nodes)

    def _build_pattern(
        self, g: ModelWithGraphStructure, fct: typing.Callable
    ) -> BuilderWithGraphStructure:
        if fct == self.match_pattern:
            key = id(g), "match"
            if key in self._cache:
                return self._cache[key]
            onx = self.match_proto
        elif fct == self.apply_pattern:
            key = id(g), "apply"
            if key in self._cache:
                return self._cache[key]
            onx = self.apply_proto
        else:
            raise AssertionError(
                f"Function {fct} is not {self.match_pattern} or {self.apply_pattern}."
            )

        g2 = g.make_opset()
        for name in onx.input:
            g2.make_input(name)
        for node in onx.node:
            g2.make_node_with_proto(node)
        for name in onx.output:
            g2.make_output(name)
        g2._build()
        self._cache[key] = g2
        return g2


def make_pattern_rule(
    match_pattern: typing.Callable,
    apply_pattern: typing.Callable,
    validate_mapping: typing.Callable | None = None,
    verbose: int = 0,
    opsets: dict[str, "onnxscript.Opset"] | None = None,  # noqa: F821
) -> orp.RewriteRule:
    """
    Creates a rewriting rule.

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
    import onnxscript

    if opsets is None:
        opsets = dict(
            op=onnxscript.opset18, msft_op=onnxscript.values.Opset("com.microsoft", 1)
        )

    if verbose > 5:
        print(f"[make_pattern_rule] Converting {match_pattern} into ONNX.")
    match = onnxscript.script(**opsets)(match_pattern).to_function_proto()
    if verbose > 5:
        print("[make_pattern_rule] done.")
        print(f"[make_pattern_rule] Converting {apply_pattern} into ONNX.")
    apply = onnxscript.script(**opsets)(apply_pattern).to_function_proto()
    if verbose > 5:
        print("[make_pattern_rule] done.")

    pat = OnnxGenericPattern(
        match,
        apply,
        validate_mapping or (lambda *_, **__: True),
        verbose=verbose,
    )
    return pat.make_rule()
