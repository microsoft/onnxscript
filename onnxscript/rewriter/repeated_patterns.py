# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import hashlib
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.helper as oh  # noqa: TID251
import onnx.numpy_helper as onh  # noqa: TID251


def node_type_frequency(
    onx: Union[Sequence[onnx.NodeProto], onnx.ModelProto, onnx.GraphProto, onnx.FunctionProto],
    min_freq: int = 2,
) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], int], int, List[Tuple[str, str]]]:
    """
    Computes the frequency of every node type in a list.

    :param onx: any object containing a sequence of NodeProto
    :param min_freq: do not consider any frequency below that threshold
    :return: 4 results,
        the frequencies of the node types,
        the frequencies of the frequencies,
        the most frequent frequency (the estimation of the number of layers),
        all types having the exact same frequency as the previous one

    .. note::
        This function assumes at least one type of node is present only once in every layer.
    """
    if isinstance(onx, onnx.ModelProto):
        return node_type_frequency(onx.graph, min_freq=min_freq)
    h = Counter((node.domain, node.op_type) for node in onx.node)
    freq = {k: v for k, v in h.items() if v >= min_freq}
    freq_freq = Counter(freq.values())
    freqs = dict(freq_freq)
    for k, v in freq_freq.items():
        for i in range(2, k):
            if k % i == 0 and i in freq_freq:
                freqs[i] += k // i * v
    ret = max((v, k) for k, v in freqs.items())
    types = [k for k, v in freq.items() if v == ret[1]]
    return freq, freqs, ret[1], types


def _serialize_attribute(attribute: Sequence[onnx.AttributeProto]) -> bytes:
    return b"/".join(a.SerializeToString() for a in attribute)


class _GraphPattern:
    def __init__(self, first_node: int):
        self.cursor = first_node
        self.first_node = first_node
        self.subgraph = set()

    def add_cursor(self):
        assert self.cursor >= 0, f"Cannot add a negative cursor ({self.cursor})"
        assert (
            self.cursor not in self.subgraph
        ), f"Cursor {self.cursor} already added in {self.subgraph}"
        self.subgraph.add(self.cursor)


class _GraphIterator:
    def __init__(self, graph: "_GraphPatterns", node_index: int):
        self.graph = graph
        self.node_index = node_index
        self.io_index = None
        self.io_kind = None
        self.io_name = None
        self.o_suc = None
        self.o_suc_index = None

    def __str__(self) -> str:
        if self.node_index is None:
            return "it()"
        indices = [
            self.node_index,
            self.io_index,
            "N" if self.io_kind is None else ("I" if self.io_kind else "O"),
            "." if self.o_suc is None else self.o_suc,
            self.io_name,
        ]
        s = ", ".join(map(str, indices))
        return f"it({s})"

    def next(self):
        # assumes node.output is never empty
        node = self.graph.nodes[self.node_index]
        if self.io_name is None:
            self.io_index = 0
            self.o_suc = 0
            self.io_kind = bool(node.input)
            self.io_name = node.input[0] if self.io_kind else node.output[0]
            self.o_suc_index = (
                None
                if self.io_kind or not self.graph.successors[self.io_name]
                else self.graph.successors[self.io_name][self.o_suc]
            )
        else:
            if self.io_kind:
                self.io_index += 1
                self.o_suc = 0
                if self.io_index >= len(node.input):
                    self.io_kind = False
                    self.io_index = 0
                self.io_name = (
                    node.input[self.io_index] if self.io_kind else node.output[self.io_index]
                )
                self.o_suc_index = (
                    None
                    if not self.graph.successors[self.io_name]
                    else self.graph.successors[self.io_name][self.o_suc]
                )
            else:
                self.io_name = node.output[self.io_index]
                if self.io_name not in self.graph.successors:
                    self.io_name = None
                    return False
                self.o_suc += 1
                if self.o_suc < len(self.graph.successors[self.io_name]):
                    self.o_suc_index = self.graph.successors[self.io_name][self.o_suc]
                    return True
                self.o_suc = 0
                self.io_index += 1
                if self.io_index >= len(node.output):
                    self.io_name = None
                    self.o_suc_index = None
                    return False
                self.io_name = node.output[self.io_index]
                self.o_suc_index = (
                    None
                    if not self.graph.successors[self.io_name]
                    else self.graph.successors[self.io_name][self.o_suc]
                )
        return True

    def get_name(self, node_index: int) -> str:
        node = self.graph.nodes[node_index]
        if self.io_kind is None:
            return None
        if self.io_kind:
            name = node.input[self.io_index]
        else:
            name = node.output[self.io_index]
        assert (
            node_index != self.node_index or name == self.io_name
        ), f"Inconsistency with node_index={node_index}, name={name!r}, self={self!r}"
        return name

    def get_node_index(self, node_index: int) -> int:
        node = self.graph.nodes[node_index]
        if self.io_kind is None:
            return None
        if self.io_kind:
            name = node.input[self.io_index]
            index = self.graph.predecessor.get(name, -1)
        else:
            name = node.output[self.io_index]
            suc = self.graph.successors.get(name, [])
            if not suc:
                return -1
            # It is tricky here because the order of the successors
            # is not necessarily the same.
            if self.o_suc == 0 and len(suc) == 1:
                # Only one possible.
                index = suc[self.o_suc]
            else:
                assert self.o_suc_index is not None, (
                    f"Unable to guess the forward node, node_index={node_index}, "
                    f"self={self}, mapped={self.graph.mapped}"
                )
                expected_sig = self.graph.signatures[self.o_suc_index]
                sigs = {self.graph.signatures[s]: s for s in suc}
                assert len(sigs) == len(suc), (
                    f"Unable to distinguish between successors signatares: {sigs}, "
                    f"node_index={node_index}, type is "
                    f"{self.graph.nodes[node_index].op_type!r} "
                    f"name is {self.graph.nodes[node_index].name!r}, self={self}"
                )
                if expected_sig not in sigs:
                    # Cannot find the expected successor
                    return -1
                index = sigs[expected_sig]
                return index

        assert node_index != self.node_index or name == self.io_name, (
            f"Inconsistency with node_index={node_index}, "
            f"self.io_index={self.io_index!r}, name={name!r}, self={self!r}"
        )
        return index


class _GraphPredecessorSuccessors:
    def __init__(
        self,
        nodes: List[onnx.NodeProto],
        initializer: Optional[onnx.TensorProto] = None,
    ):
        self.nodes = nodes
        self.initializer = {init.name: init for init in initializer} if initializer else {}
        self.build_edges()

    def make_sig(self, node: onnx.NodeProto) -> str:
        hash = (
            f"H{hashlib.sha256(_serialize_attribute(node.attribute)).hexdigest()[:20]}"
            if node.attribute
            else ""
        )
        sigi = []
        for i in node.input:
            if i in self.initializer:
                cst = self.initializer[i]
                shape = tuple(cst.dims)
                if len(shape) <= 1:
                    size = np.prod(shape)
                    if size < 1024:
                        t = onh.to_array(cst).ravel()
                        if t.size < 16:
                            c = ",".join(str(x) for x in t.ravel())
                        else:
                            c = ",".join(str(x) for x in t.ravel()[:16])
                        sigi.append(c)
                    else:
                        sigi.append("CC")
                else:
                    sigi.append("C")
            else:
                p = self.predecessor.get(i, -1)
                sigi.append(self.nodes[p].op_type if p >= 0 else "")
        sig = (
            f"{node.domain}/{node.op_type}/{len(node.input)}-{len(node.output)}"
            f"{hash}//{'/'.join(sigi)}"
        )
        return sig

    def build_edges(self):
        self.successors: Dict[str, Dict[str, int]] = {}
        self.predecessor: Dict[str, int] = {}
        self.signatures: Dict[int, str] = {}
        self.result_names = set()
        for node_index, node in enumerate(self.nodes):
            self.result_names |= set(node.input) | set(node.output)
            for i in node.output:
                self.predecessor[i] = node_index
            for i in node.input:
                if i not in self.successors:
                    self.successors[i] = []
                self.successors[i].append(node_index)
            sig = self.make_sig(node)
            self.signatures[node_index] = sig


def make_function_from_nodes(
    nodes: List[onnx.NodeProto], name: str = "function", domain: str = "repeated"
) -> onnx.FunctionProto:
    """
    Creates a function from a list of nodes.
    Looks into inputs not created by one of the nodes, looks into unused outputs.
    Opset versions are all set to one.

    :param nodes: list of nodes
    :param name: function name
    :param domain: domain name
    :return: function proto
    """
    gr = _GraphPredecessorSuccessors(nodes)
    domains = sorted({n.domain for n in nodes})
    inputs = sorted(
        k for k in gr.result_names if k not in gr.predecessor or gr.predecessor[k] is None
    )
    outputs = sorted(
        k for k in gr.result_names if k not in gr.successors or not gr.successors[k]
    )
    return oh.make_function(
        name,
        domain,
        inputs,
        outputs,
        nodes,
        opset_imports=[oh.make_opsetid(n, 1) for n in domains],
    )


class _GraphPatterns(_GraphPredecessorSuccessors):
    def __init__(
        self,
        nodes: List[onnx.NodeProto],
        cursor: Sequence[int],
        initializer: Optional[onnx.TensorProto] = None,
    ):
        super().__init__(nodes, initializer)
        self.pats = [_GraphPattern(c) for c in cursor]
        self.current: List[_GraphIterator] = []
        self.processed_indices = set()
        self.mapped: Dict[int : List[int]] = {}

    def validate_cursor(self, verbose: int = 0):
        # op_types
        if any(p.cursor < 0 for p in self.pats):
            if verbose > 2:
                print("[_GraphPatterns.validate_cursor] INVALID (-1)")
            return False
        # already processed
        if any(p.cursor in self.processed_indices for p in self.pats):
            if verbose > 2:
                print("[_GraphPatterns.validate_cursor] INVALID (processed)")
            return False
        nodes = [self.nodes[p.cursor] for p in self.pats]
        rec = {(n.op_type, len(n.input), len(n.output), len(n.attribute)) for n in nodes}
        if len(rec) != 1:
            if verbose > 2:
                print("[_GraphPatterns.validate_cursor] INVALID (not unique type)")
            return False
        n_atts = rec.pop()[-1]
        if n_atts == 0:
            if verbose > 2:
                print("[_GraphPatterns.validate_cursor] VALID (1)")
            return True

        # Needs to check attributes
        base = _serialize_attribute(nodes[0].attribute)
        for n in nodes[1:]:
            get = _serialize_attribute(n.attribute)
            if get != base:
                if verbose > 2:
                    print("[_GraphPatterns.validate_cursor] INVALID (not the same attribute)")
                return False
        if verbose > 2:
            print("[_GraphPatterns.validate_cursor] VALID (2)")
        return True

    def add_cursor(self):
        bug = set()
        for pi, p in enumerate(self.pats):
            assert p.cursor not in bug, (
                f"Every cursor pi={pi}, should be different but "
                f"{[p.cursor for p in self.pats]}"
            )
            bug.add(p.cursor)
            p.add_cursor()
            if p.cursor not in self.mapped:
                self.mapped[p.cursor] = set()
        for p in self.pats:
            for pp in self.pats:
                self.mapped[p.cursor].add(pp.cursor)

    def apply_path(self, node_index: int) -> int:
        if not self.current:
            return node_index
        for p in self.current:
            node_index = p.get_node_index(node_index)
        return node_index

    def set_cursor(self):
        bug = set()
        for pi, p in enumerate(self.pats):
            p.cursor = self.apply_path(p.first_node)
            assert p.cursor is not None, (
                f"Wonrg cursor for p.first_node={p.first_node} and "
                f"path={'/'.join(map(str,self.current))}, pi={pi}"
            )
            if p.cursor >= 0:
                if p.cursor in bug:
                    # This means one input is shared accross multiple patterns.
                    # This cannot be possible.
                    p.cursor = -1
                else:
                    bug.add(p.cursor)

    def next_valid(self):
        i = self.pats[0].cursor
        self.current.append(_GraphIterator(self, i))
        return self.next_not_valid()

    def next_not_valid(self):
        has_next = self.current[-1].next()
        while not has_next:
            self.current.pop()
            if not self.current:
                return False
            has_next = self.current[-1].next()
        self.set_cursor()
        return True

    def add_processed_cursor(self):
        if any(p.cursor == -1 for p in self.pats):
            return
        for p in self.pats:
            if p.cursor != -1:
                self.processed_indices.add(p.cursor)

    def process(
        self, name: str = "RepeatedPattern", verbose: int = 0
    ) -> Optional[Tuple[Union[List[int], List[List[int]]], Tuple[onnx.FunctionProto, onnx.FunctionProto]]]:
        """Main function, looks for repeated patterns."""
        valid = self.validate_cursor(verbose=verbose)
        n_iter = 0
        while True and n_iter < len(self.nodes):
            if verbose > 1:
                node = self.nodes[self.pats[0].cursor]
                print(
                    f"[_GraphPatterns.process] it={n_iter}: {node.op_type!r}: "
                    f"{','.join(str(p.cursor) for p in self.pats)}"
                )
            if valid:
                if verbose:
                    node = self.nodes[self.pats[0].cursor]
                    print(
                        f"[_GraphPatterns.process] add node type "
                        f"{node.op_type}({', '.join(node.input)})"
                    )
                self.add_cursor()
                self.add_processed_cursor()
                is_next = self.next_valid()
            else:
                self.add_processed_cursor()
                if not self.current:
                    # No current cursor.
                    break
                is_next = self.next_not_valid()
            if not is_next:
                valid = True
                break
            valid = self.validate_cursor(verbose=verbose)
            n_iter += 1

        if self.pats[0].subgraph:
            indices = sorted(self.pats[0].subgraph)
            nodes = [self.nodes[i] for i in indices]
            proto = make_function_from_nodes(nodes, domain="repeated")
            pattern = (
                proto,
                oh.make_function(
                    "repeated",
                    "pattern",
                    proto.input,
                    proto.output,
                    [
                        oh.make_node(
                            name,
                            proto.input,
                            proto.output,
                            domain="repeated",
                        )
                    ],
                    opset_imports=[oh.make_opsetid("repeated", 1)],
                ),
            )
            return indices, pattern
        return None


def find_largest_repeated_pattern(
    onx: Union[Sequence[onnx.NodeProto], onnx.ModelProto, onnx.GraphProto, onnx.FunctionProto],
    min_freq: int = 2,
    verbose: int = 0,
    all_instances: bool = False,
    name: str = "RepeatedPattern",
    replace: bool = False,
) -> Optional[Tuple[Union[List[int], List[List[int]]], Tuple[onnx.FunctionProto, onnx.FunctionProto]]]:
    """
    Finds the largest repeated pattern in a graph.

    :param onx: any object containing a sequence of NodeProto
    :param min_freq: do not consider any frequency below that threshold
    :param verbose: verbosity
    :param all_instances: if True, returns all instances
    :param name: function name (used to return the pattern)
    :param replace: does replace the matched node with local function,
        all_instance must be True, it only works if the input is a ModelProto.
    :return: list of node indices in the pattern, the pattern as two FunctionProto

    .. code-block:: python

        import onnx
        from onnxscript.rewriter.repeated_patterns import find_largest_repeated_pattern

        onx = onnx.load("<filename>", load_external_data=False)
        res = find_largest_repeated_pattern(onx, all_instances=True, replace=True, verbose=1)
        if res:
            print(f"number of repeated instances", len(res[0]))
            onnx.save(onx, "<filename>.modified.onnx")
    """
    if isinstance(onx, onnx.ModelProto):
        assert not replace or all_instances, f"incompatible replace={replace}, all_instances={all_instances}"
        res = find_largest_repeated_pattern(
            onx.graph,
            min_freq=min_freq,
            verbose=verbose,
            all_instances=all_instances or replace,
            name=name,
        )
        if res is None:
            return res
        # Let's adjust the domain.
        subgraphs, pattern = res
        ds = {d.domain: d.version for d in onx.opset_import}
        for d in pattern[0].opset_import:
            d.version = ds[d.domain]

        if replace:
            # Runs the replacement inplace
            local_function = pattern[0]
            onx.functions.append(local_function)
            nodes = list(onx.graph.node)
            for index, subgraph in enumerate(subgraphs):
                subnodes = [nodes[node_index] for node_index in subgraph]
                temp = make_function_from_nodes(subnodes)
                max_index = max(subgraph)
                for index in subgraph:
                    nodes[index] = None
                nodes[max_index] = oh.make_node(local_function.name, temp.input, temp.output, domain=local_function.domain)

            del onx.graph.node[:]
            onx.graph.node.extend([n for n in nodes if n is not None])
        return subgraphs, pattern

    assert not replace, f"replace is not allowed on type {type(onx)}"
    _freq, _freqs, npats, types = node_type_frequency(onx, min_freq)
    if not types:
        return None
    if verbose:
        print(f"[find_largest_repeated_pattern] number of patterns: {npats}")
        print(f"[find_largest_repeated_pattern] frequencies of frequencies: {_freqs}")
        print(f"[find_largest_repeated_pattern] candidates: {types}")

    # initialization
    keep = None
    all_patterns = None
    nodes = list(onx.node)
    for candidate in types:
        if verbose:
            print(f"[find_largest_repeated_pattern] tries: {candidate}")
        cursor = []
        for i, n in enumerate(nodes):
            if (n.domain, n.op_type) == candidate:
                cursor.append(i)
        patterns = _GraphPatterns(
            nodes, cursor, initializer=onx.initializer if hasattr(onx, "initializer") else None
        )
        res = patterns.process(verbose=verbose, name=name)
        if res is not None:
            if verbose:
                print(
                    f"[find_largest_repeated_pattern] found a pattern of length {len(res[0])}"
                )
            if keep is None or len(keep[0]) < len(res[0]):
                keep = res
                all_patterns = [sorted(p.subgraph) for p in patterns.pats]
            if len(keep[0]) > 1:
                break
        elif verbose:
            print("[find_largest_repeated_pattern] no found pattern")
    if keep is None:
        return keep
    if all_instances:
        return all_patterns, keep[1]
    return keep
