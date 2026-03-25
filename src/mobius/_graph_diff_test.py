# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the _graph_diff module.

Tests the three public functions: ``canonicalize_graph``,
``diff_graphs``, and ``render_markdown``.
"""

from __future__ import annotations

import onnx_ir as ir

from mobius._graph_diff import (
    canonicalize_graph,
    diff_graphs,
    render_markdown,
)

# ------------------------------------------------------------------
# Helpers — build tiny test graphs using onnx_ir directly
# ------------------------------------------------------------------


def _simple_add_graph(
    *,
    input_name: str = "x",
    bias_name: str = "bias",
    output_name: str = "y",
    node_name: str = "add_0",
) -> ir.Graph:
    """Return a graph: y = x + bias."""
    import numpy as np

    x = ir.val(
        input_name,
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([1, 4]),
    )
    bias_tensor = ir.Tensor(np.ones(4, dtype=np.float32), name=bias_name)
    bias = ir.Value(
        name=bias_name,
        const_value=bias_tensor,
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([4]),
    )
    add_node = ir.Node("", "Add", [x, bias], name=node_name)
    out = add_node.outputs[0]
    out.name = output_name
    return ir.Graph(
        [x],
        [out],
        nodes=[add_node],
        initializers=[bias],
    )


def _add_relu_graph(
    *,
    input_name: str = "a",
    bias_name: str = "b",
    mid_name: str = "mid",
    output_name: str = "z",
) -> ir.Graph:
    """Return a graph: z = Relu(a + b)."""
    import numpy as np

    x = ir.val(
        input_name,
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([1, 4]),
    )
    bias_tensor = ir.Tensor(np.ones(4, dtype=np.float32), name=bias_name)
    bias = ir.Value(
        name=bias_name,
        const_value=bias_tensor,
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([4]),
    )
    add_node = ir.Node("", "Add", [x, bias], name="add")
    add_out = add_node.outputs[0]
    add_out.name = mid_name

    relu_node = ir.Node("", "Relu", [add_out], name="relu")
    relu_out = relu_node.outputs[0]
    relu_out.name = output_name

    return ir.Graph(
        [x],
        [relu_out],
        nodes=[add_node, relu_node],
        initializers=[bias],
    )


# ------------------------------------------------------------------
# canonicalize_graph tests
# ------------------------------------------------------------------


class TestCanonicalizeGraph:
    """Tests for canonicalize_graph."""

    def test_empty_graph(self) -> None:
        graph = ir.Graph([], [], nodes=[])
        canon = canonicalize_graph(graph)

        assert canon["interface"] == {
            "inputs": [],
            "outputs": [],
        }
        assert canon["initializers"] == []
        assert canon["nodes"] == []
        assert canon["op_sequence"] == []

    def test_ignores_names(self) -> None:
        """Two structurally identical graphs with different names."""
        g1 = _simple_add_graph(
            input_name="input_0",
            bias_name="weight_0",
            output_name="result",
            node_name="node_add",
        )
        g2 = _simple_add_graph(
            input_name="x",
            bias_name="b",
            output_name="y",
            node_name="my_add",
        )
        c1 = canonicalize_graph(g1)
        c2 = canonicalize_graph(g2)

        assert c1 == c2

    def test_captures_op_sequence(self) -> None:
        g = _add_relu_graph()
        canon = canonicalize_graph(g)
        assert canon["op_sequence"] == ["Add", "Relu"]

    def test_captures_interface(self) -> None:
        g = _simple_add_graph()
        canon = canonicalize_graph(g)
        iface = canon["interface"]
        assert len(iface["inputs"]) == 1
        assert iface["inputs"][0]["dtype"] == "FLOAT"
        assert len(iface["outputs"]) == 1

    def test_captures_initializers(self) -> None:
        g = _simple_add_graph()
        canon = canonicalize_graph(g)
        assert len(canon["initializers"]) == 1
        init = canon["initializers"][0]
        assert init["dtype"] == "FLOAT"
        assert init["shape"] == ["4"]

    def test_captures_attributes(self) -> None:
        x = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([2, 3]),
        )
        concat = ir.Node(
            "",
            "Concat",
            [x, x],
            [ir.AttrInt64("axis", 1)],
            name="concat",
        )
        out = concat.outputs[0]
        out.name = "out"
        g = ir.Graph([x], [out], nodes=[concat])
        canon = canonicalize_graph(g)
        assert canon["nodes"][0]["attributes"] == {"axis": 1}

    def test_symbolic_dims_erased(self) -> None:
        """Differently-named symbolic dims produce equal canonical forms."""
        x1 = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["batch", 4]),
        )
        add1 = ir.Node("", "Add", [x1, x1], name="add")
        o1 = add1.outputs[0]
        o1.name = "y"
        g1 = ir.Graph([x1], [o1], nodes=[add1])

        x2 = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["B", 4]),
        )
        add2 = ir.Node("", "Add", [x2, x2], name="add")
        o2 = add2.outputs[0]
        o2.name = "y"
        g2 = ir.Graph([x2], [o2], nodes=[add2])

        assert canonicalize_graph(g1) == canonicalize_graph(g2)

    def test_captures_num_outputs(self) -> None:
        x = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([2, 3]),
        )
        split = ir.Node(
            "",
            "Split",
            [x],
            [ir.AttrInt64("num_outputs", 2)],
            num_outputs=2,
            name="split",
        )
        o1 = split.outputs[0]
        o1.name = "o1"
        o2 = split.outputs[1]
        o2.name = "o2"
        g = ir.Graph([x], [o1, o2], nodes=[split])
        canon = canonicalize_graph(g)
        assert canon["nodes"][0]["num_outputs"] == 2


# ------------------------------------------------------------------
# diff_graphs tests
# ------------------------------------------------------------------


class TestDiffGraphs:
    """Tests for diff_graphs."""

    def test_no_changes(self) -> None:
        g = _simple_add_graph()
        c = canonicalize_graph(g)
        changes = diff_graphs(c, c)
        assert changes == []

    def test_added_node(self) -> None:
        base = _simple_add_graph()
        head = _add_relu_graph()
        cb = canonicalize_graph(base)
        ch = canonicalize_graph(head)
        changes = diff_graphs(cb, ch)
        types = {c["type"] for c in changes}
        assert "added_node" in types
        added = [c for c in changes if c["type"] == "added_node"]
        assert any("Relu" in c["details"] for c in added)

    def test_removed_node(self) -> None:
        base = _add_relu_graph()
        head = _simple_add_graph()
        cb = canonicalize_graph(base)
        ch = canonicalize_graph(head)
        changes = diff_graphs(cb, ch)
        types = {c["type"] for c in changes}
        assert "removed_node" in types
        removed = [c for c in changes if c["type"] == "removed_node"]
        assert any("Relu" in c["details"] for c in removed)

    def test_changed_attributes(self) -> None:
        x = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([2, 3]),
        )
        n1 = ir.Node(
            "",
            "Concat",
            [x, x],
            [ir.AttrInt64("axis", 0)],
            name="c",
        )
        o1 = n1.outputs[0]
        o1.name = "out"
        g1 = ir.Graph([x], [o1], nodes=[n1])

        x2 = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([2, 3]),
        )
        n2 = ir.Node(
            "",
            "Concat",
            [x2, x2],
            [ir.AttrInt64("axis", 1)],
            name="c2",
        )
        o2 = n2.outputs[0]
        o2.name = "out2"
        g2 = ir.Graph([x2], [o2], nodes=[n2])

        changes = diff_graphs(canonicalize_graph(g1), canonicalize_graph(g2))
        types = {c["type"] for c in changes}
        assert "changed_attrs" in types
        attr_changes = [c for c in changes if c["type"] == "changed_attrs"]
        assert any("axis" in c["details"] for c in attr_changes)

    def test_interface_change(self) -> None:
        """Detect when an output is added."""
        import numpy as np

        x = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 4]),
        )
        bias_t = ir.Tensor(np.ones(4, dtype=np.float32), name="b")
        bias = ir.Value(
            name="b",
            const_value=bias_t,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([4]),
        )
        add = ir.Node("", "Add", [x, bias], name="add")
        out = add.outputs[0]
        out.name = "y"
        g1 = ir.Graph([x], [out], nodes=[add], initializers=[bias])

        # Build g2 with two outputs (identity branch)
        x2 = ir.val(
            "x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 4]),
        )
        bias_t2 = ir.Tensor(np.ones(4, dtype=np.float32), name="b")
        bias2 = ir.Value(
            name="b",
            const_value=bias_t2,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([4]),
        )
        add2 = ir.Node("", "Add", [x2, bias2], name="add")
        out2 = add2.outputs[0]
        out2.name = "y"
        ident = ir.Node("", "Identity", [out2], name="ident")
        out3 = ident.outputs[0]
        out3.name = "y2"
        g2 = ir.Graph(
            [x2],
            [out2, out3],
            nodes=[add2, ident],
            initializers=[bias2],
        )

        changes = diff_graphs(canonicalize_graph(g1), canonicalize_graph(g2))
        types = {c["type"] for c in changes}
        assert "interface_change" in types

    def test_changed_connectivity(self) -> None:
        """Detect when node input wiring changes."""
        import numpy as np

        # Graph 1: Add(x, bias)
        x1 = ir.val("x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
        bias_t1 = ir.Tensor(np.ones(4, dtype=np.float32), name="b")
        bias1 = ir.Value(
            name="b",
            const_value=bias_t1,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([4]),
        )
        add1 = ir.Node("", "Add", [x1, bias1], name="add")
        out1 = add1.outputs[0]
        out1.name = "y"
        g1 = ir.Graph([x1], [out1], nodes=[add1], initializers=[bias1])

        # Graph 2: Add(x, x) — same op, different wiring
        x2 = ir.val("x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
        bias_t2 = ir.Tensor(np.ones(4, dtype=np.float32), name="b")
        bias2 = ir.Value(
            name="b",
            const_value=bias_t2,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([4]),
        )
        add2 = ir.Node("", "Add", [x2, x2], name="add")
        out2 = add2.outputs[0]
        out2.name = "y"
        g2 = ir.Graph([x2], [out2], nodes=[add2], initializers=[bias2])

        changes = diff_graphs(canonicalize_graph(g1), canonicalize_graph(g2))
        types = {c["type"] for c in changes}
        assert "changed_connectivity" in types
        conn = [c for c in changes if c["type"] == "changed_connectivity"]
        assert any("input_ids" in c["details"] for c in conn)

    def test_changed_connectivity_swapped_inputs(self) -> None:
        """Detect when two inputs to the same node are swapped."""
        import numpy as np

        # Graph 1: Sub(x, bias)
        x1 = ir.val("x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
        bias_t1 = ir.Tensor(np.ones(4, dtype=np.float32), name="b")
        bias1 = ir.Value(
            name="b",
            const_value=bias_t1,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([4]),
        )
        sub1 = ir.Node("", "Sub", [x1, bias1], name="sub")
        out1 = sub1.outputs[0]
        out1.name = "y"
        g1 = ir.Graph([x1], [out1], nodes=[sub1], initializers=[bias1])

        # Graph 2: Sub(bias, x) — same op, swapped inputs
        x2 = ir.val("x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 4]))
        bias_t2 = ir.Tensor(np.ones(4, dtype=np.float32), name="b")
        bias2 = ir.Value(
            name="b",
            const_value=bias_t2,
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([4]),
        )
        sub2 = ir.Node("", "Sub", [bias2, x2], name="sub")
        out2 = sub2.outputs[0]
        out2.name = "y"
        g2 = ir.Graph([x2], [out2], nodes=[sub2], initializers=[bias2])

        changes = diff_graphs(canonicalize_graph(g1), canonicalize_graph(g2))
        types = {c["type"] for c in changes}
        assert "changed_connectivity" in types


# ------------------------------------------------------------------
# render_markdown tests
# ------------------------------------------------------------------


class TestRenderMarkdown:
    """Tests for render_markdown."""

    def test_no_changes(self) -> None:
        md = render_markdown({})
        assert "Architecture Diff" in md
        assert "No architecture changes" in md
        assert "<!-- arch-diff-bot -->" in md

    def test_with_changes(self) -> None:
        diffs = {
            "llama": {
                "model": {
                    "changes": [
                        {
                            "type": "added_node",
                            "details": "+ Relu",
                        },
                    ],
                    "_base_ops": ["Add"],
                    "_head_ops": ["Add", "Relu"],
                    "_base_node_count": 1,
                    "_head_node_count": 2,
                }
            }
        }
        md = render_markdown(diffs)
        assert "Architecture Diff" in md
        # Summary table present
        assert "| llama" in md
        assert "1 |" in md
        # Details section present
        assert "<details>" in md
        assert "Added nodes" in md
        assert "Relu" in md
        # Legend present
        assert "Legend" in md

    def test_no_change_model_shows_no_change_emoji(self) -> None:
        diffs = {
            "bert": {
                "model": {
                    "changes": [],
                    "_base_ops": ["Add"],
                    "_head_ops": ["Add"],
                    "_base_node_count": 1,
                    "_head_node_count": 1,
                }
            }
        }
        md = render_markdown(diffs)
        assert "⚪" in md

    def test_interface_change_is_major(self) -> None:
        diffs = {
            "qwen2": {
                "model": {
                    "changes": [
                        {
                            "type": "interface_change",
                            "details": "output count 1 → 2",
                        },
                    ],
                    "_base_ops": [],
                    "_head_ops": [],
                    "_base_node_count": 0,
                    "_head_node_count": 0,
                }
            }
        }
        md = render_markdown(diffs)
        assert "🔴" in md

    def test_empty_diffs_dict(self) -> None:
        """Empty diffs dict produces no-change output."""
        md = render_markdown({})
        assert "No architecture changes" in md
        # Summary table header still present
        assert "| Model |" in md
        # No details sections
        assert "<details>" not in md

    def test_commit_shas_displayed(self) -> None:
        """Passing base_ref and head_ref shows comparison line without links."""
        md = render_markdown({}, base_ref="abc1234", head_ref="def5678")
        assert "Comparing `abc1234` → `def5678`" in md

    def test_commit_shas_as_links_when_repo_url_provided(self) -> None:
        """Passing repo_url renders SHAs as clickable GitHub links."""
        md = render_markdown(
            {},
            base_ref="abc1234",
            head_ref="def5678",
            repo_url="https://github.com/onnxruntime/mobius",
        )
        assert "[`abc1234`](https://github.com/onnxruntime/mobius/commit/abc1234)" in md
        assert "[`def5678`](https://github.com/onnxruntime/mobius/commit/def5678)" in md

    def test_commit_shas_omitted_when_not_provided(self) -> None:
        """Without refs, no comparison line appears."""
        md = render_markdown({})
        assert "Comparing" not in md

    def test_multiple_models(self) -> None:
        """Multiple model types are each rendered in the summary table."""
        diffs = {
            "llama": {
                "model": {
                    "changes": [
                        {"type": "added_node", "details": "+ Relu"},
                    ],
                    "_base_ops": ["Add"],
                    "_head_ops": ["Add", "Relu"],
                    "_base_node_count": 1,
                    "_head_node_count": 2,
                }
            },
            "bert": {
                "model": {
                    "changes": [
                        {"type": "changed_attrs", "details": "node[0] Concat: axis: 0 → 1"},
                    ],
                    "_base_ops": ["Concat"],
                    "_head_ops": ["Concat"],
                    "_base_node_count": 1,
                    "_head_node_count": 1,
                }
            },
        }
        md = render_markdown(diffs)
        assert "| llama" in md
        assert "| bert" in md
        # Both detail sections rendered (sorted alphabetically: bert < llama)
        assert "bert / model" in md
        assert "llama / model" in md
        # Different severity emojis
        assert "🟡" in md  # llama: moderate (added_node)
        assert "🔵" in md  # bert: minor (changed_attrs)

    def test_multiple_sub_models(self) -> None:
        """Multiple sub-models within one model type."""
        diffs = {
            "whisper": {
                "encoder": {
                    "changes": [],
                    "_base_ops": ["Add"],
                    "_head_ops": ["Add"],
                    "_base_node_count": 1,
                    "_head_node_count": 1,
                },
                "decoder": {
                    "changes": [
                        {"type": "added_node", "details": "+ Softmax"},
                    ],
                    "_base_ops": ["Add"],
                    "_head_ops": ["Add", "Softmax"],
                    "_base_node_count": 1,
                    "_head_node_count": 2,
                },
            }
        }
        md = render_markdown(diffs)
        assert "| whisper | decoder" in md
        assert "| whisper | encoder" in md
        # encoder has no changes → no-change emoji
        assert "⚪" in md


# ------------------------------------------------------------------
# Integration: canonicalize → diff round-trip
# ------------------------------------------------------------------


class TestRoundTrip:
    """End-to-end tests combining canonicalize + diff."""

    def test_identical_graph_round_trip(self) -> None:
        g = _add_relu_graph()
        canon = canonicalize_graph(g)
        assert diff_graphs(canon, canon) == []

    def test_structural_change_detected(self) -> None:
        base = _simple_add_graph()
        head = _add_relu_graph()
        changes = diff_graphs(canonicalize_graph(base), canonicalize_graph(head))
        assert len(changes) > 0
