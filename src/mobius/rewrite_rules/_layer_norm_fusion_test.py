# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnx import TensorProto, helper, numpy_helper
from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius.rewrite_rules._layer_norm_fusion import (
    layer_norm_fusion_rules,
)
from mobius.rewrite_rules._testing_utils import count_ops


def _build_decomposed_layernorm(
    hidden: int = 8,
    *,
    include_bias: bool = True,
    epsilon: float = 1e-5,
) -> ir.Model:
    """Build a model with decomposed LayerNorm from primitive ops."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, hidden])
    out_name = "y"
    y = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, 4, hidden])

    inits = [
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes"),
        numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="two"),
        numpy_helper.from_array(np.array(epsilon, dtype=np.float32), name="eps"),
        numpy_helper.from_array(np.ones(hidden, dtype=np.float32), name="gamma"),
    ]
    if include_bias:
        inits.append(numpy_helper.from_array(np.zeros(hidden, dtype=np.float32), name="beta"))

    nodes = [
        helper.make_node("ReduceMean", ["x", "axes"], ["mean"], keepdims=1),
        helper.make_node("Sub", ["x", "mean"], ["diff"]),
        helper.make_node("Pow", ["diff", "two"], ["sq"]),
        helper.make_node("ReduceMean", ["sq", "axes"], ["var"], keepdims=1),
        helper.make_node("Add", ["var", "eps"], ["var_eps"]),
        helper.make_node("Sqrt", ["var_eps"], ["std"]),
        helper.make_node("Div", ["diff", "std"], ["normalized"]),
        helper.make_node("Mul", ["normalized", "gamma"], ["scaled"]),
    ]

    if include_bias:
        nodes.append(helper.make_node("Add", ["scaled", "beta"], [out_name]))
    else:
        # Rename last node output
        nodes[-1] = helper.make_node("Mul", ["normalized", "gamma"], [out_name])

    graph = helper.make_graph(nodes, "test_ln", [x], [y], inits)
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    return ir.from_proto(proto)


def _build_non_layernorm_model() -> ir.Model:
    """Build a model with ReduceMean + Mul that is NOT a LayerNorm."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8])

    inits = [
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes"),
        numpy_helper.from_array(np.ones(8, dtype=np.float32), name="gamma"),
    ]
    nodes = [
        helper.make_node("ReduceMean", ["x", "axes"], ["mean"], keepdims=1),
        helper.make_node("Mul", ["mean", "gamma"], ["y"]),
    ]

    graph = helper.make_graph(nodes, "test", [x], [y], inits)
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    return ir.from_proto(proto)


class TestLayerNormFusionRules:
    def test_rules_returns_rule_set(self):
        rules = layer_norm_fusion_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_fuses_decomposed_layernorm_with_bias(self):
        """Full decomposed LayerNorm → single LayerNormalization."""
        model = _build_decomposed_layernorm(include_bias=True)
        counts_before = count_ops(model)
        assert counts_before["ReduceMean"] == 2
        assert counts_before["Pow"] == 1
        assert counts_before["Sub"] == 1

        rewrite(
            model,
            pattern_rewrite_rules=layer_norm_fusion_rules(),
        )

        counts_after = count_ops(model)
        assert counts_after["LayerNormalization"] == 1
        assert counts_after.get("ReduceMean", 0) == 0
        assert counts_after.get("Sub", 0) == 0
        assert counts_after.get("Pow", 0) == 0
        assert counts_after.get("Sqrt", 0) == 0
        assert counts_after.get("Div", 0) == 0

    def test_fuses_decomposed_layernorm_without_bias(self):
        """Bias-free decomposed LayerNorm → LayerNormalization."""
        model = _build_decomposed_layernorm(include_bias=False)
        counts_before = count_ops(model)
        assert counts_before["ReduceMean"] == 2
        assert counts_before.get("Add", 0) == 1  # only var + eps

        rewrite(
            model,
            pattern_rewrite_rules=layer_norm_fusion_rules(),
        )

        counts_after = count_ops(model)
        assert counts_after["LayerNormalization"] == 1
        assert counts_after.get("ReduceMean", 0) == 0

    def test_preserves_epsilon_value(self):
        """The fused op should carry the original epsilon value."""
        eps = 1e-6
        model = _build_decomposed_layernorm(epsilon=eps)

        rewrite(
            model,
            pattern_rewrite_rules=layer_norm_fusion_rules(),
        )

        ln_nodes = [n for n in model.graph if n.op_type == "LayerNormalization"]
        assert len(ln_nodes) == 1
        fused_eps = ln_nodes[0].attributes.get_float("epsilon")
        assert abs(fused_eps - eps) < 1e-10

    def test_preserves_non_matching_model(self):
        """Models without the decomposed pattern are not affected."""
        model = _build_non_layernorm_model()
        counts_before = count_ops(model)

        rewrite(
            model,
            pattern_rewrite_rules=layer_norm_fusion_rules(),
        )

        counts_after = count_ops(model)
        assert counts_after.get("LayerNormalization", 0) == 0
        assert counts_after == counts_before

    def test_multiple_layernorms(self):
        """Model with two sequential decomposed LNs → both get fused."""
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])
        y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8])
        inits = [
            numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes"),
            numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="two"),
            numpy_helper.from_array(np.array(1e-5, dtype=np.float32), name="eps"),
            numpy_helper.from_array(np.ones(8, dtype=np.float32), name="g1"),
            numpy_helper.from_array(np.zeros(8, dtype=np.float32), name="b1"),
            numpy_helper.from_array(np.ones(8, dtype=np.float32), name="g2"),
            numpy_helper.from_array(np.zeros(8, dtype=np.float32), name="b2"),
        ]

        # First LN
        nodes = [
            helper.make_node("ReduceMean", ["x", "axes"], ["m1"], keepdims=1),
            helper.make_node("Sub", ["x", "m1"], ["d1"]),
            helper.make_node("Pow", ["d1", "two"], ["s1"]),
            helper.make_node("ReduceMean", ["s1", "axes"], ["v1"], keepdims=1),
            helper.make_node("Add", ["v1", "eps"], ["ve1"]),
            helper.make_node("Sqrt", ["ve1"], ["st1"]),
            helper.make_node("Div", ["d1", "st1"], ["n1"]),
            helper.make_node("Mul", ["n1", "g1"], ["sc1"]),
            helper.make_node("Add", ["sc1", "b1"], ["ln1"]),
        ]
        # Second LN
        nodes += [
            helper.make_node("ReduceMean", ["ln1", "axes"], ["m2"], keepdims=1),
            helper.make_node("Sub", ["ln1", "m2"], ["d2"]),
            helper.make_node("Pow", ["d2", "two"], ["s2"]),
            helper.make_node("ReduceMean", ["s2", "axes"], ["v2"], keepdims=1),
            helper.make_node("Add", ["v2", "eps"], ["ve2"]),
            helper.make_node("Sqrt", ["ve2"], ["st2"]),
            helper.make_node("Div", ["d2", "st2"], ["n2"]),
            helper.make_node("Mul", ["n2", "g2"], ["sc2"]),
            helper.make_node("Add", ["sc2", "b2"], ["y"]),
        ]

        graph = helper.make_graph(nodes, "test", [x_info], [y_info], inits)
        proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        model = ir.from_proto(proto)

        counts_before = count_ops(model)
        assert counts_before["ReduceMean"] == 4

        rewrite(
            model,
            pattern_rewrite_rules=layer_norm_fusion_rules(),
        )

        counts_after = count_ops(model)
        assert counts_after["LayerNormalization"] == 2
        assert counts_after.get("ReduceMean", 0) == 0
