# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
from onnx import TensorProto, helper, numpy_helper
from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius.rewrite_rules._gelu_fusion import (
    gelu_fusion_rules,
)
from mobius.rewrite_rules._testing_utils import count_ops


def _build_exact_gelu(*, half_first: bool = False) -> ir.Model:
    """Build a model with decomposed exact GeLU.

    GeLU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8])

    inits = [
        numpy_helper.from_array(np.array(math.sqrt(2), dtype=np.float32), name="sqrt2"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="half"),
    ]

    nodes = [
        helper.make_node("Div", ["x", "sqrt2"], ["x_div"]),
        helper.make_node("Erf", ["x_div"], ["erf_out"]),
        helper.make_node("Add", ["erf_out", "one"], ["add_one"]),
        helper.make_node("Mul", ["x", "add_one"], ["mul_x"]),
    ]
    if half_first:
        nodes.append(helper.make_node("Mul", ["half", "mul_x"], ["y"]))
    else:
        nodes.append(helper.make_node("Mul", ["mul_x", "half"], ["y"]))

    graph = helper.make_graph(nodes, "test_gelu", [x], [y], inits)
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    return ir.from_proto(proto)


def _build_approx_gelu(*, half_first: bool = False) -> ir.Model:
    """Build a model with decomposed approximate (tanh) GeLU.

    GeLU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8])

    inits = [
        numpy_helper.from_array(np.array(3.0, dtype=np.float32), name="three"),
        numpy_helper.from_array(np.array(0.044715, dtype=np.float32), name="coeff"),
        numpy_helper.from_array(
            np.array(math.sqrt(2.0 / math.pi), dtype=np.float32),
            name="sqrt_2pi",
        ),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="half"),
    ]

    nodes = [
        helper.make_node("Pow", ["x", "three"], ["x3"]),
        helper.make_node("Mul", ["coeff", "x3"], ["scaled_cube"]),
        helper.make_node("Add", ["x", "scaled_cube"], ["inner"]),
        helper.make_node("Mul", ["sqrt_2pi", "inner"], ["scaled"]),
        helper.make_node("Tanh", ["scaled"], ["tanh_out"]),
        helper.make_node("Add", ["tanh_out", "one"], ["add_one"]),
        helper.make_node("Mul", ["x", "add_one"], ["mul_x"]),
    ]
    if half_first:
        nodes.append(helper.make_node("Mul", ["half", "mul_x"], ["y"]))
    else:
        nodes.append(helper.make_node("Mul", ["mul_x", "half"], ["y"]))

    graph = helper.make_graph(nodes, "test_approx_gelu", [x], [y], inits)
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    return ir.from_proto(proto)


def _build_non_gelu_model() -> ir.Model:
    """Build a model with Erf but NOT in a GeLU pattern."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8])

    nodes = [
        helper.make_node("Erf", ["x"], ["erf_out"]),
        helper.make_node("Abs", ["erf_out"], ["y"]),
    ]

    graph = helper.make_graph(nodes, "test", [x], [y])
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    return ir.from_proto(proto)


class TestGeluFusionRules:
    def test_rules_returns_rule_set(self):
        rules = gelu_fusion_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_fuses_exact_gelu(self):
        """Decomposed exact GeLU → single Gelu op."""
        model = _build_exact_gelu()
        counts_before = count_ops(model)
        assert counts_before["Erf"] == 1
        assert counts_before["Div"] == 1

        rewrite(model, pattern_rewrite_rules=gelu_fusion_rules())

        counts_after = count_ops(model)
        assert counts_after["Gelu"] == 1
        assert counts_after.get("Erf", 0) == 0
        assert counts_after.get("Div", 0) == 0

    def test_fuses_exact_gelu_half_first(self):
        """Variant: Mul(0.5, Mul(x, ...)) ordering."""
        model = _build_exact_gelu(half_first=True)

        rewrite(model, pattern_rewrite_rules=gelu_fusion_rules())

        counts_after = count_ops(model)
        assert counts_after["Gelu"] == 1
        assert counts_after.get("Erf", 0) == 0

    def test_exact_gelu_has_approximate_none(self):
        """Fused exact Gelu must have approximate='none'."""
        model = _build_exact_gelu()
        rewrite(model, pattern_rewrite_rules=gelu_fusion_rules())

        gelu_nodes = [n for n in model.graph if n.op_type == "Gelu"]
        assert len(gelu_nodes) == 1
        approx = gelu_nodes[0].attributes.get("approximate", None)
        approx_val = approx.value if approx else "none"
        assert approx_val == "none"

    def test_fuses_approx_gelu(self):
        """Decomposed tanh-approximate GeLU → Gelu(approximate='tanh')."""
        model = _build_approx_gelu()
        counts_before = count_ops(model)
        assert counts_before["Tanh"] == 1
        assert counts_before["Pow"] == 1

        rewrite(model, pattern_rewrite_rules=gelu_fusion_rules())

        counts_after = count_ops(model)
        assert counts_after["Gelu"] == 1
        assert counts_after.get("Tanh", 0) == 0
        assert counts_after.get("Pow", 0) == 0

    def test_fuses_approx_gelu_half_first(self):
        """Approx GeLU variant: Mul(0.5, Mul(x, ...))."""
        model = _build_approx_gelu(half_first=True)

        rewrite(model, pattern_rewrite_rules=gelu_fusion_rules())

        counts_after = count_ops(model)
        assert counts_after["Gelu"] == 1
        assert counts_after.get("Tanh", 0) == 0

    def test_approx_gelu_has_approximate_tanh(self):
        """Fused approximate Gelu must have approximate='tanh'."""
        model = _build_approx_gelu()
        rewrite(model, pattern_rewrite_rules=gelu_fusion_rules())

        gelu_nodes = [n for n in model.graph if n.op_type == "Gelu"]
        assert len(gelu_nodes) == 1
        approx = gelu_nodes[0].attributes.get("approximate", None)
        assert approx is not None
        assert approx.value == "tanh"

    def test_preserves_non_matching_model(self):
        """Model with Erf but not in a GeLU pattern is not affected."""
        model = _build_non_gelu_model()
        counts_before = count_ops(model)
        assert counts_before["Erf"] == 1

        rewrite(model, pattern_rewrite_rules=gelu_fusion_rules())

        counts_after = count_ops(model)
        assert counts_after.get("Gelu", 0) == 0
        assert counts_after["Erf"] == 1
