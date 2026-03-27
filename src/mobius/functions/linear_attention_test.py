# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LinearAttention ir.Function builder."""

from __future__ import annotations

import pytest

from mobius.functions.linear_attention import linear_attention


class TestLinearAttentionSeparateQKV:
    """Tests for LinearAttention with separate Q, K, V inputs."""

    def test_function_has_six_inputs(self):
        func = linear_attention(q_num_heads=2, kv_num_heads=4, scale=0.25)
        assert len(func.graph.inputs) == 6
        names = [inp.name for inp in func.graph.inputs]
        assert names == [
            "query",
            "key",
            "value",
            "past_state",
            "decay",
            "beta",
        ]

    def test_function_has_two_outputs(self):
        func = linear_attention(q_num_heads=2, kv_num_heads=4, scale=0.25)
        assert len(func.graph.outputs) == 2
        names = [out.name for out in func.graph.outputs]
        assert "output" in names[0]
        assert "present_state" in names[1]

    def test_function_name_and_domain(self):
        func = linear_attention(q_num_heads=2, kv_num_heads=4, scale=0.25)
        assert func.name == "LinearAttention"
        assert func.domain == "com.microsoft"

    def test_attributes_present(self):
        func = linear_attention(
            q_num_heads=2,
            kv_num_heads=4,
            scale=0.25,
            update_rule="gated_delta",
        )
        attr_names = set(func.attributes.keys())
        assert "q_num_heads" in attr_names
        assert "kv_num_heads" in attr_names
        assert "scale" in attr_names
        assert "update_rule" in attr_names

    def test_graph_contains_scan(self):
        func = linear_attention(q_num_heads=2, kv_num_heads=4, scale=0.25)
        op_types = {n.op_type for n in func.graph}
        assert "Scan" in op_types

    def test_gqa_ratio_1_no_tile(self):
        """When q_num_heads == kv_num_heads, no Tile is needed."""
        func = linear_attention(q_num_heads=4, kv_num_heads=4, scale=0.25)
        op_types = [n.op_type for n in func.graph]
        assert "Tile" not in op_types

    def test_gqa_ratio_gt1_has_tile(self):
        """When kv_num_heads > q_num_heads, Tile expands heads."""
        func = linear_attention(q_num_heads=2, kv_num_heads=8, scale=0.25)
        op_types = [n.op_type for n in func.graph]
        assert "Tile" in op_types

    def test_invalid_gqa_ratio_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            linear_attention(q_num_heads=3, kv_num_heads=5, scale=0.25)

    def test_invalid_update_rule_raises(self):
        with pytest.raises(ValueError, match="Unknown update_rule"):
            linear_attention(
                q_num_heads=2,
                kv_num_heads=4,
                update_rule="invalid",
            )


class TestLinearAttentionUpdateRules:
    """Tests for different update rule variants."""

    @pytest.mark.parametrize(
        "rule",
        ["linear", "gated", "delta", "gated_delta"],
    )
    def test_valid_update_rules_build(self, rule):
        func = linear_attention(
            q_num_heads=2,
            kv_num_heads=4,
            scale=0.25,
            update_rule=rule,
        )
        assert func.attributes["update_rule"].value == rule
        # All variants should produce a valid graph with Scan
        op_types = {n.op_type for n in func.graph}
        assert "Scan" in op_types

    def test_gated_delta_scan_body_has_exp(self):
        """gated_delta uses Exp for decay in the Scan body."""
        func = linear_attention(
            q_num_heads=2,
            kv_num_heads=4,
            scale=0.25,
            update_rule="gated_delta",
        )
        # Find the Scan node and inspect its body
        scan_nodes = [n for n in func.graph if n.op_type == "Scan"]
        assert len(scan_nodes) == 1
        body = scan_nodes[0].attributes["body"].value
        body_ops = {n.op_type for n in body}
        assert "Exp" in body_ops  # decay uses Exp
        assert "Sub" in body_ops  # delta uses Sub(v, retrieval)

    def test_linear_scan_body_no_exp_no_sub(self):
        """Linear rule: no decay (Exp) and no beta (Sub)."""
        func = linear_attention(
            q_num_heads=2,
            kv_num_heads=4,
            scale=0.25,
            update_rule="linear",
        )
        scan_nodes = [n for n in func.graph if n.op_type == "Scan"]
        assert len(scan_nodes) == 1
        body = scan_nodes[0].attributes["body"].value
        body_ops = {n.op_type for n in body}
        assert "Exp" not in body_ops
        assert "Sub" not in body_ops


class TestLinearAttentionDecayShapes:
    """Tests verifying decay shape handling (3D → 4D internally)."""

    def test_decay_reshape_present(self):
        """Function should Reshape decay from 3D to 4D."""
        func = linear_attention(q_num_heads=2, kv_num_heads=4, scale=0.25)
        # Count Reshape ops — should have reshapes for
        # q, k, v, decay (3D→4D) and output (4D→3D)
        reshape_count = sum(1 for n in func.graph if n.op_type == "Reshape")
        assert reshape_count >= 5  # q, k, v, decay, output

    def test_beta_transpose_present(self):
        """Function should Transpose beta from (B,T,H) to (B,H,T)."""
        func = linear_attention(q_num_heads=2, kv_num_heads=4, scale=0.25)
        transpose_count = sum(1 for n in func.graph if n.op_type == "Transpose")
        # At least: q, k, v (3D→4D), decay (3D→4D), beta,
        # q/k/v/decay/beta (4D→T-first for Scan), output (T→B first)
        assert transpose_count >= 6
