# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GatedDeltaNet linear attention component."""

from __future__ import annotations

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
    make_config,
)
from mobius.components._gated_deltanet import GatedDeltaNet


class TestGatedDeltaNet:
    """Tests for the GatedDeltaNet linear attention module."""

    def _make_deltanet_config(self, **overrides):
        defaults = dict(
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
        )
        defaults.update(overrides)
        return make_config(**defaults)

    def test_projection_parameters(self):
        config = self._make_deltanet_config()
        dn = GatedDeltaNet(config)
        param_names = [n for n, _ in dn.named_parameters()]
        assert any("in_proj_qkv" in n for n in param_names)
        assert any("in_proj_z" in n for n in param_names)
        assert any("in_proj_b" in n for n in param_names)
        assert any("in_proj_a" in n for n in param_names)
        assert any("out_proj" in n for n in param_names)

    def test_conv1d_weight_shape(self):
        config = self._make_deltanet_config()
        dn = GatedDeltaNet(config)
        # conv_dim = key_dim*2 + value_dim = 2*32 + 4*16 = 128
        key_dim = 2 * 16  # num_k_heads * head_k_dim
        value_dim = 4 * 16  # num_v_heads * head_v_dim
        conv_dim = key_dim * 2 + value_dim
        assert list(dn.conv1d.weight.shape) == [conv_dim, 1, 4]

    def test_learnable_params(self):
        config = self._make_deltanet_config()
        dn = GatedDeltaNet(config)
        # dt_bias and A_log: (num_v_heads,)
        assert list(dn.dt_bias.shape) == [4]
        assert list(dn.A_log.shape) == [4]

    def test_in_proj_qkv_shape(self):
        config = self._make_deltanet_config()
        dn = GatedDeltaNet(config)
        key_dim = 2 * 16
        value_dim = 4 * 16
        # in_proj_qkv: (key_dim*2 + value_dim, hidden_size)
        assert list(dn.in_proj_qkv.weight.shape) == [
            key_dim * 2 + value_dim,
            64,
        ]

    def test_forward_builds_graph(self):
        config = self._make_deltanet_config()
        dn = GatedDeltaNet(config)
        builder, op, graph = create_test_builder()

        hidden = create_test_input(builder, "hidden", [1, 1, 64])
        key_dim = 2 * 16
        value_dim = 4 * 16
        conv_dim = key_dim * 2 + value_dim
        conv_state = create_test_input(builder, "conv_state", [1, conv_dim, 3])
        rec_state = create_test_input(builder, "rec_state", [1, 4, 16, 16])

        output, new_conv, new_rec = dn(op, hidden, conv_state, rec_state)
        builder._adapt_outputs([output, new_conv, new_rec])
        assert graph.num_nodes() > 0
        # Function ops called by name
        assert count_op_type(graph, "CausalConvWithState") >= 1
        assert count_op_type(graph, "LinearAttention") >= 1
        # No Scan/Conv in parent graph — all inside functions
        assert count_op_type(graph, "Scan") == 0
        assert count_op_type(graph, "Conv") == 0

    def test_forward_has_sigmoid_for_beta(self):
        config = self._make_deltanet_config()
        dn = GatedDeltaNet(config)
        builder, op, graph = create_test_builder()

        key_dim = 2 * 16
        value_dim = 4 * 16
        conv_dim = key_dim * 2 + value_dim

        hidden = create_test_input(builder, "hidden", [1, 1, 64])
        conv_state = create_test_input(builder, "conv_state", [1, conv_dim, 3])
        rec_state = create_test_input(builder, "rec_state", [1, 4, 16, 16])

        dn(op, hidden, conv_state, rec_state)
        # Sigmoid for beta gate
        assert count_op_type(graph, "Sigmoid") >= 1

    def test_forward_has_softplus(self):
        config = self._make_deltanet_config()
        dn = GatedDeltaNet(config)
        builder, op, graph = create_test_builder()

        key_dim = 2 * 16
        value_dim = 4 * 16
        conv_dim = key_dim * 2 + value_dim

        hidden = create_test_input(builder, "hidden", [1, 1, 64])
        conv_state = create_test_input(builder, "conv_state", [1, conv_dim, 3])
        rec_state = create_test_input(builder, "rec_state", [1, 4, 16, 16])

        dn(op, hidden, conv_state, rec_state)
        assert count_op_type(graph, "Softplus") >= 1

    def test_gqa_head_expansion_deferred_to_function(self):
        """GQA expansion is inside LinearAttention function.

        The parent graph should NOT have Tile ops for head expansion —
        Q/K are passed at their native head count.
        """
        config = self._make_deltanet_config(
            linear_num_value_heads=8,
            linear_num_key_heads=2,
        )
        dn = GatedDeltaNet(config)
        assert dn.num_v_heads == 8
        assert dn.num_k_heads == 2
        builder, op, graph = create_test_builder()

        key_dim = 2 * 16
        value_dim = 8 * 16
        conv_dim = key_dim * 2 + value_dim

        hidden = create_test_input(builder, "hidden", [1, 1, 64])
        conv_state = create_test_input(builder, "conv_state", [1, conv_dim, 3])
        rec_state = create_test_input(builder, "rec_state", [1, 8, 16, 16])

        output, new_conv, new_rec = dn(op, hidden, conv_state, rec_state)
        builder._adapt_outputs([output, new_conv, new_rec])
        # No Tile in parent — GQA is inside the function
        assert count_op_type(graph, "Tile") == 0
        assert count_op_type(graph, "LinearAttention") >= 1

    def test_l2_norm_uses_decomposed_ops(self):
        """L2 normalization uses ReduceSumSquare/Sqrt decomposition.

        LpNormalization is not yet supported by ORT (<1.25), so
        the graph must contain the decomposed form instead.
        """
        config = self._make_deltanet_config()
        dn = GatedDeltaNet(config)
        builder, op, graph = create_test_builder()

        key_dim = 2 * 16
        value_dim = 4 * 16
        conv_dim = key_dim * 2 + value_dim

        hidden = create_test_input(builder, "hidden", [1, 1, 64])
        conv_state = create_test_input(builder, "conv_state", [1, conv_dim, 3])
        rec_state = create_test_input(builder, "rec_state", [1, 4, 16, 16])

        dn(op, hidden, conv_state, rec_state)
        # Decomposed L2 norm: Sqrt(Max(ReduceSumSquare(...), eps))
        assert count_op_type(graph, "ReduceSumSquare") >= 2
        assert count_op_type(graph, "Sqrt") >= 2
        # LpNormalization must NOT appear (unsupported by ORT <1.25)
        assert count_op_type(graph, "LpNormalization") == 0
