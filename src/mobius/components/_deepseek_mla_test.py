# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DeepSeek Multi-head Latent Attention (MLA) component."""

from __future__ import annotations

import pytest

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
    make_config,
)
from mobius.components._deepseek_mla import DeepSeekMLA

# DeepSeek MLA config: uses low-rank KV compression with separate
# nope (non-positional) and rope (rotary) head dimensions.
# qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
_MLA_DEFAULTS = dict(
    hidden_size=64,
    num_attention_heads=4,
    q_lora_rank=16,
    kv_lora_rank=8,
    qk_nope_head_dim=12,
    qk_rope_head_dim=4,
    v_head_dim=8,
    rope_interleave=False,
    rms_norm_eps=1e-6,
)


def _mla_config(**overrides):
    """Create a test config for DeepSeek MLA."""
    kw = {**_MLA_DEFAULTS, **overrides}
    return make_config(**kw)


class TestDeepSeekMLAInit:
    """Tests for MLA module initialization and parameter shapes."""

    def test_q_lora_projections(self):
        """With q_lora_rank > 0: q_a_proj, q_a_layernorm, q_b_proj."""
        config = _mla_config()
        mla = DeepSeekMLA(config)
        # q_a_proj: (hidden, q_lora_rank) = (64, 16)
        assert list(mla.q_a_proj.weight.shape) == [16, 64]
        # q_b_proj: (q_lora_rank, num_heads * qk_head_dim) = (16, 4*16)
        qk_head_dim = 12 + 4  # nope + rope
        assert list(mla.q_b_proj.weight.shape) == [4 * qk_head_dim, 16]
        assert mla.q_a_layernorm is not None

    def test_no_q_lora_uses_direct_proj(self):
        """With q_lora_rank=None: single q_proj, no layernorm."""
        config = _mla_config(q_lora_rank=None)
        mla = DeepSeekMLA(config)
        qk_head_dim = 12 + 4
        assert list(mla.q_proj.weight.shape) == [4 * qk_head_dim, 64]
        assert not hasattr(mla, "q_a_proj")

    def test_zero_q_lora_uses_direct_proj(self):
        """With q_lora_rank=0: same as None, direct projection."""
        config = _mla_config(q_lora_rank=0)
        mla = DeepSeekMLA(config)
        assert hasattr(mla, "q_proj")
        assert not hasattr(mla, "q_a_proj")

    def test_kv_a_proj_shape(self):
        """kv_a_proj_with_mqa: (hidden, kv_lora_rank + rope_dim)."""
        config = _mla_config()
        mla = DeepSeekMLA(config)
        # (kv_lora_rank + qk_rope_head_dim, hidden) = (8 + 4, 64)
        assert list(mla.kv_a_proj_with_mqa.weight.shape) == [12, 64]

    def test_kv_b_proj_shape(self):
        """kv_b_proj: decompresses latent KV into per-head k_nope + v."""
        config = _mla_config()
        mla = DeepSeekMLA(config)
        # (num_heads * (nope + v_dim), kv_lora_rank)
        # = (4 * (12 + 8), 8) = (80, 8)
        assert list(mla.kv_b_proj.weight.shape) == [80, 8]

    def test_kv_a_layernorm_dim(self):
        """kv_a_layernorm normalizes the kv_lora_rank dimension."""
        config = _mla_config()
        mla = DeepSeekMLA(config)
        assert list(mla.kv_a_layernorm.weight.shape) == [8]

    def test_o_proj_shape(self):
        """o_proj: (hidden, num_heads * v_head_dim)."""
        config = _mla_config()
        mla = DeepSeekMLA(config)
        # (hidden, num_heads * v_dim) = (64, 4 * 8)
        assert list(mla.o_proj.weight.shape) == [64, 32]

    def test_default_scale(self):
        """Default scale = 1/sqrt(qk_head_dim)."""
        config = _mla_config()
        mla = DeepSeekMLA(config)
        qk_head_dim = 12 + 4  # 16
        assert mla.scaling == pytest.approx(qk_head_dim**-0.5)

    def test_custom_scale(self):
        config = _mla_config()
        mla = DeepSeekMLA(config, scale=0.25)
        assert mla.scaling == pytest.approx(0.25)

    def test_head_dim_composition(self):
        """qk_head_dim = qk_nope_head_dim + qk_rope_head_dim."""
        config = _mla_config(qk_nope_head_dim=20, qk_rope_head_dim=6)
        mla = DeepSeekMLA(config)
        assert mla.qk_head_dim == 26
        assert mla.qk_nope_head_dim == 20
        assert mla.qk_rope_head_dim == 6


class TestDeepSeekMLAForward:
    """Tests for MLA forward pass graph construction."""

    def test_forward_builds_graph_with_q_lora(self):
        """Forward with Q LoRA path produces valid graph."""
        config = _mla_config()
        mla = DeepSeekMLA(config)
        builder, op, graph = create_test_builder()

        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])
        cos = create_test_input(builder, "cos", [1, 8, 4])
        sin = create_test_input(builder, "sin", [1, 8, 4])

        output, (present_key, present_value) = mla(
            op,
            hidden,
            attention_bias=bias,
            position_embeddings=(cos, sin),
        )
        builder._adapt_outputs([output, present_key, present_value])

        assert graph.num_nodes() > 0
        assert count_op_type(graph, "Attention") >= 1

    def test_forward_without_q_lora(self):
        """Forward without Q LoRA (direct q_proj)."""
        config = _mla_config(q_lora_rank=None)
        mla = DeepSeekMLA(config)
        builder, op, graph = create_test_builder()

        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])
        cos = create_test_input(builder, "cos", [1, 8, 4])
        sin = create_test_input(builder, "sin", [1, 8, 4])

        output, (pk, pv) = mla(
            op,
            hidden,
            attention_bias=bias,
            position_embeddings=(cos, sin),
        )
        builder._adapt_outputs([output, pk, pv])

        assert graph.num_nodes() > 0
        assert count_op_type(graph, "Attention") >= 1

    def test_forward_with_past_kv(self):
        """Forward with cached KV for decode step."""
        config = _mla_config()
        mla = DeepSeekMLA(config)
        builder, op, graph = create_test_builder()

        # Decode step: seq_len=1, past_seq_len=7
        qk_head_dim = 12 + 4  # nope + rope
        hidden = create_test_input(builder, "hidden", [1, 1, 64])
        bias = create_test_input(builder, "bias", [1, 4, 1, 8])
        cos = create_test_input(builder, "cos", [1, 1, 4])
        sin = create_test_input(builder, "sin", [1, 1, 4])
        # past_key: (B, past_seq, num_heads, qk_head_dim)
        past_key = create_test_input(builder, "pk", [1, 7, 4, qk_head_dim])
        # past_value: (B, past_seq, num_heads, v_head_dim)
        past_value = create_test_input(builder, "pv", [1, 7, 4, 8])

        output, (pk, pv) = mla(
            op,
            hidden,
            attention_bias=bias,
            position_embeddings=(cos, sin),
            past_key_value=(past_key, past_value),
        )
        builder._adapt_outputs([output, pk, pv])

        assert count_op_type(graph, "Attention") >= 1

    def test_rope_interleave_flag(self):
        """rope_interleave config propagates to module."""
        config = _mla_config(rope_interleave=True)
        mla = DeepSeekMLA(config)
        assert mla._rope_interleave is True

        config2 = _mla_config(rope_interleave=False)
        mla2 = DeepSeekMLA(config2)
        assert mla2._rope_interleave is False

    def test_graph_has_split_ops(self):
        """MLA uses Split to separate nope/rope and kv/rope portions."""
        config = _mla_config()
        mla = DeepSeekMLA(config)
        builder, op, graph = create_test_builder()

        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])
        cos = create_test_input(builder, "cos", [1, 8, 4])
        sin = create_test_input(builder, "sin", [1, 8, 4])

        output, (pk, pv) = mla(
            op,
            hidden,
            attention_bias=bias,
            position_embeddings=(cos, sin),
        )
        builder._adapt_outputs([output, pk, pv])

        # MLA splits: q_nope/q_rope, k_pass/k_rope, k_nope/v
        assert count_op_type(graph, "Split") >= 3

    def test_graph_has_concat_ops(self):
        """MLA concatenates nope+rope for final Q and K."""
        config = _mla_config()
        mla = DeepSeekMLA(config)
        builder, op, graph = create_test_builder()

        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])
        cos = create_test_input(builder, "cos", [1, 8, 4])
        sin = create_test_input(builder, "sin", [1, 8, 4])

        output, (pk, pv) = mla(
            op,
            hidden,
            attention_bias=bias,
            position_embeddings=(cos, sin),
        )
        builder._adapt_outputs([output, pk, pv])

        # Q = concat(q_nope, q_rope), K = concat(k_nope, k_rope_expanded)
        assert count_op_type(graph, "Concat") >= 2

    def test_graph_has_rms_normalization(self):
        """MLA uses RMSNormalization for q_a_layernorm and kv_a_layernorm."""
        config = _mla_config()
        mla = DeepSeekMLA(config)
        builder, op, graph = create_test_builder()

        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])
        cos = create_test_input(builder, "cos", [1, 8, 4])
        sin = create_test_input(builder, "sin", [1, 8, 4])

        output, (pk, pv) = mla(
            op,
            hidden,
            attention_bias=bias,
            position_embeddings=(cos, sin),
        )
        builder._adapt_outputs([output, pk, pv])

        # q_a_layernorm + kv_a_layernorm = 2
        assert count_op_type(graph, "RMSNormalization") >= 2


class TestDeepSeekMLADimensions:
    """Tests for different MLA dimension configurations."""

    def test_larger_kv_lora_rank(self):
        """Larger kv_lora_rank increases compressed KV dimension."""
        config = _mla_config(kv_lora_rank=32)
        mla = DeepSeekMLA(config)
        # kv_a_proj: (kv_lora_rank + rope_dim, hidden) = (32 + 4, 64)
        assert list(mla.kv_a_proj_with_mqa.weight.shape) == [36, 64]
        # kv_a_layernorm: (kv_lora_rank,) = (32,)
        assert list(mla.kv_a_layernorm.weight.shape) == [32]

    def test_larger_rope_dim(self):
        """Larger qk_rope_head_dim affects kv_a_proj and q_b_proj."""
        config = _mla_config(qk_rope_head_dim=8)
        mla = DeepSeekMLA(config)
        assert mla.qk_head_dim == 12 + 8  # nope + rope = 20
        # kv_a_proj: (kv_lora_rank + rope_dim, hidden) = (8 + 8, 64)
        assert list(mla.kv_a_proj_with_mqa.weight.shape) == [16, 64]

    def test_v_head_dim_independent_of_qk(self):
        """V head dim can differ from QK head dim (key MLA feature)."""
        config = _mla_config(qk_nope_head_dim=24, qk_rope_head_dim=8, v_head_dim=16)
        mla = DeepSeekMLA(config)
        assert mla.qk_head_dim == 32  # 24 + 8
        assert mla.v_head_dim == 16
        # o_proj: (hidden, num_heads * v_dim) = (64, 4*16)
        assert list(mla.o_proj.weight.shape) == [64, 64]

    def test_forward_with_custom_dims(self):
        """Forward pass works with non-default MLA dimensions."""
        config = _mla_config(
            num_attention_heads=2,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_nope_head_dim=24,
            qk_rope_head_dim=8,
            v_head_dim=16,
        )
        mla = DeepSeekMLA(config)
        builder, op, graph = create_test_builder()

        hidden = create_test_input(builder, "hidden", [1, 4, 64])
        bias = create_test_input(builder, "bias", [1, 2, 4, 4])
        cos = create_test_input(builder, "cos", [1, 4, 8])
        sin = create_test_input(builder, "sin", [1, 4, 8])

        output, (pk, pv) = mla(
            op,
            hidden,
            attention_bias=bias,
            position_embeddings=(cos, sin),
        )
        builder._adapt_outputs([output, pk, pv])

        assert count_op_type(graph, "Attention") >= 1
