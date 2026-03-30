# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for _weight_utils.py — shared weight preprocessing utilities."""

from __future__ import annotations

import pytest
import torch

from mobius._weight_utils import (
    merge_lora_weights,
    preprocess_awq_weights,
    preprocess_gptq_weights,
    split_codegen_qkv,
    split_fused_qkv,
    split_gate_up_proj,
    split_interleaved_qkv,
    strip_prefix,
    tie_word_embeddings,
    vlm_decoder_weights,
    vlm_embedding_weights,
)


class TestSplitFusedQKV:
    """Tests for split_fused_qkv."""

    def test_mha_equal_heads(self):
        """MHA: num_heads == num_kv_heads, all sizes equal."""
        num_heads = 4
        num_kv_heads = 4
        head_dim = 8
        hidden = 16
        total = (num_heads + 2 * num_kv_heads) * head_dim
        weight = torch.arange(total * hidden).reshape(total, hidden).float()

        q, k, v = split_fused_qkv(weight, num_heads, num_kv_heads, head_dim)

        assert q.shape == (num_heads * head_dim, hidden)
        assert k.shape == (num_kv_heads * head_dim, hidden)
        assert v.shape == (num_kv_heads * head_dim, hidden)
        # Values are contiguous slices of the original
        torch.testing.assert_close(q, weight[: num_heads * head_dim])

    def test_gqa_fewer_kv_heads(self):
        """GQA: num_kv_heads < num_heads."""
        num_heads = 8
        num_kv_heads = 2
        head_dim = 4
        q_size = num_heads * head_dim  # 32
        kv_size = num_kv_heads * head_dim  # 8
        total = q_size + 2 * kv_size  # 48
        weight = torch.randn(total)

        q, k, v = split_fused_qkv(weight, num_heads, num_kv_heads, head_dim)

        assert q.shape == (q_size,)
        assert k.shape == (kv_size,)
        assert v.shape == (kv_size,)
        # Verify the splits reconstruct the original
        torch.testing.assert_close(torch.cat([q, k, v]), weight)

    def test_mqa_single_kv_head(self):
        """MQA: num_kv_heads == 1."""
        num_heads = 16
        num_kv_heads = 1
        head_dim = 64
        hidden = 32
        total = (num_heads + 2) * head_dim
        weight = torch.randn(total, hidden)

        q, k, v = split_fused_qkv(weight, num_heads, num_kv_heads, head_dim)

        assert q.shape == (num_heads * head_dim, hidden)
        assert k.shape == (head_dim, hidden)
        assert v.shape == (head_dim, hidden)

    def test_splits_are_views(self):
        """Splits should be views of the original tensor (no copy)."""
        weight = torch.randn(32, 16)
        q, _k, _v = split_fused_qkv(weight, 4, 2, 4)
        assert q.data_ptr() == weight.data_ptr()

    def test_1d_bias_tensor(self):
        """Works with 1D bias tensors too."""
        # num_heads=4, num_kv_heads=2, head_dim=4 → total = 16+8+8 = 32
        bias = torch.randn(32)
        q, k, v = split_fused_qkv(bias, 4, 2, 4)
        assert q.shape == (16,)
        assert k.shape == (8,)
        assert v.shape == (8,)


class TestSplitGateUpProj:
    """Tests for split_gate_up_proj."""

    def test_basic_split(self):
        """Splits a 2D weight at intermediate_size."""
        intermediate_size = 64
        hidden = 32
        weight = torch.randn(2 * intermediate_size, hidden)

        gate, up = split_gate_up_proj(weight, intermediate_size)

        assert gate.shape == (intermediate_size, hidden)
        assert up.shape == (intermediate_size, hidden)
        torch.testing.assert_close(torch.cat([gate, up]), weight)

    def test_1d_bias(self):
        """Works with 1D bias tensors."""
        intermediate_size = 128
        bias = torch.randn(2 * intermediate_size)

        gate, up = split_gate_up_proj(bias, intermediate_size)

        assert gate.shape == (intermediate_size,)
        assert up.shape == (intermediate_size,)

    def test_splits_are_views(self):
        """Splits should be views of the original tensor."""
        weight = torch.randn(256, 64)
        gate, _up = split_gate_up_proj(weight, 128)
        assert gate.data_ptr() == weight.data_ptr()


class TestStripPrefix:
    """Tests for strip_prefix."""

    def test_strips_prefix(self):
        """Basic prefix stripping."""
        state_dict = {
            "model.layers.0.weight": torch.tensor(1.0),
            "model.layers.1.weight": torch.tensor(2.0),
            "model.embed.weight": torch.tensor(3.0),
        }
        result = strip_prefix(state_dict, "model")
        assert set(result.keys()) == {
            "layers.0.weight",
            "layers.1.weight",
            "embed.weight",
        }

    def test_drops_non_matching_keys(self):
        """Keys without the prefix are excluded."""
        state_dict = {
            "model.weight": torch.tensor(1.0),
            "other.weight": torch.tensor(2.0),
        }
        result = strip_prefix(state_dict, "model")
        assert list(result.keys()) == ["weight"]

    def test_trailing_dot_optional(self):
        """Prefix with or without trailing dot gives same result."""
        state_dict = {"foo.bar": torch.tensor(1.0)}
        result_no_dot = strip_prefix(state_dict, "foo")
        result_with_dot = strip_prefix(state_dict, "foo.")
        assert result_no_dot == result_with_dot

    def test_empty_state_dict(self):
        """Empty input returns empty output."""
        result = strip_prefix({}, "model")
        assert result == {}

    def test_preserves_tensor_values(self):
        """Tensor values are preserved (not copied)."""
        t = torch.randn(3, 4)
        result = strip_prefix({"prefix.key": t}, "prefix")
        assert result["key"].data_ptr() == t.data_ptr()


class TestSplitFusedQkvValidation:
    """Tests for QKV dimension mismatch detection."""

    def test_wrong_dim_raises(self):
        weight = torch.randn(100, 64)  # wrong size
        with pytest.raises(ValueError, match="QKV weight dim 0 is 100"):
            split_fused_qkv(weight, num_heads=8, num_kv_heads=2, head_dim=8)

    def test_too_small_raises(self):
        # Expected: 8*8 + 2*2*8 = 96, but provide 80
        weight = torch.randn(80, 64)
        with pytest.raises(ValueError, match="expected 96"):
            split_fused_qkv(weight, num_heads=8, num_kv_heads=2, head_dim=8)


class TestSplitGateUpProjValidation:
    """Tests for gate_up dimension mismatch detection."""

    def test_wrong_dim_raises(self):
        weight = torch.randn(100, 64)  # wrong size
        with pytest.raises(ValueError, match="gate_up weight dim 0 is 100"):
            split_gate_up_proj(weight, intermediate_size=64)

    def test_odd_size_raises(self):
        # Expected: 2*32 = 64, but provide 63
        weight = torch.randn(63, 16)
        with pytest.raises(ValueError, match="expected 64"):
            split_gate_up_proj(weight, intermediate_size=32)


class TestTieWordEmbeddings:
    """Tests for tie_word_embeddings."""

    def test_copies_embed_to_head(self):
        """If lm_head is missing, copies embed_tokens."""
        t = torch.randn(4, 8)
        sd = {"model.embed_tokens.weight": t}
        tie_word_embeddings(sd)
        assert "lm_head.weight" in sd
        assert sd["lm_head.weight"].data_ptr() == t.data_ptr()

    def test_copies_head_to_embed(self):
        """If embed_tokens is missing, copies lm_head."""
        t = torch.randn(4, 8)
        sd = {"lm_head.weight": t}
        tie_word_embeddings(sd)
        assert "model.embed_tokens.weight" in sd
        assert sd["model.embed_tokens.weight"].data_ptr() == t.data_ptr()

    def test_both_present_no_change(self):
        """If both present, no change."""
        t1 = torch.randn(4, 8)
        t2 = torch.randn(4, 8)
        sd = {"model.embed_tokens.weight": t1, "lm_head.weight": t2}
        tie_word_embeddings(sd)
        assert sd["model.embed_tokens.weight"].data_ptr() == t1.data_ptr()
        assert sd["lm_head.weight"].data_ptr() == t2.data_ptr()

    def test_neither_present_no_change(self):
        """If neither present, no change."""
        sd = {"other.weight": torch.randn(4)}
        tie_word_embeddings(sd)
        assert list(sd.keys()) == ["other.weight"]

    def test_custom_keys(self):
        """Custom embed/head keys."""
        t = torch.randn(4, 8)
        sd = {"enc.embed.weight": t}
        tie_word_embeddings(sd, embed_key="enc.embed.weight", head_key="dec.head.weight")
        assert "dec.head.weight" in sd


class TestVlmDecoderWeights:
    """Tests for vlm_decoder_weights."""

    def test_strips_prefix_and_ties(self):
        """Strips language_model. prefix and ties embeddings."""
        embed = torch.randn(4, 8)
        sd = {
            "language_model.model.layers.0.weight": torch.randn(4),
            "language_model.model.embed_tokens.weight": embed,
            "language_model.model.norm.weight": torch.randn(4),
            "vision_model.encoder.weight": torch.randn(4),
        }
        result = vlm_decoder_weights(sd, tie=True)
        assert "model.layers.0.weight" in result
        assert "model.embed_tokens.weight" in result
        assert "lm_head.weight" in result
        # Vision weights are excluded
        assert not any("vision" in k for k in result)

    def test_no_tie(self):
        """Without tie, lm_head is not added."""
        sd = {
            "language_model.model.embed_tokens.weight": torch.randn(4, 8),
        }
        result = vlm_decoder_weights(sd, tie=False)
        assert "lm_head.weight" not in result

    def test_returns_new_dict(self):
        """Returns a new dict, doesn't modify input."""
        sd = {"language_model.w": torch.randn(4)}
        result = vlm_decoder_weights(sd)
        assert "w" in result
        assert "language_model.w" in sd  # original unchanged

    def test_custom_prefix(self):
        """Works with a non-default prefix."""
        sd = {"decoder.layers.0.weight": torch.randn(4)}
        result = vlm_decoder_weights(sd, prefix="decoder.")
        assert "layers.0.weight" in result


class TestVlmEmbeddingWeights:
    """Tests for vlm_embedding_weights."""

    def test_filters_and_strips(self):
        """Filters embed_tokens and strips prefixes."""
        t = torch.randn(4, 8)
        sd = {
            "language_model.model.embed_tokens.weight": t,
            "language_model.model.layers.0.weight": torch.randn(4),
            "vision_model.encoder.weight": torch.randn(4),
        }
        result = vlm_embedding_weights(sd)
        assert list(result.keys()) == ["embed_tokens.weight"]
        assert result["embed_tokens.weight"].data_ptr() == t.data_ptr()

    def test_strips_shorter_prefix(self):
        """Falls through to shorter prefix."""
        t = torch.randn(4, 8)
        sd = {"language_model.embed_tokens.weight": t}
        result = vlm_embedding_weights(sd)
        assert list(result.keys()) == ["embed_tokens.weight"]

    def test_no_prefix_match(self):
        """Keys without matching prefix kept as-is."""
        t = torch.randn(4, 8)
        sd = {"embed_tokens.weight": t}
        result = vlm_embedding_weights(sd)
        assert list(result.keys()) == ["embed_tokens.weight"]

    def test_empty_when_no_keyword(self):
        """Returns empty dict if no keys match keyword."""
        sd = {"language_model.layers.0.weight": torch.randn(4)}
        result = vlm_embedding_weights(sd)
        assert result == {}

    def test_custom_keyword(self):
        """Custom keyword filter."""
        t = torch.randn(4, 8)
        sd = {
            "model.word_embedding.weight": t,
            "model.layers.0.weight": torch.randn(4),
        }
        result = vlm_embedding_weights(sd, keyword="word_embedding", prefixes=("model.",))
        assert list(result.keys()) == ["word_embedding.weight"]


class TestPreprocessGptqWeights:
    """Tests for GPTQ weight preprocessing.

    Uses realistic GPTQ shapes for INT4, group_size=32:
    K=256, N=128 → K_packed=32, n_groups=8, blob_size=16.
    """

    K = 256
    N = 128
    BITS = 4
    GROUP_SIZE = 32
    K_PACKED = K * BITS // 32  # 32
    N_GROUPS = K // GROUP_SIZE  # 8
    BLOB_SIZE = GROUP_SIZE * BITS // 8  # 16
    # qzeros packs (32 // BITS)=8 zero points per int32
    N_GROUPS_PACKED = N_GROUPS * BITS // 32  # 1

    def test_qweight_renamed_to_weight(self):
        sd = {
            "q_proj.qweight": torch.randint(0, 255, (self.K_PACKED, self.N), dtype=torch.int32)
        }
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert "q_proj.weight" in result
        assert "q_proj.qweight" not in result

    def test_qweight_shape_3d(self):
        """Weight must be [N, n_blocks, blob_size] for MatMulNBits."""
        sd = {
            "q_proj.qweight": torch.randint(0, 255, (self.K_PACKED, self.N), dtype=torch.int32)
        }
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        w = result["q_proj.weight"]
        assert w.shape == (self.N, self.N_GROUPS, self.BLOB_SIZE)
        assert w.dtype == torch.uint8

    def test_qzeros_renamed_to_zero_points(self):
        sd = {
            "q_proj.qweight": torch.randint(
                0, 255, (self.K_PACKED, self.N), dtype=torch.int32
            ),
            "q_proj.qzeros": torch.randint(
                0, 255, (max(1, self.N_GROUPS_PACKED), self.N), dtype=torch.int32
            ),
        }
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert "q_proj.zero_points" in result
        assert "q_proj.qzeros" not in result

    def test_g_idx_dropped(self):
        sd = {
            "q_proj.g_idx": torch.arange(self.K),
            "q_proj.scales": torch.randn(self.N_GROUPS, self.N),
        }
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert not any(k.endswith(".g_idx") for k in result)
        assert "q_proj.scales" in result

    def test_g_idx_nontrivial_warns(self, caplog):
        """Non-trivial g_idx (desc_act) should emit a warning."""
        import logging

        sd = {
            "q_proj.g_idx": torch.tensor([7, 3, 0, 1]),  # not sequential
        }
        with caplog.at_level(logging.WARNING):
            result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert not any(k.endswith(".g_idx") for k in result)
        assert "desc_act" in caplog.text

    def test_scales_transposed(self):
        sd = {"q_proj.scales": torch.randn(self.N_GROUPS, self.N)}
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert result["q_proj.scales"].shape == (self.N, self.N_GROUPS)

    def test_non_gptq_keys_pass_through(self):
        t = torch.randn(4, 8)
        sd = {"model.embed_tokens.weight": t, "lm_head.weight": t.clone()}
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert "model.embed_tokens.weight" in result
        assert "lm_head.weight" in result
        assert torch.equal(result["model.embed_tokens.weight"], t)

    def test_missing_qweight_raises(self):
        """Qzeros without matching qweight raises ValueError."""
        sd = {"q_proj.qzeros": torch.zeros(1, self.N, dtype=torch.int32)}
        with pytest.raises(ValueError, match=r"Missing q_proj\.qweight"):
            preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)


class TestPreprocessAwqWeights:
    """Tests for AWQ weight preprocessing.

    Uses same realistic shapes as GPTQ tests: INT4, group_size=32,
    K=256, N=128.
    """

    K = 256
    N = 128
    BITS = 4
    GROUP_SIZE = 32
    K_PACKED = K * BITS // 32  # 32
    N_GROUPS = K // GROUP_SIZE  # 8
    BLOB_SIZE = GROUP_SIZE * BITS // 8  # 16
    N_GROUPS_PACKED = N_GROUPS * BITS // 32  # 1

    def test_qweight_renamed_to_weight(self):
        sd = {
            "q_proj.qweight": torch.randint(0, 255, (self.K_PACKED, self.N), dtype=torch.int32)
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert "q_proj.weight" in result
        assert "q_proj.qweight" not in result

    def test_qweight_shape_3d(self):
        """Weight must be [N, n_blocks, blob_size] for MatMulNBits."""
        sd = {
            "q_proj.qweight": torch.randint(0, 255, (self.K_PACKED, self.N), dtype=torch.int32)
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        w = result["q_proj.weight"]
        assert w.shape == (self.N, self.N_GROUPS, self.BLOB_SIZE)
        assert w.dtype == torch.uint8

    def test_qzeros_renamed_to_zero_points(self):
        sd = {
            "q_proj.qweight": torch.randint(
                0, 255, (self.K_PACKED, self.N), dtype=torch.int32
            ),
            "q_proj.qzeros": torch.randint(
                0,
                255,
                (max(1, self.N_GROUPS_PACKED), self.N),
                dtype=torch.int32,
            ),
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert "q_proj.zero_points" in result
        assert "q_proj.qzeros" not in result

    def test_zero_point_offset_subtracted(self):
        """AWQ zero points have +1 offset that must be subtracted."""
        # Create qzeros where every byte is 0x05 (all nibbles = 5).
        # After unpacking to uint8 we get 5 per element, after -1 → 4.
        sd = {
            "q_proj.qweight": torch.randint(
                0, 255, (self.K_PACKED, self.N), dtype=torch.int32
            ),
            "q_proj.qzeros": torch.full(
                (max(1, self.N_GROUPS_PACKED), self.N),
                0x05050505,
                dtype=torch.int32,
            ),
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        zp = result["q_proj.zero_points"]
        # Each byte was 0x05=5, after -1 → 4
        assert (zp == 4).all()

    def test_no_g_idx_handling(self):
        """AWQ does not have g_idx — unknown keys should pass through."""
        sd = {
            "q_proj.scales": torch.randn(self.N_GROUPS, self.N),
            "q_proj.some_extra": torch.randn(4),
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert "q_proj.some_extra" in result

    def test_scales_transposed(self):
        sd = {"q_proj.scales": torch.randn(self.N_GROUPS, self.N)}
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert result["q_proj.scales"].shape == (self.N, self.N_GROUPS)

    def test_same_qweight_reshape_as_gptq(self):
        """AWQ and GPTQ should produce identical qweight reshapes."""
        qweight = torch.randint(0, 255, (self.K_PACKED, self.N), dtype=torch.int32)
        gptq_result = preprocess_gptq_weights(
            {"p.qweight": qweight.clone(), "p.scales": torch.randn(self.N_GROUPS, self.N)},
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
        )
        awq_result = preprocess_awq_weights(
            {"p.qweight": qweight.clone(), "p.scales": torch.randn(self.N_GROUPS, self.N)},
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
        )
        assert torch.equal(gptq_result["p.weight"], awq_result["p.weight"])

    def test_zero_valued_qzeros_no_underflow(self):
        """When raw qzeros bytes are 0, subtraction must not underflow uint8."""
        sd = {
            "q_proj.qweight": torch.randint(
                0, 255, (self.K_PACKED, self.N), dtype=torch.int32
            ),
            "q_proj.qzeros": torch.zeros(
                max(1, self.N_GROUPS_PACKED), self.N, dtype=torch.int32
            ),
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        zp = result["q_proj.zero_points"]
        # 0 - 1 clamped to 0; no uint8 wrap-around to 255
        assert zp.dtype == torch.uint8
        assert (zp == 0).all()

    def test_zero_point_offset_per_nibble(self):
        """AWQ -1 offset must apply per-nibble, not per-byte.

        0x88 = low nibble 8, high nibble 8.  After per-nibble -1:
        low = 7, high = 7 → repacked = 0x77.
        A byte-level subtract would give 0x87 (wrong).
        """
        sd = {
            "q_proj.qweight": torch.randint(
                0, 255, (self.K_PACKED, self.N), dtype=torch.int32
            ),
            "q_proj.qzeros": torch.full(
                (max(1, self.N_GROUPS_PACKED), self.N),
                -2004318072,  # 0x88888888 as signed int32
                dtype=torch.int32,
            ),
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        zp = result["q_proj.zero_points"]
        # Both nibbles decremented: 0x88 → 0x77
        assert (zp == 0x77).all()

    def test_missing_qweight_raises(self):
        """Qzeros without matching qweight raises ValueError."""
        sd = {"q_proj.qzeros": torch.zeros(1, self.N, dtype=torch.int32)}
        with pytest.raises(ValueError, match=r"Missing q_proj\.qweight"):
            preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)


class TestMergeLoraWeights:
    """Tests for merge_lora_weights()."""

    def test_basic_merge(self):
        """LoRA delta is added to base weight: W' = W + (alpha/rank) * B @ A."""
        base = {"layer.weight": torch.zeros(4, 8)}
        lora = {
            "layer.lora_A.weight": torch.ones(2, 8),  # rank=2
            "layer.lora_B.weight": torch.ones(4, 2),
        }
        result = merge_lora_weights(base, lora, default_alpha=2.0)
        # scale = alpha/rank = 2/2 = 1.0
        # delta = B @ A = ones(4,2) @ ones(2,8) = 2*ones(4,8)
        expected = torch.full((4, 8), 2.0)
        torch.testing.assert_close(result["layer.weight"], expected)

    def test_alpha_from_tensor(self):
        """Per-layer alpha tensor overrides default_alpha."""
        base = {"layer.weight": torch.zeros(4, 8)}
        lora = {
            "layer.lora_A.weight": torch.eye(2, 8),
            "layer.lora_B.weight": torch.eye(4, 2),
            "layer.lora_A.alpha": torch.tensor(4.0),
        }
        result = merge_lora_weights(base, lora, default_alpha=1.0)
        # scale = alpha/rank = 4/2 = 2.0
        # delta = B @ A = eye(4,2) @ eye(2,8) -> (4,8) with 1s in top-left 2x2
        # scaled delta: 2.0 * delta
        w = result["layer.weight"]
        assert w[0, 0].item() == pytest.approx(2.0)
        assert w[2, 0].item() == pytest.approx(0.0)

    def test_default_alpha_equals_rank(self):
        """When no alpha provided, scale = rank/rank = 1.0."""
        base = {"layer.weight": torch.zeros(4, 8)}
        lora = {
            "layer.lora_A.weight": torch.ones(2, 8),
            "layer.lora_B.weight": torch.ones(4, 2),
        }
        result = merge_lora_weights(base, lora)
        # scale = rank/rank = 1.0, delta = 2*ones(4,8)
        expected = torch.full((4, 8), 2.0)
        torch.testing.assert_close(result["layer.weight"], expected)

    def test_preserves_base_dtype(self):
        """Merged result keeps the base weight's dtype (e.g. float16)."""
        base = {"layer.weight": torch.zeros(4, 8, dtype=torch.float16)}
        lora = {
            "layer.lora_A.weight": torch.ones(2, 8),
            "layer.lora_B.weight": torch.ones(4, 2),
        }
        result = merge_lora_weights(base, lora, default_alpha=2.0)
        assert result["layer.weight"].dtype == torch.float16

    def test_multiple_layers(self):
        """Merges LoRA for multiple layers independently."""
        base = {
            "attn.q_proj.weight": torch.zeros(4, 4),
            "attn.v_proj.weight": torch.ones(4, 4),
        }
        lora = {
            "attn.q_proj.lora_A.weight": torch.eye(1, 4),
            "attn.q_proj.lora_B.weight": torch.eye(4, 1),
            "attn.v_proj.lora_A.weight": torch.eye(1, 4),
            "attn.v_proj.lora_B.weight": torch.eye(4, 1),
        }
        result = merge_lora_weights(base, lora, default_alpha=1.0)
        # Both get identity delta (rank=1, alpha=1, scale=1)
        assert result["attn.q_proj.weight"][0, 0].item() == pytest.approx(1.0)
        assert result["attn.v_proj.weight"][0, 0].item() == pytest.approx(2.0)

    def test_missing_base_key_warns(self, caplog):
        """Warns when LoRA targets a weight not in base model."""
        import logging

        base = {"other.weight": torch.zeros(4, 4)}
        lora = {
            "missing.lora_A.weight": torch.ones(2, 4),
            "missing.lora_B.weight": torch.ones(4, 2),
        }
        with caplog.at_level(logging.WARNING):
            result = merge_lora_weights(base, lora)
        assert "not found in base model" in caplog.text
        # base unchanged
        torch.testing.assert_close(result["other.weight"], torch.zeros(4, 4))

    def test_orphan_lora_a_warns(self, caplog):
        """Warns when lora_A has no matching lora_B."""
        import logging

        base = {"layer.weight": torch.zeros(4, 4)}
        lora = {"layer.lora_A.weight": torch.ones(2, 4)}
        with caplog.at_level(logging.WARNING):
            merge_lora_weights(base, lora)
        assert "without matching lora_B" in caplog.text

    def test_non_lora_keys_ignored(self):
        """Non-LoRA keys in lora_state_dict are silently ignored."""
        base = {"layer.weight": torch.zeros(4, 4)}
        lora = {"some_other_key": torch.ones(4, 4)}
        result = merge_lora_weights(base, lora)
        torch.testing.assert_close(result["layer.weight"], torch.zeros(4, 4))

    def test_modifies_base_in_place(self):
        """Returns the same dict object (modified in-place)."""
        base = {"layer.weight": torch.zeros(4, 8)}
        lora = {
            "layer.lora_A.weight": torch.ones(2, 8),
            "layer.lora_B.weight": torch.ones(4, 2),
        }
        result = merge_lora_weights(base, lora)
        assert result is base


class TestSplitInterleavedQKV:
    """Tests for split_interleaved_qkv (GPT-NeoX / Persimmon layout)."""

    def test_2d_weight_mha(self):
        """MHA weight with interleaved [h0_q, h0_k, h0_v, h1_q, ...] layout."""
        num_heads, head_dim, hidden = 4, 8, 32
        # Build a known interleaved pattern so we can verify the split
        qs, ks, vs = [], [], []
        for h in range(num_heads):
            qs.append(torch.full((head_dim, hidden), float(h)))
            ks.append(torch.full((head_dim, hidden), float(h) + 0.1))
            vs.append(torch.full((head_dim, hidden), float(h) + 0.2))
        # Interleave: [q0, k0, v0, q1, k1, v1, ...]
        parts = []
        for h in range(num_heads):
            parts.extend([qs[h], ks[h], vs[h]])
        fused = torch.cat(parts, dim=0)  # [3*hidden, hidden]

        q, k, v = split_interleaved_qkv(fused, num_heads, num_heads, head_dim)

        assert q.shape == (num_heads * head_dim, hidden)
        assert k.shape == (num_heads * head_dim, hidden)
        assert v.shape == (num_heads * head_dim, hidden)
        torch.testing.assert_close(q, torch.cat(qs))
        torch.testing.assert_close(k, torch.cat(ks))
        torch.testing.assert_close(v, torch.cat(vs))

    def test_1d_bias(self):
        """Bias vector (1D) with interleaved layout."""
        num_heads, head_dim = 4, 8
        bias = torch.arange(num_heads * 3 * head_dim, dtype=torch.float32)
        q, k, v = split_interleaved_qkv(bias, num_heads, num_heads, head_dim)

        assert q.shape == (num_heads * head_dim,)
        assert k.shape == (num_heads * head_dim,)
        assert v.shape == (num_heads * head_dim,)
        # Reconstruct from known interleaved pattern
        reshaped = bias.reshape(num_heads, 3, head_dim)
        torch.testing.assert_close(q, reshaped[:, 0].reshape(-1))
        torch.testing.assert_close(k, reshaped[:, 1].reshape(-1))
        torch.testing.assert_close(v, reshaped[:, 2].reshape(-1))

    def test_wrong_dim_raises(self):
        """Wrong leading dimension raises ValueError."""
        with pytest.raises(ValueError, match="expected 96"):
            split_interleaved_qkv(
                torch.zeros(100, 32), num_heads=4, num_kv_heads=4, head_dim=8
            )

    def test_gqa_raises(self):
        """GQA (num_kv_heads != num_heads) is not supported."""
        with pytest.raises(ValueError, match="requires MHA"):
            split_interleaved_qkv(torch.zeros(96, 32), num_heads=4, num_kv_heads=2, head_dim=8)


class TestSplitCodegenQKV:
    """Tests for split_codegen_qkv (QVK model-parallel layout)."""

    def test_basic_split(self):
        """Verify Q/V/K extraction from mp-interleaved layout."""
        num_heads, head_dim, hidden, mp_num = 4, 8, 32, 2
        local_dim = num_heads * head_dim // mp_num  # 16

        # Build known pattern: each mp-block has [q, v, k] chunks
        q_full = torch.ones(num_heads * head_dim, hidden) * 1.0
        v_full = torch.ones(num_heads * head_dim, hidden) * 2.0
        k_full = torch.ones(num_heads * head_dim, hidden) * 3.0

        # Interleave by mp blocks: [q_mp0, v_mp0, k_mp0, q_mp1, v_mp1, k_mp1]
        parts = []
        for mp in range(mp_num):
            s = mp * local_dim
            e = s + local_dim
            parts.extend([q_full[s:e], v_full[s:e], k_full[s:e]])
        fused = torch.cat(parts, dim=0)  # [3*hidden, hidden]

        q, k, v = split_codegen_qkv(fused, num_heads, head_dim, mp_num)

        assert q.shape == (num_heads * head_dim, hidden)
        assert k.shape == (num_heads * head_dim, hidden)
        assert v.shape == (num_heads * head_dim, hidden)
        torch.testing.assert_close(q, q_full)
        torch.testing.assert_close(k, k_full)
        torch.testing.assert_close(v, v_full)

    def test_wrong_dim_raises(self):
        """Wrong leading dimension raises ValueError."""
        with pytest.raises(ValueError, match="expected 96"):
            split_codegen_qkv(torch.zeros(100, 32), num_heads=4, head_dim=8)

    def test_indivisible_mp_num_raises(self):
        """mp_num that doesn't divide hidden raises ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            split_codegen_qkv(torch.zeros(96, 32), num_heads=4, head_dim=8, mp_num=3)
