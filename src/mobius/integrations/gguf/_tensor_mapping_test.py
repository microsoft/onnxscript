# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GGUF → HF tensor name mapping."""

from __future__ import annotations

import pytest

from mobius.integrations.gguf._tensor_mapping import (
    build_gguf_to_hf_map,
    map_gguf_to_hf_names,
)


class TestMapGGUFToHFNames:
    """Tests for map_gguf_to_hf_names()."""

    # ---- Llama / Mistral / Qwen2 (same naming convention) ----

    @pytest.mark.parametrize(
        "gguf_name, expected",
        [
            (
                "token_embd.weight",
                "model.embed_tokens.weight",
            ),
            ("output.weight", "lm_head.weight"),
            ("output_norm.weight", "model.norm.weight"),
            (
                "blk.0.attn_q.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ),
            (
                "blk.0.attn_k.weight",
                "model.layers.0.self_attn.k_proj.weight",
            ),
            (
                "blk.0.attn_v.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ),
            (
                "blk.0.attn_output.weight",
                "model.layers.0.self_attn.o_proj.weight",
            ),
            (
                "blk.0.attn_norm.weight",
                "model.layers.0.input_layernorm.weight",
            ),
            (
                "blk.0.ffn_gate.weight",
                "model.layers.0.mlp.gate_proj.weight",
            ),
            (
                "blk.0.ffn_up.weight",
                "model.layers.0.mlp.up_proj.weight",
            ),
            (
                "blk.0.ffn_down.weight",
                "model.layers.0.mlp.down_proj.weight",
            ),
            (
                "blk.0.ffn_norm.weight",
                "model.layers.0.post_attention_layernorm.weight",
            ),
            # Higher layer index
            (
                "blk.31.attn_q.weight",
                "model.layers.31.self_attn.q_proj.weight",
            ),
        ],
    )
    def test_llama_mapping(self, gguf_name: str, expected: str) -> None:
        assert map_gguf_to_hf_names(gguf_name, "llama") == expected

    @pytest.mark.parametrize("arch", ["mistral", "qwen2", "qwen3"])
    def test_llama_family_aliases(self, arch: str) -> None:
        """Mistral/Qwen2/Qwen3 share llama naming convention."""
        result = map_gguf_to_hf_names("blk.0.attn_q.weight", arch)
        assert result == "model.layers.0.self_attn.q_proj.weight"

    # ---- Bias tensors ----

    def test_bias_suffix(self) -> None:
        result = map_gguf_to_hf_names("blk.0.attn_q.bias", "llama")
        assert result == "model.layers.0.self_attn.q_proj.bias"

    # ---- Skip rules ----

    def test_skip_tokenizer(self) -> None:
        assert map_gguf_to_hf_names("tokenizer.ggml.tokens", "llama") is None

    def test_skip_rope_freqs(self) -> None:
        assert map_gguf_to_hf_names("rope_freqs", "llama") is None

    def test_skip_attn_rot_embd(self) -> None:
        assert map_gguf_to_hf_names("blk.0.attn_rot_embd.weight", "llama") is None

    def test_unknown_tensor_returns_none(self) -> None:
        assert map_gguf_to_hf_names("blk.0.unknown_tensor.weight", "llama") is None

    # ---- Gemma family ----

    def test_gemma2_extras(self) -> None:
        assert (
            map_gguf_to_hf_names("blk.0.pre_ffn_norm.weight", "gemma2")
            == "model.layers.0.pre_feedforward_layernorm.weight"
        )
        assert (
            map_gguf_to_hf_names("blk.0.post_ffn_norm.weight", "gemma2")
            == "model.layers.0.post_feedforward_layernorm.weight"
        )

    def test_gemma2_inherits_llama_base(self) -> None:
        result = map_gguf_to_hf_names("blk.0.attn_q.weight", "gemma2")
        assert result == "model.layers.0.self_attn.q_proj.weight"

    # ---- Phi-3 ----

    def test_phi3_fused_qkv(self) -> None:
        result = map_gguf_to_hf_names("blk.0.attn_qkv.weight", "phi3")
        assert result == "model.layers.0.self_attn.qkv_proj.weight"

    def test_phi3_fused_gate_up(self) -> None:
        result = map_gguf_to_hf_names("blk.0.ffn_up.weight", "phi3")
        assert result == "model.layers.0.mlp.gate_up_proj.weight"

    # ---- GPT-2 ----

    def test_gpt2_mapping(self) -> None:
        assert map_gguf_to_hf_names("token_embd.weight", "gpt2") == "transformer.wte.weight"
        assert (
            map_gguf_to_hf_names("blk.0.attn_qkv.weight", "gpt2")
            == "transformer.h.0.attn.c_attn.weight"
        )
        assert (
            map_gguf_to_hf_names("blk.0.ffn_up.weight", "gpt2")
            == "transformer.h.0.mlp.c_fc.weight"
        )
        assert map_gguf_to_hf_names("position_embd.weight", "gpt2") == "transformer.wpe.weight"

    # ---- Falcon ----

    def test_falcon_mapping(self) -> None:
        assert (
            map_gguf_to_hf_names("token_embd.weight", "falcon")
            == "transformer.word_embeddings.weight"
        )
        assert map_gguf_to_hf_names("blk.0.attn_qkv.weight", "falcon") == (
            "transformer.h.0.self_attention.query_key_value.weight"
        )

    # ---- Mamba ----

    def test_mamba_mapping(self) -> None:
        assert (
            map_gguf_to_hf_names("token_embd.weight", "mamba") == "backbone.embeddings.weight"
        )
        assert (
            map_gguf_to_hf_names("blk.0.ssm_in.weight", "mamba")
            == "backbone.layers.0.mixer.in_proj.weight"
        )
        assert (
            map_gguf_to_hf_names("blk.0.ssm_conv1d.weight", "mamba")
            == "backbone.layers.0.mixer.conv1d.weight"
        )
        assert (
            map_gguf_to_hf_names("blk.0.ssm_a.weight", "mamba")
            == "backbone.layers.0.mixer.A_log.weight"
        )

    # ---- MoE ----

    def test_moe_extras(self) -> None:
        assert (
            map_gguf_to_hf_names("blk.0.ffn_gate_inp.weight", "qwen2moe")
            == "model.layers.0.mlp.gate.weight"
        )
        assert map_gguf_to_hf_names("blk.0.ffn_gate_exps.weight", "qwen2moe") == (
            "model.layers.0.mlp.experts.gate_proj.weight"
        )

    # ---- Unsupported architecture ----

    def test_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            map_gguf_to_hf_names("token_embd.weight", "unknown")


class TestBuildGGUFToHFMap:
    """Tests for build_gguf_to_hf_map()."""

    def test_builds_complete_map(self) -> None:
        gguf_names = [
            "token_embd.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "tokenizer.ggml.tokens",  # should be skipped
            "rope_freqs",  # should be skipped
        ]
        result = build_gguf_to_hf_map(gguf_names, "llama")
        assert len(result) == 4
        assert "tokenizer.ggml.tokens" not in result
        assert "rope_freqs" not in result
        assert result["blk.0.attn_q.weight"] == "model.layers.0.self_attn.q_proj.weight"

    def test_empty_input(self) -> None:
        assert build_gguf_to_hf_map([], "llama") == {}
