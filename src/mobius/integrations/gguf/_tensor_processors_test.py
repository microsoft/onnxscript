# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GGUF tensor processors."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from mobius.integrations.gguf._tensor_processors import (
    process_tensors,
)


class TestProcessTensorsLlama:
    """Tests for Llama/Mistral Q/K reverse permutation."""

    def _make_config(
        self,
        model_type: str = "llama",
        num_heads: int = 8,
        num_kv_heads: int = 8,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            model_type=model_type,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
        )

    def test_qk_roundtrip(self) -> None:
        """Verify permute → reverse_permute is identity."""
        config = self._make_config(num_heads=4, num_kv_heads=4)
        # Create a known weight matrix
        original_q = torch.randn(64, 128)
        original_k = torch.randn(64, 128)

        # Simulate what llama.cpp does: permute
        q_perm = self._forward_permute(original_q, 4)
        k_perm = self._forward_permute(original_k, 4)

        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": q_perm,
            "model.layers.0.self_attn.k_proj.weight": k_perm,
            "model.layers.0.self_attn.v_proj.weight": (torch.randn(64, 128)),
        }
        result = process_tensors(state_dict, config)

        # After reverse permute, should match original
        torch.testing.assert_close(
            result["model.layers.0.self_attn.q_proj.weight"],
            original_q,
        )
        torch.testing.assert_close(
            result["model.layers.0.self_attn.k_proj.weight"],
            original_k,
        )

    def test_gqa_different_head_counts(self) -> None:
        """Test GQA with num_kv_heads < num_attention_heads."""
        config = self._make_config(num_heads=8, num_kv_heads=2)
        original_k = torch.randn(32, 128)
        k_perm = self._forward_permute(original_k, 2)

        state_dict = {
            "model.layers.0.self_attn.k_proj.weight": k_perm,
        }
        result = process_tensors(state_dict, config)
        torch.testing.assert_close(
            result["model.layers.0.self_attn.k_proj.weight"],
            original_k,
        )

    def test_v_proj_untouched(self) -> None:
        """V projection should NOT be permuted."""
        config = self._make_config()
        v_weight = torch.randn(64, 128)
        state_dict = {
            "model.layers.0.self_attn.v_proj.weight": (v_weight.clone()),
        }
        result = process_tensors(state_dict, config)
        torch.testing.assert_close(
            result["model.layers.0.self_attn.v_proj.weight"],
            v_weight,
        )

    def test_mistral_uses_llama_processor(self) -> None:
        """Mistral should use the same processor as Llama."""
        config = self._make_config(model_type="mistral", num_heads=4, num_kv_heads=4)
        original = torch.randn(64, 128)
        perm = self._forward_permute(original, 4)
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": perm,
        }
        result = process_tensors(state_dict, config)
        torch.testing.assert_close(
            result["model.layers.0.self_attn.q_proj.weight"],
            original,
        )

    @staticmethod
    def _forward_permute(weights: torch.Tensor, n_head: int) -> torch.Tensor:
        """Simulate llama.cpp's forward permutation.

        Reference: convert_hf_to_gguf.py permute()
        """
        dim = weights.shape[0] // n_head // 2
        w = weights.reshape(n_head, dim, 2, *weights.shape[1:])
        return w.swapaxes(1, 2).reshape(weights.shape)


class TestProcessTensorsGemma:
    """Tests for Gemma norm weight offset."""

    def test_norm_weights_restored(self) -> None:
        config = SimpleNamespace(model_type="gemma2")
        state_dict = {
            "model.layers.0.input_layernorm.weight": (torch.tensor([0.0, 1.0, -1.0])),
            "model.norm.weight": torch.tensor([0.5]),
            "model.layers.0.self_attn.q_proj.weight": (torch.tensor([1.0, 2.0])),
        }
        result = process_tensors(state_dict, config)

        # Norm weights: +1
        torch.testing.assert_close(
            result["model.layers.0.input_layernorm.weight"],
            torch.tensor([1.0, 2.0, 0.0]),
        )
        torch.testing.assert_close(
            result["model.norm.weight"],
            torch.tensor([1.5]),
        )
        # Non-norm weights: unchanged
        torch.testing.assert_close(
            result["model.layers.0.self_attn.q_proj.weight"],
            torch.tensor([1.0, 2.0]),
        )


class TestProcessTensorsGPT2:
    """Tests for GPT-2 weight transpose."""

    def test_attn_weights_transposed(self) -> None:
        config = SimpleNamespace(model_type="gpt2")
        w = torch.randn(3, 5)
        state_dict = {
            "transformer.h.0.attn.c_attn.weight": w.clone(),
            "transformer.h.0.attn.c_attn.bias": (torch.randn(5)),
        }
        result = process_tensors(state_dict, config)

        # Weight should be transposed
        torch.testing.assert_close(
            result["transformer.h.0.attn.c_attn.weight"],
            w.T,
        )
        # Bias should be unchanged
        assert result["transformer.h.0.attn.c_attn.bias"].shape == (5,)

    def test_ffn_weights_transposed(self) -> None:
        config = SimpleNamespace(model_type="gpt2")
        w = torch.randn(3, 5)
        state_dict = {
            "transformer.h.0.mlp.c_fc.weight": w.clone(),
            "transformer.h.0.mlp.c_proj.weight": w.clone(),
        }
        result = process_tensors(state_dict, config)
        torch.testing.assert_close(result["transformer.h.0.mlp.c_fc.weight"], w.T)
        torch.testing.assert_close(result["transformer.h.0.mlp.c_proj.weight"], w.T)


class TestProcessTensorsMamba:
    """Tests for Mamba tensor fixes."""

    def test_conv1d_unsqueeze(self) -> None:
        config = SimpleNamespace(model_type="mamba")
        w = torch.randn(16, 4)
        state_dict = {
            "backbone.layers.0.mixer.conv1d.weight": (w.clone()),
        }
        result = process_tensors(state_dict, config)
        assert result["backbone.layers.0.mixer.conv1d.weight"].shape == (16, 1, 4)


class TestProcessTensorsNoop:
    """Test that unknown architectures pass through."""

    def test_unknown_model_type_noop(self) -> None:
        config = SimpleNamespace(model_type="some_unknown_model")
        original = {"a.weight": torch.tensor([1.0])}
        result = process_tensors(dict(original), config)
        torch.testing.assert_close(result["a.weight"], original["a.weight"])

    def test_no_model_type_noop(self) -> None:
        config = SimpleNamespace()
        original = {"a.weight": torch.tensor([1.0])}
        result = process_tensors(dict(original), config)
        torch.testing.assert_close(result["a.weight"], original["a.weight"])
