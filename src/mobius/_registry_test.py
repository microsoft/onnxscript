# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mobius._registry import (
    ModelRegistration,
    _detect_fallback_registration,
    registry,
)


def _make_hf_config(**kwargs) -> SimpleNamespace:
    """Create a fake HuggingFace config object for testing."""
    defaults: dict = {
        "model_type": "test_model",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "vocab_size": 32000,
        "architectures": ["TestModelForCausalLM"],
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestDetectFallbackRegistration:
    """Tests for _detect_fallback_registration heuristic."""

    def test_standard_causal_lm_returns_causal_model(self):
        from mobius.models import CausalLMModel

        config = _make_hf_config()
        result = _detect_fallback_registration(config)
        assert result is not None
        assert result.module_class is CausalLMModel

    def test_moe_config_returns_moe_model(self):
        from mobius.models import MoECausalLMModel

        config = _make_hf_config(num_local_experts=8)
        result = _detect_fallback_registration(config)
        assert result is not None
        assert result.module_class is MoECausalLMModel

    def test_encoder_decoder_rejected(self):
        config = _make_hf_config(is_encoder_decoder=True)
        assert _detect_fallback_registration(config) is None

    def test_vision_config_rejected(self):
        config = _make_hf_config(vision_config={"hidden_size": 768})
        assert _detect_fallback_registration(config) is None

    def test_audio_config_rejected(self):
        config = _make_hf_config(audio_config={"hidden_size": 768})
        assert _detect_fallback_registration(config) is None

    def test_no_architectures_field_rejected(self):
        config = _make_hf_config(architectures=None)
        assert _detect_fallback_registration(config) is None

    def test_empty_architectures_rejected(self):
        config = _make_hf_config(architectures=[])
        assert _detect_fallback_registration(config) is None

    def test_non_causal_architecture_rejected(self):
        config = _make_hf_config(architectures=["TestModelForMaskedLM"])
        assert _detect_fallback_registration(config) is None

    def test_seq2seq_architecture_rejected(self):
        config = _make_hf_config(architectures=["TestModelForConditionalGeneration"])
        assert _detect_fallback_registration(config) is None

    def test_missing_hidden_size_rejected(self):
        config = _make_hf_config(hidden_size=0)
        assert _detect_fallback_registration(config) is None

    def test_missing_num_layers_rejected(self):
        config = _make_hf_config(num_hidden_layers=0)
        assert _detect_fallback_registration(config) is None

    def test_missing_num_heads_rejected(self):
        config = _make_hf_config(num_attention_heads=0)
        assert _detect_fallback_registration(config) is None

    def test_missing_vocab_size_rejected(self):
        config = _make_hf_config(vocab_size=0)
        assert _detect_fallback_registration(config) is None

    def test_missing_field_attribute_rejected(self):
        """Config that lacks fields entirely (not just zero)."""
        config = SimpleNamespace(
            model_type="bare_model",
            architectures=["BareForCausalLM"],
        )
        assert _detect_fallback_registration(config) is None

    def test_returns_model_registration_type(self):
        result = _detect_fallback_registration(_make_hf_config())
        assert isinstance(result, ModelRegistration)

    def test_moe_with_single_expert_returns_causal(self):
        """num_local_experts=1 is not MoE — should fall back to CausalLM."""
        from mobius.models import CausalLMModel

        config = _make_hf_config(num_local_experts=1)
        result = _detect_fallback_registration(config)
        assert result is not None
        assert result.module_class is CausalLMModel

    def test_registered_type_uses_registry_not_fallback(self):
        """Verify that explicitly registered types bypass fallback."""
        assert "llama" in registry
        # The fallback function itself doesn't check the registry —
        # it's the caller's responsibility. Verify the registry
        # takes precedence.
        from mobius.models import CausalLMModel

        reg_cls = registry.get("llama")
        assert reg_cls is CausalLMModel

    def test_fallback_task_is_none_defaults_to_text_gen(self):
        """Fallback registration has task=None, meaning text-generation."""
        result = _detect_fallback_registration(_make_hf_config())
        assert result is not None
        # task is None → caller uses default "text-generation"
        assert result.task is None

    def test_ssm_d_state_rejected(self):
        """Mamba-like models with d_state are rejected."""
        config = _make_hf_config(d_state=16, d_conv=4)
        assert _detect_fallback_registration(config) is None

    def test_ssm_recurrent_block_type_rejected(self):
        """RecurrentGemma-like models with recurrent_block_type are rejected."""
        config = _make_hf_config(recurrent_block_type="recurrent")
        assert _detect_fallback_registration(config) is None

    def test_ssm_ssm_cfg_rejected(self):
        """Models with ssm_cfg dict are rejected."""
        config = _make_hf_config(ssm_cfg={"d_state": 16})
        assert _detect_fallback_registration(config) is None

    def test_rwkv_rescale_every_not_rejected(self):
        """RWKV is now properly registered; rescale_every no longer triggers SSM rejection."""
        from mobius.models.base import CausalLMModel

        config = _make_hf_config(rescale_every=6)
        # A fake CausalLM config with rescale_every is now handled by the
        # standard fallback path (not rejected as an SSM model) because
        # RWKV has its own registry entry under model_type='rwkv'.
        result = _detect_fallback_registration(config)
        assert result is not None
        assert result.module_class is CausalLMModel


class TestRegistryBasics:
    """Smoke tests for core registry functionality."""

    def test_registry_contains_llama(self):
        assert "llama" in registry

    def test_registry_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown model_type"):
            registry.get("nonexistent_model_xyz")

    def test_registry_has_expected_count(self):
        # Sanity: registry should have at least 100 entries
        assert len(registry) >= 100

    def test_registry_suggests_close_matches(self):
        """Fuzzy matching suggests similar model types."""
        with pytest.raises(KeyError, match=r"Did you mean"):
            registry.get("llma")  # typo for 'llama'

    def test_registry_no_suggestion_for_random_string(self):
        """Very different strings don't produce suggestions."""
        with pytest.raises(KeyError, match=r"Use registry\.register"):
            registry.get("zzzzzzzzz_not_a_model")
