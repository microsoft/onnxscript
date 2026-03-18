# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for _config_resolver.py — pure function input/output tests."""

from __future__ import annotations

import pytest

from mobius._config_resolver import (
    _config_from_hf,
    _default_task_for_model,
    _dict_to_pretrained_config,
)
from mobius._configs import (
    ArchitectureConfig,
    WhisperConfig,
)


def _fake_hf_config(model_type: str, **overrides):
    """Create a minimal HF-config-like object for testing."""
    defaults = {
        "model_type": model_type,
        "vocab_size": 100,
        "max_position_embeddings": 32,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "head_dim": 16,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
    }
    defaults.update(overrides)
    return type("FakeHFConfig", (), defaults)()


# ── Top 5 HF config formats ─────────────────────────────────────────────


class TestConfigFromHfLlama:
    """Llama config resolution — the baseline decoder-only format."""

    def test_resolves_to_architecture_config(self):
        hf = _fake_hf_config("llama")
        result = _config_from_hf(hf)
        assert isinstance(result, ArchitectureConfig)

    def test_core_fields_extracted(self):
        hf = _fake_hf_config(
            "llama",
            vocab_size=32000,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            num_hidden_layers=32,
        )
        result = _config_from_hf(hf)
        assert result.vocab_size == 32000
        assert result.hidden_size == 4096
        assert result.num_attention_heads == 32
        assert result.num_key_value_heads == 8
        assert result.head_dim == 128

    def test_no_qkv_bias(self):
        hf = _fake_hf_config("llama")
        result = _config_from_hf(hf)
        assert result.attn_qkv_bias is False

    def test_default_task_is_text_generation(self):
        assert _default_task_for_model("llama") == "text-generation"


class TestConfigFromHfQwen:
    """Qwen2/3 config resolution — attention bias and QK norm."""

    def test_qwen2_has_qkv_bias(self):
        hf = _fake_hf_config("qwen2")
        result = _config_from_hf(hf)
        assert result.attn_qkv_bias is True

    def test_qwen2_head_dim_inferred(self):
        hf = _fake_hf_config("qwen2", head_dim=None, hidden_size=2048, num_attention_heads=16)
        result = _config_from_hf(hf)
        assert result.head_dim == 2048 // 16

    def test_qwen3_has_qk_norm(self):
        hf = _fake_hf_config("qwen3", head_dim=128)
        result = _config_from_hf(hf)
        assert result.attn_qk_norm is True

    def test_qwen2_no_qk_norm(self):
        hf = _fake_hf_config("qwen2")
        result = _config_from_hf(hf)
        assert result.attn_qk_norm is False


class TestConfigFromHfPhi:
    """Phi3 config resolution — partial rotary and su rope."""

    def test_phi3_resolves(self):
        hf = _fake_hf_config(
            "phi3",
            rope_scaling={"rope_type": "su", "long_factor": [1.0] * 48},
        )
        result = _config_from_hf(hf)
        assert isinstance(result, ArchitectureConfig)
        assert result.rope_type == "su"

    def test_phi3_partial_rotary(self):
        hf = _fake_hf_config("phi3", partial_rotary_factor=0.5)
        result = _config_from_hf(hf)
        assert result.partial_rotary_factor == pytest.approx(0.5)


class TestConfigFromHfGemma:
    """Gemma config resolution — including nested rope_scaling."""

    def test_gemma_resolves(self):
        hf = _fake_hf_config("gemma")
        result = _config_from_hf(hf)
        assert isinstance(result, ArchitectureConfig)

    def test_gemma3_text_has_qk_norm(self):
        hf = _fake_hf_config("gemma3_text")
        result = _config_from_hf(hf)
        assert result.attn_qk_norm is True

    def test_gemma3_nested_rope_scaling(self):
        """Gemma3 stores per-attention-type rope configs.

        When config.rope_theta is set, it takes priority.  The nested
        ``full_attention.rope_theta`` is only used as fallback.
        """
        hf = _fake_hf_config(
            "gemma3_text",
            rope_theta=None,  # force fallback to nested lookup
            rope_scaling={
                "full_attention": {
                    "rope_type": "linear",
                    "factor": 8.0,
                    "rope_theta": 500_000.0,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                },
            },
        )
        result = _config_from_hf(hf)
        assert result.rope_type == "linear"
        assert result.rope_theta == pytest.approx(500_000.0)
        assert result.rope_local_base_freq == pytest.approx(10_000.0)


class TestConfigFromHfMistral:
    """Mistral config resolution — sliding window."""

    def test_mistral_resolves(self):
        hf = _fake_hf_config("mistral")
        result = _config_from_hf(hf)
        assert isinstance(result, ArchitectureConfig)

    def test_mistral_sliding_window(self):
        hf = _fake_hf_config("mistral", sliding_window=4096)
        result = _config_from_hf(hf)
        assert result.sliding_window == 4096

    def test_mistral_no_qkv_bias(self):
        hf = _fake_hf_config("mistral")
        result = _config_from_hf(hf)
        assert result.attn_qkv_bias is False


# ── Unknown/malformed config ────────────────────────────────────────────


class TestUnknownMalformedConfig:
    """Unknown model types are handled by the registry, not from_transformers().

    from_transformers() extracts config fields regardless of model_type.
    The registry raises KeyError for unregistered architectures.
    """

    def test_unknown_model_type_still_extracts_config(self):
        hf = _fake_hf_config("totally_unknown_model_xyz")
        config = _config_from_hf(hf)
        assert isinstance(config, ArchitectureConfig)
        assert config.hidden_size == 64

    def test_unknown_model_type_preserves_all_fields(self):
        hf = _fake_hf_config("bogus_arch_42")
        config = _config_from_hf(hf)
        assert isinstance(config, ArchitectureConfig)
        assert config.num_attention_heads == 4

    def test_missing_model_type_attribute(self):
        """Config object without model_type falls through to ArchitectureConfig default."""
        hf = type(
            "NoModelType",
            (),
            {
                "vocab_size": 100,
                "hidden_size": 64,
                "num_attention_heads": 4,
            },
        )()
        # No model_type → registry lookup skipped → ArchitectureConfig.from_transformers
        # which requires model_type, so should raise
        with pytest.raises((ValueError, AttributeError)):
            _config_from_hf(hf)


# ── Gemma OffsetRMSNorm +1.0 config edge case ──────────────────────────


class TestGemmaOffsetRMSNorm:
    """Gemma OffsetRMSNorm +1.0 edge case.

    The OffsetRMSNorm class uses ``1.0 + weight`` for the effective multiplier.
    This is a model-level behavior, but config resolution must correctly
    preserve the rms_norm_eps and related fields that feed into the norm.
    """

    def test_gemma_rms_norm_eps_preserved(self):
        hf = _fake_hf_config("gemma", rms_norm_eps=1e-6)
        result = _config_from_hf(hf)
        assert result.rms_norm_eps == pytest.approx(1e-6)

    def test_gemma_layer_norm_eps_fallback(self):
        """When rms_norm_eps missing, falls back to layer_norm_eps."""
        hf = _fake_hf_config("gemma", rms_norm_eps=None, layer_norm_eps=1e-5)
        result = _config_from_hf(hf)
        assert result.rms_norm_eps == pytest.approx(1e-5)

    def test_gemma_custom_norm_eps(self):
        """Gemma2 uses 1e-6 by default; ensure custom values propagate."""
        hf = _fake_hf_config("gemma2", rms_norm_eps=1e-10)
        result = _config_from_hf(hf)
        assert result.rms_norm_eps == pytest.approx(1e-10)

    def test_gemma3_text_config_for_offset_norm(self):
        """Gemma3_text models use OffsetRMSNorm — config must resolve cleanly."""
        hf = _fake_hf_config(
            "gemma3_text",
            rms_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
        )
        result = _config_from_hf(hf)
        assert result.rms_norm_eps == pytest.approx(1e-6)
        assert result.hidden_act == "gelu_pytorch_tanh"
        assert result.attn_qk_norm is True


# ── DeepSeek MLA-specific config fields ─────────────────────────────────


class TestDeepSeekMLA:
    """DeepSeek-V2/V3 Multi-Latent Attention config extraction."""

    def _deepseek_config(self, **overrides):
        defaults = dict(
            model_type="deepseek_v2",
            vocab_size=102400,
            hidden_size=2048,
            intermediate_size=10944,
            num_hidden_layers=27,
            num_attention_heads=16,
            num_key_value_heads=16,
            head_dim=128,
            hidden_act="silu",
            max_position_embeddings=4096,
            pad_token_id=0,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            rope_scaling=None,
            # MLA-specific fields
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            # MoE fields
            n_routed_experts=64,
            num_experts_per_tok=6,
            moe_intermediate_size=1408,
            shared_expert_intermediate_size=5632,
            n_shared_experts=2,
            first_k_dense_replace=1,
            scoring_func="softmax",
            topk_method="greedy",
        )
        defaults.update(overrides)
        return type("FakeDeepSeekConfig", (), defaults)()

    def test_mla_fields_extracted(self):
        hf = self._deepseek_config()
        result = _config_from_hf(hf)
        assert result.q_lora_rank == 1536
        assert result.kv_lora_rank == 512
        assert result.qk_nope_head_dim == 128
        assert result.qk_rope_head_dim == 64
        assert result.v_head_dim == 128

    def test_rope_interleave_auto_enabled(self):
        """rope_interleave is auto-enabled when qk_rope_head_dim > 0."""
        hf = self._deepseek_config(qk_rope_head_dim=64)
        result = _config_from_hf(hf)
        assert result.rope_interleave is True

    def test_rope_interleave_disabled_without_mla(self):
        """No MLA (qk_rope_head_dim=None) → rope_interleave stays False."""
        hf = self._deepseek_config(qk_rope_head_dim=None)
        result = _config_from_hf(hf)
        assert result.rope_interleave is False

    def test_moe_fields_extracted(self):
        hf = self._deepseek_config()
        result = _config_from_hf(hf)
        assert result.num_local_experts == 64
        assert result.num_experts_per_tok == 6
        assert result.moe_intermediate_size == 1408
        assert result.shared_expert_intermediate_size == 5632
        assert result.n_shared_experts == 2

    def test_deepseek_v3_model_type(self):
        hf = self._deepseek_config(model_type="deepseek_v3")
        result = _config_from_hf(hf)
        assert result.q_lora_rank == 1536
        assert result.kv_lora_rank == 512


# ── Whisper encoder-decoder nesting ─────────────────────────────────────


class TestWhisperEncoderDecoder:
    """Whisper encoder-decoder config resolution via registry config_class."""

    def _whisper_hf_config(self, **overrides):
        defaults = dict(
            model_type="whisper",
            vocab_size=51865,
            hidden_size=512,
            d_model=512,
            num_attention_heads=8,
            encoder_attention_heads=8,
            decoder_attention_heads=8,
            encoder_layers=6,
            decoder_layers=6,
            encoder_ffn_dim=2048,
            decoder_ffn_dim=2048,
            num_hidden_layers=6,
            num_mel_bins=80,
            max_source_positions=1500,
            max_target_positions=448,
            scale_embedding=False,
            decoder_start_token_id=50258,
            activation_function="gelu",
            pad_token_id=0,
            tie_word_embeddings=True,
        )
        defaults.update(overrides)
        return type("FakeWhisperConfig", (), defaults)()

    def test_whisper_routes_to_whisper_config(self):
        """_config_from_hf routes whisper to WhisperConfig via registry."""
        hf = self._whisper_hf_config()
        result = _config_from_hf(hf)
        assert isinstance(result, WhisperConfig)

    def test_encoder_fields_extracted(self):
        hf = self._whisper_hf_config()
        result = _config_from_hf(hf)
        assert result.encoder_layers == 6
        assert result.encoder_attention_heads == 8
        assert result.encoder_ffn_dim == 2048

    def test_decoder_fields_extracted(self):
        hf = self._whisper_hf_config()
        result = _config_from_hf(hf)
        assert result.num_hidden_layers == 6
        assert result.num_attention_heads == 8
        assert result.head_dim == 512 // 8

    def test_whisper_speech_specific_fields(self):
        hf = self._whisper_hf_config()
        result = _config_from_hf(hf)
        assert result.num_mel_bins == 80
        assert result.max_source_positions == 1500
        assert result.max_target_positions == 448
        assert result.decoder_start_token_id == 50258

    def test_whisper_default_task(self):
        assert _default_task_for_model("whisper") == "speech-to-text"

    def test_whisper_bias_flags(self):
        """Whisper has both QKV and output projection biases."""
        hf = self._whisper_hf_config()
        result = _config_from_hf(hf)
        assert result.attn_qkv_bias is True
        assert result.attn_o_bias is True

    def test_whisper_tie_word_embeddings(self):
        hf = self._whisper_hf_config(tie_word_embeddings=True)
        result = _config_from_hf(hf)
        assert result.tie_word_embeddings is True


# ── _dict_to_pretrained_config ──────────────────────────────────────────


class TestDictToPretrainedConfig:
    """Tests for dict → PretrainedConfig conversion with nested configs."""

    def test_flat_dict(self):
        d = {"model_type": "llama", "hidden_size": 4096, "vocab_size": 32000}
        config = _dict_to_pretrained_config(d)
        assert config.model_type == "llama"
        assert config.hidden_size == 4096

    def test_nested_text_config(self):
        d = {
            "model_type": "qwen3_vl",
            "text_config": {"model_type": "qwen3_vl_text", "hidden_size": 2048},
        }
        config = _dict_to_pretrained_config(d)
        assert hasattr(config, "text_config")
        assert config.text_config.model_type == "qwen3_vl_text"
        assert config.text_config.hidden_size == 2048

    def test_nested_vision_config(self):
        d = {
            "model_type": "llava",
            "vision_config": {"hidden_size": 1024, "num_hidden_layers": 24},
        }
        config = _dict_to_pretrained_config(d)
        assert hasattr(config, "vision_config")
        assert config.vision_config.hidden_size == 1024

    def test_non_nested_keys_untouched(self):
        """Keys not in the nested_keys list stay as-is (dicts stay dicts)."""
        d = {"model_type": "test", "custom_config": {"key": "value"}}
        config = _dict_to_pretrained_config(d)
        assert isinstance(config.custom_config, dict)

    def test_thinker_config_nested(self):
        """Qwen3-ASR thinker_config nesting works."""
        d = {
            "model_type": "qwen3_asr",
            "thinker_config": {
                "model_type": "qwen3",
                "text_config": {"model_type": "qwen3", "hidden_size": 1024},
            },
        }
        config = _dict_to_pretrained_config(d)
        assert config.thinker_config.model_type == "qwen3"
        assert config.thinker_config.text_config.model_type == "qwen3"
