# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ArchitectureConfig."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import pytest

from mobius._configs import (
    DEFAULT_INT,
    ArchitectureConfig,
    AudioConfig,
    QuantizationConfig,
    VisionConfig,
    _extract_audio_config,
    _extract_mrope_fields,
    _extract_rope_config,
    _extract_vision_config,
    _nested_rope_theta,
    _nested_rope_type,
    _normalize_rope_scaling,
)


class TestArchitectureConfig:
    def test_default_values(self):
        config = ArchitectureConfig()
        assert config.vocab_size == DEFAULT_INT
        assert config.hidden_size == DEFAULT_INT
        assert config.num_hidden_layers == DEFAULT_INT
        assert config.rms_norm_eps == pytest.approx(1e-6)
        assert config.rope_type == "default"
        assert config.rope_theta == pytest.approx(10_000.0)
        assert config.partial_rotary_factor == pytest.approx(1.0)
        assert config.attn_qkv_bias is False
        assert config.attn_o_bias is False
        assert config.attn_qk_norm is False
        assert config.mlp_bias is False
        assert config.tie_word_embeddings is False

    def test_custom_values(self):
        config = ArchitectureConfig(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            hidden_act="silu",
        )
        assert config.vocab_size == 32000
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
        assert config.num_key_value_heads == 8
        assert config.head_dim == 128
        assert config.hidden_act == "silu"

    def test_is_dataclass(self):
        config = ArchitectureConfig()
        assert dataclasses.is_dataclass(config)

    def test_from_transformers_extracts_common_architectures(self):
        """Spot-check that from_transformers works for common model types."""

        class FakeLlamaConfig:
            model_type = "llama"
            num_attention_heads = 32
            num_key_value_heads = 8
            num_hidden_layers = 32
            vocab_size = 32000
            hidden_size = 4096
            intermediate_size = 11008
            hidden_act = "silu"
            max_position_embeddings = 4096
            head_dim = 128
            pad_token_id = 0
            rms_norm_eps = 1e-5
            rope_theta = 10000.0
            rope_scaling = None

        config = ArchitectureConfig.from_transformers(FakeLlamaConfig())
        assert config.vocab_size == 32000
        assert config.hidden_size == 4096

    def test_from_transformers_unknown_model_type_still_extracts_config(self):
        """from_transformers() does not gate on model type — the registry handles validation."""

        class FakeConfig:
            model_type = "unsupported_model"
            num_attention_heads = 8
            num_key_value_heads = 4
            num_hidden_layers = 2
            vocab_size = 1000
            hidden_size = 256
            intermediate_size = 512
            hidden_act = "silu"
            max_position_embeddings = 1024
            head_dim = 32
            pad_token_id = 0
            rms_norm_eps = 1e-6
            rope_theta = 10000.0
            rope_scaling = None

        config = ArchitectureConfig.from_transformers(FakeConfig())
        assert config.vocab_size == 1000
        assert config.hidden_size == 256

    def test_from_transformers_llama(self):
        class FakeLlamaConfig:
            model_type = "llama"
            num_attention_heads = 32
            num_key_value_heads = 8
            num_hidden_layers = 32
            vocab_size = 32000
            hidden_size = 4096
            intermediate_size = 11008
            hidden_act = "silu"
            max_position_embeddings = 4096
            head_dim = 128
            pad_token_id = 0
            rms_norm_eps = 1e-5
            rope_theta = 10000.0
            rope_scaling = None

        config = ArchitectureConfig.from_transformers(FakeLlamaConfig())
        assert config.vocab_size == 32000
        assert config.hidden_size == 4096
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 8
        assert config.head_dim == 128
        assert config.hidden_act == "silu"
        assert config.rope_type == "default"
        assert config.attn_qkv_bias is False
        assert config.attn_qk_norm is False
        assert config.tie_word_embeddings is False

    def test_from_transformers_qwen2_has_qkv_bias(self):
        class FakeQwen2Config:
            model_type = "qwen2"
            num_attention_heads = 16
            num_key_value_heads = 4
            num_hidden_layers = 24
            vocab_size = 151936
            hidden_size = 2048
            intermediate_size = 5504
            hidden_act = "silu"
            max_position_embeddings = 32768
            head_dim = None
            pad_token_id = 0
            rms_norm_eps = 1e-6
            rope_theta = 1000000.0
            rope_scaling = None

        config = ArchitectureConfig.from_transformers(FakeQwen2Config())
        assert config.attn_qkv_bias is True
        assert config.head_dim == 2048 // 16  # inferred from hidden_size / num_heads

    def test_from_transformers_qwen3_has_qk_norm(self):
        class FakeQwen3Config:
            model_type = "qwen3"
            num_attention_heads = 16
            num_key_value_heads = 4
            num_hidden_layers = 24
            vocab_size = 151936
            hidden_size = 2048
            intermediate_size = 5504
            hidden_act = "silu"
            max_position_embeddings = 32768
            head_dim = 128
            pad_token_id = 0
            rms_norm_eps = 1e-6
            rope_theta = 1000000.0
            rope_scaling = None

        config = ArchitectureConfig.from_transformers(FakeQwen3Config())
        assert config.attn_qk_norm is True

    def test_from_transformers_rope_scaling(self):
        class FakeConfig:
            model_type = "llama"
            num_attention_heads = 32
            num_key_value_heads = 8
            num_hidden_layers = 32
            vocab_size = 32000
            hidden_size = 4096
            intermediate_size = 11008
            hidden_act = "silu"
            max_position_embeddings = 131072
            head_dim = 128
            pad_token_id = 0
            rms_norm_eps = 1e-5
            rope_theta = 500000.0
            rope_scaling: ClassVar[dict] = {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            }

        config = ArchitectureConfig.from_transformers(FakeConfig())
        assert config.rope_type == "llama3"
        assert config.original_max_position_embeddings == 8192


class TestExtractRopeConfig:
    """Unit tests for _extract_rope_config helper."""

    def test_defaults_when_no_rope_attrs(self):
        """Bare config with no rope attrs yields sensible defaults."""

        class Bare:
            pass

        result = _extract_rope_config(Bare())
        assert result.rope_type == "default"
        assert result.rope_theta == pytest.approx(10_000.0)
        assert result.rope_scaling is None
        assert result.partial_rotary_factor == pytest.approx(1.0)
        assert result.rope_local_base_freq is None
        assert result.original_max_position_embeddings is None

    def test_rope_theta_from_config_attr(self):
        class Cfg:
            rope_theta = 500_000.0
            rope_scaling = None

        result = _extract_rope_config(Cfg())
        assert result.rope_theta == pytest.approx(500_000.0)

    def test_rope_type_from_rope_scaling(self):
        class Cfg:
            rope_scaling: ClassVar[dict] = {"rope_type": "llama3", "factor": 8.0}

        result = _extract_rope_config(Cfg())
        assert result.rope_type == "llama3"

    def test_partial_rotary_factor(self):
        class Cfg:
            partial_rotary_factor = 0.5
            rope_scaling = None

        result = _extract_rope_config(Cfg())
        assert result.partial_rotary_factor == pytest.approx(0.5)

    def test_partial_rotary_factor_zero_is_preserved(self):
        """partial_rotary_factor=0.0 must NOT be replaced by default 1.0."""

        class Cfg:
            partial_rotary_factor = 0.0
            rope_scaling = None

        result = _extract_rope_config(Cfg())
        assert result.partial_rotary_factor == pytest.approx(0.0)

    def test_rope_theta_zero_is_preserved(self):
        """rope_theta=0.0 must NOT be replaced by default 10000.0."""

        class Cfg:
            rope_theta = 0.0
            rope_scaling = None

        result = _extract_rope_config(Cfg())
        assert result.rope_theta == pytest.approx(0.0)

    def test_mrope_interleaved_from_rope_scaling(self):
        class Cfg:
            rope_scaling: ClassVar[dict] = {
                "mrope_interleaved": True,
                "mrope_section": [16, 24, 24],
            }

        result = _extract_mrope_fields(Cfg())
        assert result["mrope_interleaved"] is True
        assert result["mrope_section"] == [16, 24, 24]

    def test_mrope_interleaved_from_rope_parameters(self):
        class Cfg:
            rope_scaling = None
            rope_parameters: ClassVar[dict] = {
                "mrope_interleaved": True,
                "mrope_section": [8, 16, 8],
            }

        result = _extract_mrope_fields(Cfg())
        assert result["mrope_interleaved"] is True
        assert result["mrope_section"] == [8, 16, 8]

    def test_original_max_position_embeddings(self):
        class Cfg:
            original_max_position_embeddings = 8192
            rope_scaling = None

        result = _extract_rope_config(Cfg())
        assert result.original_max_position_embeddings == 8192


class TestExtractVisionConfig:
    """Unit tests for _extract_vision_config helper."""

    def test_no_vision_returns_empty(self):
        """Config with no vision_config yields empty dict."""

        class Cfg:
            pass

        result = _extract_vision_config(Cfg(), None, "llama")
        assert result == {}

    def test_basic_vision_config(self):
        """Extract standard vision fields into VisionConfig."""

        class VC:
            hidden_size = 1024
            intermediate_size = 4096
            num_hidden_layers = 24
            num_attention_heads = 16
            image_size = 384
            patch_size = 14
            layer_norm_eps = 1e-6

        class Cfg:
            vision_config = VC()
            mm_tokens_per_image = None
            image_token_id = 32000

        result = _extract_vision_config(Cfg(), None, "llava")
        assert "vision" in result
        assert isinstance(result["vision"], VisionConfig)
        assert result["vision"].hidden_size == 1024
        assert result["vision"].num_hidden_layers == 24
        assert result["vision"].image_token_id == 32000

    def test_vision_config_as_dict(self):
        """vision_config can be a plain dict (some HF configs)."""

        class Cfg:
            vision_config: ClassVar[dict] = {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "image_size": 224,
                "patch_size": 16,
            }
            mm_tokens_per_image = None
            image_token_id = None

        result = _extract_vision_config(Cfg(), None, "llava")
        assert result["vision"].hidden_size == 768
        assert result["vision"].patch_size == 16

    def test_mrope_section_from_composite_vl(self):
        """VL models pass mrope_section through vision helper."""

        class TextCfg:
            rope_scaling: ClassVar[dict] = {"mrope_section": [16, 24, 24]}
            vision_config = None

        class ParentCfg:
            vision_config = type(
                "VC",
                (),
                {
                    "hidden_size": 1024,
                    "intermediate_size": 4096,
                    "num_hidden_layers": 24,
                    "num_attention_heads": 16,
                    "image_size": 384,
                    "patch_size": 14,
                    "layer_norm_eps": 1e-6,
                },
            )()
            mm_tokens_per_image = None
            image_token_id = 151655

        result = _extract_vision_config(TextCfg(), ParentCfg(), "qwen2_vl")
        assert result["vision"].mrope_section == [16, 24, 24]

    def test_phi4mm_hardcoded_vision(self):
        """phi4mm uses hardcoded SigLIP vision encoder params."""

        class Cfg:
            vision_config = None
            mm_tokens_per_image = None
            image_token_id = None
            special_image_token_id = 200010
            embd_layer: ClassVar[dict] = {"image_embd_layer": {"crop_size": 448}}

        result = _extract_vision_config(Cfg(), None, "phi4mm")
        assert result["vision"].hidden_size == 1152
        assert result["vision"].num_hidden_layers == 27
        assert result["vision"].image_token_id == 200010


class TestExtractAudioConfig:
    """Unit tests for _extract_audio_config helper."""

    def test_no_audio_returns_empty(self):
        """Config with no audio attributes yields empty dict."""

        class Cfg:
            pass

        result = _extract_audio_config(Cfg(), None, "llama")
        assert result == {}

    def test_audio_processor_config(self):
        """Extract audio fields from audio_processor.config dict."""

        class Cfg:
            audio_processor: ClassVar[dict] = {
                "config": {
                    "attention_dim": 512,
                    "attention_heads": 8,
                    "num_blocks": 6,
                    "linear_units": 2048,
                    "kernel_size": 31,
                    "input_size": 80,
                    "nemo_conv_settings": {"conv_channels": 256},
                    "relative_attention_bias_args": {"t5_bias_max_distance": 64},
                }
            }

        result = _extract_audio_config(Cfg(), None, "phi4mm")
        assert "audio" in result
        assert isinstance(result["audio"], AudioConfig)
        assert result["audio"].attention_dim == 512
        assert result["audio"].attention_heads == 8
        assert result["audio"].t5_bias_max_distance == 64

    def test_qwen3_asr_thinker_config(self):
        """Qwen3-ASR extracts audio from thinker_config."""

        class Cfg:
            thinker_config = type(
                "TC",
                (),
                {
                    "audio_config": type(
                        "AC",
                        (),
                        {
                            "d_model": 1280,
                            "encoder_layers": 32,
                            "encoder_attention_heads": 20,
                            "encoder_ffn_dim": 5120,
                            "num_mel_bins": 128,
                            "max_source_positions": 1500,
                            "downsample_hidden_size": 1024,
                            "output_dim": 2048,
                            "activation_function": "gelu",
                        },
                    )(),
                    "audio_token_id": 151646,
                    "audio_start_token_id": 151647,
                    "audio_end_token_id": 151648,
                    "classify_num": None,
                },
            )()

        result = _extract_audio_config(Cfg(), None, "qwen3_asr")
        assert result["audio"].d_model == 1280
        assert result["audio"].encoder_layers == 32
        assert result["audio"].audio_token_id == 151646

    def test_phi4mm_audio_token_id(self):
        """phi4mm extracts audio_token_id from audio_config attr."""

        class Cfg:
            audio_config: ClassVar[dict] = {"audio_token_id": 200011}

        result = _extract_audio_config(Cfg(), None, "phi4mm")
        assert result["audio"].token_id == 200011


class TestExtractRopeConfigFallbacks:
    """Tests for older HF format fallbacks and nested rope_scaling."""

    def test_rope_type_from_type_key(self):
        """Older HF configs use 'type' instead of 'rope_type'."""

        class Cfg:
            rope_scaling: ClassVar[dict] = {"type": "dynamic", "factor": 2.0}

        result = _extract_rope_config(Cfg())
        assert result.rope_type == "dynamic"

    def test_rope_type_from_nested_full_attention(self):
        """Gemma3 nests rope config under full_attention key."""

        class Cfg:
            rope_scaling: ClassVar[dict] = {
                "full_attention": {
                    "rope_type": "linear",
                    "factor": 8.0,
                    "rope_theta": 100_000.0,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                },
            }

        result = _extract_rope_config(Cfg())
        assert result.rope_type == "linear"
        assert result.rope_theta == pytest.approx(100_000.0)
        assert result.rope_local_base_freq == pytest.approx(10_000.0)
        # rope_scaling should be normalized to the full_attention sub-dict
        assert result.rope_scaling["factor"] == pytest.approx(8.0)
        assert "full_attention" not in result.rope_scaling


class TestNestedRopeHelpers:
    """Unit tests for nested rope helpers.

    Covers _nested_rope_theta, _nested_rope_type, _normalize_rope_scaling.
    """

    def test_nested_rope_theta_found(self):
        scaling = {"full_attention": {"rope_theta": 50_000.0}}
        assert _nested_rope_theta(scaling, "full_attention") == pytest.approx(50_000.0)

    def test_nested_rope_theta_missing_key(self):
        assert _nested_rope_theta({}, "full_attention") is None

    def test_nested_rope_theta_not_dict(self):
        assert _nested_rope_theta({"full_attention": "not_a_dict"}, "full_attention") is None

    def test_nested_rope_type_found(self):
        scaling = {"full_attention": {"rope_type": "linear"}}
        assert _nested_rope_type(scaling, "full_attention") == "linear"

    def test_nested_rope_type_missing_key(self):
        assert _nested_rope_type({}, "sliding_attention") is None

    def test_normalize_rope_scaling_empty(self):
        assert _normalize_rope_scaling({}) == {}

    def test_normalize_rope_scaling_flat(self):
        flat = {"rope_type": "llama3", "factor": 8.0}
        assert _normalize_rope_scaling(flat) == flat

    def test_normalize_rope_scaling_gemma3(self):
        nested = {
            "full_attention": {"rope_type": "linear", "factor": 8.0},
            "sliding_attention": {"rope_type": "default"},
        }
        result = _normalize_rope_scaling(nested)
        assert result == {"rope_type": "linear", "factor": 8.0}


class TestVisionConfigBidirectionalSync:
    """Tests for __post_init__ VisionConfig ↔ flat field sync."""

    def test_nested_vision_config_works(self):
        vc = VisionConfig(hidden_size=64, num_attention_heads=4)
        config = ArchitectureConfig(vision=vc)
        assert config.vision.hidden_size == 64
        assert config.vision.num_attention_heads == 4

    def test_no_vision_fields_keeps_vision_none(self):
        config = ArchitectureConfig(hidden_size=128)
        assert config.vision is None


class TestQuantizationConfig:
    def test_defaults(self):
        qc = QuantizationConfig()
        assert qc.bits == 4
        assert qc.group_size == 128
        assert qc.quant_method == "none"
        assert qc.sym is True

    def test_from_transformers_gptq_dict(self):
        """Parse a GPTQ quantization_config dict."""
        hf = type(
            "HFConfig",
            (),
            {
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                    "sym": True,
                }
            },
        )()
        qc = QuantizationConfig.from_transformers(hf)
        assert qc is not None
        assert qc.quant_method == "gptq"
        assert qc.bits == 4
        assert qc.group_size == 128
        assert qc.sym is True

    def test_from_transformers_awq(self):
        hf = type(
            "HFConfig",
            (),
            {
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 64,
                    "sym": False,
                }
            },
        )()
        qc = QuantizationConfig.from_transformers(hf)
        assert qc is not None
        assert qc.quant_method == "awq"
        assert qc.group_size == 64
        assert qc.sym is False

    def test_from_transformers_no_quant_config(self):
        hf = type("HFConfig", (), {})()
        assert QuantizationConfig.from_transformers(hf) is None

    def test_from_transformers_none_quant_config(self):
        hf = type("HFConfig", (), {"quantization_config": None})()
        assert QuantizationConfig.from_transformers(hf) is None

    def test_from_transformers_method_none_returns_none(self):
        hf = type("HFConfig", (), {"quantization_config": {"quant_method": "none"}})()
        assert QuantizationConfig.from_transformers(hf) is None

    def test_from_transformers_to_dict_object(self):
        """HF QuantizationConfig objects have a to_dict() method."""
        inner = type(
            "QC",
            (),
            {
                "to_dict": lambda self: {
                    "quant_method": "gptq",
                    "bits": 8,
                    "group_size": 32,
                    "sym": False,
                }
            },
        )()
        hf = type("HFConfig", (), {"quantization_config": inner})()
        qc = QuantizationConfig.from_transformers(hf)
        assert qc is not None
        assert qc.bits == 8
        assert qc.group_size == 32

    def test_architecture_config_has_quantization_field(self):
        config = ArchitectureConfig()
        assert config.quantization is None

    def test_architecture_config_accepts_quantization(self):
        qc = QuantizationConfig(bits=4, group_size=128, quant_method="gptq")
        config = ArchitectureConfig(quantization=qc)
        assert config.quantization is not None
        assert config.quantization.quant_method == "gptq"


class TestArchitectureConfigValidate:
    """Tests for ArchitectureConfig.validate()."""

    def _make_valid_config(self, **overrides):
        defaults = dict(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=256,
            head_dim=16,
            num_key_value_heads=2,
            intermediate_size=128,
        )
        defaults.update(overrides)
        return ArchitectureConfig(**defaults)

    def test_valid_config_passes(self):
        config = self._make_valid_config()
        config.validate()  # Should not raise

    def test_zero_hidden_size_fails(self):
        config = self._make_valid_config(hidden_size=0)
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            config.validate()

    def test_negative_hidden_size_fails(self):
        config = self._make_valid_config(hidden_size=-1)
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            config.validate()

    def test_zero_num_heads_fails(self):
        config = self._make_valid_config(num_attention_heads=0)
        with pytest.raises(ValueError, match="num_attention_heads must be positive"):
            config.validate()

    def test_zero_vocab_size_fails(self):
        config = self._make_valid_config(vocab_size=0)
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            config.validate()

    def test_zero_num_layers_fails(self):
        config = self._make_valid_config(num_hidden_layers=0)
        with pytest.raises(ValueError, match="num_hidden_layers must be positive"):
            config.validate()

    def test_zero_head_dim_fails(self):
        config = self._make_valid_config(head_dim=0)
        with pytest.raises(ValueError, match="head_dim must be positive"):
            config.validate()

    def test_heads_not_dividing_kv_heads_fails(self):
        config = self._make_valid_config(
            num_attention_heads=5,
            num_key_value_heads=3,
            hidden_size=80,
            head_dim=16,
        )
        with pytest.raises(ValueError, match="divisible by num_key_value_heads"):
            config.validate()

    def test_hidden_size_not_divisible_by_heads_fails(self):
        # Only applies when head_dim is not explicitly set (DEFAULT_INT)
        config = self._make_valid_config(
            hidden_size=65,
            num_attention_heads=4,
            head_dim=DEFAULT_INT,
            num_key_value_heads=2,
        )
        with pytest.raises(ValueError, match=r"hidden_size.*divisible by num_attention_heads"):
            config.validate()

    def test_hidden_size_not_divisible_by_heads_ok_with_explicit_head_dim(self):
        # Models like Qwen3.5 set head_dim explicitly, so the check is skipped
        config = self._make_valid_config(
            hidden_size=65,
            num_attention_heads=4,
            head_dim=16,
            num_key_value_heads=2,
        )
        config.validate()  # Should not raise

    def test_zero_intermediate_size_fails(self):
        config = self._make_valid_config(intermediate_size=0)
        with pytest.raises(ValueError, match="intermediate_size must be positive"):
            config.validate()

    def test_negative_intermediate_size_fails(self):
        config = self._make_valid_config(intermediate_size=-1)
        with pytest.raises(ValueError, match="intermediate_size must be positive"):
            config.validate()

    def test_none_intermediate_size_passes(self):
        config = self._make_valid_config(intermediate_size=None)
        config.validate()  # None means model doesn't use MLP

    def test_multiple_errors_reported(self):
        config = self._make_valid_config(
            hidden_size=0,
            vocab_size=0,
            num_hidden_layers=0,
        )
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        msg = str(exc_info.value)
        assert "hidden_size" in msg
        assert "vocab_size" in msg
        assert "num_hidden_layers" in msg
