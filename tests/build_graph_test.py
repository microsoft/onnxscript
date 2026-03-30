# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests: build ONNX graphs for all supported architectures (no weights).

These tests verify that each model architecture can construct a valid ONNX
graph without downloading any weights. They are fast and require no network
access. Run with::

    pytest tests/build_graph_test.py -v

To run a single model::

    pytest tests/build_graph_test.py -k "qwen2"
"""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import pytest
from _test_configs import (
    ALL_CAUSAL_LM_CONFIGS,
    ALL_CONFIGS,
    AUTO_GENERATED_CONFIGS,
    DETECTION_CONFIGS,
    ENCODER_CONFIGS,
    LONGROPE_FACTORS,
    SEQ2SEQ_CONFIGS,
    SSM_CONFIGS,
    TINY_HEAD_DIM,
    TINY_HEADS,
    TINY_HIDDEN,
    TINY_INTERMEDIATE,
    TINY_KV_HEADS,
    TINY_LAYERS,
    TINY_VOCAB,
    VISION_CONFIGS,
)

from mobius._builder import (
    DTYPE_MAP,
    build_from_module,
)
from mobius._config_resolver import _default_task_for_model
from mobius._configs import (
    ArchitectureConfig,
    AudioConfig,
    CodePredictorConfig,
    SpeakerEncoderConfig,
    TTSConfig,
    VisionConfig,
)
from mobius._registry import registry
from mobius.tasks import (
    CausalLMTask,
    Phi4MMMultiModalTask,
    Qwen3VLVisionLanguageTask,
    get_task,
)

# Minimal configs for each architecture. These are hand-crafted small configs
# that exercise each model class without needing to download from HuggingFace.


def _base_config(config_cls=None, **overrides) -> ArchitectureConfig:
    if config_cls is None:
        config_cls = overrides.pop("_config_cls", ArchitectureConfig)
    else:
        overrides.pop("_config_cls", None)
    defaults = dict(
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_INTERMEDIATE,
        num_attention_heads=TINY_HEADS,
        num_key_value_heads=TINY_KV_HEADS,
        head_dim=TINY_HEAD_DIM,
        num_hidden_layers=TINY_LAYERS,
        vocab_size=TINY_VOCAB,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        pad_token_id=0,
    )
    defaults.update(overrides)
    # Filter out fields not accepted by the config class (e.g. MambaConfig
    # doesn't have max_position_embeddings or rope_* fields).
    if dataclasses.is_dataclass(config_cls):
        valid_fields = {f.name for f in dataclasses.fields(config_cls)}
        defaults = {k: v for k, v in defaults.items() if k in valid_fields}
    return config_cls(**defaults)


# Semantic test IDs for model_types that intentionally appear more than once
# with different config overrides. Keyed by (model_type, occurrence_index).
_SEMANTIC_IDS: dict[tuple[str, int], str] = {
    ("deepseek_v2", 0): "deepseek_v2_mla",
    ("deepseek_v2", 1): "deepseek_v2_no_mla",
    ("deepseek_v2", 2): "deepseek_v2_mla_dense",
    ("qwen3_5_text", 0): "qwen3_5_text_default",
    ("qwen3_5_text", 1): "qwen3_5_text_linear_attn",
    ("qwen3_next", 0): "qwen3_next_hybrid",
    ("qwen3_next", 1): "qwen3_next_all_full_attn",
    ("qwen3_next", 2): "qwen3_next_all_linear_attn",
    ("falcon_h1", 0): "falcon_h1_alibi",
    ("falcon_h1", 1): "falcon_h1_parallel_attn",
    ("jamba", 0): "jamba_hybrid_moe",
    ("jamba", 1): "jamba_all_attention",
    ("bamba", 0): "bamba_hybrid",
    ("bamba", 1): "bamba_all_attention",
    ("gemma3n_text", 0): "gemma3n_text_sliding",
    ("gemma3n_text", 1): "gemma3n_text_full_attn",
    ("granite", 0): "granite_default",
    ("granite", 1): "granite_scaling",
    ("phi3small", 0): "phi3small_default",
    ("phi3small", 1): "phi3small_rotary_025",
}


def _make_params(
    configs: list[tuple[str, dict, bool]],
) -> list:
    """Create pytest.param entries with stable unique IDs.

    Duplicate model_types get semantic IDs from ``_SEMANTIC_IDS`` when
    available, falling back to ``<model_type>_<index>`` otherwise.
    """
    from collections import Counter

    stripped = [(mt, ov) for mt, ov, _ in configs]
    counts = Counter(mt for mt, _ in stripped)
    seen: dict[str, int] = {}
    params = []
    for model_type, overrides in stripped:
        if counts[model_type] > 1:
            idx = seen.get(model_type, 0)
            seen[model_type] = idx + 1
            test_id = _SEMANTIC_IDS.get((model_type, idx), f"{model_type}_{idx}")
        else:
            test_id = model_type
        params.append(pytest.param(model_type, overrides, id=test_id))
    return params


# Configs imported from _test_configs — strip the is_representative flag
# for use with pytest.parametrize.
_MODEL_CONFIGS: list[tuple[str, dict]] = [(mt, ov) for mt, ov, _ in ALL_CAUSAL_LM_CONFIGS]

_MODEL_PARAMS = _make_params(ALL_CAUSAL_LM_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _MODEL_PARAMS)
class TestBuildGraph:
    """Verify that each model type builds a valid ONNX graph."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        """Build a model graph from a tiny config and verify basic structure."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        # Basic structure checks
        assert model.graph is not None
        assert len(model.graph.inputs) > 0, "Model should have inputs"
        assert len(model.graph.outputs) > 0, "Model should have outputs"

        # Check expected inputs exist
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names

        # Check outputs include logits and KV cache
        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names

        # Check KV cache / hybrid cache outputs
        num_layers = config.num_hidden_layers
        layer_types = config.layer_types or []
        for i in range(num_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            if ltype == "mlp":
                continue  # MLP layers are stateless — no cache outputs
            if ltype in ("linear_attention",):
                assert f"present.{i}.conv_state" in output_names, (
                    f"Missing present.{i}.conv_state"
                )
                assert f"present.{i}.recurrent_state" in output_names, (
                    f"Missing present.{i}.recurrent_state"
                )
            elif ltype in ("mamba", "mamba2"):
                assert f"present.{i}.conv_state" in output_names, (
                    f"Missing present.{i}.conv_state"
                )
                assert f"present.{i}.ssm_state" in output_names, (
                    f"Missing present.{i}.ssm_state"
                )
            else:
                assert f"present.{i}.key" in output_names, f"Missing present.{i}.key"
                assert f"present.{i}.value" in output_names, f"Missing present.{i}.value"

    def test_graph_has_initializers(self, model_type: str, config_overrides: dict):
        """Verify the graph has initializers (parameters) even without weight values."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0, "Model should have initializers"

        # Check for expected parameter patterns (allow model-specific naming)
        has_embed = any(
            "embed_tokens" in n or "word_embeddings" in n or "wte" in n for n in init_names
        )
        has_attn = any(
            "self_attn" in n or "self_attention" in n or "attention" in n or ".attn." in n
            for n in init_names
        )
        has_mlp = any("mlp" in n or "expert" in n or "feed_forward" in n for n in init_names)
        assert has_embed, "Should have embedding parameters"
        assert has_attn, "Should have attention parameters"
        assert has_mlp, "Should have MLP parameters"


# === Encoder-only model configs (imported from _test_configs) ===
_ENCODER_MODEL_CONFIGS: list[tuple[str, dict]] = [(mt, ov) for mt, ov, _ in ENCODER_CONFIGS]

_ENCODER_MODEL_PARAMS = _make_params(ENCODER_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _ENCODER_MODEL_PARAMS)
class TestBuildEncoderGraph:
    """Verify that encoder-only model types build valid ONNX graphs."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        assert "token_type_ids" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "last_hidden_state" in output_names
        # No KV cache outputs for encoder-only models
        assert not any(n.startswith("present.") for n in output_names)

    def test_graph_has_initializers(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        has_embed = any("word_embeddings" in n or "embed" in n for n in init_names)
        has_attn = any(
            "self_attn" in n or "self_attention" in n or "attention" in n or ".attn." in n
            for n in init_names
        )
        has_mlp = any(
            "mlp" in n or "ffn" in n or "feed_forward" in n or "intermediate" in n
            for n in init_names
        )
        assert has_embed, "Should have word embedding parameters"
        assert has_attn, "Should have attention parameters"
        assert has_mlp, "Should have MLP parameters"


# === Encoder-decoder model configs (imported from _test_configs) ===
_SEQ2SEQ_MODEL_CONFIGS: list[tuple[str, dict]] = [(mt, ov) for mt, ov, _ in SEQ2SEQ_CONFIGS]

_SEQ2SEQ_MODEL_PARAMS = _make_params(SEQ2SEQ_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _SEQ2SEQ_MODEL_PARAMS)
class TestBuildSeq2SeqGraph:
    """Verify that encoder-decoder model types build valid ONNX graphs."""

    def test_encoder_graph_builds(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["encoder"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "last_hidden_state" in output_names

    def test_package_has_encoder_and_decoder(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert "encoder" in pkg
        assert "decoder" in pkg

        dec_outputs = {out.name for out in pkg["decoder"].graph.outputs}
        assert "logits" in dec_outputs


# === Vision model configs (imported from _test_configs) ===
_VISION_MODEL_CONFIGS: list[tuple[str, dict]] = [(mt, ov) for mt, ov, _ in VISION_CONFIGS]

_VISION_MODEL_PARAMS = _make_params(VISION_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _VISION_MODEL_PARAMS)
class TestBuildVisionGraph:
    """Verify that vision model types build valid ONNX graphs."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "pixel_values" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "last_hidden_state" in output_names


# === Object detection model configs (imported from _test_configs) ===
_DETECTION_MODEL_CONFIGS: list[tuple[str, dict]] = [
    (mt, ov) for mt, ov, _ in DETECTION_CONFIGS
]

_DETECTION_MODEL_PARAMS = _make_params(DETECTION_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _DETECTION_MODEL_PARAMS)
class TestBuildDetectionGraph:
    """Verify that object detection model types build valid ONNX graphs."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "pixel_values" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names
        assert "pred_boxes" in output_names


# === SSM (Mamba/Mamba2) configs ===
_SSM_MODEL_PARAMS = _make_params(SSM_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _SSM_MODEL_PARAMS)
class TestBuildSSMGraph:
    """Verify that SSM (Mamba/Mamba2) model types build valid ONNX graphs."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names

        # SSM models carry conv_state + ssm_state per layer
        num_layers = config.num_hidden_layers
        for i in range(num_layers):
            assert f"present.{i}.conv_state" in output_names
            assert f"present.{i}.ssm_state" in output_names

    def test_graph_has_initializers(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0, "Model should have initializers"

        has_embed = any("embeddings" in n for n in init_names)
        has_mixer = any("mixer" in n for n in init_names)
        has_norm = any("norm" in n for n in init_names)
        assert has_embed, "Should have embedding parameters"
        assert has_mixer, "Should have mixer (SSM) parameters"
        assert has_norm, "Should have norm parameters"


class TestBuildGraphLoRA:
    """Verify LoRA-specific structure in Phi4MM graph."""

    def _phi4mm_config(self):
        return _base_config(
            partial_rotary_factor=0.5,
            rope_type="longrope",
            rope_scaling={
                "short_factor": LONGROPE_FACTORS,
                "long_factor": LONGROPE_FACTORS,
            },
            original_max_position_embeddings=128,
            vision=VisionConfig(
                lora={"r": 4, "lora_alpha": 8},
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            audio=AudioConfig(
                lora={"r": 8, "lora_alpha": 16},
                attention_dim=32,
                attention_heads=2,
                num_blocks=1,
                linear_units=64,
                kernel_size=3,
                input_size=16,
                conv_channels=32,
                t5_bias_max_distance=10,
            ),
            image_token_id=200010,
        )

    def test_lora_initializers_present(self):
        config = self._phi4mm_config()
        model_cls = registry.get("phi4mm")
        module = model_cls(config)
        task = Phi4MMMultiModalTask()
        pkg = task.build(module, config)
        # LoRA adapters live in the decoder model (pkg["model"])
        decoder = pkg["model"]

        init_names = list(decoder.graph.initializers)
        lora_names = [n for n in init_names if "lora" in n]
        assert len(lora_names) > 0, "Phi4MM should have LoRA initializers"

        # Each layer should have LoRA for q/k/v/o_proj and gate/up/down_proj
        # Each proj has lora_A and lora_B for both vision and speech adapters
        vision_a = [n for n in lora_names if "lora_A.vision" in n]
        vision_b = [n for n in lora_names if "lora_B.vision" in n]
        speech_a = [n for n in lora_names if "lora_A.speech" in n]
        speech_b = [n for n in lora_names if "lora_B.speech" in n]
        assert len(vision_a) > 0, "Should have vision lora_A"
        assert len(vision_b) > 0, "Should have vision lora_B"
        assert len(speech_a) > 0, "Should have speech lora_A"
        assert len(speech_b) > 0, "Should have speech lora_B"


class TestBuildGraphQuantized:
    """Verify quantized model graphs use MatMulNBits."""

    TINY_LAYERS = 2
    NUM_PROJECTIONS_PER_LAYER = 7  # q, k, v, o, gate, up, down

    def _quantized_config(self, sym=True):
        from mobius._configs import QuantizationConfig

        qc = QuantizationConfig(bits=4, group_size=32, quant_method="gptq", sym=sym)
        return _base_config(
            num_hidden_layers=self.TINY_LAYERS,
            quantization=qc,
        )

    def test_matmulnbits_count(self):
        """Each layer has 7 projections → 2 layers = 14 MatMulNBits ops."""
        config = self._quantized_config()
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        matmulnbits = [n for n in model.graph if n.op_type == "MatMulNBits"]
        expected = self.TINY_LAYERS * self.NUM_PROJECTIONS_PER_LAYER
        assert len(matmulnbits) == expected, (
            f"Expected {expected} MatMulNBits, got {len(matmulnbits)}"
        )

    def test_scales_initializers_present(self):
        """Quantized projections should have scales initializers."""
        config = self._quantized_config()
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        scales_names = [n for n in init_names if ".scales" in n]
        expected = self.TINY_LAYERS * self.NUM_PROJECTIONS_PER_LAYER
        assert len(scales_names) == expected, (
            f"Expected {expected} scales initializers, got {len(scales_names)}"
        )

    def test_asymmetric_has_zero_points(self):
        """Asymmetric quantization should have zero_points initializers."""
        config = self._quantized_config(sym=False)
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        zp_names = [n for n in init_names if ".zero_points" in n]
        expected = self.TINY_LAYERS * self.NUM_PROJECTIONS_PER_LAYER
        assert len(zp_names) == expected, (
            f"Expected {expected} zero_points, got {len(zp_names)}"
        )

    def test_lm_head_stays_fp(self):
        """lm_head should remain a standard Linear (MatMul), not quantized."""
        config = self._quantized_config()
        model_cls = registry.get("llama")
        module = model_cls(config)
        assert type(module.lm_head).__name__ == "Linear"

    def test_no_quantization_no_matmulnbits(self):
        """Without quantization config, no MatMulNBits ops should exist."""
        config = _base_config(num_hidden_layers=1)
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        matmulnbits = [n for n in model.graph if n.op_type == "MatMulNBits"]
        assert len(matmulnbits) == 0, "Non-quantized model should have no MatMulNBits"

    def test_awq_produces_matmulnbits(self):
        """AWQ quantization should also produce MatMulNBits ops."""
        from mobius._configs import QuantizationConfig

        qc = QuantizationConfig(bits=4, group_size=32, quant_method="awq", sym=False)
        config = _base_config(num_hidden_layers=1, quantization=qc)
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        matmulnbits = [n for n in model.graph if n.op_type == "MatMulNBits"]
        expected = 1 * self.NUM_PROJECTIONS_PER_LAYER
        assert len(matmulnbits) == expected


class TestBuildGraphVisionLanguage:
    """Verify multimodal models build correctly."""

    def test_phi4mm_multimodal_graph(self):
        """Build Phi4MM with Phi4MMMultiModalTask and verify 4-model split."""
        config = _base_config(
            partial_rotary_factor=0.5,
            rope_type="longrope",
            rope_scaling={
                "short_factor": LONGROPE_FACTORS,
                "long_factor": LONGROPE_FACTORS,
            },
            original_max_position_embeddings=128,
            vision=VisionConfig(
                lora={"r": 4, "lora_alpha": 8},
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            audio=AudioConfig(
                lora={"r": 8, "lora_alpha": 16},
                attention_dim=32,
                attention_heads=2,
                num_blocks=1,
                linear_units=64,
                kernel_size=3,
                input_size=16,
                conv_channels=32,
                t5_bias_max_distance=10,
            ),
            image_token_id=200010,
        )
        model_cls = registry.get("phi4mm")
        module = model_cls(config)
        task = Phi4MMMultiModalTask()
        pkg = task.build(module, config)

        # Verify 4-model package structure
        assert "vision" in pkg, "Should have vision model"
        assert "speech" in pkg, "Should have speech model"
        assert "embedding" in pkg, "Should have embedding model"
        assert "model" in pkg, "Should have decoder model"

        # Vision model: pixel_values + image_sizes → image_features
        vision = pkg["vision"]
        v_inputs = {inp.name for inp in vision.graph.inputs}
        v_outputs = {out.name for out in vision.graph.outputs}
        assert "pixel_values" in v_inputs
        assert "image_sizes" in v_inputs
        assert "image_features" in v_outputs
        v_inits = list(vision.graph.initializers)
        assert any("img_processor" in n for n in v_inits), (
            "Vision model should have SigLIP initializers"
        )

        # Speech model: audio_embeds + metadata → audio_features (single output)
        speech = pkg["speech"]
        s_inputs = {inp.name for inp in speech.graph.inputs}
        s_outputs = {out.name for out in speech.graph.outputs}
        assert "audio_embeds" in s_inputs
        assert "audio_sizes" in s_inputs
        assert "audio_projection_mode" in s_inputs
        assert "audio_features" in s_outputs

        # Embedding model: input_ids + features → inputs_embeds
        emb = pkg["embedding"]
        e_inputs = {inp.name for inp in emb.graph.inputs}
        e_outputs = {out.name for out in emb.graph.outputs}
        assert "input_ids" in e_inputs
        assert "image_features" in e_inputs
        assert "audio_features" in e_inputs
        assert "inputs_embeds" in e_outputs

        # Decoder model (pkg["model"]): inputs_embeds → logits + KV cache
        decoder = pkg["model"]
        d_inputs = {inp.name for inp in decoder.graph.inputs}
        d_outputs = {out.name for out in decoder.graph.outputs}
        assert "inputs_embeds" in d_inputs
        assert "attention_mask" in d_inputs
        assert "position_ids" in d_inputs
        assert "logits" in d_outputs

    def test_llava_vision_language_graph(self):
        """Build LLaVA with 3-model split and verify all components."""
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=32000,
        )
        model_cls = registry.get("llava")
        module = model_cls(config)
        task_name = _default_task_for_model("llava")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision", "embedding"}

        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        vision = pkg["vision"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_internvl2_vision_language_graph(self):
        """Build InternVL2 with 3-model split and verify all components."""
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=32000,
        )
        model_cls = registry.get("internvl_chat")
        module = model_cls(config)
        task_name = _default_task_for_model("internvl_chat")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision", "embedding"}

        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        vision = pkg["vision"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

        # Verify aliases also resolve to InternVL2Model
        from mobius.models.internvl import InternVL2Model

        for alias in ("internvl2", "internvl"):
            alias_cls = registry.get(alias)
            assert alias_cls is InternVL2Model, f"{alias} should map to InternVL2Model"

    def test_qwen2_5_vl_graph(self):
        """Build Qwen2.5-VL with its auto-detected 3-model task."""
        config = _base_config(
            attn_qkv_bias=True,
            mrope_section=[8, 12, 12],
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=14,
                in_channels=3,
                out_hidden_size=64,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            fullatt_block_indexes=[1],
            image_token_id=151655,
        )
        model_cls = registry.get("qwen2_5_vl")
        module = model_cls(config)
        task_name = _default_task_for_model("qwen2_5_vl")
        task = get_task(task_name)
        pkg = task.build(module, config)

        # 3-model split: decoder, vision, embedding
        assert set(pkg.keys()) == {"decoder", "vision", "embedding"}

        # Decoder: inputs_embeds → logits + KV cache
        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        # Vision: pixel_values → image_features
        vision = pkg["vision"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        # Embedding: input_ids + image_features → inputs_embeds
        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_qwen2_5_vl_text_graph(self):
        """Build Qwen2.5-VL text-only model."""
        config = _base_config(attn_qkv_bias=True, mrope_section=[8, 12, 12])
        model_cls = registry.get("qwen2_5_vl_text")
        module = model_cls(config)
        task_name = _default_task_for_model("qwen2_5_vl_text")
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]
        assert model.graph is not None
        assert "logits" in {out.name for out in model.graph.outputs}

    def test_qwen3_vl_graph(self):
        """Build Qwen3-VL with its auto-detected 3-model task."""
        config = _base_config(
            attn_qk_norm=True,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=16,
                in_channels=3,
                out_hidden_size=64,
                num_position_embeddings=16,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0],
            image_token_id=151655,
            mrope_section=[8, 12, 12],
        )
        model_cls = registry.get("qwen3_vl")
        module = model_cls(config)
        task_name = _default_task_for_model("qwen3_vl")
        task = get_task(task_name)
        pkg = task.build(module, config)

        # 3-model split produces decoder, vision, embedding
        assert "decoder" in pkg
        assert "vision" in pkg
        assert "embedding" in pkg

        # Decoder should have logits output and inputs_embeds input
        decoder = pkg["decoder"]
        assert "logits" in {out.name for out in decoder.graph.outputs}
        assert "inputs_embeds" in {inp.name for inp in decoder.graph.inputs}

    def test_qwen35_vl_graph(self):
        """Build Qwen3.5-VL with its auto-detected 3-model task."""
        config = _base_config(
            attn_qk_norm=True,
            partial_rotary_factor=0.5,
            layer_types=["linear_attention", "full_attention"],
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=16,
                in_channels=3,
                out_hidden_size=64,
                num_position_embeddings=16,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0],
            image_token_id=248056,
            mrope_section=[8, 12, 12],
        )
        model_cls = registry.get("qwen3_5_vl")
        module = model_cls(config)
        task_name = _default_task_for_model("qwen3_5_vl")
        task = get_task(task_name)
        pkg = task.build(module, config)

        # 3-model split produces decoder, vision, embedding
        assert "decoder" in pkg
        assert "vision" in pkg
        assert "embedding" in pkg

        # Decoder should have logits output and inputs_embeds input
        decoder = pkg["decoder"]
        assert "logits" in {out.name for out in decoder.graph.outputs}
        assert "inputs_embeds" in {inp.name for inp in decoder.graph.inputs}

        # Verify hybrid cache: linear_attention layer gets conv_state/recurrent_state,
        # full_attention layer gets key/value
        output_names = {out.name for out in decoder.graph.outputs}
        assert "present.0.conv_state" in output_names
        assert "present.0.recurrent_state" in output_names
        assert "present.1.key" in output_names
        assert "present.1.value" in output_names

    def test_qwen3_vl_single_model_graph(self):
        """Build Qwen3-VL with single-model Qwen3VLVisionLanguageTask."""
        config = _base_config(
            attn_qk_norm=True,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=16,
                in_channels=3,
                out_hidden_size=64,
                num_position_embeddings=16,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0],
            image_token_id=151655,
            mrope_section=[8, 12, 12],
        )
        model_cls = registry.get("qwen3_vl_single")
        module = model_cls(config)
        task = Qwen3VLVisionLanguageTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        assert model.graph is not None
        assert "logits" in {out.name for out in model.graph.outputs}

    def test_gemma3_multimodal_graph(self):
        """Build Gemma3 multimodal model with 3-model split."""
        config = _base_config(
            attn_qk_norm=True,
            rope_local_base_freq=10_000.0,
            layer_types=["full_attention", "sliding_attention"],
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            mm_tokens_per_image=4,
            image_token_id=255999,
        )
        model_cls = registry.get("gemma3_multimodal")
        module = model_cls(config)
        task_name = _default_task_for_model("gemma3_multimodal")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision", "embedding"}
        assert "pixel_values" in {i.name for i in pkg["vision"].graph.inputs}
        assert "logits" in {o.name for o in pkg["decoder"].graph.outputs}

    def test_blip2_vision_language_graph(self):
        """Build BLIP-2 with ViT + Q-Former + LLM 3-model split."""
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=50265,
            # Q-Former config
            num_query_tokens=4,
            qformer_hidden_size=32,
            qformer_num_hidden_layers=1,
            qformer_num_attention_heads=2,
            qformer_intermediate_size=64,
        )
        model_cls = registry.get("blip-2")
        module = model_cls(config)
        task_name = _default_task_for_model("blip-2")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision", "embedding"}

        # Decoder: inputs_embeds → logits + KV cache
        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        # Vision: pixel_values → image_features (via ViT + Q-Former)
        vision = pkg["vision"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        # Embedding: input_ids + image_features → inputs_embeds
        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_llava_aliases_build(self):
        """LLaVA aliases (llava_next, llava_onevision, video_llava, etc.) all build."""
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=32000,
        )
        for model_type in (
            "aya_vision",
            "cohere2_vision",
            "deepseek_vl",
            "deepseek_vl_hybrid",
            "glm4v",
            "glm4v_moe",
            "got_ocr2",
            "instructblipvideo",
            "janus",
            "llava_next",
            "llava_next_video",
            "llava_onevision",
            "ovis2",
            "smolvlm",
            "video_llava",
            "vipllava",
            "chameleon",
            "florence2",
            "fuyu",
            "idefics2",
            "idefics3",
            "instructblip",
            "molmo",
            "paligemma",
            "pixtral",
        ):
            model_cls = registry.get(model_type)
            module = model_cls(config)
            task_name = _default_task_for_model(model_type)
            task = get_task(task_name)
            pkg = task.build(module, config)

            assert set(pkg.keys()) == {"decoder", "vision", "embedding"}, (
                f"{model_type} should produce 3 models"
            )
            assert "logits" in {o.name for o in pkg["decoder"].graph.outputs}, (
                f"{model_type} decoder missing logits"
            )
            assert "pixel_values" in {i.name for i in pkg["vision"].graph.inputs}, (
                f"{model_type} vision missing pixel_values"
            )

    def test_mllama_vision_language_graph(self):
        """Build Mllama (Llama 3.2 Vision) with cross-attention decoder."""
        from mobius._configs import MllamaConfig

        config = _base_config(
            config_cls=MllamaConfig,
            num_hidden_layers=3,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=32000,
            cross_attention_layers=[1],
        )
        model_cls = registry.get("mllama")
        module = model_cls(config)
        task_name = _default_task_for_model("mllama")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision", "embedding"}

        decoder = pkg["decoder"]
        dec_inputs = {i.name for i in decoder.graph.inputs}
        assert "inputs_embeds" in dec_inputs
        assert "logits" in {o.name for o in decoder.graph.outputs}

        # Cross-attention states must be a decoder input
        assert "cross_attention_states" in dec_inputs

        # Cross-attention layers (layer 1) should use a different
        # past-sequence-length dim than self-attention layers (0, 2)
        kv_shapes = {}
        for inp in decoder.graph.inputs:
            if inp.name.startswith("past_key_values."):
                kv_shapes[inp.name] = str(inp.shape)
        assert kv_shapes["past_key_values.1.key"] != kv_shapes["past_key_values.0.key"]
        assert kv_shapes["past_key_values.0.key"] == kv_shapes["past_key_values.2.key"]

        vision = pkg["vision"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_deepseek_ocr2_graph(self):
        """Build DeepSeek-OCR-2 with 3-model VL split."""
        config = _base_config(
            # LLM decoder: DeepSeek-V2 non-MLA + MoE
            qk_nope_head_dim=0,
            qk_rope_head_dim=0,
            v_head_dim=0,
            num_local_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            scoring_func="softmax",
            topk_method="greedy",
            first_k_dense_replace=1,
            n_shared_experts=2,
            image_token_id=100015,
        )
        model_cls = registry.get("deepseek_vl_v2")
        module = model_cls(config)
        task_name = _default_task_for_model("deepseek_vl_v2")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision", "embedding"}

        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        vision = pkg["vision"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_vl_aliases_resolve(self):
        """Verify VL alias model_types resolve to the same class and task."""
        from mobius.models.qwen35 import (
            Qwen35VL3ModelCausalLMModel,
        )
        from mobius.models.qwen_vl import (
            Qwen25VLCausalLMModel,
            Qwen25VLTextModel,
        )

        assert registry.get("qwen2_vl") is Qwen25VLCausalLMModel
        assert registry.get("qwen2_vl") is registry.get("qwen2_5_vl")
        assert _default_task_for_model("qwen2_vl") == "qwen-vl"

        assert registry.get("qwen2_vl_text") is Qwen25VLTextModel
        assert registry.get("qwen2_vl_text") is registry.get("qwen2_5_vl_text")

        assert registry.get("qwen3_5") is Qwen35VL3ModelCausalLMModel
        assert registry.get("qwen3_5") is registry.get("qwen3_5_vl")
        assert _default_task_for_model("qwen3_5") == "hybrid-qwen-vl"


class TestBuildGraphDtype:
    """Verify dtype casting for model initializers."""

    @pytest.mark.parametrize(
        "dtype_str,expected",
        [
            ("f16", "FLOAT16"),
            ("bf16", "BFLOAT16"),
        ],
    )
    def test_dtype_casts_float_initializers(self, dtype_str, expected):
        """Build with dtype and verify Parameter-derived initializers are cast."""
        config = _base_config()
        config.dtype = DTYPE_MAP[dtype_str]
        model_cls = registry.get("llama")
        module = model_cls(config)
        model = build_from_module(module, config)["model"]

        expected_dtype = ir.DataType[expected]
        for name, init in model.graph.initializers.items():
            if init.dtype == ir.DataType.INT64:
                continue
            # Lifted scalar constants (e.g. const_1.0_f32) stay f32
            if name.startswith("const_"):
                continue
            assert init.dtype == expected_dtype, (
                f"Initializer '{name}' dtype is {init.dtype}, expected {expected_dtype}"
            )


class TestBuildGraphMultiModal:
    """Verify Phi4MM builds with Phi4MMMultiModalTask (4-model split)."""

    def test_phi4mm_multimodal_graph(self):
        """Build Phi4MM 4-model split and verify all components."""
        config = _base_config(
            partial_rotary_factor=0.5,
            rope_type="longrope",
            rope_scaling={
                "short_factor": LONGROPE_FACTORS,
                "long_factor": LONGROPE_FACTORS,
            },
            original_max_position_embeddings=128,
            vision=VisionConfig(
                lora={"r": 4, "lora_alpha": 8},
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            audio=AudioConfig(
                lora={"r": 8, "lora_alpha": 16},
                attention_dim=32,
                attention_heads=2,
                num_blocks=1,
                linear_units=64,
                kernel_size=3,
                input_size=16,
                conv_channels=32,
                t5_bias_max_distance=10,
                token_id=200011,
            ),
            image_token_id=200010,
        )
        model_cls = registry.get("phi4mm")
        module = model_cls(config)
        task = Phi4MMMultiModalTask()
        pkg = task.build(module, config)

        # Verify 4 models in package
        assert len(pkg) == 4, f"Expected 4 models, got {len(pkg)}"
        for key in ("vision", "speech", "embedding", "model"):
            assert key in pkg, f"Missing model: {key}"

        # Vision model has SigLIP encoder initializers
        vision_inits = list(pkg["vision"].graph.initializers)
        assert any("img_processor" in n for n in vision_inits), (
            "Vision model should have SigLIP initializers"
        )

        # Speech model has Conformer encoder initializers
        speech_inits = list(pkg["speech"].graph.initializers)
        assert any("encoder" in n for n in speech_inits), (
            "Speech model should have Conformer initializers"
        )

        # Decoder model (pkg["model"]) has LoRA initializers
        decoder_inits = list(pkg["model"].graph.initializers)
        assert any("lora" in n for n in decoder_inits), (
            "Decoder model should have LoRA initializers"
        )

    def test_phi4_multimodal_alias_resolves(self):
        """Verify phi4_multimodal alias resolves to same class as phi4mm."""
        from mobius.models.phi import Phi4MMMultiModalModel

        assert registry.get("phi4_multimodal") is Phi4MMMultiModalModel
        assert registry.get("phi4_multimodal") is registry.get("phi4mm")
        assert _default_task_for_model("phi4_multimodal") == "phi4mm-multimodal"


class TestBuildGraphWhisper:
    """Verify Whisper encoder-decoder builds with SpeechToTextTask."""

    def _whisper_config(self):
        from mobius._configs import WhisperConfig

        return WhisperConfig(
            vocab_size=512,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_hidden_layers=TINY_LAYERS,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_HEADS,
            head_dim=TINY_HIDDEN // TINY_HEADS,
            hidden_act="gelu",
            pad_token_id=0,
            tie_word_embeddings=True,
            attn_qkv_bias=True,
            attn_o_bias=True,
            encoder_layers=TINY_LAYERS,
            encoder_attention_heads=TINY_HEADS,
            encoder_ffn_dim=TINY_INTERMEDIATE,
            num_mel_bins=16,
            max_source_positions=100,
            max_target_positions=50,
            scale_embedding=True,
        )

    def test_whisper_package_builds(self):
        """Build Whisper with SpeechToTextTask and verify encoder + decoder."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        task = SpeechToTextTask()
        pkg = build_from_module(module, config, task=task)

        assert "encoder" in pkg
        assert "decoder" in pkg

    def test_whisper_encoder_io(self):
        """Verify encoder inputs/outputs."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechToTextTask())
        encoder = pkg["encoder"]

        input_names = {inp.name for inp in encoder.graph.inputs}
        output_names = {out.name for out in encoder.graph.outputs}
        assert "input_features" in input_names
        assert "encoder_hidden_states" in output_names

    def test_whisper_decoder_io(self):
        """Verify decoder inputs/outputs including KV cache."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechToTextTask())
        decoder = pkg["decoder"]

        input_names = {inp.name for inp in decoder.graph.inputs}
        output_names = {out.name for out in decoder.graph.outputs}

        assert "decoder_input_ids" in input_names
        assert "encoder_hidden_states" in input_names
        assert "position_ids" in input_names
        assert "logits" in output_names

        # KV cache inputs/outputs
        for i in range(TINY_LAYERS):
            assert f"past_key_values.{i}.key" in input_names
            assert f"past_key_values.{i}.value" in input_names
            assert f"present.{i}.key" in output_names
            assert f"present.{i}.value" in output_names

    def test_whisper_encoder_has_initializers(self):
        """Verify encoder has conv and layer norm initializers."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechToTextTask())
        encoder = pkg["encoder"]

        init_names = list(encoder.graph.initializers)
        assert any("conv1" in n for n in init_names), "Should have conv1 initializers"
        assert any("conv2" in n for n in init_names), "Should have conv2 initializers"
        assert any("self_attn" in n for n in init_names), "Should have attention initializers"
        assert any("layer_norm" in n for n in init_names), "Should have LayerNorm initializer"

    def test_whisper_decoder_has_initializers(self):
        """Verify decoder has embedding, attention, cross-attention, and proj_out initializers."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechToTextTask())
        decoder = pkg["decoder"]

        init_names = list(decoder.graph.initializers)
        assert any("embed_tokens" in n for n in init_names), "Should have token embeddings"
        assert any("embed_positions" in n for n in init_names), (
            "Should have position embeddings"
        )
        assert any("self_attn" in n for n in init_names), "Should have self-attention"
        assert any("encoder_attn" in n for n in init_names), "Should have cross-attention"
        assert any("proj_out" in n for n in init_names), "Should have proj_out"

    def test_whisper_registry_lookup(self):
        """Verify whisper model_type is properly registered."""
        model_cls = registry.get("whisper")
        from mobius.models.whisper import WhisperForConditionalGeneration

        assert model_cls is WhisperForConditionalGeneration


class TestBuildGraphQwen3ASR:
    """Verify Qwen3-ASR 3-model split with SpeechLanguageTask."""

    def _asr_config(self):
        return _base_config(
            attn_qk_norm=True,
            hidden_act="silu",
            mrope_section=[24, 20, 20],
            mrope_interleaved=True,
            audio=AudioConfig(
                d_model=64,
                encoder_layers=2,
                encoder_attention_heads=4,
                encoder_ffn_dim=128,
                num_mel_bins=128,
                max_source_positions=256,
                downsample_hidden_size=32,
                output_dim=64,
                audio_token_id=100,
            ),
        )

    def test_package_builds_3_models(self):
        """Build Qwen3-ASR and verify 3-model package."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())

        assert "audio_encoder" in pkg
        assert "embedding" in pkg
        assert "decoder" in pkg

    def test_audio_encoder_io(self):
        """Verify audio encoder inputs/outputs."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())
        encoder = pkg["audio_encoder"]

        input_names = {inp.name for inp in encoder.graph.inputs}
        assert "input_features" in input_names

        output_names = {out.name for out in encoder.graph.outputs}
        assert "audio_features" in output_names

    def test_embedding_io(self):
        """Verify embedding model inputs/outputs."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())
        embedding = pkg["embedding"]

        input_names = {inp.name for inp in embedding.graph.inputs}
        assert "input_ids" in input_names
        assert "audio_features" in input_names

        output_names = {out.name for out in embedding.graph.outputs}
        assert "inputs_embeds" in output_names

    def test_decoder_io(self):
        """Verify decoder has MRoPE position_ids and KV cache."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())
        decoder = pkg["decoder"]

        input_names = {inp.name for inp in decoder.graph.inputs}
        assert "inputs_embeds" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names

        output_names = {out.name for out in decoder.graph.outputs}
        assert "logits" in output_names
        assert "present.0.key" in output_names
        assert "present.0.value" in output_names

    def test_registry_lookup(self):
        """Verify qwen3_asr is registered with speech-language task."""
        model_cls = registry.get("qwen3_asr")
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )

        assert model_cls is Qwen3ASRForConditionalGeneration
        assert _default_task_for_model("qwen3_asr") == "speech-language"

    def test_qwen3_forced_aligner_alias_resolves(self):
        """Verify qwen3_forced_aligner alias resolves to same class as qwen3_asr."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )

        assert registry.get("qwen3_forced_aligner") is Qwen3ASRForConditionalGeneration
        assert registry.get("qwen3_forced_aligner") is registry.get("qwen3_asr")
        assert _default_task_for_model("qwen3_forced_aligner") == "speech-language"

    def test_3model_pipeline_runs_with_ort(self):
        """Run audio_encoder → embedding with ORT.

        Guards against audio token count mismatches: the number of
        AUDIO_TOKEN_ID positions in input_ids must equal the number of
        audio feature rows from the encoder, otherwise the embedding
        Gather goes out of bounds.
        """
        import numpy as np

        from mobius._testing.ort_inference import (
            OnnxModelSession,
        )
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.rewrite_rules._testing_utils import (
            fill_random_weights,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())

        for model in pkg.values():
            fill_random_weights(model)

        # Step 1: Audio encoder — random mel input
        enc_sess = OnnxModelSession(pkg["audio_encoder"])
        mel = np.random.randn(1, config.audio.num_mel_bins, 100).astype(np.float32)
        enc_out = enc_sess.run({"input_features": mel})
        audio_features = enc_out["audio_features"]
        num_audio_tokens = audio_features.shape[1]
        # Flatten to 2D: (num_audio_tokens, output_dim)
        audio_features_2d = audio_features.reshape(-1, audio_features.shape[-1])
        enc_sess.close()

        # Step 2: Embedding — mix text + audio tokens
        # Build input_ids with exactly num_audio_tokens audio pad tokens
        # Use the config's audio_token_id (must be within vocab_size)
        audio_token_id = config.audio.audio_token_id
        prefix = [1, 2, 3]  # mock system/user tokens
        suffix = [4, 5]  # mock footer tokens
        input_ids = np.array(
            [prefix + [audio_token_id] * num_audio_tokens + suffix],
            dtype=np.int64,
        )

        embed_sess = OnnxModelSession(pkg["embedding"])
        embed_out = embed_sess.run(
            {
                "input_ids": input_ids,
                "audio_features": audio_features_2d,
            }
        )
        inputs_embeds = embed_out["inputs_embeds"]
        embed_sess.close()

        seq_len = inputs_embeds.shape[1]
        assert seq_len == input_ids.shape[1]
        assert inputs_embeds.shape[2] == config.hidden_size

        # Step 3: Decoder — single forward pass with MRoPE
        decoder_sess = OnnxModelSession(pkg["decoder"])
        past_kv = {}
        for i in range(config.num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )

        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        # MRoPE: (3, 1, seq_len)
        position_ids = np.stack([pos, pos, pos])

        dec_out = decoder_sess.run(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": np.ones((1, seq_len), dtype=np.int64),
                "position_ids": position_ids,
                **past_kv,
            }
        )
        decoder_sess.close()

        logits = dec_out["logits"]
        assert logits.shape[0] == 1
        assert logits.shape[1] == seq_len


class TestBuildGraphQwen3TTS:
    """Verify Qwen3-TTS 4-model split with TTSTask."""

    def _tts_config(self):
        """Tiny config mimicking 0.6B: hidden_size=64 but text_hidden_size=128."""
        return _base_config(
            attn_qk_norm=True,
            hidden_act="silu",
            rope_scaling={
                "rope_type": "default",
                "mrope_section": [24, 20, 20],
            },
            mrope_interleaved=True,
            tts=TTSConfig(
                text_hidden_size=TINY_INTERMEDIATE,  # 128 (larger than hidden)
                text_vocab_size=TINY_VOCAB,
                num_code_groups=4,  # Fewer groups for testing
                code_predictor=CodePredictorConfig(
                    hidden_size=TINY_HIDDEN,
                    intermediate_size=TINY_INTERMEDIATE,
                    num_hidden_layers=2,
                    num_attention_heads=TINY_HEADS,
                    num_key_value_heads=TINY_KV_HEADS,
                    head_dim=TINY_HEAD_DIM,
                    vocab_size=TINY_VOCAB,
                    num_code_groups=4,
                ),
                speaker_encoder=SpeakerEncoderConfig(
                    mel_dim=32,
                    enc_dim=TINY_HIDDEN,
                    enc_channels=[16, 16, 16, 16, 48],
                    enc_kernel_sizes=[5, 3, 3, 3, 1],
                    enc_dilations=[1, 2, 3, 4, 1],
                    enc_attention_channels=16,
                    enc_res2net_scale=2,
                    enc_se_channels=16,
                ),
            ),
        )

    def test_package_builds_4_models(self):
        """Build Qwen3-TTS and verify 4-model package."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())

        assert "talker" in pkg
        assert "code_predictor" in pkg
        assert "embedding" in pkg
        assert "speaker_encoder" in pkg

    def test_talker_io(self):
        """Verify talker has inputs_embeds, logits, last_hidden_state, KV cache."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())
        talker = pkg["talker"]

        input_names = {inp.name for inp in talker.graph.inputs}
        assert "inputs_embeds" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names
        assert "past_key_values.0.key" in input_names

        output_names = {out.name for out in talker.graph.outputs}
        assert "logits" in output_names
        assert "last_hidden_state" in output_names
        assert "present.0.key" in output_names

    def test_code_predictor_io(self):
        """Verify code predictor takes inputs_embeds and step_index."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())
        cp = pkg["code_predictor"]

        input_names = {inp.name for inp in cp.graph.inputs}
        assert "inputs_embeds" in input_names
        assert "step_index" in input_names
        assert "position_ids" in input_names
        assert "attention_mask" in input_names

        output_names = {out.name for out in cp.graph.outputs}
        assert "logits" in output_names

        # Verify 2D position_ids (1D RoPE, not 3D MRoPE)
        pos_input = next(i for i in cp.graph.inputs if i.name == "position_ids")
        assert len(pos_input.shape) == 2  # (batch, seq_len)

    def test_embedding_io(self):
        """Verify embedding model takes text_ids + codec_ids."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())
        embedding = pkg["embedding"]

        input_names = {inp.name for inp in embedding.graph.inputs}
        assert "text_ids" in input_names
        assert "codec_ids" in input_names

        output_names = {out.name for out in embedding.graph.outputs}
        assert "text_embeds" in output_names
        assert "codec_embeds" in output_names

    def test_speaker_encoder_io(self):
        """Verify speaker encoder takes mel_input."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())
        se = pkg["speaker_encoder"]

        input_names = {inp.name for inp in se.graph.inputs}
        assert "mel_input" in input_names

        output_names = {out.name for out in se.graph.outputs}
        assert "speaker_embedding" in output_names

    def test_registry_lookup(self):
        """Verify qwen3_tts is registered with tts task."""
        model_cls = registry.get("qwen3_tts")
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )

        assert model_cls is Qwen3TTSForConditionalGeneration
        assert _default_task_for_model("qwen3_tts") == "tts"


class TestBuildVAEGraph:
    """Verify VAE (AutoencoderKL) graph construction."""

    def _vae_config(self):
        from mobius._diffusers_configs import VAEConfig

        return VAEConfig(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=32,
            act_fn="silu",
            mid_block_add_attention=True,
            use_quant_conv=True,
            use_post_quant_conv=True,
        )

    def test_decoder_graph_builds(self):
        from mobius.models.vae import AutoencoderKLModel
        from mobius.tasks import VAETask

        config = self._vae_config()
        module = AutoencoderKLModel(config)
        task = VAETask()
        pkg = task.build(module, config)
        model = pkg["decoder"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "latent_sample" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "sample" in output_names

    def test_package_has_encoder_and_decoder(self):
        from mobius.models.vae import AutoencoderKLModel
        from mobius.tasks import VAETask

        config = self._vae_config()
        module = AutoencoderKLModel(config)
        task = VAETask()
        pkg = task.build(module, config)

        assert "encoder" in pkg
        assert "decoder" in pkg

        # Encoder: sample → latent_dist
        enc_inputs = {inp.name for inp in pkg["encoder"].graph.inputs}
        enc_outputs = {out.name for out in pkg["encoder"].graph.outputs}
        assert "sample" in enc_inputs
        assert "latent_dist" in enc_outputs

        # Decoder: latent_sample → sample
        dec_inputs = {inp.name for inp in pkg["decoder"].graph.inputs}
        dec_outputs = {out.name for out in pkg["decoder"].graph.outputs}
        assert "latent_sample" in dec_inputs
        assert "sample" in dec_outputs

    def test_decoder_has_initializers(self):
        from mobius.models.vae import AutoencoderKLModel
        from mobius.tasks import VAETask

        config = self._vae_config()
        module = AutoencoderKLModel(config)
        task = VAETask()
        pkg = task.build(module, config)
        model = pkg["decoder"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        has_conv = any("conv" in n for n in init_names)
        has_norm = any("norm" in n for n in init_names)
        assert has_conv, "Should have conv initializers"
        assert has_norm, "Should have norm initializers"


class TestBuildAudioGraph:
    """Verify audio encoder-only models build valid ONNX graphs."""

    def test_wav2vec2_graph_builds(self):
        from mobius.models.wav2vec2 import Wav2Vec2Model
        from mobius.tasks import AudioFeatureExtractionTask

        config = _base_config()
        module = Wav2Vec2Model(config)
        task = AudioFeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_values" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "last_hidden_state" in output_names

    def test_wav2vec2_has_initializers(self):
        from mobius.models.wav2vec2 import Wav2Vec2Model
        from mobius.tasks import AudioFeatureExtractionTask

        config = _base_config()
        module = Wav2Vec2Model(config)
        task = AudioFeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        has_feature_extractor = any("feature_extractor" in n for n in init_names)
        has_attention = any("attention" in n for n in init_names)
        assert has_feature_extractor, "Should have feature extractor initializers"
        assert has_attention, "Should have attention initializers"

    def test_audio_aliases_build(self):
        """Audio model aliases (hubert, wavlm, musicgen, etc.) all build."""
        from mobius.tasks import AudioFeatureExtractionTask

        config = _base_config()
        task = AudioFeatureExtractionTask()
        for model_type in (
            "data2vec-audio",
            "hubert",
            "wavlm",
            "mctct",
            "musicgen",
            "seamless_m4t",
            "seamless_m4t_v2",
            "sew",
            "sew-d",
            "speecht5",
            "unispeech",
            "unispeech-sat",
            "voxtral_encoder",
            "wav2vec2",
            "wav2vec2-bert",
            "wav2vec2-conformer",
        ):
            model_cls = registry.get(model_type)
            module = model_cls(config)
            pkg = task.build(module, config)
            model = pkg["model"]
            assert model.graph is not None, f"{model_type} graph should build"

            input_names = {inp.name for inp in model.graph.inputs}
            assert "input_values" in input_names, f"{model_type} missing input_values"

            output_names = {out.name for out in model.graph.outputs}
            assert "last_hidden_state" in output_names, (
                f"{model_type} missing last_hidden_state"
            )

            init_names = list(model.graph.initializers)
            assert len(init_names) > 0, f"{model_type} should have initializers"


class TestBuildUNetGraph:
    """Verify UNet2DConditionModel graph construction."""

    def _unet_config(self):
        from mobius._diffusers_configs import UNet2DConfig

        return UNet2DConfig(
            in_channels=4,
            out_channels=4,
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=32,
            cross_attention_dim=32,
            attention_head_dim=8,
        )

    def test_unet_graph_builds(self):
        from mobius.models.unet import UNet2DConditionModel
        from mobius.tasks import DenoisingTask

        config = self._unet_config()
        module = UNet2DConditionModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "noise_pred" in output_names

    def test_unet_has_initializers(self):
        from mobius.models.unet import UNet2DConditionModel
        from mobius.tasks import DenoisingTask

        config = self._unet_config()
        module = UNet2DConditionModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        has_time_emb = any("time_embedding" in n for n in init_names)
        has_conv = any("conv" in n for n in init_names)
        has_mid = any("mid_block" in n for n in init_names)
        assert has_time_emb, "Should have time embedding initializers"
        assert has_conv, "Should have conv initializers"
        assert has_mid, "Should have mid block initializers"


class TestBuildDiTGraph:
    """Verify DiT transformer denoiser graph construction."""

    def test_dit_graph_builds(self):
        from mobius.models.dit import DiTConfig, DiTTransformer2DModel
        from mobius.tasks import DenoisingTask

        config = DiTConfig(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            cross_attention_dim=32,
            caption_channels=32,
            sample_size=8,
        )
        module = DiTTransformer2DModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "noise_pred" in output_names


class TestBuildHunyuanDiTGraph:
    """Verify HunyuanDiT transformer denoiser graph construction."""

    def test_hunyuan_dit_graph_builds(self):
        from mobius.models.hunyuan_dit import (
            HunyuanDiT2DModel,
            HunyuanDiTConfig,
        )
        from mobius.tasks import DenoisingTask

        config = HunyuanDiTConfig(
            in_channels=4,
            patch_size=2,
            hidden_size=64,
            num_layers=4,
            num_attention_heads=4,
            cross_attention_dim=32,
            mlp_ratio=4.0,
            learn_sigma=True,
            sample_size=8,
        )
        module = HunyuanDiT2DModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "noise_pred" in output_names


class TestBuildControlNetGraph:
    """Verify ControlNet model graph construction."""

    def test_controlnet_graph_builds(self):
        from mobius.models.controlnet import ControlNetConfig, ControlNetModel
        from mobius.tasks import ControlNetTask

        config = ControlNetConfig(
            in_channels=4,
            conditioning_channels=3,
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=32,
            cross_attention_dim=32,
            attention_head_dim=8,
        )
        module = ControlNetModel(config)
        task = ControlNetTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "controlnet_cond" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "mid_block_res" in output_names
        down_res = [n for n in output_names if n.startswith("down_block_res_")]
        assert len(down_res) > 0, "Should have down block residual outputs"


class TestBuildVideoVAEGraph:
    """Verify Video VAE (3D autoencoder) graph construction."""

    def test_video_decoder_graph_builds(self):
        from mobius.models.video_vae import VideoAutoencoderModel, VideoVAEConfig
        from mobius.tasks import VAETask

        config = VideoVAEConfig(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=32,
        )
        module = VideoAutoencoderModel(config)
        task = VAETask()
        pkg = task.build(module, config)
        model = pkg["decoder"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "latent_sample" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "sample" in output_names

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        has_conv = any("conv" in n for n in init_names)
        assert has_conv, "Should have 3D conv initializers"


class TestBuildSD3Graph:
    """Verify SD3 (MMDiT) transformer denoiser graph construction."""

    def test_sd3_graph_builds(self):
        from mobius.models.flux_sd3 import SD3Config, SD3Transformer2DModel
        from mobius.tasks import DenoisingTask

        config = SD3Config(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            joint_attention_dim=32,
            caption_projection_dim=32,
            cross_attention_dim=32,
            sample_size=8,
        )
        module = SD3Transformer2DModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "noise_pred" in {out.name for out in model.graph.outputs}


class TestBuildFluxGraph:
    """Verify Flux transformer denoiser graph construction."""

    def test_flux_graph_builds(self):
        from mobius.models.flux_sd3 import FluxConfig, FluxTransformer2DModel
        from mobius.tasks import DenoisingTask

        config = FluxConfig(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            hidden_size=64,
            num_layers=1,
            num_single_layers=2,
            num_attention_heads=4,
            joint_attention_dim=32,
            cross_attention_dim=32,
            sample_size=8,
        )
        module = FluxTransformer2DModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "noise_pred" in {out.name for out in model.graph.outputs}


class TestBuildCogVideoXGraph:
    """Verify CogVideoX 3D video transformer graph construction."""

    def test_cogvideox_graph_builds(self):
        from mobius._diffusers_configs import CogVideoXConfig
        from mobius.models.cogvideox import CogVideoXTransformer3DModel
        from mobius.tasks import VideoDenoisingTask

        config = CogVideoXConfig(
            num_attention_heads=2,
            attention_head_dim=32,
            in_channels=4,
            out_channels=4,
            time_embed_dim=64,
            text_embed_dim=32,
            num_layers=2,
            patch_size=2,
            sample_height=8,
            sample_width=8,
            sample_frames=9,
            temporal_compression_ratio=4,
            max_text_seq_length=8,
            spatial_interpolation_scale=1.0,
            temporal_interpolation_scale=1.0,
            norm_eps=1e-5,
            cross_attention_dim=32,
        )
        module = CogVideoXTransformer3DModel(config)
        task = VideoDenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "noise_pred" in {out.name for out in model.graph.outputs}

        # Verify 5D sample shape
        sample_input = next(inp for inp in model.graph.inputs if inp.name == "sample")
        assert len(sample_input.shape) == 5  # [B, T, C, H, W]


class TestBuildAdapterGraph:
    """Verify T2I-Adapter and IP-Adapter graph construction."""

    def test_t2i_adapter_graph_builds(self):
        from mobius.models.adapters import T2IAdapterConfig, T2IAdapterModel
        from mobius.tasks import AdapterTask

        config = T2IAdapterConfig(in_channels=3, channels=(32, 64), num_res_blocks=1)
        module = T2IAdapterModel(config)
        task = AdapterTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "condition" in input_names
        output_names = {out.name for out in model.graph.outputs}
        assert any(n.startswith("feature_") for n in output_names)

    def test_ip_adapter_graph_builds(self):
        from mobius.models.adapters import IPAdapterConfig, IPAdapterModel
        from mobius.tasks import AdapterTask

        config = IPAdapterConfig(image_embed_dim=32, cross_attention_dim=64, num_tokens=4)
        module = IPAdapterModel(config)
        task = AdapterTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "image_embeds" in input_names
        output_names = {out.name for out in model.graph.outputs}
        assert "adapter_output" in output_names


class TestBuildQwenImageGraph:
    """Verify QwenImage transformer denoiser graph construction."""

    def test_qwen_image_transformer_graph_builds(self):
        from mobius._diffusers_configs import QwenImageConfig
        from mobius.models.qwen_image import QwenImageTransformer2DModel
        from mobius.tasks import DenoisingTask

        config = QwenImageConfig(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            num_layers=2,
            attention_head_dim=32,
            num_attention_heads=2,
            joint_attention_dim=64,
            cross_attention_dim=64,
        )
        module = QwenImageTransformer2DModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "noise_pred" in {out.name for out in model.graph.outputs}

    def test_qwen_image_vae_encoder_decoder_graphs_build(self):
        from mobius._diffusers_configs import QwenImageVAEConfig
        from mobius.models.qwen_image_vae import AutoencoderKLQwenImageModel
        from mobius.tasks import QwenImageVAETask

        config = QwenImageVAEConfig(
            base_dim=8,
            z_dim=4,
            dim_mult=(1, 2),
            num_res_blocks=1,
            temperal_downsample=(False,),
        )
        module = AutoencoderKLQwenImageModel(config)
        task = QwenImageVAETask()
        pkg = task.build(module, config)

        enc = pkg["encoder"]
        assert enc.graph is not None
        assert "sample" in {inp.name for inp in enc.graph.inputs}
        assert "latent_dist" in {out.name for out in enc.graph.outputs}

        dec = pkg["decoder"]
        assert dec.graph is not None
        assert "latent_sample" in {inp.name for inp in dec.graph.inputs}
        assert "sample" in {out.name for out in dec.graph.outputs}


class TestBuildCodecGraph:
    """Verify codec tokenizer (Qwen3-TTS-Tokenizer-12Hz) graph construction."""

    @staticmethod
    def _codec_config():
        from mobius._configs import (
            ArchitectureConfig,
            CodecDecoderConfig,
            CodecEncoderConfig,
        )

        return ArchitectureConfig(
            # Use decoder's transformer dims as top-level (from exporter)
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            intermediate_size=64,
            vocab_size=256,
            max_position_embeddings=128,
            rms_norm_eps=1e-5,
            codec_decoder=CodecDecoderConfig(
                codebook_dim=32,
                codebook_size=64,
                latent_dim=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=8,
                rms_norm_eps=1e-5,
                rope_theta=10000.0,
                max_position_embeddings=128,
                decoder_dim=96,
                num_quantizers=4,
                upsample_rates=[2, 2, 2, 2],
                upsampling_ratios=[2, 2],
            ),
            codec_encoder=CodecEncoderConfig(
                codebook_dim=16,
                codebook_size=64,
                hidden_size=32,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=8,
                rope_theta=10000.0,
                max_position_embeddings=128,
                num_quantizers=8,
                num_semantic_quantizers=1,
            ),
        )

    def test_package_builds_2_models(self):
        """Build codec tokenizer and verify 2-model package."""
        from mobius.models.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizerV2Model,
        )
        from mobius.tasks import CodecTask

        config = self._codec_config()
        module = Qwen3TTSTokenizerV2Model(config)
        pkg = build_from_module(module, config, task=CodecTask())

        assert "decoder" in pkg
        assert "encoder" in pkg

    def test_decoder_io(self):
        """Verify decoder: codes → waveform."""
        from mobius.models.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizerV2Model,
        )
        from mobius.tasks import CodecTask

        config = self._codec_config()
        module = Qwen3TTSTokenizerV2Model(config)
        pkg = build_from_module(module, config, task=CodecTask())
        decoder = pkg["decoder"]

        input_names = {inp.name for inp in decoder.graph.inputs}
        assert "codes" in input_names

        output_names = {out.name for out in decoder.graph.outputs}
        assert "waveform" in output_names

    def test_encoder_io(self):
        """Verify encoder: waveform → codes."""
        from mobius.models.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizerV2Model,
        )
        from mobius.tasks import CodecTask

        config = self._codec_config()
        module = Qwen3TTSTokenizerV2Model(config)
        pkg = build_from_module(module, config, task=CodecTask())
        encoder = pkg["encoder"]

        input_names = {inp.name for inp in encoder.graph.inputs}
        assert "waveform" in input_names

        output_names = {out.name for out in encoder.graph.outputs}
        assert "codes" in output_names

    def test_registry_lookup(self):
        """Verify qwen3_tts_tokenizer_12hz is registered with codec task."""
        model_cls = registry.get("qwen3_tts_tokenizer_12hz")
        from mobius.models.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizerV2Model,
        )

        assert model_cls is Qwen3TTSTokenizerV2Model
        assert _default_task_for_model("qwen3_tts_tokenizer_12hz") == "codec"


# ===========================================================================
# SSM (Mamba) model tests
# ===========================================================================


class TestBuildMambaGraph:
    """Verify Mamba SSM model builds with SSMCausalLMTask."""

    def _mamba_config(self):
        from mobius._configs import MambaConfig

        return MambaConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_HIDDEN * 2,  # expand=2
            num_hidden_layers=TINY_LAYERS,
            state_size=8,
            conv_kernel=4,
            expand=2,
            time_step_rank=4,
            layer_norm_epsilon=1e-5,
            use_conv_bias=True,
            tie_word_embeddings=True,
        )

    def test_mamba_builds(self):
        """Build Mamba model and verify basic graph structure."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import MambaCausalLMModel
        from mobius.tasks import SSMCausalLMTask

        config = self._mamba_config()
        module = MambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=SSMCausalLMTask())
        model = pkg["model"]

        assert model.graph is not None
        assert len(model.graph.inputs) > 0
        assert len(model.graph.outputs) > 0

    def test_mamba_inputs_no_attention(self):
        """Verify Mamba has input_ids but NOT attention_mask or position_ids."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import MambaCausalLMModel
        from mobius.tasks import SSMCausalLMTask

        config = self._mamba_config()
        module = MambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=SSMCausalLMTask())
        model = pkg["model"]

        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "attention_mask" not in input_names
        assert "position_ids" not in input_names

    def test_mamba_ssm_state_io(self):
        """Verify conv_state + ssm_state per layer in inputs/outputs."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import MambaCausalLMModel
        from mobius.tasks import SSMCausalLMTask

        config = self._mamba_config()
        module = MambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=SSMCausalLMTask())
        model = pkg["model"]

        input_names = {inp.name for inp in model.graph.inputs}
        output_names = {out.name for out in model.graph.outputs}

        for i in range(config.num_hidden_layers):
            assert f"past_states.{i}.conv_state" in input_names
            assert f"past_states.{i}.ssm_state" in input_names
            assert f"present.{i}.conv_state" in output_names
            assert f"present.{i}.ssm_state" in output_names

    def test_mamba_logits_output(self):
        """Verify logits are in the outputs."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import MambaCausalLMModel
        from mobius.tasks import SSMCausalLMTask

        config = self._mamba_config()
        module = MambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=SSMCausalLMTask())
        model = pkg["model"]

        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names

    def test_mamba_has_initializers(self):
        """Verify model has SSM-specific parameters."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import MambaCausalLMModel
        from mobius.tasks import SSMCausalLMTask

        config = self._mamba_config()
        module = MambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=SSMCausalLMTask())
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        # Check for Mamba-specific parameters
        assert any("embeddings" in n for n in init_names)
        assert any("mixer" in n for n in init_names)
        assert any("norm" in n for n in init_names)

    def test_mamba_registry_lookup(self):
        """Verify 'mamba' is registered and uses SSM task."""
        model_cls = registry.get("mamba")
        from mobius.models.mamba import MambaCausalLMModel

        assert model_cls is MambaCausalLMModel
        assert _default_task_for_model("mamba") == "ssm-text-generation"

    def test_mamba_preprocess_weights_ssm_nesting(self):
        """Verify preprocess_weights maps flat mixer SSM params to nested ssm."""
        import torch

        from mobius.models.mamba import MambaCausalLMModel

        config = self._mamba_config()
        module = MambaCausalLMModel(config)

        # Simulate HF weight names (flat mixer SSM params)
        state_dict = {
            "model.layers.0.mixer.A_log": torch.zeros(1),
            "model.layers.0.mixer.D": torch.zeros(1),
            "model.layers.0.mixer.x_proj.weight": torch.zeros(1),
            "model.layers.0.mixer.dt_proj.weight": torch.zeros(1),
            "model.layers.0.mixer.dt_proj.bias": torch.zeros(1),
            "model.layers.0.mixer.in_proj.weight": torch.zeros(1),
            "model.layers.0.mixer.conv1d.weight": torch.zeros(1),
            "model.layers.0.mixer.out_proj.weight": torch.zeros(1),
            "model.layers.0.norm.weight": torch.zeros(1),
            "model.embeddings.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(1),
        }
        result = module.preprocess_weights(state_dict)

        # SSM params should be nested under .mixer.ssm.
        assert "model.layers.0.mixer.ssm.A_log" in result
        assert "model.layers.0.mixer.ssm.D" in result
        assert "model.layers.0.mixer.ssm.x_proj.weight" in result
        assert "model.layers.0.mixer.ssm.dt_proj.weight" in result
        assert "model.layers.0.mixer.ssm.dt_proj.bias" in result
        # Non-SSM mixer params stay flat
        assert "model.layers.0.mixer.in_proj.weight" in result
        assert "model.layers.0.mixer.conv1d.weight" in result
        assert "model.layers.0.mixer.out_proj.weight" in result


# ===========================================================================
# FalconMamba SSM model tests
# ===========================================================================


class TestBuildFalconMambaGraph:
    """Verify FalconMamba reuses MambaCausalLMModel via registry."""

    def _falcon_mamba_config(self):
        from mobius._configs import MambaConfig

        return MambaConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_HIDDEN * 2,
            num_hidden_layers=TINY_LAYERS,
            state_size=8,
            conv_kernel=4,
            expand=2,
            time_step_rank=4,
            layer_norm_epsilon=1e-5,
            use_conv_bias=True,
            tie_word_embeddings=True,
        )

    def test_falcon_mamba_registry_lookup(self):
        """Verify 'falcon_mamba' maps to MambaCausalLMModel."""
        model_cls = registry.get("falcon_mamba")
        from mobius.models.mamba import MambaCausalLMModel

        assert model_cls is MambaCausalLMModel
        assert _default_task_for_model("falcon_mamba") == "ssm-text-generation"

    def test_falcon_mamba_builds(self):
        """Build FalconMamba model and verify graph structure."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import MambaCausalLMModel
        from mobius.tasks import SSMCausalLMTask

        config = self._falcon_mamba_config()
        module = MambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=SSMCausalLMTask())
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        output_names = {out.name for out in model.graph.outputs}
        assert "input_ids" in input_names
        assert "logits" in output_names
        # SSM state I/O
        for i in range(config.num_hidden_layers):
            assert f"past_states.{i}.conv_state" in input_names
            assert f"present.{i}.conv_state" in output_names


# ===========================================================================
# Standalone Mamba2/SSD model tests
# ===========================================================================


class TestBuildMamba2Graph:
    """Verify standalone Mamba2/SSD model builds correctly."""

    def _mamba2_config(self):
        from mobius._configs import Mamba2Config

        return Mamba2Config(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_HIDDEN * 2,
            num_hidden_layers=TINY_LAYERS,
            num_heads=8,
            head_dim=16,
            state_size=8,
            n_groups=1,
            conv_kernel=4,
            expand=2,
            layer_norm_epsilon=1e-5,
            use_conv_bias=True,
        )

    def test_mamba2_registry_lookup(self):
        """Verify 'mamba2' maps to Mamba2CausalLMModel."""
        model_cls = registry.get("mamba2")
        from mobius.models.mamba import Mamba2CausalLMModel

        assert model_cls is Mamba2CausalLMModel
        assert _default_task_for_model("mamba2") == "ssm2-text-generation"

    def test_mamba2_builds(self):
        """Build standalone Mamba2 model and verify graph structure."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import Mamba2CausalLMModel
        from mobius.tasks import SSM2CausalLMTask

        config = self._mamba2_config()
        module = Mamba2CausalLMModel(config)
        pkg = build_from_module(module, config, task=SSM2CausalLMTask())
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        output_names = {out.name for out in model.graph.outputs}
        assert "input_ids" in input_names
        assert "logits" in output_names

    def test_mamba2_state_io(self):
        """Verify 4D SSM state shapes in graph I/O."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import Mamba2CausalLMModel
        from mobius.tasks import SSM2CausalLMTask

        config = self._mamba2_config()
        module = Mamba2CausalLMModel(config)
        pkg = build_from_module(module, config, task=SSM2CausalLMTask())
        model = pkg["model"]

        input_names = {inp.name for inp in model.graph.inputs}
        output_names = {out.name for out in model.graph.outputs}

        # Every layer should have conv_state + ssm_state
        for i in range(config.num_hidden_layers):
            assert f"past_states.{i}.conv_state" in input_names
            assert f"past_states.{i}.ssm_state" in input_names
            assert f"present.{i}.conv_state" in output_names
            assert f"present.{i}.ssm_state" in output_names

    def test_mamba2_preprocess_weights(self):
        """Verify SSM param nesting: mixer.A_log -> mixer.ssm.A_log."""
        from mobius.models.mamba import Mamba2CausalLMModel

        config = self._mamba2_config()
        module = Mamba2CausalLMModel(config)

        import torch

        state_dict = {
            "backbone.layers.0.mixer.A_log": torch.zeros(8),
            "backbone.layers.0.mixer.D": torch.zeros(8),
            "backbone.layers.0.mixer.dt_bias": torch.zeros(8),
            "backbone.layers.0.mixer.in_proj.weight": torch.zeros(280, 64),
            "backbone.layers.0.norm.weight": torch.zeros(64),
        }
        result = module.preprocess_weights(state_dict)

        assert "backbone.layers.0.mixer.ssm.A_log" in result
        assert "backbone.layers.0.mixer.ssm.D" in result
        assert "backbone.layers.0.mixer.ssm.dt_bias" in result
        # Non-SSM params stay as-is
        assert "backbone.layers.0.mixer.in_proj.weight" in result
        assert "backbone.layers.0.norm.weight" in result


# ===========================================================================
# Hybrid Mamba2+Attention (Bamba) model tests
# ===========================================================================


class TestBuildBambaGraph:
    """Verify Bamba hybrid Mamba2+Attention model builds correctly."""

    def _bamba_config(self):
        from mobius._configs import BambaConfig

        return BambaConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_hidden_layers=4,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            rms_norm_eps=1e-5,
            layer_types=[
                "mamba2",
                "full_attention",
                "mamba2",
                "mamba2",
            ],
            mamba_n_heads=4,
            mamba_d_head=32,
            mamba_d_state=8,
            mamba_n_groups=1,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_conv_bias=True,
            mamba_proj_bias=False,
            hidden_act="silu",
            head_dim=TINY_HIDDEN // TINY_HEADS,
        )

    def test_bamba_builds(self):
        """Build Bamba model and verify basic graph structure."""
        from mobius._builder import build_from_module
        from mobius.models.bamba import BambaCausalLMModel
        from mobius.tasks import HybridCausalLMTask

        config = self._bamba_config()
        module = BambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=HybridCausalLMTask())
        model = pkg["model"]

        assert model.graph is not None
        assert len(model.graph.inputs) > 0
        assert len(model.graph.outputs) > 0

    def test_bamba_hybrid_cache_io(self):
        """Verify mixed Mamba2/attention cache I/O."""
        from mobius._builder import build_from_module
        from mobius.models.bamba import BambaCausalLMModel
        from mobius.tasks import HybridCausalLMTask

        config = self._bamba_config()
        module = BambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=HybridCausalLMTask())
        model = pkg["model"]

        input_names = {inp.name for inp in model.graph.inputs}
        output_names = {out.name for out in model.graph.outputs}

        assert "past_key_values.0.conv_state" in input_names
        assert "past_key_values.0.ssm_state" in input_names
        assert "present.0.conv_state" in output_names
        assert "present.0.ssm_state" in output_names
        assert "past_key_values.1.key" in input_names
        assert "past_key_values.1.value" in input_names
        assert "present.1.key" in output_names
        assert "present.1.value" in output_names
        assert "past_key_values.2.conv_state" in input_names
        assert "past_key_values.3.conv_state" in input_names

    def test_bamba_registry_lookup(self):
        """Verify bamba is registered and uses hybrid task."""
        model_cls = registry.get("bamba")
        from mobius.models.bamba import BambaCausalLMModel

        assert model_cls is BambaCausalLMModel
        assert _default_task_for_model("bamba") == "hybrid-text-generation"

    def test_bamba_preprocess_weights(self):
        """Verify preprocess_weights nests SSM params under mamba.ssm."""
        import torch

        from mobius.models.bamba import BambaCausalLMModel

        config = self._bamba_config()
        module = BambaCausalLMModel(config)

        state_dict = {
            "model.layers.0.mamba.A_log": torch.zeros(4),
            "model.layers.0.mamba.D": torch.zeros(4),
            "model.layers.0.mamba.dt_bias": torch.zeros(4),
            "model.layers.0.mamba.in_proj.weight": torch.zeros(1),
            "model.layers.0.mamba.conv1d.weight": torch.zeros(1),
            "model.layers.0.mamba.norm.weight": torch.zeros(1),
            "model.layers.0.mamba.out_proj.weight": torch.zeros(1),
            "model.layers.1.self_attn.q_proj.weight": torch.zeros(1),
            "model.embed_tokens.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(1),
        }
        result = module.preprocess_weights(state_dict)

        assert "model.layers.0.mamba.ssm.A_log" in result
        assert "model.layers.0.mamba.ssm.D" in result
        assert "model.layers.0.mamba.ssm.dt_bias" in result
        assert "model.layers.0.mamba.in_proj.weight" in result
        assert "model.layers.1.self_attn.q_proj.weight" in result


# ===========================================================================
# Hybrid Mamba2+Attention+MLP (NemotronH) model tests
# ===========================================================================


class TestBuildNemotronHGraph:
    """Verify NemotronH hybrid model weight renaming."""

    def _nemotron_h_config(self):
        from mobius._configs import NemotronHConfig

        # 4 layers: mamba2, mlp, full_attention, mlp
        return NemotronHConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_hidden_layers=4,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            rms_norm_eps=1e-5,
            layer_types=["mamba2", "mlp", "full_attention", "mlp"],
            mamba_n_heads=TINY_KV_HEADS,
            mamba_d_head=TINY_HEAD_DIM,
            mamba_d_state=16,
            mamba_n_groups=1,
            mamba_d_conv=4,
            mamba_expand=2,
            hidden_act="relu2",
            head_dim=TINY_HEAD_DIM,
        )

    def test_nemotron_h_preprocess_weights(self):
        """Verify preprocess_weights routes by layer type and nests SSM params."""
        import torch

        from mobius.models.nemotron_h import NemotronHCausalLMModel

        config = self._nemotron_h_config()
        module = NemotronHCausalLMModel(config)

        # Simulate HF NemotronH weight names (backbone.* prefix,
        # all layer mixers named "mixer.*" regardless of type)
        state_dict = {
            # Embeddings & final norm
            "backbone.embeddings.weight": torch.zeros(1),
            "backbone.norm_f.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(1),
            # Layer 0: mamba2 — SSM params + mixer params
            "backbone.layers.0.norm.weight": torch.zeros(1),
            "backbone.layers.0.mixer.A_log": torch.zeros(4),
            "backbone.layers.0.mixer.D": torch.zeros(4),
            "backbone.layers.0.mixer.dt_bias": torch.zeros(4),
            "backbone.layers.0.mixer.in_proj.weight": torch.zeros(1),
            "backbone.layers.0.mixer.conv1d.weight": torch.zeros(1),
            "backbone.layers.0.mixer.out_proj.weight": torch.zeros(1),
            "backbone.layers.0.mixer.norm.weight": torch.zeros(1),
            # Layer 1: mlp
            "backbone.layers.1.norm.weight": torch.zeros(1),
            "backbone.layers.1.mixer.up_proj.weight": torch.zeros(1),
            "backbone.layers.1.mixer.down_proj.weight": torch.zeros(1),
            # Layer 2: full_attention
            "backbone.layers.2.norm.weight": torch.zeros(1),
            "backbone.layers.2.mixer.q_proj.weight": torch.zeros(1),
            "backbone.layers.2.mixer.k_proj.weight": torch.zeros(1),
            "backbone.layers.2.mixer.v_proj.weight": torch.zeros(1),
            "backbone.layers.2.mixer.o_proj.weight": torch.zeros(1),
            # Layer 3: mlp
            "backbone.layers.3.norm.weight": torch.zeros(1),
            "backbone.layers.3.mixer.up_proj.weight": torch.zeros(1),
            "backbone.layers.3.mixer.down_proj.weight": torch.zeros(1),
        }
        result = module.preprocess_weights(state_dict)

        # Global renames: backbone.embeddings -> model.embed_tokens,
        # backbone.norm_f -> model.norm
        assert "model.embed_tokens.weight" in result
        assert "model.norm.weight" in result

        # Layer 0 (mamba2): SSM params nested under mamba.ssm
        assert "model.layers.0.mamba.ssm.A_log" in result
        assert "model.layers.0.mamba.ssm.D" in result
        assert "model.layers.0.mamba.ssm.dt_bias" in result
        # Non-SSM mamba params stay under mamba.*
        assert "model.layers.0.mamba.in_proj.weight" in result
        assert "model.layers.0.mamba.conv1d.weight" in result
        assert "model.layers.0.mamba.out_proj.weight" in result
        assert "model.layers.0.mamba.norm.weight" in result

        # Layer 1 (mlp): mixer -> mlp
        assert "model.layers.1.mlp.up_proj.weight" in result
        assert "model.layers.1.mlp.down_proj.weight" in result

        # Layer 2 (full_attention): mixer -> self_attn
        assert "model.layers.2.self_attn.q_proj.weight" in result
        assert "model.layers.2.self_attn.k_proj.weight" in result
        assert "model.layers.2.self_attn.v_proj.weight" in result
        assert "model.layers.2.self_attn.o_proj.weight" in result

        # Layer 3 (mlp): mixer -> mlp
        assert "model.layers.3.mlp.up_proj.weight" in result
        assert "model.layers.3.mlp.down_proj.weight" in result

        # Per-layer norms keep their names
        assert "model.layers.0.norm.weight" in result
        assert "model.layers.2.norm.weight" in result

        # No original backbone.* keys should remain
        for key in result:
            assert not key.startswith("backbone."), f"Unrenamed key: {key}"


# ===========================================================================
# Hybrid SSM+Attention (Jamba) model tests
# ===========================================================================


class TestBuildJambaGraph:
    """Verify Jamba hybrid SSM+Attention model builds correctly."""

    def _jamba_config(self):
        from mobius._configs import JambaConfig

        # 4 layers: attn_layer_period=2, attn_layer_offset=1
        # → layer 0: mamba, 1: attention, 2: mamba, 3: attention
        # expert_layer_period=2, expert_layer_offset=1
        # → layer 0: dense, 1: MoE, 2: dense, 3: MoE
        return JambaConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_hidden_layers=4,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            head_dim=TINY_HIDDEN // TINY_HEADS,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            layer_types=["mamba", "full_attention", "mamba", "full_attention"],
            mamba_d_state=8,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_dt_rank=4,
            attn_layer_period=2,
            attn_layer_offset=1,
            expert_layer_period=2,
            expert_layer_offset=1,
            num_local_experts=2,
            num_experts_per_tok=1,
        )

    def test_jamba_builds(self):
        """Build Jamba model and verify basic graph structure."""
        from mobius._builder import build_from_module
        from mobius.models.jamba import JambaCausalLMModel
        from mobius.tasks import HybridCausalLMTask

        config = self._jamba_config()
        module = JambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=HybridCausalLMTask())
        model = pkg["model"]

        assert model.graph is not None
        assert len(model.graph.inputs) > 0

    def test_jamba_has_logits_output(self):
        """Jamba model should produce logits output."""
        from mobius._builder import build_from_module
        from mobius.models.jamba import JambaCausalLMModel
        from mobius.tasks import HybridCausalLMTask

        config = self._jamba_config()
        module = JambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=HybridCausalLMTask())
        model = pkg["model"]

        output_names = [o.name for o in model.graph.outputs]
        assert "logits" in output_names

    def test_jamba_hybrid_cache_io(self):
        """Verify Jamba has mixed cache outputs (mamba + attention)."""
        from mobius._builder import build_from_module
        from mobius.models.jamba import JambaCausalLMModel
        from mobius.tasks import HybridCausalLMTask

        config = self._jamba_config()
        module = JambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=HybridCausalLMTask())
        model = pkg["model"]

        output_names = [o.name for o in model.graph.outputs]
        # Mamba layers (0, 2): conv_state + ssm_state
        assert "present.0.conv_state" in output_names
        assert "present.0.ssm_state" in output_names
        assert "present.2.conv_state" in output_names
        assert "present.2.ssm_state" in output_names
        # Attention layers (1, 3): key + value
        assert "present.1.key" in output_names
        assert "present.1.value" in output_names
        assert "present.3.key" in output_names
        assert "present.3.value" in output_names

    def test_jamba_registry_lookup(self):
        """Jamba should be in the registry as JambaCausalLMModel."""
        from mobius._registry import registry

        model_cls = registry.get("jamba")
        assert model_cls.__name__ == "JambaCausalLMModel"

    def test_jamba_preprocess_weights_moe_renames(self):
        """Verify MoE expert weight renames and SSM nesting."""
        import torch

        from mobius.models.jamba import JambaCausalLMModel

        config = self._jamba_config()
        module = JambaCausalLMModel(config)

        state_dict = {
            # SSM param: should nest under mamba.ssm
            "model.layers.0.mamba.A_log": torch.zeros(1),
            "model.layers.0.mamba.D": torch.zeros(1),
            "model.layers.0.mamba.dt_layernorm.weight": torch.zeros(1),
            # MoE router → gate
            "model.layers.1.feed_forward.router.weight": torch.zeros(1),
            # Non-mamba params pass through
            "model.layers.0.mamba.in_proj.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(1),
        }
        result = module.preprocess_weights(state_dict)

        # SSM params nested
        assert "model.layers.0.mamba.ssm.A_log" in result
        assert "model.layers.0.mamba.ssm.D" in result
        assert "model.layers.0.mamba.ssm.dt_layernorm.weight" in result
        # MoE gate renamed
        assert "model.layers.1.feed_forward.gate.weight" in result
        # Non-SSM stays flat
        assert "model.layers.0.mamba.in_proj.weight" in result


# ===========================================================================
# Registry completeness
# ===========================================================================

# Model types exercised by non-parametrized test classes above (VLM,
# whisper, audio, TTS, diffusion, etc.).  Keep sorted for readability.
_SPECIALIZED_TEST_MODEL_TYPES: set[str] = {
    # VLM alias tests (test_llava_aliases_build)
    "aya_vision",
    "chameleon",
    "cohere2_vision",
    "deepseek_vl",
    "deepseek_vl_hybrid",
    "florence2",
    "fuyu",
    "glm4v",
    "glm4v_moe",
    "got_ocr2",
    "idefics2",
    "idefics3",
    "instructblip",
    "instructblipvideo",
    "internvl",
    "internvl_chat",
    "internvl2",
    "janus",
    "llava_next",
    "llava_next_video",
    "llava_onevision",
    "molmo",
    "ovis2",
    "paligemma",
    "pixtral",
    "smolvlm",
    "video_llava",
    "vipllava",
    # VLM dedicated tests
    "blip-2",
    "deepseek_vl_v2",
    "gemma3_multimodal",
    "llava",
    "mllama",
    "phi4_multimodal",
    "phi4mm",
    "qwen2_5_vl",
    "qwen2_5_vl_text",
    "qwen2_vl",
    "qwen2_vl_text",
    "qwen3_5",
    "qwen3_5_vl",
    "qwen3_vl",
    "qwen3_vl_single",
    # Audio alias tests (test_audio_aliases_build)
    "data2vec-audio",
    "hubert",
    "mctct",
    "musicgen",
    "seamless_m4t",
    "seamless_m4t_v2",
    "sew",
    "sew-d",
    "speecht5",
    "unispeech",
    "unispeech-sat",
    "voxtral_encoder",
    "wav2vec2",
    "wav2vec2-bert",
    "wav2vec2-conformer",
    "wavlm",
    # Audio/TTS dedicated tests
    "qwen3_asr",
    "qwen3_forced_aligner",
    "qwen3_tts",
    "qwen3_tts_tokenizer_12hz",
    "whisper",
    # SSM dedicated tests
    "falcon_mamba",
    "mamba",
    "mamba2",
    # Hybrid SSM+Attention dedicated tests
    "bamba",
    "jamba",
}

# Registered model types that truly have no test coverage yet.
# This set should be empty or near-empty. If a NEW model is registered
# and is not in any tested set, the completeness test below will
# fail — forcing the developer to add a test or acknowledge the gap.
_KNOWN_UNTESTED_MODEL_TYPES: set[str] = set()


class TestRegistryCompleteness:
    """Ensure every registered model type has a test config entry."""

    def test_all_registered_models_have_test_coverage(self):
        """Every model_type in the registry must be accounted for.

        A model_type is *covered* if it appears in parametrized test
        configs, auto-generated configs, a specialized test class, or
        the known-untested allowlist.  New registrations that aren't
        covered anywhere will cause this test to fail.
        """
        all_covered = (
            {mt for mt, _, _ in ALL_CONFIGS}
            | {mt for mt, _, _ in AUTO_GENERATED_CONFIGS}
            | _SPECIALIZED_TEST_MODEL_TYPES
            | _KNOWN_UNTESTED_MODEL_TYPES
        )
        registered = set(registry.architectures())
        missing = registered - all_covered
        assert not missing, (
            f"Registered model types without test coverage: "
            f"{sorted(missing)}. Add a test config to "
            "tests/_test_configs.py, a specialized test class, or "
            "acknowledge in _KNOWN_UNTESTED_MODEL_TYPES."
        )

    def test_known_untested_is_minimal(self):
        """Entries in _KNOWN_UNTESTED should still be registered.

        If a model_type is removed from the registry, it should also
        be removed from _KNOWN_UNTESTED_MODEL_TYPES.  If a test is
        added for it, it should move to the appropriate config list or
        _SPECIALIZED_TEST_MODEL_TYPES.
        """
        registered = set(registry.architectures())
        stale = _KNOWN_UNTESTED_MODEL_TYPES - registered
        assert not stale, (
            f"Entries in _KNOWN_UNTESTED_MODEL_TYPES that are no longer "
            f"registered: {sorted(stale)}. Remove them."
        )


class TestBuildStaticCacheGraph:
    """Verify StaticCacheCausalLMTask builds a valid graph."""

    MAX_SEQ_LEN = 128

    def _build_static_cache_model(self, model_type: str = "qwen2", **config_overrides):
        """Build a model with StaticCacheCausalLMTask and return (model, config)."""
        from mobius.tasks import StaticCacheCausalLMTask

        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = StaticCacheCausalLMTask(max_seq_len=self.MAX_SEQ_LEN)
        pkg = task.build(module, config)
        return pkg["model"], config

    def test_static_cache_graph_builds(self):
        """Build a Qwen2 model with StaticCacheCausalLMTask."""
        model, _ = self._build_static_cache_model()

        assert model.graph is not None
        assert len(model.graph.inputs) > 0
        assert len(model.graph.outputs) > 0

    def test_static_cache_graph_inputs(self):
        """Verify expected inputs: standard + per-layer caches + shared."""
        model, config = self._build_static_cache_model()
        input_names = {inp.name for inp in model.graph.inputs}
        num_layers = config.num_hidden_layers

        # Standard inputs
        assert "input_ids" in input_names
        assert "position_ids" in input_names

        # No attention_mask in static cache mode — causal masking is
        # handled by is_causal=1 on the Attention op.
        assert "attention_mask" not in input_names

        # Per-layer static cache inputs
        for i in range(num_layers):
            assert f"key_cache.{i}" in input_names, f"Missing key_cache.{i}"
            assert f"value_cache.{i}" in input_names, f"Missing value_cache.{i}"

        # Shared cache management inputs
        assert "write_indices" in input_names
        assert "nonpad_kv_seqlen" in input_names

        # Exact count: 2 standard + 2*num_layers caches + 2 shared
        expected_count = 2 + 2 * num_layers + 2
        assert len(model.graph.inputs) == expected_count, (
            f"Expected {expected_count} inputs, got {len(model.graph.inputs)}"
        )

    def test_static_cache_graph_outputs(self):
        """Verify outputs: logits + updated caches per layer."""
        model, config = self._build_static_cache_model()
        output_names = {out.name for out in model.graph.outputs}
        num_layers = config.num_hidden_layers

        assert "logits" in output_names

        # Updated caches per layer (not present.{i}.key/value)
        for i in range(num_layers):
            assert f"updated_key_cache.{i}" in output_names, f"Missing updated_key_cache.{i}"
            assert f"updated_value_cache.{i}" in output_names, (
                f"Missing updated_value_cache.{i}"
            )

        # Should NOT have dynamic cache outputs
        assert not any(n.startswith("present.") for n in output_names), (
            "Static cache graph should not have present.* outputs"
        )

        # Exact count: 1 logits + 2*num_layers updated caches
        expected_count = 1 + 2 * num_layers
        assert len(model.graph.outputs) == expected_count, (
            f"Expected {expected_count} outputs, got {len(model.graph.outputs)}"
        )

    def test_static_cache_has_tensorscatter_and_attention(self):
        """Verify graph contains TensorScatter and Attention ops."""
        model, _ = self._build_static_cache_model()

        op_types = {n.op_type for n in model.graph}
        assert "TensorScatter" in op_types, "Static cache graph should use TensorScatter"
        assert "Attention" in op_types, "Static cache graph should use Attention"

    def test_static_cache_has_initializers(self):
        """Verify the graph has model parameters."""
        model, _ = self._build_static_cache_model()

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        assert any("embed_tokens" in n for n in init_names)
        assert any("self_attn" in n for n in init_names)
        assert any("mlp" in n for n in init_names)

    def test_static_cache_graph_validates(self):
        """Verify the graph survives a serialization round-trip."""
        model, _config = self._build_static_cache_model()
        proto = ir.serde.serialize_model(model)
        assert len(proto.SerializeToString()) > 0

    def test_static_cache_attention_is_causal(self):
        """Verify Attention ops use is_causal=1 in static cache mode."""
        model, config = self._build_static_cache_model()

        attention_nodes = [n for n in model.graph if n.op_type == "Attention"]
        assert len(attention_nodes) == config.num_hidden_layers

        for node in attention_nodes:
            is_causal = node.attributes.get("is_causal")
            assert is_causal is not None, (
                f"Attention node {node.name} missing is_causal attribute"
            )
            assert is_causal.as_int() == 1, (
                f"Attention node {node.name} should have is_causal=1"
            )

    def test_static_cache_attention_no_attn_mask_input(self):
        """Verify Attention ops do NOT receive attn_mask in static cache mode."""
        model, config = self._build_static_cache_model()

        attention_nodes = [n for n in model.graph if n.op_type == "Attention"]
        assert len(attention_nodes) == config.num_hidden_layers

        for node in attention_nodes:
            # Input 3 (0-indexed) is attn_mask — should be empty/None
            attn_mask_input = node.inputs[3]
            assert attn_mask_input is None or attn_mask_input.name == "", (
                f"Attention node {node.name} should not have attn_mask "
                f"connected, but got input: {attn_mask_input}"
            )

    def test_static_cache_moe_graph_builds(self):
        """Build a MoE model (qwen2_moe) with StaticCacheCausalLMTask."""
        model, _config = self._build_static_cache_model(
            model_type="qwen2_moe",
            num_local_experts=4,
            num_experts_per_tok=2,
            attn_qkv_bias=True,
        )

        assert model.graph is not None
        assert len(model.graph.inputs) > 0
        assert len(model.graph.outputs) > 0

        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "position_ids" in input_names
        assert "attention_mask" not in input_names

        # Verify TensorScatter and Attention ops are present
        op_types = {n.op_type for n in model.graph}
        assert "TensorScatter" in op_types
        assert "Attention" in op_types
