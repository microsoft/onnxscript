# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for model building — end-to-end graph construction for all architectures."""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._config_resolver import _default_task_for_model
from mobius._configs import VisionConfig
from mobius._registry import (
    MODEL_MAP,
    ModelRegistry,
    registry,
)
from mobius._testing import make_config
from mobius.models.base import CausalLMModel, TextModel
from mobius.tasks import CausalLMTask, get_task


class TestTextModel:
    def test_text_model_params(self):
        config = make_config()
        model = TextModel(config)
        param_names = [n for n, _ in model.named_parameters()]
        assert any("embed_tokens" in n for n in param_names)
        assert any("norm" in n for n in param_names)
        assert any("layers" in n for n in param_names)

    def test_text_model_num_layers(self):
        config = make_config(num_hidden_layers=4)
        model = TextModel(config)
        assert len(model.layers) == 4


class TestCausalLMModel:
    def test_causal_lm_model_has_lm_head(self):
        config = make_config()
        model = CausalLMModel(config)
        param_names = [n for n, _ in model.named_parameters()]
        assert any("lm_head" in n for n in param_names)

    def test_preprocess_weights_tied_embeddings(self):
        config = make_config(tie_word_embeddings=True)
        model = CausalLMModel(config)

        weight = torch.zeros(100, 64)
        sd = {"lm_head.weight": weight}
        sd = model.preprocess_weights(sd)
        assert "model.embed_tokens.weight" in sd
        assert sd["model.embed_tokens.weight"] is weight

    def test_preprocess_weights_no_tied(self):
        config = make_config(tie_word_embeddings=False)
        model = CausalLMModel(config)
        sd = {"lm_head.weight": torch.zeros(100, 64)}
        sd = model.preprocess_weights(sd)
        assert "model.embed_tokens.weight" not in sd


class TestBuildFromModule:
    def test_build_base_model(self):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        assert isinstance(model, ir.Model)
        assert model.graph.num_nodes() > 0
        assert len(model.graph.inputs) > 0
        assert len(model.graph.outputs) > 0

    def test_build_model_inputs(self):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        input_names = [v.name for v in model.graph.inputs]
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names
        assert "past_key_values.0.key" in input_names
        assert "past_key_values.0.value" in input_names

    def test_build_model_outputs(self):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        output_names = [v.name for v in model.graph.outputs]
        assert "logits" in output_names
        assert "present.0.key" in output_names
        assert "present.0.value" in output_names

    def test_build_model_num_kv_caches(self):
        config = make_config(num_hidden_layers=3)
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        output_names = [v.name for v in model.graph.outputs]
        for i in range(3):
            assert f"present.{i}.key" in output_names
            assert f"present.{i}.value" in output_names

    def test_build_model_has_initializers(self):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        init_names = list(model.graph.initializers.keys())
        assert len(init_names) > 0
        assert any("embed_tokens" in n for n in init_names)
        assert any("lm_head" in n for n in init_names)

    def test_build_model_save_load_roundtrip(self, tmp_path):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]
        path = str(tmp_path / "test_roundtrip.onnx")
        ir.save(model, path)
        loaded = ir.load(path)
        assert loaded.graph.num_nodes() == model.graph.num_nodes()

    def test_build_with_task_instance(self):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config, task=CausalLMTask())["model"]
        assert isinstance(model, ir.Model)
        assert model.graph.num_nodes() > 0

    def test_build_with_task_string(self):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config, task="text-generation")["model"]
        assert isinstance(model, ir.Model)
        assert model.graph.num_nodes() > 0


# Parametrized tests for all model classes that use the base config
_BASE_CONFIG_MODELS = [
    "llama",
    "mistral",
    "qwen2",
    "chatglm",
    "ernie4_5",
    "granite",
    "internlm2",
    "olmo",
    "phi",
    "phi3",
    "phi3small",
    "qwen",
    "qwen3",
    "qwen2_5_vl_text",
    "qwen3_vl_text",
    "nemotron",
    # Newly registered Llama-compatible models
    "cohere",
    "cohere2",
    "diffllama",
    "doge",
    "dots1",
    "exaone4",
    "glm",
    "glm4",
    "helium",
    "hunyuan_v1_dense",
    "llama4_text",
    "ministral",
    "ministral3",
    "nanochat",
    "olmo2",
    "olmo3",
    "qwen3_5_text",
    "solar_open",
    "stablelm",
    "starcoder2",
    "youtu",
]


@pytest.mark.parametrize("arch", _BASE_CONFIG_MODELS)
def test_build_model_architecture(arch):
    """Test that each architecture builds a valid ONNX model."""
    config = make_config()
    model_class = registry.get(arch)
    module = model_class(config)
    model = build_from_module(module, config)["model"]
    assert isinstance(model, ir.Model)
    assert model.graph.num_nodes() > 0
    assert len(model.graph.outputs) >= 1
    output_names = [v.name for v in model.graph.outputs]
    assert "logits" in output_names


def test_build_gemma_model():
    config = make_config()
    module = registry.get("gemma")(config)
    model = build_from_module(module, config)["model"]
    assert model.graph.num_nodes() > 0


def test_build_gemma2_model():
    from mobius._configs import Gemma2Config

    base_config = make_config(sliding_window=8)
    # Convert to Gemma2Config with Gemma2-specific fields
    config = Gemma2Config(
        **{f.name: getattr(base_config, f.name) for f in dataclasses.fields(base_config)},
        attn_logit_softcapping=0.0,
        final_logit_softcapping=0.0,
        query_pre_attn_scalar=None,
    )
    module = registry.get("gemma2")(config)
    model = build_from_module(module, config)["model"]
    assert model.graph.num_nodes() > 0


def test_build_gemma3_text_model():
    config = make_config(
        sliding_window=8,
        layer_types=["full_attention", "sliding_attention"],
        attn_qk_norm=True,
        rope_local_base_freq=10_000.0,
    )
    module = registry.get("gemma3_text")(config)
    model = build_from_module(module, config)["model"]
    assert model.graph.num_nodes() > 0


def test_build_gemma3_model():
    config = make_config(
        sliding_window=8,
        layer_types=["full_attention", "sliding_attention"],
        attn_qk_norm=True,
        rope_local_base_freq=10_000.0,
    )
    module = registry.get("gemma3")(config)
    model = build_from_module(module, config)["model"]
    assert model.graph.num_nodes() > 0


def test_build_smollm3_model():
    config = make_config(
        sliding_window=8,
        layer_types=["full_attention", "sliding_attention"],
    )
    module = registry.get("smollm3")(config)
    model = build_from_module(module, config)["model"]
    assert model.graph.num_nodes() > 0


_MOE_MODELS = ["phimoe", "gptoss", "granitemoe", "mixtral", "olmoe", "qwen2_moe", "qwen3_moe"]


@pytest.mark.parametrize("arch", _MOE_MODELS)
def test_build_moe_model_architecture(arch):
    """Test that MoE architectures build a valid ONNX model."""
    config = make_config(num_local_experts=4, num_experts_per_tok=2)
    model_class = registry.get(arch)
    module = model_class(config)
    model = build_from_module(module, config)["model"]
    assert isinstance(model, ir.Model)
    assert model.graph.num_nodes() > 0
    output_names = [v.name for v in model.graph.outputs]
    assert "logits" in output_names
    # MoE models should have expert parameters
    init_names = list(model.graph.initializers.keys())
    assert any("block_sparse_moe" in n for n in init_names)
    assert any("experts" in n for n in init_names)


def test_build_qwen3_next_model():
    """Test Qwen3-Next hybrid DeltaNet + MoE architecture."""
    config = make_config(
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        partial_rotary_factor=0.25,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        norm_topk_prob=True,
        attn_qk_norm=True,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )
    model_class = registry.get("qwen3_next")
    module = model_class(config)
    model = build_from_module(module, config)["model"]
    assert isinstance(model, ir.Model)
    assert model.graph.num_nodes() > 0
    output_names = [v.name for v in model.graph.outputs]
    assert "logits" in output_names


def test_build_gemma3_multimodal_model():
    """Test Gemma3 multimodal model with 3-model split."""
    from mobius.tasks import VisionLanguageTask

    config = make_config(
        sliding_window=8,
        layer_types=["full_attention", "sliding_attention"],
        attn_qk_norm=True,
        rope_local_base_freq=10_000.0,
        vision=VisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=32,
            patch_size=8,
            norm_eps=1e-6,
            image_token_id=999,
        ),
        image_token_id=999,
    )
    module = registry.get("gemma3_multimodal")(config)
    task = VisionLanguageTask()
    pkg = task.build(module, config)
    assert set(pkg.keys()) == {"decoder", "vision", "embedding"}
    assert "logits" in {v.name for v in pkg["decoder"].graph.outputs}
    assert "pixel_values" in {v.name for v in pkg["vision"].graph.inputs}
    assert "inputs_embeds" in {v.name for v in pkg["embedding"].graph.outputs}


def test_build_qwen3_vl_model():
    """Test Qwen3-VL full VL model with 3-model split task."""
    config = make_config(
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
    module = registry.get("qwen3_vl")(config)
    task_name = _default_task_for_model("qwen3_vl")
    task = get_task(task_name)
    pkg = task.build(module, config)

    # 3-model split: decoder, vision, embedding
    assert "decoder" in pkg
    assert "vision" in pkg
    assert "embedding" in pkg

    decoder = pkg["decoder"]
    assert isinstance(decoder, ir.Model)
    assert decoder.graph.num_nodes() > 0
    assert "logits" in [v.name for v in decoder.graph.outputs]
    assert "inputs_embeds" in [v.name for v in decoder.graph.inputs]


def test_build_qwen3_vl_single_model():
    """Test Qwen3-VL single-model with DeepStack vision encoder."""
    from mobius.tasks import Qwen3VLVisionLanguageTask

    config = make_config(
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
    module = registry.get("qwen3_vl_single")(config)
    task = Qwen3VLVisionLanguageTask()
    pkg = task.build(module, config)
    model = pkg["model"]
    assert isinstance(model, ir.Model)
    assert model.graph.num_nodes() > 0
    output_names = [v.name for v in model.graph.outputs]
    assert "logits" in output_names
    input_names = [v.name for v in model.graph.inputs]
    assert "pixel_values" in input_names
    assert "grid_thw" in input_names
    assert "position_ids" in input_names


class TestModelRegistry:
    def test_registry_not_empty(self):
        assert len(registry) > 0

    def test_registry_has_architectures(self):
        assert len(registry) >= 50

    def test_all_registry_values_are_callable(self):
        for name in registry.architectures():
            cls = registry.get(name)
            assert callable(cls), f"registry['{name}'] is not callable"

    def test_register_custom_architecture(self):
        reg = ModelRegistry()
        reg.register("test_arch", CausalLMModel)
        assert "test_arch" in reg
        assert reg.get("test_arch") is CausalLMModel

    def test_get_unknown_raises(self):
        reg = ModelRegistry()
        with pytest.raises(KeyError, match="Unknown model_type"):
            reg.get("nonexistent")

    def test_model_map_backward_compat(self):
        """MODEL_MAP dict still works for backward compatibility."""
        assert len(MODEL_MAP) >= 50
        assert "llama" in MODEL_MAP
