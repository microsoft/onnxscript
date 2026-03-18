# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for task classes.

Covers: CausalLM, VisionLanguage, Seq2Seq, Denoising, VAE,
FeatureExtraction, ImageClassification, SSM, SSM2, AudioFeatureExtraction,
ObjectDetection, SpeechToText.
"""

from __future__ import annotations

import onnx_ir as ir
import pytest

from mobius._configs import VisionConfig
from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius.models.base import CausalLMModel
from mobius.models.gemma3 import Gemma3MultiModalModel
from mobius.tasks import (
    TASK_REGISTRY,
    AudioFeatureExtractionTask,
    CausalLMTask,
    DenoisingTask,
    FeatureExtractionTask,
    ImageClassificationTask,
    ModelTask,
    ObjectDetectionTask,
    Seq2SeqTask,
    SpeechToTextTask,
    SSM2CausalLMTask,
    SSMCausalLMTask,
    VAETask,
    VisionLanguageTask,
    get_task,
)


class TestGetTask:
    def test_get_task_by_name(self):
        task = get_task("text-generation")
        assert isinstance(task, CausalLMTask)

    def test_get_task_by_instance(self):
        instance = CausalLMTask()
        assert get_task(instance) is instance

    def test_get_task_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nonexistent-task")

    def test_task_registry_has_text_generation(self):
        assert "text-generation" in TASK_REGISTRY


class TestCausalLMTask:
    def test_build_returns_package(self):
        config = make_config()
        module = CausalLMModel(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg

    def test_build_inputs(self):
        config = make_config()
        module = CausalLMModel(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = [v.name for v in model.graph.inputs]
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names
        # 2 layers x 2 (key, value)
        assert "past_key_values.0.key" in input_names
        assert "past_key_values.1.value" in input_names

    def test_build_outputs(self):
        config = make_config()
        module = CausalLMModel(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = [v.name for v in model.graph.outputs]
        assert "logits" in output_names
        assert "present.0.key" in output_names
        assert "present.0.value" in output_names

    def test_build_producer_info(self):
        config = make_config()
        module = CausalLMModel(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        assert model.producer_name == "mobius"


class TestCustomTask:
    """Test that users can create custom tasks."""

    def test_subclass_model_task(self):
        class MyTask(ModelTask):
            def build(self, module, config):
                # Minimal: just create an empty model
                graph = ir.Graph([], [], nodes=[], name="custom")
                model = ir.Model(graph, ir_version=10)
                return ModelPackage({"model": model})

        task = MyTask()
        config = make_config()
        module = CausalLMModel(config)
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert pkg["model"].graph.name == "custom"


class TestCustomModuleWithTask:
    """Test the user story: custom module + standard task."""

    def test_custom_module_with_causal_lm_task(self):
        """A user-defined module should work with CausalLMTask.

        It follows the expected forward signature.
        """
        config = make_config()

        # Re-use CausalLMModel as a "custom" module — it has the right signature
        custom_module = CausalLMModel(config)
        task = CausalLMTask()
        pkg = task.build(custom_module, config)

        assert isinstance(pkg, ModelPackage)
        model = pkg["model"]
        output_names = [v.name for v in model.graph.outputs]
        assert "logits" in output_names


def _make_multimodal_config():
    return make_config(
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


class TestVisionLanguageTask:
    def test_task_registered(self):
        assert "vision-language" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("vision-language")
        assert isinstance(task, VisionLanguageTask)

    def test_build_returns_3_model_package(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert set(pkg.keys()) == {"decoder", "vision", "embedding"}

    def test_decoder_has_inputs_embeds(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        input_names = [v.name for v in decoder.graph.inputs]
        assert "inputs_embeds" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names

    def test_decoder_has_kv_cache(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        input_names = [v.name for v in decoder.graph.inputs]
        assert "past_key_values.0.key" in input_names
        assert "past_key_values.0.value" in input_names

    def test_decoder_outputs_logits(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        output_names = [v.name for v in decoder.graph.outputs]
        assert "logits" in output_names
        assert "present.0.key" in output_names

    def test_vision_has_pixel_values(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        vision = pkg["vision"]
        input_names = [v.name for v in vision.graph.inputs]
        assert "pixel_values" in input_names
        output_names = [v.name for v in vision.graph.outputs]
        assert "image_features" in output_names

    def test_embedding_fuses_features(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        embed = pkg["embedding"]
        input_names = [v.name for v in embed.graph.inputs]
        assert "input_ids" in input_names
        assert "image_features" in input_names
        output_names = [v.name for v in embed.graph.outputs]
        assert "inputs_embeds" in output_names

    def test_build_producer_info(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        for model in pkg.values():
            assert model.producer_name == "mobius"


# ── Seq2SeqTask ──────────────────────────────────────────────────────────


class TestSeq2SeqTask:
    def _make_seq2seq(self):
        from mobius.models.bart import BartForConditionalGeneration

        config = make_config(
            hidden_act="gelu",
            num_decoder_layers=2,
            max_position_embeddings=64,
        )
        module = BartForConditionalGeneration(config)
        return config, module

    def test_task_registered(self):
        assert "seq2seq" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("seq2seq")
        assert isinstance(task, Seq2SeqTask)

    def test_build_returns_encoder_and_decoder(self):
        config, module = self._make_seq2seq()
        task = Seq2SeqTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "encoder" in pkg
        assert "decoder" in pkg

    def test_encoder_inputs(self):
        config, module = self._make_seq2seq()
        task = Seq2SeqTask()
        pkg = task.build(module, config)
        encoder = pkg["encoder"]
        input_names = {v.name for v in encoder.graph.inputs}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names

    def test_encoder_outputs(self):
        config, module = self._make_seq2seq()
        task = Seq2SeqTask()
        pkg = task.build(module, config)
        encoder = pkg["encoder"]
        output_names = {v.name for v in encoder.graph.outputs}
        assert "last_hidden_state" in output_names

    def test_decoder_inputs(self):
        config, module = self._make_seq2seq()
        task = Seq2SeqTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        input_names = {v.name for v in decoder.graph.inputs}
        assert "input_ids" in input_names
        assert "encoder_hidden_states" in input_names
        assert "attention_mask" in input_names
        # Self-attention KV cache
        assert "past_key_values.0.self.key" in input_names
        assert "past_key_values.0.self.value" in input_names
        # Cross-attention KV cache
        assert "past_key_values.0.cross.key" in input_names
        assert "past_key_values.0.cross.value" in input_names

    def test_decoder_outputs(self):
        config, module = self._make_seq2seq()
        task = Seq2SeqTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        output_names = {v.name for v in decoder.graph.outputs}
        assert "logits" in output_names
        assert "present.0.self.key" in output_names
        assert "present.0.self.value" in output_names
        assert "present.0.cross.key" in output_names
        assert "present.0.cross.value" in output_names

    def test_build_producer_info(self):
        config, module = self._make_seq2seq()
        task = Seq2SeqTask()
        pkg = task.build(module, config)
        for model in pkg.values():
            assert model.producer_name == "mobius"


# ── DenoisingTask ────────────────────────────────────────────────────────


class TestDenoisingTask:
    def _make_denoiser(self):
        from mobius.models.dit import (
            DiTConfig,
            DiTTransformer2DModel,
        )

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
        return config, module

    def test_task_registered(self):
        assert "denoising" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("denoising")
        assert isinstance(task, DenoisingTask)

    def test_build_returns_single_model(self):
        config, module = self._make_denoiser()
        task = DenoisingTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg
        assert len(pkg) == 1

    def test_inputs(self):
        config, module = self._make_denoiser()
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names

    def test_input_types(self):
        config, module = self._make_denoiser()
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        inputs_by_name = {v.name: v for v in model.graph.inputs}
        assert inputs_by_name["sample"].dtype == ir.DataType.FLOAT
        assert inputs_by_name["timestep"].dtype == ir.DataType.INT64
        assert inputs_by_name["encoder_hidden_states"].dtype == ir.DataType.FLOAT

    def test_outputs(self):
        config, module = self._make_denoiser()
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "noise_pred" in output_names
        assert len(model.graph.outputs) == 1

    def test_build_producer_info(self):
        config, module = self._make_denoiser()
        task = DenoisingTask()
        pkg = task.build(module, config)
        assert pkg["model"].producer_name == "mobius"


# ── VAETask ──────────────────────────────────────────────────────────────


class TestVAETask:
    def _make_vae(self):
        from mobius._diffusers_configs import VAEConfig
        from mobius.models.vae import AutoencoderKLModel

        config = VAEConfig(
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
        module = AutoencoderKLModel(config)
        return config, module

    def test_task_registered(self):
        assert "vae" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("vae")
        assert isinstance(task, VAETask)

    def test_build_returns_encoder_and_decoder(self):
        config, module = self._make_vae()
        task = VAETask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "encoder" in pkg
        assert "decoder" in pkg
        assert len(pkg) == 2

    def test_encoder_inputs(self):
        config, module = self._make_vae()
        task = VAETask()
        pkg = task.build(module, config)
        encoder = pkg["encoder"]
        input_names = {v.name for v in encoder.graph.inputs}
        assert "sample" in input_names

    def test_encoder_outputs(self):
        config, module = self._make_vae()
        task = VAETask()
        pkg = task.build(module, config)
        encoder = pkg["encoder"]
        output_names = {v.name for v in encoder.graph.outputs}
        assert "latent_dist" in output_names

    def test_decoder_inputs(self):
        config, module = self._make_vae()
        task = VAETask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        input_names = {v.name for v in decoder.graph.inputs}
        assert "latent_sample" in input_names

    def test_decoder_outputs(self):
        config, module = self._make_vae()
        task = VAETask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        output_names = {v.name for v in decoder.graph.outputs}
        assert "sample" in output_names

    def test_build_producer_info(self):
        config, module = self._make_vae()
        task = VAETask()
        pkg = task.build(module, config)
        for model in pkg.values():
            assert model.producer_name == "mobius"


# ── FeatureExtractionTask ────────────────────────────────────────────────


class TestFeatureExtractionTask:
    def _make_encoder(self):
        from mobius._configs import EncoderConfig
        from mobius.models.bert import BertModel

        config = EncoderConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            hidden_act="gelu",
            pad_token_id=0,
            max_position_embeddings=32,
            type_vocab_size=2,
            attn_qkv_bias=True,
            attn_o_bias=True,
        )
        module = BertModel(config)
        return config, module

    def test_task_registered(self):
        assert "feature-extraction" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("feature-extraction")
        assert isinstance(task, FeatureExtractionTask)

    def test_build_returns_single_model(self):
        config, module = self._make_encoder()
        task = FeatureExtractionTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg
        assert len(pkg) == 1

    def test_inputs(self):
        config, module = self._make_encoder()
        task = FeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        assert "token_type_ids" in input_names

    def test_no_kv_cache(self):
        config, module = self._make_encoder()
        task = FeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert not any("past_key_values" in n for n in input_names)

    def test_outputs(self):
        config, module = self._make_encoder()
        task = FeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "last_hidden_state" in output_names
        assert len(model.graph.outputs) == 1


# ── ImageClassificationTask ─────────────────────────────────────────────


class TestImageClassificationTask:
    def _make_vision(self):
        from mobius._configs import EncoderConfig
        from mobius.models.vit import ViTModel

        config = EncoderConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            hidden_act="gelu",
            pad_token_id=0,
            max_position_embeddings=32,
            image_size=32,
            patch_size=8,
            num_channels=3,
            attn_qkv_bias=True,
            attn_o_bias=True,
        )
        module = ViTModel(config)
        return config, module

    def test_task_registered(self):
        assert "image-classification" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("image-classification")
        assert isinstance(task, ImageClassificationTask)

    def test_build_returns_single_model(self):
        config, module = self._make_vision()
        task = ImageClassificationTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg
        assert len(pkg) == 1

    def test_inputs(self):
        config, module = self._make_vision()
        task = ImageClassificationTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "pixel_values" in input_names

    def test_outputs(self):
        config, module = self._make_vision()
        task = ImageClassificationTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "last_hidden_state" in output_names


# ── ObjectDetectionTask ──────────────────────────────────────────────────


class TestObjectDetectionTask:
    def _make_detector(self):
        from mobius._configs import YolosConfig
        from mobius.models.yolos import YolosForObjectDetection

        config = YolosConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            hidden_act="gelu",
            pad_token_id=0,
            max_position_embeddings=32,
            image_size=32,
            patch_size=8,
            num_channels=3,
            num_labels=10,
            num_detection_tokens=5,
            attn_qkv_bias=True,
            attn_o_bias=True,
        )
        module = YolosForObjectDetection(config)
        return config, module

    def test_task_registered(self):
        assert "object-detection" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("object-detection")
        assert isinstance(task, ObjectDetectionTask)

    def test_build_returns_single_model(self):
        config, module = self._make_detector()
        task = ObjectDetectionTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg

    def test_inputs(self):
        config, module = self._make_detector()
        task = ObjectDetectionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "pixel_values" in input_names

    def test_outputs(self):
        config, module = self._make_detector()
        task = ObjectDetectionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "logits" in output_names
        assert "pred_boxes" in output_names
        assert len(model.graph.outputs) == 2


# ── SSMCausalLMTask ──────────────────────────────────────────────────────


class TestSSMCausalLMTask:
    def _make_mamba(self):
        from mobius._configs import MambaConfig
        from mobius.models.mamba import MambaCausalLMModel

        config = MambaConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            pad_token_id=0,
            state_size=16,
            conv_kernel=4,
            expand=2,
            time_step_rank=4,
        )
        module = MambaCausalLMModel(config)
        return config, module

    def test_task_registered(self):
        assert "ssm-text-generation" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("ssm-text-generation")
        assert isinstance(task, SSMCausalLMTask)

    def test_build_returns_single_model(self):
        config, module = self._make_mamba()
        task = SSMCausalLMTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg

    def test_inputs_have_ssm_states(self):
        config, module = self._make_mamba()
        task = SSMCausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "input_ids" in input_names
        assert "past_states.0.conv_state" in input_names
        assert "past_states.0.ssm_state" in input_names
        assert "past_states.1.conv_state" in input_names

    def test_no_kv_cache(self):
        config, module = self._make_mamba()
        task = SSMCausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert not any("past_key_values" in n for n in input_names)
        assert "attention_mask" not in input_names

    def test_outputs(self):
        config, module = self._make_mamba()
        task = SSMCausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "logits" in output_names
        assert "present.0.conv_state" in output_names
        assert "present.0.ssm_state" in output_names


# ── SSM2CausalLMTask ─────────────────────────────────────────────────────


class TestSSM2CausalLMTask:
    def _make_mamba2(self):
        from mobius._configs import Mamba2Config
        from mobius.models.mamba import Mamba2CausalLMModel

        config = Mamba2Config(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=16,
            pad_token_id=0,
            num_heads=8,
            state_size=16,
            n_groups=2,
            conv_kernel=4,
            expand=2,
        )
        module = Mamba2CausalLMModel(config)
        return config, module

    def test_task_registered(self):
        assert "ssm2-text-generation" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("ssm2-text-generation")
        assert isinstance(task, SSM2CausalLMTask)

    def test_build_returns_single_model(self):
        config, module = self._make_mamba2()
        task = SSM2CausalLMTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg

    def test_inputs_have_ssm_states(self):
        config, module = self._make_mamba2()
        task = SSM2CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "input_ids" in input_names
        assert "past_states.0.conv_state" in input_names
        assert "past_states.0.ssm_state" in input_names

    def test_outputs(self):
        config, module = self._make_mamba2()
        task = SSM2CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "logits" in output_names
        assert "present.0.conv_state" in output_names
        assert "present.0.ssm_state" in output_names


# ── AudioFeatureExtractionTask ───────────────────────────────────────────


class TestAudioFeatureExtractionTask:
    def _make_audio(self):
        from mobius.models.wav2vec2 import Wav2Vec2Model

        config = make_config(
            hidden_act="gelu",
            attn_qkv_bias=True,
            attn_o_bias=True,
        )
        # Wav2Vec2Model uses getattr for conv_channels/conv_kernel_sizes
        # with defaults. We override with tiny values to keep graphs small.
        config.conv_channels = [1, 32, 32]
        config.conv_kernel_sizes = [5, 3]
        module = Wav2Vec2Model(config)
        return config, module

    def test_task_registered(self):
        assert "audio-feature-extraction" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("audio-feature-extraction")
        assert isinstance(task, AudioFeatureExtractionTask)

    def test_build_returns_single_model(self):
        config, module = self._make_audio()
        task = AudioFeatureExtractionTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg
        assert len(pkg) == 1

    def test_inputs(self):
        config, module = self._make_audio()
        task = AudioFeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "input_values" in input_names

    def test_outputs(self):
        config, module = self._make_audio()
        task = AudioFeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "last_hidden_state" in output_names


# ── SpeechToTextTask ─────────────────────────────────────────────────────


class TestSpeechToTextTask:
    def _make_whisper(self):
        from mobius._configs import WhisperConfig
        from mobius.models.whisper import (
            WhisperForConditionalGeneration,
        )

        config = WhisperConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            hidden_act="gelu",
            pad_token_id=0,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=128,
            num_mel_bins=40,
            max_source_positions=64,
            max_target_positions=32,
            attn_qkv_bias=True,
            attn_o_bias=True,
        )
        module = WhisperForConditionalGeneration(config)
        return config, module

    def test_task_registered(self):
        assert "speech-to-text" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("speech-to-text")
        assert isinstance(task, SpeechToTextTask)

    def test_build_returns_encoder_and_decoder(self):
        config, module = self._make_whisper()
        task = SpeechToTextTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "encoder" in pkg
        assert "decoder" in pkg
        assert len(pkg) == 2

    def test_encoder_inputs(self):
        config, module = self._make_whisper()
        task = SpeechToTextTask()
        pkg = task.build(module, config)
        encoder = pkg["encoder"]
        input_names = {v.name for v in encoder.graph.inputs}
        assert "input_features" in input_names

    def test_encoder_outputs(self):
        config, module = self._make_whisper()
        task = SpeechToTextTask()
        pkg = task.build(module, config)
        encoder = pkg["encoder"]
        output_names = {v.name for v in encoder.graph.outputs}
        assert "encoder_hidden_states" in output_names

    def test_decoder_inputs(self):
        config, module = self._make_whisper()
        task = SpeechToTextTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        input_names = {v.name for v in decoder.graph.inputs}
        assert "decoder_input_ids" in input_names
        assert "encoder_hidden_states" in input_names
        assert "position_ids" in input_names
        assert "past_key_values.0.key" in input_names
        assert "past_key_values.0.value" in input_names

    def test_decoder_outputs(self):
        config, module = self._make_whisper()
        task = SpeechToTextTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        output_names = {v.name for v in decoder.graph.outputs}
        assert "logits" in output_names
        assert "present.0.key" in output_names
        assert "present.0.value" in output_names
