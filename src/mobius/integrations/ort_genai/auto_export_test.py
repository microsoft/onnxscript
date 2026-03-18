# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ORT-GenAI auto-export pipeline."""

from __future__ import annotations

import json
import os
from unittest import mock

import numpy as np
import pytest

from mobius.integrations.ort_genai.auto_export import (
    _copy_tokenizer_files,
    _resolve_ort_genai_model_type,
    _write_processor_config,
)


class TestResolveOrtGenaiModelType:
    def test_known_model_type(self):
        assert _resolve_ort_genai_model_type("qwen3") == "qwen2"
        assert _resolve_ort_genai_model_type("gemma2") == "gemma"
        assert _resolve_ort_genai_model_type("llama") == "llama"

    def test_unknown_model_type_passthrough(self):
        assert _resolve_ort_genai_model_type("my_custom") == "my_custom"

    def test_phi4mm_model_types(self):
        assert _resolve_ort_genai_model_type("phi4mm") == "phi4mm"
        assert _resolve_ort_genai_model_type("phi4_multimodal") == "phi4mm"
        assert _resolve_ort_genai_model_type("phi") == "phi"


class TestWriteProcessorConfig:
    def test_no_vision_returns_none(self, tmp_path):
        config = mock.MagicMock(spec=[])
        del config.vision  # ensure no vision attribute
        assert _write_processor_config(config, str(tmp_path)) is None

    def test_writes_vision_config(self, tmp_path):
        vision = mock.MagicMock()
        vision.image_size = 224
        vision.patch_size = 16
        config = mock.MagicMock()
        config.vision = vision

        path = _write_processor_config(config, str(tmp_path))
        assert path is not None
        with open(path) as f:
            data = json.load(f)
        assert data["image_size"] == 224
        assert data["patch_size"] == 16


class TestCopyTokenizerFiles:
    def test_copies_available_files(self, tmp_path):
        # Create a fake tokenizer file to "download"
        fake_src = tmp_path / "src"
        fake_src.mkdir()
        (fake_src / "tokenizer.json").write_text('{"test": true}')

        with mock.patch("huggingface_hub.hf_hub_download") as mock_dl:
            mock_dl.side_effect = lambda model_id, filename: (
                str(fake_src / filename)
                if (fake_src / filename).exists()
                else (_ for _ in ()).throw(OSError("not found"))
            )

            dst = tmp_path / "output"
            dst.mkdir()
            copied = _copy_tokenizer_files("fake/model", str(dst))

        assert "tokenizer.json" in copied
        assert (dst / "tokenizer.json").exists()


@pytest.mark.integration
class TestAutoExportEndToEnd:
    """Integration test: auto_export with a tiny model (no real download)."""

    def test_auto_export_produces_genai_config(self, tmp_path):
        """Mock build() to return a tiny package, verify genai_config."""
        import onnx_ir as ir

        from mobius._builder import build_from_module
        from mobius._configs import ArchitectureConfig
        from mobius._registry import registry
        from mobius.integrations.ort_genai.genai_config import (
            GenaiConfigGenerator,
        )

        # Build a tiny model
        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
        )
        module = registry.get("qwen2")(config)
        pkg = build_from_module(module, config)

        # Fill with random weights
        rng = np.random.default_rng(42)
        for model in pkg.values():
            for init in model.graph.initializers.values():
                if init.const_value is None:
                    shape = list(init.shape)
                    init.const_value = ir.Tensor(rng.standard_normal(shape).astype(np.float32))

        # Generate genai_config from the config
        gen = GenaiConfigGenerator.from_config(config, "qwen2")
        genai_config = gen.generate()

        assert "model" in genai_config
        assert genai_config["model"]["type"] == "qwen2"
        assert genai_config["model"]["vocab_size"] == 256
        assert genai_config["model"]["decoder"]["num_hidden_layers"] == 2

        # Save models and config
        output_dir = str(tmp_path / "export")
        os.makedirs(output_dir)
        pkg.save(output_dir, progress_bar=False)
        gen.write(output_dir)

        assert os.path.exists(os.path.join(output_dir, "model.onnx"))
        assert os.path.exists(os.path.join(output_dir, "genai_config.json"))

        with open(os.path.join(output_dir, "genai_config.json")) as f:
            saved = json.load(f)
        assert saved["model"]["type"] == "qwen2"

    def test_phi4mm_detection_and_config(self, tmp_path):
        """Simulate phi4mm auto-export: verify detection and config."""
        import onnx_ir as ir

        from mobius._builder import build_from_module
        from mobius._configs import ArchitectureConfig, AudioConfig, VisionConfig
        from mobius.integrations.ort_genai.genai_config import (
            GenaiConfigGenerator,
        )
        from mobius.models.phi import Phi4MMMultiModalModel
        from mobius.tasks import Phi4MMMultiModalTask

        # Build a tiny phi4mm model
        # LongRoPE requires rope_scaling with long/short factors
        # inv_freq has int(head_dim * partial_rotary_factor) // 2 elements
        rope_dim = int(16 * 0.75) // 2  # head_dim=16, partial_rotary_factor=0.75
        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=1,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="longrope",
            rope_theta=10000.0,
            partial_rotary_factor=0.75,
            original_max_position_embeddings=128,
            rope_scaling={
                "long_factor": [1.0] * rope_dim,
                "short_factor": [1.0] * rope_dim,
            },
            pad_token_id=0,
            image_token_id=200010,
            vision=VisionConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=28,
                patch_size=14,
                lora={"r": 4, "lora_alpha": 8},
            ),
            audio=AudioConfig(
                attention_dim=64,
                attention_heads=4,
                num_blocks=1,
                linear_units=128,
                kernel_size=3,
                input_size=80,
                token_id=200011,
                lora={"r": 4, "lora_alpha": 8},
            ),
        )
        module = Phi4MMMultiModalModel(config)
        pkg = build_from_module(module, config, task=Phi4MMMultiModalTask())

        # Verify 4-model split
        assert "vision" in pkg
        assert "speech" in pkg
        assert "embedding" in pkg
        assert "model" in pkg

        # Simulate auto_export detection logic
        is_vlm = "vision" in pkg and "embedding" in pkg
        has_speech = "speech" in pkg
        ort_model_type = "phi"  # HF model_type for phi4mm
        if ort_model_type == "phi" and has_speech:
            ort_model_type = "phi4mm"

        assert ort_model_type == "phi4mm"
        assert is_vlm
        assert has_speech

        # Build genai_config using the same logic as auto_export
        generator = GenaiConfigGenerator.from_config(config, ort_model_type)
        vision_kwargs = {
            "spatial_merge_size": None,
            "config_filename": "vision_processor.json",
            "input_names": {
                "pixel_values": "pixel_values",
                "image_sizes": "image_sizes",
            },
        }
        generator.with_vision(image_token_id=config.image_token_id, **vision_kwargs)
        generator.with_speech(audio_token_id=config.audio.token_id)

        genai_config = generator.generate()

        # Verify config structure
        model = genai_config["model"]
        assert model["type"] == "phi4mm"
        assert model["image_token_id"] == 200010
        assert model["audio_token_id"] == 200011

        # All 4 model sections present
        assert "decoder" in model
        assert "vision" in model
        assert "speech" in model
        assert "embedding" in model

        # Vision uses phi4mm-specific inputs
        assert model["vision"]["inputs"]["pixel_values"] == "pixel_values"
        assert model["vision"]["inputs"]["image_sizes"] == "image_sizes"
        assert "image_grid_thw" not in model["vision"]["inputs"]
        assert "spatial_merge_size" not in model["vision"]
        assert model["vision"]["config_filename"] == "vision_processor.json"

        # Speech section
        assert model["speech"]["inputs"]["audio_embeds"] == "audio_embeds"
        assert model["speech"]["inputs"]["audio_sizes"] == "audio_sizes"
        assert model["speech"]["inputs"]["audio_projection_mode"] == "audio_projection_mode"

        # Embedding includes audio_features
        assert model["embedding"]["inputs"]["audio_features"] == "audio_features"

        # Decoder uses inputs_embeds (multimodal)
        assert "inputs_embeds" in model["decoder"]["inputs"]

        # Save and verify files
        output_dir = str(tmp_path / "phi4mm_export")
        os.makedirs(output_dir)

        # Fill with random weights so save() doesn't complain
        rng = np.random.default_rng(42)
        for model in pkg.values():
            for init in model.graph.initializers.values():
                if init.const_value is None:
                    shape = list(init.shape)
                    init.const_value = ir.Tensor(rng.standard_normal(shape).astype(np.float32))

        pkg.save(output_dir, progress_bar=False)
        generator.write(output_dir)

        assert os.path.exists(os.path.join(output_dir, "genai_config.json"))
        # 4-model split produces subdirectories
        assert os.path.exists(os.path.join(output_dir, "vision"))
        assert os.path.exists(os.path.join(output_dir, "speech"))
        assert os.path.exists(os.path.join(output_dir, "embedding"))

        with open(os.path.join(output_dir, "genai_config.json")) as f:
            saved = json.load(f)
        assert saved["model"]["type"] == "phi4mm"
        assert "speech" in saved["model"]
