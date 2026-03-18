# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for onnxruntime-genai inference.

Verifies that exported ONNX models work end-to-end with the
onnxruntime-genai inference runtime. Requires the ``ort-genai``
extra to be installed::

    pip install mobius-ai[ort-genai]

Run::

    pytest tests/ort_genai_test.py -m integration -sv
"""

from __future__ import annotations

import json
import os

import pytest

ort_genai = pytest.importorskip("onnxruntime_genai")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_genai_config(config, output_dir: str, *, model_type: str = "qwen2_5_vl") -> None:
    """Write a minimal genai_config.json for a VL 3-model split.

    Includes image_token_id, vision_start_token_id, and spatial_merge_size
    which are required for correct 3D M-RoPE position_ids computation
    in ORT GenAI's multimodal pipeline.
    """
    genai_config = {
        "model": {
            "bos_token_id": 151643,
            "context_length": 4096,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "filename": "decoder/model.onnx",
                "head_size": config.head_dim,
                "hidden_size": config.hidden_size,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "num_attention_heads": config.num_attention_heads,
                "num_hidden_layers": config.num_hidden_layers,
                "num_key_value_heads": config.num_key_value_heads,
            },
            "embedding": {
                "filename": "embedding/model.onnx",
                "inputs": {
                    "input_ids": "input_ids",
                    "image_features": "image_features",
                },
                "outputs": {
                    "inputs_embeds": "inputs_embeds",
                },
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
            },
            # Required for 3D M-RoPE position_ids computation.
            # Without these, ORT GenAI falls back to 1D positions,
            # producing wrong output for image inputs.
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "vision": {
                "filename": "vision/model.onnx",
                "config_filename": "processor_config.json",
                "spatial_merge_size": getattr(config, "spatial_merge_size", 2),
                "tokens_per_second": 2.0,
                "inputs": {
                    "pixel_values": "pixel_values",
                    "image_grid_thw": "image_grid_thw",
                },
                "outputs": {
                    "image_features": "image_features",
                },
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
            },
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "type": model_type,
            "vocab_size": config.vocab_size,
        },
        "search": {
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": True,
            "length_penalty": 1.0,
            "max_length": 4096,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }
    path = os.path.join(output_dir, "genai_config.json")
    with open(path, "w") as f:
        json.dump(genai_config, f, indent=4)


def _copy_tokenizer(model_id: str, output_dir: str) -> None:
    """Copy tokenizer files and processor config from the HuggingFace cache."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    processor.save_pretrained(output_dir)

    # ORT GenAI expects ort-extensions processor_config.json format
    _write_processor_config(processor, output_dir)


def _write_processor_config(processor, output_dir: str) -> None:
    """Write processor_config.json in the ort-extensions format."""
    ip = processor.image_processor
    processor_config = {
        "processor": {
            "name": "qwen2_5_image_processor",
            "transforms": [
                {
                    "operation": {
                        "name": "decode_image",
                        "type": "DecodeImage",
                        "attrs": {"color_space": "RGB"},
                    }
                },
                {
                    "operation": {
                        "name": "convert_to_rgb",
                        "type": "ConvertRGB",
                    }
                },
                {
                    "operation": {
                        "name": "resize",
                        "type": "Resize",
                        "attrs": {
                            "width": 540,
                            "height": 360,
                            "smart_resize": 1,
                            "min_pixels": ip.size.get("shortest_edge", 3136),
                            "max_pixels": ip.size.get("longest_edge", 12845056),
                            "patch_size": ip.patch_size,
                            "merge_size": ip.merge_size,
                        },
                    }
                },
                {
                    "operation": {
                        "name": "rescale",
                        "type": "Rescale",
                        "attrs": {"rescale_factor": ip.rescale_factor},
                    }
                },
                {
                    "operation": {
                        "name": "normalize",
                        "type": "Normalize",
                        "attrs": {
                            "mean": list(ip.image_mean),
                            "std": list(ip.image_std),
                            "qwen2_5_vl": 1,
                        },
                    }
                },
                {
                    "operation": {
                        "name": "patch_image",
                        "type": "PatchImage",
                        "attrs": {
                            "patch_size": ip.patch_size,
                            "temporal_patch_size": ip.temporal_patch_size,
                            "merge_size": ip.merge_size,
                        },
                    }
                },
            ],
        }
    }
    with open(os.path.join(output_dir, "processor_config.json"), "w") as f:
        json.dump(processor_config, f, indent=2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_MODELS = [
    pytest.param("Qwen/Qwen2.5-VL-3B-Instruct", id="qwen2.5-vl-3b"),
]


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.parametrize("model_id", _MODELS)
class TestOrtGenaiQwen25VL:
    """End-to-end tests with onnxruntime-genai for Qwen2.5-VL."""

    def test_text_generation(self, model_id: str, tmp_path):
        """Build, save, and run text generation with onnxruntime-genai."""
        from mobius import build

        # Build 3-model package with weights
        pkg = build(model_id, dtype="f32", load_weights=True)
        assert "decoder" in pkg
        assert "vision" in pkg
        assert "embedding" in pkg

        # Save in flat layout for ORT GenAI
        output_dir = str(tmp_path / "qwen25vl")
        pkg.save(output_dir)
        _write_genai_config(pkg.config, output_dir)
        _copy_tokenizer(model_id, output_dir)

        # Verify files exist
        assert os.path.isfile(os.path.join(output_dir, "model.onnx"))
        assert os.path.isfile(os.path.join(output_dir, "vision.onnx"))
        assert os.path.isfile(os.path.join(output_dir, "embedding.onnx"))
        assert os.path.isfile(os.path.join(output_dir, "genai_config.json"))
        assert os.path.isfile(os.path.join(output_dir, "tokenizer.json"))

        # Load with onnxruntime-genai
        model = ort_genai.Model(output_dir)
        tokenizer = ort_genai.Tokenizer(model)

        # Text-only generation (no image)
        prompt = "The capital of France is"
        input_ids = tokenizer.encode(prompt)

        params = ort_genai.GeneratorParams(model)
        params.set_search_options(max_length=len(input_ids) + 20)

        generator = ort_genai.Generator(model, params)
        generator.append_tokens(input_ids)

        generated_tokens = list(input_ids)
        max_new = 20
        for _ in range(max_new):
            if generator.is_done():
                break
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            generated_tokens.append(new_token)

        output_text = tokenizer.decode(generated_tokens)
        print(f"\nPrompt: {prompt!r}")
        print(f"Output: {output_text!r}")

        # Basic sanity: output should be longer than input
        assert len(generated_tokens) > len(input_ids), (
            "Model should generate at least one token"
        )
        del generator

    def test_multimodal_image_generation(self, model_id: str, tmp_path):
        """Verify image generation works end-to-end with ORT GenAI.

        Uses the multimodal processor to process an image, then generates
        text describing it. This is the same pipeline as the example scripts.
        """
        from mobius import build

        pkg = build(model_id, dtype="f32", load_weights=True)

        output_dir = str(tmp_path / "qwen25vl")
        pkg.save(output_dir)
        _write_genai_config(pkg.config, output_dir)
        _copy_tokenizer(model_id, output_dir)

        model = ort_genai.Model(output_dir)
        processor = model.create_multimodal_processor()
        tokenizer = ort_genai.Tokenizer(model)

        # Build prompt with image placeholder
        prompt = (
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "What is in this image?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        image_path = os.path.abspath("testdata/pipeline-cat-chonk.jpeg")
        images = ort_genai.Images.open(image_path)

        inputs = processor(prompt, images=images)

        params = ort_genai.GeneratorParams(model)
        params.set_search_options(max_length=200)
        params.set_inputs(inputs)

        output_ids = model.generate(params)[0]
        output_text = tokenizer.decode(output_ids)
        print(f"\n[ORT GenAI] Image generation output: {output_text!r}")

        # The model should produce meaningful output about the image
        assert len(output_ids) > 10, (
            "Model should generate substantial output for image description"
        )


# ---------------------------------------------------------------------------
# Qwen3-VL ORT GenAI tests
# ---------------------------------------------------------------------------

_QWEN3VL_MODELS = [
    pytest.param("Qwen/Qwen3-VL-2B-Instruct", id="qwen3-vl-2b"),
]


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.parametrize("model_id", _QWEN3VL_MODELS)
class TestOrtGenaiQwen3VL:
    """End-to-end tests with onnxruntime-genai for Qwen3-VL.

    Qwen3-VL uses the same 3-model I/O contract as Qwen2.5-VL.
    ORT GenAI loads it with model type ``qwen2_5_vl`` since the
    runtime pipeline is identical for text-only generation.
    """

    def test_text_generation(self, model_id: str, tmp_path):
        """Build, save, and run text generation with onnxruntime-genai."""
        from mobius import build

        pkg = build(model_id, dtype="f32", load_weights=True)
        assert "decoder" in pkg
        assert "vision" in pkg
        assert "embedding" in pkg

        output_dir = str(tmp_path / "qwen3vl")
        pkg.save(output_dir)
        # Use qwen2_5_vl model type — ORT GenAI doesn't have qwen3_vl yet
        _write_genai_config(pkg.config, output_dir, model_type="qwen2_5_vl")
        _copy_tokenizer(model_id, output_dir)

        model = ort_genai.Model(output_dir)
        tokenizer = ort_genai.Tokenizer(model)

        prompt = "The capital of France is"
        input_ids = tokenizer.encode(prompt)

        params = ort_genai.GeneratorParams(model)
        params.set_search_options(max_length=len(input_ids) + 20)

        generator = ort_genai.Generator(model, params)
        generator.append_tokens(input_ids)

        generated_tokens = list(input_ids)
        for _ in range(20):
            if generator.is_done():
                break
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            generated_tokens.append(new_token)

        output_text = tokenizer.decode(generated_tokens)
        print(f"\nPrompt: {prompt!r}")
        print(f"Output: {output_text!r}")

        assert len(generated_tokens) > len(input_ids), (
            "Model should generate at least one token"
        )
        del generator
