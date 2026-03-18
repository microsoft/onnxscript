#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL generation with onnxruntime-genai.

Builds the 3-model ONNX package (decoder, vision encoder, embedding),
saves it in the flat layout expected by onnxruntime-genai, and runs
text generation — with or without an image.

Qwen3-VL uses the same 3-model I/O contract as Qwen2.5-VL, so it can
be loaded by onnxruntime-genai with ``model.type = "qwen2_5_vl"``.

Requirements::

    pip install mobius-ai[ort-genai]

Usage::

    # Text-only generation:
    python examples/qwen3_vl_ort_genai.py

    # With an image:
    python examples/qwen3_vl_ort_genai.py --image testdata/pipeline-cat-chonk.jpeg

    # Compare ORT GenAI output with HuggingFace transformers:
    python examples/qwen3_vl_ort_genai.py --image testdata/pipeline-cat-chonk.jpeg --compare-hf

    # Use a pre-built model directory:
    python examples/qwen3_vl_ort_genai.py --model-dir output/qwen3vl/

    # Build and save (skip inference):
    python examples/qwen3_vl_ort_genai.py --save-to output/qwen3vl/
"""

from __future__ import annotations

import argparse
import json
import os

import onnxruntime_genai as og

# ---------------------------------------------------------------------------
# Model export helpers
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 50


def _write_genai_config(config, output_dir: str) -> None:
    """Write genai_config.json for the Qwen3-VL 3-model split.

    Uses ``qwen2_5_vl`` model type since onnxruntime-genai does not
    yet have a native ``qwen3_vl`` handler — the I/O contract is
    identical for the 3-model pipeline.
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
            },
            "vision": {
                "filename": "vision/model.onnx",
                "spatial_merge_size": 2,
                "inputs": {
                    "pixel_values": "pixel_values",
                    "image_grid_thw": "image_grid_thw",
                },
                "outputs": {
                    "image_features": "image_features",
                },
            },
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "image_token_id": 151655,
            "vision_start_token_id": 151652,
            "type": "qwen2_5_vl",
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
    with open(os.path.join(output_dir, "genai_config.json"), "w") as f:
        json.dump(genai_config, f, indent=4)


def _copy_tokenizer(model_id: str, output_dir: str) -> None:
    """Copy tokenizer files and processor config from the HuggingFace cache."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    processor.save_pretrained(output_dir)

    # ORT GenAI expects ort-extensions processor_config.json format
    _write_processor_config(processor, output_dir)


def _write_processor_config(processor, output_dir: str) -> None:
    """Write processor_config.json in the ort-extensions format.

    NOTE: The ``width`` / ``height`` in the Resize step are default hints.
    Call ``_update_resize_for_image`` before running multimodal inference
    so that the ORT GenAI processor resizes the image to the same
    dimensions that HuggingFace would compute.
    """
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
                            "width": 960,
                            "height": 672,
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


def _update_resize_for_image(
    model_dir: str, image_path: str, patch_size: int = 14, merge_size: int = 2
) -> None:
    """Update processor_config.json resize dims to match HF smart_resize.

    The ORT GenAI processor uses the configured width/height as the
    target resize.  HuggingFace instead computes target dimensions from
    the original image size and pixel constraints.  This helper bridges
    that gap by computing the HF-equivalent dimensions for the given
    image and writing them into the processor config.
    """
    from PIL import Image

    img = Image.open(image_path)
    w, h = img.size
    factor = patch_size * merge_size  # 28

    new_h = max(factor, round(h / factor) * factor)
    new_w = max(factor, round(w / factor) * factor)

    config_path = os.path.join(model_dir, "processor_config.json")
    with open(config_path) as f:
        config = json.load(f)

    resize = config["processor"]["transforms"][2]["operation"]["attrs"]
    min_pix = resize.get("min_pixels", 3136)
    max_pix = resize.get("max_pixels", 12845056)

    pixels = new_h * new_w
    if pixels > max_pix:
        scale = (max_pix / pixels) ** 0.5
        new_h = max(factor, round(h * scale / factor) * factor)
        new_w = max(factor, round(w * scale / factor) * factor)
    elif pixels < min_pix:
        scale = (min_pix / pixels) ** 0.5
        new_h = max(factor, round(h * scale / factor) * factor)
        new_w = max(factor, round(w * scale / factor) * factor)

    resize["width"] = new_w
    resize["height"] = new_h

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def build_and_export(model_id: str, output_dir: str) -> None:
    """Build the 3-model ONNX package and save for onnxruntime-genai."""
    from mobius import build

    print(f"Building {model_id!r} ...")
    pkg = build(model_id, dtype="f32", load_weights=True)
    print(f"Package components: {list(pkg.keys())}")

    print(f"Saving to {output_dir} ...")
    pkg.save(output_dir)
    _write_genai_config(pkg.config, output_dir)
    _copy_tokenizer(model_id, output_dir)
    print("Export complete.")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(model_dir: str, prompt: str, max_new_tokens: int) -> str:
    """Run text generation with onnxruntime-genai."""
    print(f"Loading model from {model_dir} ...")
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)

    input_ids = tokenizer.encode(prompt)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(input_ids) + max_new_tokens)

    generator = og.Generator(model, params)
    generator.append_tokens(input_ids)

    print(f"\nPrompt: {prompt}")
    print("-" * 40)

    generated = list(input_ids)
    tokenizer_stream = tokenizer.create_stream()
    for _ in range(max_new_tokens):
        if generator.is_done():
            break
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        generated.append(token)
        print(tokenizer_stream.decode(token), end="", flush=True)

    print()
    print("-" * 40)

    output = tokenizer.decode(generated)
    del generator
    return output


def generate_with_image(
    model_dir: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
) -> None:
    """Run multimodal generation with onnxruntime-genai.

    Uses the ORT GenAI multimodal processor to encode the image
    into pixel_values + image_grid_thw alongside the tokenized prompt.
    """
    # Compute HF-equivalent resize dimensions for this image
    _update_resize_for_image(model_dir, image_path)

    from transformers import AutoProcessor

    # The chat template encodes <|image_pad|> tokens for the image
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print(f"Loading model from {model_dir} ...")
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)
    ort_processor = model.create_multimodal_processor()

    # Load the image via ORT GenAI's image processor
    images = og.Images.open(image_path)
    inputs = ort_processor(text, images=images)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_new_tokens + 512)

    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    print(f"\nPrompt: {prompt}")
    print(f"Image:  {image_path}")
    print("-" * 40)

    tokenizer_stream = tokenizer.create_stream()
    tokens_generated = 0
    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(token), end="", flush=True)
        tokens_generated += 1
        if tokens_generated >= max_new_tokens:
            break

    print()
    print("-" * 40)

    del generator


# ---------------------------------------------------------------------------
# HuggingFace transformers generation (for comparison)
# ---------------------------------------------------------------------------


def generate_text_hf(model_id: str, prompt: str, max_new_tokens: int) -> str:
    """Run text-only generation with HuggingFace transformers."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[HF] Loading {model_id} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], return_tensors="pt").to("cpu")

    print(f"\n[HF] Prompt: {prompt}")
    print("-" * 40)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen_ids = output_ids[:, inputs.input_ids.shape[1] :]
    output = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    print(output)
    print("-" * 40)
    return output


def generate_with_image_hf(
    model_id: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
) -> str:
    """Run multimodal generation with HuggingFace transformers."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[HF] Loading {model_id} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to("cpu")

    print(f"\n[HF] Prompt: {prompt}")
    print(f"[HF] Image:  {image_path}")
    print("-" * 40)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen_ids = output_ids[:, inputs.input_ids.shape[1] :]
    output = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    print(output)
    print("-" * 40)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL text generation with onnxruntime-genai.",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Pre-built model directory. Skip export if provided.",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help="Export the model to DIR and exit (no inference).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt (default depends on whether --image is used).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to image file for multimodal generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum tokens to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help="Also run with HuggingFace transformers and compare outputs.",
    )
    args = parser.parse_args()

    if args.save_to:
        build_and_export(args.model_id, args.save_to)
        return

    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = os.path.join("output", "qwen3_vl")
        if not os.path.isfile(os.path.join(model_dir, "genai_config.json")):
            build_and_export(args.model_id, model_dir)

    if args.image:
        prompt = args.prompt or "Describe this image in detail."
        print("=" * 60)
        print("ORT GenAI")
        print("=" * 60)
        generate_with_image(model_dir, prompt, args.image, args.max_new_tokens)
        if args.compare_hf:
            print("\n" + "=" * 60)
            print("HuggingFace Transformers")
            print("=" * 60)
            generate_with_image_hf(
                args.model_id,
                prompt,
                args.image,
                args.max_new_tokens,
            )
    else:
        prompt = args.prompt or DEFAULT_PROMPT
        print("=" * 60)
        print("ORT GenAI")
        print("=" * 60)
        generate(model_dir, prompt, args.max_new_tokens)
        if args.compare_hf:
            print("\n" + "=" * 60)
            print("HuggingFace Transformers")
            print("=" * 60)
            generate_text_hf(args.model_id, prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()
