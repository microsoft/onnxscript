#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 text generation — standalone greedy decoding (no onnxruntime-genai).

Demonstrates a fully manual autoregressive generation loop for Qwen3.5
models with the **hybrid DeltaNet + full-attention** architecture.

All Qwen3.5 models use a mix of:
- **Full-attention layers**: standard KV cache (key + value).
- **DeltaNet layers**: recurrent state (conv_state + recurrent_state).

This example handles both state types in the decoding loop.

Usage::

    # Text-only generation (default: Qwen/Qwen3.5-0.8B):
    python examples/qwen35_text_generation.py

    # Different model:
    python examples/qwen35_text_generation.py --model Qwen/Qwen3.5-2B

    # Compare output with HuggingFace transformers:
    python examples/qwen35_text_generation.py --compare-hf

    # With an image (uses 3-model VL pipeline):
    python examples/qwen35_text_generation.py --model Qwen/Qwen3.5-0.8B --image testdata/pipeline-cat-chonk.jpeg

    # Save the ONNX model to disk without running inference:
    python examples/qwen35_text_generation.py --save-to output/qwen35/

    # Run on GPU:
    python examples/qwen35_text_generation.py --device cuda
"""

from __future__ import annotations

import argparse

import ml_dtypes
import numpy as np
import transformers

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32

DTYPE_MAP = {"f16": np.float16, "f32": np.float32, "bf16": ml_dtypes.bfloat16}


# ---------------------------------------------------------------------------
# Hybrid state initialization
# ---------------------------------------------------------------------------


def init_hybrid_states(config, dtype: np.dtype = np.float32) -> dict[str, np.ndarray]:
    """Initialize per-layer states for the hybrid architecture.

    Full-attention layers get empty KV caches (past_seq_len=0).
    DeltaNet layers get zero-filled conv_state and recurrent_state
    (fixed-size carry tensors).
    """
    batch_size = 1
    states: dict[str, np.ndarray] = {}
    layer_types = config.layer_types or []

    # DeltaNet dimensions — these must be present in the config
    num_k_heads = config.linear_num_key_heads
    head_k_dim = config.linear_key_head_dim
    num_v_heads = config.linear_num_value_heads
    head_v_dim = config.linear_value_head_dim
    conv_kernel = config.linear_conv_kernel_dim
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    conv_dim = key_dim * 2 + value_dim

    for i in range(config.num_hidden_layers):
        ltype = layer_types[i] if i < len(layer_types) else "full_attention"

        if ltype == "linear_attention":
            # DeltaNet: fixed-size conv_state and recurrent_state
            states[f"past_key_values.{i}.conv_state"] = np.zeros(
                (batch_size, conv_dim, conv_kernel - 1), dtype=dtype
            )
            states[f"past_key_values.{i}.recurrent_state"] = np.zeros(
                (batch_size, num_v_heads, head_k_dim, head_v_dim),
                dtype=dtype,
            )
        else:
            # Full attention: empty KV cache (grows with each step)
            states[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, config.head_dim),
                dtype=dtype,
            )
            states[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, config.head_dim),
                dtype=dtype,
            )

    return states


def update_states(
    states: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
    config,
) -> dict[str, np.ndarray]:
    """Copy present-state outputs back into the past-state inputs."""
    layer_types = config.layer_types or []
    new_states: dict[str, np.ndarray] = {}

    for i in range(config.num_hidden_layers):
        ltype = layer_types[i] if i < len(layer_types) else "full_attention"

        if ltype == "linear_attention":
            new_states[f"past_key_values.{i}.conv_state"] = outputs[f"present.{i}.conv_state"]
            new_states[f"past_key_values.{i}.recurrent_state"] = outputs[
                f"present.{i}.recurrent_state"
            ]
        else:
            new_states[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
            new_states[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

    return new_states


# ---------------------------------------------------------------------------
# Text-only generation (single ONNX model, 1D position_ids)
# ---------------------------------------------------------------------------


def generate(
    session: OnnxModelSession,
    tokenizer,
    prompt: str,
    config,
    *,
    dtype: np.dtype = np.float32,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Greedy autoregressive generation with the hybrid architecture.

    Uses a single ONNX model built with ``Qwen35CausalLMModel`` (text-only
    mode).  The model takes ``input_ids`` and 1D ``position_ids``.

    Because DeltaNet layers only support single-token decode, every
    token (including the prompt) is processed one at a time.
    """
    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = tokens["input_ids"].astype(np.int64)
    batch_size = 1
    prompt_len = input_ids.shape[1]

    states = init_hybrid_states(config, dtype=dtype)
    past_seq_len = 0
    generated_ids: list[int] = []

    # Process prompt tokens one at a time (DeltaNet requires seq_len=1)
    for t in range(prompt_len):
        cur_token = input_ids[:, t : t + 1]  # (1, 1)
        total_seq_len = past_seq_len + 1

        feeds = {
            "input_ids": cur_token,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": np.array([[past_seq_len]], dtype=np.int64),
            **states,
        }

        outputs = session.run(feeds)
        states = update_states(states, outputs, config)
        past_seq_len = total_seq_len

    # Generate new tokens
    logits = outputs["logits"]
    next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)

    for _ in range(max_new_tokens):
        token_id = int(next_token[0, 0])
        generated_ids.append(token_id)

        new_text = tokenizer.decode([token_id], skip_special_tokens=True)
        print(new_text, end="", flush=True)

        if token_id == tokenizer.eos_token_id:
            break

        cur_input_ids = next_token.astype(np.int64)
        total_seq_len = past_seq_len + 1

        feeds = {
            "input_ids": cur_input_ids,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": np.array([[past_seq_len]], dtype=np.int64),
            **states,
        }

        outputs = session.run(feeds)
        states = update_states(states, outputs, config)
        past_seq_len = total_seq_len

        logits = outputs["logits"]
        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)

    print()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 3-model VL generation (vision + embedding + decoder, 3D position_ids)
# ---------------------------------------------------------------------------


def compute_mrope_position_ids(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray | None,
    image_token_id: int,
    spatial_merge_size: int = 2,
) -> np.ndarray:
    """Compute 3D MRoPE position IDs for the VL decoder.

    Returns shape ``(3, batch, seq_len)`` where the three dimensions
    are (temporal, height, width).

    For text tokens all three dimensions are identical (sequential
    position).  For image tokens the positions reflect the 2D spatial
    grid after merging.
    """
    batch_size, seq_len = input_ids.shape
    position_ids = np.zeros((3, batch_size, seq_len), dtype=np.int64)

    for b in range(batch_size):
        text_pos = 0
        image_idx = 0
        i = 0
        while i < seq_len:
            if image_grid_thw is not None and input_ids[b, i] == image_token_id:
                # Count consecutive image tokens
                img_start = i
                while i < seq_len and input_ids[b, i] == image_token_id:
                    i += 1

                # Grid dims for this image (from HF processor)
                t, h, w = image_grid_thw[image_idx]
                image_idx += 1
                merge_h = h // spatial_merge_size
                merge_w = w // spatial_merge_size

                # Assign 3D positions for each merged patch
                idx = img_start
                for ti in range(t):
                    for hi in range(merge_h):
                        for wi in range(merge_w):
                            if idx < i:
                                position_ids[0, b, idx] = text_pos + ti  # temporal
                                position_ids[1, b, idx] = text_pos + hi  # height
                                position_ids[2, b, idx] = text_pos + wi  # width
                                idx += 1

                # Advance position by the max spatial/temporal extent
                text_pos += max(t, merge_h, merge_w)
            else:
                # Text token: all 3 dims are the same
                position_ids[0, b, i] = text_pos
                position_ids[1, b, i] = text_pos
                position_ids[2, b, i] = text_pos
                text_pos += 1
                i += 1

    return position_ids


def generate_with_image(
    decoder_session: OnnxModelSession,
    vision_session: OnnxModelSession,
    embed_session: OnnxModelSession,
    processor,
    prompt: str,
    image_path: str,
    config,
    *,
    dtype: np.dtype = np.float32,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Greedy generation with the 3-model VL pipeline.

    Steps:
    1. HF processor → pixel_values, image_grid_thw, input_ids
    2. Vision encoder(pixel_values, image_grid_thw) → image_features
    3. Embedding(input_ids, image_features) → inputs_embeds
    4. Decoder(inputs_embeds, 3D position_ids, states) → logits
    5. Autoregressive decode loop
    """
    from PIL import Image

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
    # Use return_tensors="pt" because the fast image processor in
    # transformers ≥5.x only supports PyTorch tensors, then convert to numpy.
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    input_ids = inputs["input_ids"].numpy().astype(np.int64)
    pixel_values = inputs["pixel_values"].numpy().astype(dtype)
    image_grid_thw = inputs["image_grid_thw"].numpy().astype(np.int64)
    batch_size = 1

    # Step 1: Vision encoder → image features
    vision_out = vision_session.run(
        {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
    )
    image_features = vision_out["image_features"]

    # Step 2: Embedding model → fuse text + image
    embed_out = embed_session.run(
        {
            "input_ids": input_ids,
            "image_features": image_features,
        }
    )
    inputs_embeds = embed_out["inputs_embeds"]

    # Step 3: Compute 3D MRoPE position IDs for all tokens
    image_token_id = getattr(config, "image_token_id", 248056)
    spatial_merge_size = getattr(config, "spatial_merge_size", 2)
    all_position_ids = compute_mrope_position_ids(
        input_ids,
        image_grid_thw,
        image_token_id=image_token_id,
        spatial_merge_size=spatial_merge_size,
    )

    # Step 4: Decoder prefill — process one token at a time
    # (DeltaNet layers only support seq_len=1)
    states = init_hybrid_states(config, dtype=dtype)
    seq_len = inputs_embeds.shape[1]
    past_seq_len = 0

    for t in range(seq_len):
        cur_embed = inputs_embeds[:, t : t + 1, :]  # (1, 1, hidden)
        cur_pos = all_position_ids[:, :, t : t + 1]  # (3, 1, 1)
        total_seq_len = past_seq_len + 1

        feeds = {
            "inputs_embeds": cur_embed,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": cur_pos,
            **states,
        }
        outputs = decoder_session.run(feeds)
        states = update_states(states, outputs, config)
        past_seq_len = total_seq_len

    # Extract next token from last prefill step
    logits = outputs["logits"]
    next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    token_id = int(next_token[0, 0])
    generated_ids: list[int] = [token_id]

    tokenizer = processor.tokenizer
    new_text = tokenizer.decode([token_id], skip_special_tokens=True)
    print(new_text, end="", flush=True)

    # Get max position from prefill for decode continuation
    max_pos = int(all_position_ids.max())

    # Step 5: Autoregressive decode loop
    for _ in range(max_new_tokens - 1):
        if token_id == tokenizer.eos_token_id:
            break

        # For decode steps: embed the new token (no image features)
        cur_input_ids = next_token.astype(np.int64)
        embed_out = embed_session.run(
            {
                "input_ids": cur_input_ids,
                "image_features": np.zeros((0, image_features.shape[-1]), dtype=dtype),
            }
        )
        cur_embeds = embed_out["inputs_embeds"]

        # Position: all 3 dims = next sequential position
        max_pos += 1
        pos = np.array([[[max_pos]], [[max_pos]], [[max_pos]]], dtype=np.int64)

        total_seq_len = past_seq_len + 1
        feeds = {
            "inputs_embeds": cur_embeds,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": pos,
            **states,
        }
        outputs = decoder_session.run(feeds)

        logits = outputs["logits"]
        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        token_id = int(next_token[0, 0])
        generated_ids.append(token_id)

        new_text = tokenizer.decode([token_id], skip_special_tokens=True)
        print(new_text, end="", flush=True)

        states = update_states(states, outputs, config)
        past_seq_len = total_seq_len

    print()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# HuggingFace comparison
# ---------------------------------------------------------------------------


def generate_hf(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    device: str = "cpu",
) -> str:
    """Run text-only generation with HuggingFace transformers."""
    import torch

    print(f"[HF] Loading {model_id} ...")
    hf_config = transformers.AutoConfig.from_pretrained(model_id)

    # transformers ≥5.x uses a composite config for Qwen3.5 where
    # vocab_size lives under text_config. Pass text_config so the
    # modeling code can find it.
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        config=hf_config,
        dtype=torch.float32,
    ).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\n[HF] Prompt: {prompt}")
    print("-" * 40)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen_ids = output_ids[:, inputs.input_ids.shape[1] :]
    output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    print(output)
    print("-" * 40)
    return output


def generate_with_image_hf(
    model_id: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
    device: str = "cpu",
) -> str:
    """Run multimodal generation with HuggingFace transformers."""
    import torch
    from PIL import Image

    print(f"[HF] Loading {model_id} ...")
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.float32,
    ).to(device)
    processor = transformers.AutoProcessor.from_pretrained(model_id)

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
    ).to(device)

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
        description=(
            "Qwen3.5 text generation — standalone greedy decoding (no onnxruntime-genai)."
        ),
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
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
        "--save-to",
        metavar="DIR",
        default=None,
        help="Save the ONNX model package to DIR and exit (no inference).",
    )
    parser.add_argument(
        "--dtype",
        default="f32",
        choices=["f16", "bf16", "f32"],
        help="Precision type for the ONNX model (default: %(default)s).",
    )
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help="Also run with HuggingFace transformers and compare outputs.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for ONNX Runtime and PyTorch inference (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.image:
        # ---------------------------------------------------------------
        # Image mode: 3-model VL pipeline
        # ---------------------------------------------------------------
        prompt = args.prompt or "Describe this image in detail."
        print(f"Building 3-model VL pipeline for {args.model!r} ...")
        pkg = build(args.model, dtype=args.dtype, load_weights=True)
        config = pkg.config

        if args.save_to:
            pkg.save(args.save_to)
            print(f"Saved to {args.save_to}")
            return

        decoder_session = OnnxModelSession(pkg["decoder"], device=args.device)
        vision_session = OnnxModelSession(pkg["vision"], device=args.device)
        embed_session = OnnxModelSession(pkg["embedding"], device=args.device)
        processor = transformers.AutoProcessor.from_pretrained(args.model)

        print(f"\nPrompt: {prompt}")
        print(f"Image:  {args.image}")
        print("-" * 40)
        onnx_output = generate_with_image(
            decoder_session,
            vision_session,
            embed_session,
            processor,
            prompt,
            args.image,
            config,
            dtype=DTYPE_MAP[args.dtype],
            max_new_tokens=args.max_new_tokens,
        )
        print("-" * 40)

        if args.compare_hf:
            print("\n" + "=" * 60)
            print("HuggingFace Transformers")
            print("=" * 60)
            hf_output = generate_with_image_hf(
                args.model,
                prompt,
                args.image,
                args.max_new_tokens,
                device=args.device,
            )
            if onnx_output == hf_output:
                print("\n✓ Outputs match exactly!")
            else:
                print("\n✗ Outputs differ!")
                print(f"  ONNX: {onnx_output!r}")
                print(f"  HF:   {hf_output!r}")
    else:
        # ---------------------------------------------------------------
        # Text-only mode: single ONNX model
        # ---------------------------------------------------------------
        from mobius.models import Qwen35CausalLMModel

        prompt = args.prompt or DEFAULT_PROMPT
        print(f"Building text-only model for {args.model!r} ...")
        pkg = build(
            args.model,
            task="hybrid-text-generation",
            module_class=Qwen35CausalLMModel,
            dtype=args.dtype,
            load_weights=True,
        )
        config = pkg.config

        if args.save_to:
            pkg.save(args.save_to)
            print(f"Saved to {args.save_to}")
            return

        session = OnnxModelSession(pkg["model"], device=args.device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        onnx_output = generate(
            session,
            tokenizer,
            prompt,
            config,
            dtype=DTYPE_MAP[args.dtype],
            max_new_tokens=args.max_new_tokens,
        )
        print("-" * 40)

        if args.compare_hf:
            print("\n" + "=" * 60)
            print("HuggingFace Transformers")
            print("=" * 60)
            hf_output = generate_hf(
                args.model,
                prompt,
                args.max_new_tokens,
                device=args.device,
            )
            if onnx_output == hf_output:
                print("\n✓ Outputs match exactly!")
            else:
                print("\n✗ Outputs differ!")
                print(f"  ONNX: {onnx_output!r}")
                print(f"  HF:   {hf_output!r}")


if __name__ == "__main__":
    main()
