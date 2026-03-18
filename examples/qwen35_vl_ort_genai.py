#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-VL generation with onnxruntime-genai tokenizer + ONNX Runtime.

Builds the 3-model ONNX package (decoder, vision encoder, embedding),
saves it in the flat layout expected by onnxruntime-genai, and runs
text generation — with or without an image.

Qwen3.5-VL uses the same 3-model I/O contract as Qwen2.5-VL, so it can
be loaded by onnxruntime-genai with ``model.type = "qwen2_5_vl"``.

The text decoder uses a hybrid architecture (DeltaNet linear attention +
full GQA attention).  Both layer types produce state (KV cache for
full-attention, conv_state + recurrent_state for DeltaNet).
ORT GenAI's built-in generator does not yet manage DeltaNet recurrent
state, so generation uses manual ONNX Runtime sessions with
onnxruntime-genai providing the tokenizer.

Requirements::

    pip install mobius-ai[ort-genai]

Usage::

    # Text-only generation:
    python examples/qwen35_vl_ort_genai.py

    # With an image:
    python examples/qwen35_vl_ort_genai.py --image testdata/pipeline-cat-chonk.jpeg

    # Compare ORT GenAI output with HuggingFace transformers:
    python examples/qwen35_vl_ort_genai.py --image testdata/pipeline-cat-chonk.jpeg --compare-hf

    # Use a pre-built model directory:
    python examples/qwen35_vl_ort_genai.py --model-dir output/qwen35_vl/

    # Build and save (skip inference):
    python examples/qwen35_vl_ort_genai.py --save-to output/qwen35_vl/
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import onnxruntime_genai as og

# ---------------------------------------------------------------------------
# Model export helpers
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3.5-27B"
DEFAULT_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 50


def _count_kv_layers(config) -> int:
    """Count layers that produce KV cache (full_attention only).

    In the hybrid DeltaNet + full attention architecture, only
    full_attention layers maintain KV cache pairs.  DeltaNet (linear
    attention) layers use recurrent state instead.
    """
    if config.layer_types:
        return sum(1 for t in config.layer_types if t == "full_attention")
    return config.num_hidden_layers


def _write_genai_config(config, output_dir: str) -> None:
    """Write genai_config.json for the Qwen3.5-VL 3-model split.

    Uses ``qwen2_5_vl`` model type since onnxruntime-genai does not
    yet have a native ``qwen3_5_vl`` handler — the I/O contract is
    identical for the 3-model pipeline.

    Important: ``num_hidden_layers`` is set to the number of
    full-attention layers, since only those produce KV cache pairs.
    """
    num_kv_layers = _count_kv_layers(config)

    genai_config = {
        "model": {
            "bos_token_id": 248044,
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
                "num_hidden_layers": num_kv_layers,
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
            "eos_token_id": [248044],
            "pad_token_id": 248044,
            "image_token_id": 248056,
            "vision_start_token_id": 248053,
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

    # Patch tokenizer files for ORT GenAI compatibility:
    # 1. tokenizer_config.json: transformers 5.x writes
    #    "TokenizersBackend" which ORT GenAI doesn't recognise
    # 2. tokenizer.json: Qwen3.5 regex uses \p{M} (Unicode Mark)
    #    which ORT GenAI's regex engine doesn't support
    _patch_tokenizer_class(output_dir)
    _patch_tokenizer_regex(output_dir)

    # ORT GenAI expects ort-extensions processor_config.json format
    _write_processor_config(processor, output_dir)


def _patch_tokenizer_class(output_dir: str) -> None:
    """Ensure tokenizer_config.json uses a class ORT GenAI supports.

    Transformers 5.x introduced ``TokenizersBackend`` as the default
    tokenizer class for many models.  ORT GenAI 0.12.x does not
    recognise this class.  Since the actual tokenizer data lives in
    ``tokenizer.json`` (unchanged), we can safely override the
    metadata to ``Qwen2Tokenizer`` which ORT GenAI handles correctly.
    """
    config_path = os.path.join(output_dir, "tokenizer_config.json")
    if not os.path.isfile(config_path):
        return

    with open(config_path) as f:
        config = json.load(f)

    if config.get("tokenizer_class") == "TokenizersBackend":
        config["tokenizer_class"] = "Qwen2Tokenizer"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def _patch_tokenizer_regex(output_dir: str) -> None:
    r"""Fix pre-tokenizer regex in tokenizer.json for ORT GenAI.

    Qwen3.5 uses ``\p{M}`` (Unicode Mark category) in its
    pre-tokenizer regex, but ORT GenAI's regex engine does not
    support this class.  Replace ``[\p{L}\p{M}]`` with ``\p{L}``
    and remove ``\p{M}`` from negated character classes.  This is
    safe because combining marks are almost never standalone tokens
    and ``\p{L}`` already covers the letters they modify.
    """
    import re

    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_path):
        return

    with open(tokenizer_path) as f:
        data = json.load(f)

    changed = _patch_regex_in_pretokenizer(data.get("pre_tokenizer"), re)
    if changed:
        with open(tokenizer_path, "w") as f:
            json.dump(data, f, ensure_ascii=False)


def _patch_regex_in_pretokenizer(node: dict | None, re) -> bool:
    """Recursively find and patch regex patterns in tokenizer config."""
    if node is None:
        return False
    changed = False
    # Handle Split nodes with Regex patterns
    if isinstance(node.get("pattern"), dict):
        regex = node["pattern"].get("Regex", "")
        if r"\p{M}" in regex:
            # [\p{L}\p{M}]+ → \p{L}+
            patched = re.sub(r"\[\\p\{L\}\\p\{M\}\]\+", r"\\p{L}+", regex)
            # [^\s\p{L}\p{M}\p{N}] → [^\s\p{L}\p{N}]
            patched = patched.replace(r"\p{M}", "")
            if patched != regex:
                node["pattern"]["Regex"] = patched
                changed = True
    # Recurse into pretokenizer sequences
    for child in node.get("pretokenizers", []):
        changed |= _patch_regex_in_pretokenizer(child, re)
    return changed


def _write_processor_config(processor, output_dir: str) -> None:
    """Write processor_config.json in the ort-extensions format.

    NOTE: The ``width`` / ``height`` in the Resize step are default hints.
    The ORT GenAI processor resizes the image to these dimensions; for
    accurate results, they should match the dimensions that HuggingFace
    would compute for the specific input image.
    """
    ip = processor.image_processor
    patch_size = getattr(ip, "patch_size", 16)
    merge_size = getattr(ip, "merge_size", 2)
    temporal_patch_size = getattr(ip, "temporal_patch_size", 2)
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
                            "patch_size": patch_size,
                            "merge_size": merge_size,
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
                            "patch_size": patch_size,
                            "temporal_patch_size": temporal_patch_size,
                            "merge_size": merge_size,
                        },
                    }
                },
            ],
        }
    }
    with open(os.path.join(output_dir, "processor_config.json"), "w") as f:
        json.dump(processor_config, f, indent=2)


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
# Hybrid state management (DeltaNet + full-attention)
# ---------------------------------------------------------------------------


def _init_hybrid_states(hf_config) -> dict[str, np.ndarray]:
    """Initialize per-layer states for the hybrid architecture.

    Full-attention layers get empty KV caches (past_seq_len=0).
    DeltaNet layers get zero-filled conv_state and recurrent_state.
    """
    batch_size = 1
    states: dict[str, np.ndarray] = {}
    layer_types = hf_config.layer_types or []

    # DeltaNet dimensions — these must be present in the config
    num_k_heads = hf_config.linear_num_key_heads
    head_k_dim = hf_config.linear_key_head_dim
    num_v_heads = hf_config.linear_num_value_heads
    head_v_dim = hf_config.linear_value_head_dim
    conv_kernel = hf_config.linear_conv_kernel_dim
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    conv_dim = key_dim * 2 + value_dim

    for i in range(hf_config.num_hidden_layers):
        ltype = layer_types[i] if i < len(layer_types) else "full_attention"
        if ltype == "linear_attention":
            states[f"past_key_values.{i}.conv_state"] = np.zeros(
                (batch_size, conv_dim, conv_kernel - 1),
                dtype=np.float32,
            )
            states[f"past_key_values.{i}.recurrent_state"] = np.zeros(
                (batch_size, num_v_heads, head_k_dim, head_v_dim),
                dtype=np.float32,
            )
        else:
            states[f"past_key_values.{i}.key"] = np.zeros(
                (
                    batch_size,
                    hf_config.num_key_value_heads,
                    0,
                    hf_config.head_dim,
                ),
                dtype=np.float32,
            )
            states[f"past_key_values.{i}.value"] = np.zeros(
                (
                    batch_size,
                    hf_config.num_key_value_heads,
                    0,
                    hf_config.head_dim,
                ),
                dtype=np.float32,
            )

    return states


def _update_states(
    states: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
    hf_config,
) -> dict[str, np.ndarray]:
    """Copy present-state outputs back into the past-state inputs."""
    layer_types = hf_config.layer_types or []
    new_states: dict[str, np.ndarray] = {}

    for i in range(hf_config.num_hidden_layers):
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


def _compute_mrope_position_ids(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray | None,
    image_token_id: int,
    spatial_merge_size: int = 2,
) -> np.ndarray:
    """Compute 3D MRoPE position IDs for the VL decoder.

    Returns shape ``(3, batch, seq_len)`` where the three dimensions
    are (temporal, height, width).  For text tokens all three are
    identical (sequential position).
    """
    batch_size, seq_len = input_ids.shape
    position_ids = np.zeros((3, batch_size, seq_len), dtype=np.int64)

    for b in range(batch_size):
        text_pos = 0
        image_idx = 0
        i = 0
        while i < seq_len:
            if image_grid_thw is not None and input_ids[b, i] == image_token_id:
                img_start = i
                while i < seq_len and input_ids[b, i] == image_token_id:
                    i += 1

                t, h, w = image_grid_thw[image_idx]
                image_idx += 1
                merge_h = h // spatial_merge_size
                merge_w = w // spatial_merge_size

                idx = img_start
                for ti in range(t):
                    for hi in range(merge_h):
                        for wi in range(merge_w):
                            if idx < i:
                                position_ids[0, b, idx] = text_pos + ti
                                position_ids[1, b, idx] = text_pos + hi
                                position_ids[2, b, idx] = text_pos + wi
                                idx += 1

                text_pos += max(t, merge_h, merge_w)
            else:
                position_ids[0, b, i] = text_pos
                position_ids[1, b, i] = text_pos
                position_ids[2, b, i] = text_pos
                text_pos += 1
                i += 1

    return position_ids


def _load_hf_config(model_id: str):
    """Load the HuggingFace config (text config for VL models)."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    return getattr(config, "text_config", config)


def _ort_session(model_dir: str, subdir: str) -> ort.InferenceSession:
    """Load an ONNX model as an ORT InferenceSession."""
    path = os.path.join(model_dir, subdir, "model.onnx")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _run_session(
    session: ort.InferenceSession,
    feeds: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run an ORT session and return outputs as a dict."""
    output_names = [o.name for o in session.get_outputs()]
    results = session.run(output_names, feeds)
    return dict(zip(output_names, results))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(
    model_dir: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Run text-only generation using ORT GenAI tokenizer + ORT sessions.

    Uses onnxruntime-genai for tokenization and onnxruntime for the
    decoder with manual hybrid state management (DeltaNet + attention).

    The prompt is wrapped in the model's chat template so that the
    input format matches what HuggingFace ``generate()`` receives.
    """
    from transformers import AutoProcessor

    hf_config = _load_hf_config(model_id)

    # Format prompt with chat template (same as HF comparison path)
    processor = AutoProcessor.from_pretrained(model_id)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print(f"Loading model from {model_dir} ...")
    og_model = og.Model(model_dir)
    tokenizer = og.Tokenizer(og_model)

    decoder_sess = _ort_session(model_dir, "decoder")
    embed_sess = _ort_session(model_dir, "embedding")

    # Tokenize with ORT GenAI tokenizer
    input_ids = np.array([tokenizer.encode(text)], dtype=np.int64)
    batch_size = 1
    seq_len = input_ids.shape[1]

    # Get embeddings for the input tokens (no image features)
    embed_out = _run_session(
        embed_sess,
        {
            "input_ids": input_ids,
            "image_features": np.zeros((0, hf_config.hidden_size), dtype=np.float32),
        },
    )
    inputs_embeds = embed_out["inputs_embeds"]

    # Initialize hybrid states and process tokens one at a time
    # (DeltaNet layers only support seq_len=1)
    states = _init_hybrid_states(hf_config)
    past_seq_len = 0

    print(f"\nPrompt: {prompt}")
    print("-" * 40)

    # Prefill: process each prompt token individually
    for t in range(seq_len):
        cur_embed = inputs_embeds[:, t : t + 1, :]  # (1, 1, hidden)
        total_seq_len = past_seq_len + 1
        # Text-only: 3D position IDs all equal
        pos = np.array(
            [[[past_seq_len]], [[past_seq_len]], [[past_seq_len]]],
            dtype=np.int64,
        )

        feeds = {
            "inputs_embeds": cur_embed,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": pos,
            **states,
        }
        outputs = _run_session(decoder_sess, feeds)
        states = _update_states(states, outputs, hf_config)
        past_seq_len = total_seq_len

    # Greedy decode loop
    logits = outputs["logits"]
    next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    generated_ids: list[int] = []

    tokenizer_stream = tokenizer.create_stream()
    eos_token_id = hf_config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    for _ in range(max_new_tokens):
        token_id = int(next_token[0, 0])
        generated_ids.append(token_id)
        print(tokenizer_stream.decode(token_id), end="", flush=True)

        if token_id == eos_token_id:
            break

        # Embed the new token
        cur_ids = next_token.astype(np.int64)
        embed_out = _run_session(
            embed_sess,
            {
                "input_ids": cur_ids,
                "image_features": np.zeros((0, hf_config.hidden_size), dtype=np.float32),
            },
        )
        cur_embed = embed_out["inputs_embeds"]
        total_seq_len = past_seq_len + 1
        pos = np.array(
            [[[past_seq_len]], [[past_seq_len]], [[past_seq_len]]],
            dtype=np.int64,
        )

        feeds = {
            "inputs_embeds": cur_embed,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": pos,
            **states,
        }
        outputs = _run_session(decoder_sess, feeds)
        states = _update_states(states, outputs, hf_config)
        past_seq_len = total_seq_len

        logits = outputs["logits"]
        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)

    print()
    print("-" * 40)

    return tokenizer.decode(generated_ids)


def generate_with_image(
    model_dir: str,
    model_id: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
) -> str:
    """Run multimodal generation using ORT GenAI tokenizer + ORT sessions.

    Steps:
    1. HF processor → pixel_values, image_grid_thw, input_ids
    2. Vision encoder(pixel_values, image_grid_thw) → image_features
    3. Embedding(input_ids, image_features) → inputs_embeds
    4. Decoder(inputs_embeds, 3D position_ids, states) → logits
    5. Autoregressive decode loop
    """
    from PIL import Image
    from transformers import AutoProcessor

    hf_config = _load_hf_config(model_id)

    # Use HF processor for chat template and image preprocessing
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
    # Use return_tensors="pt" because the fast image processor in
    # transformers >=5.x only supports PyTorch tensors, then convert.
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    input_ids = inputs["input_ids"].numpy().astype(np.int64)
    pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
    image_grid_thw = inputs["image_grid_thw"].numpy().astype(np.int64)
    batch_size = 1

    # Load ORT sessions and ORT GenAI tokenizer
    print(f"Loading model from {model_dir} ...")
    og_model = og.Model(model_dir)
    tokenizer = og.Tokenizer(og_model)

    decoder_sess = _ort_session(model_dir, "decoder")
    vision_sess = _ort_session(model_dir, "vision")
    embed_sess = _ort_session(model_dir, "embedding")

    # Step 1: Vision encoder → image features
    vision_out = _run_session(
        vision_sess,
        {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        },
    )
    image_features = vision_out["image_features"]

    # Step 2: Embedding → fuse text tokens + image features
    embed_out = _run_session(
        embed_sess,
        {
            "input_ids": input_ids,
            "image_features": image_features,
        },
    )
    inputs_embeds = embed_out["inputs_embeds"]

    # Step 3: Compute 3D MRoPE position IDs
    image_token_id = getattr(hf_config, "image_token_id", 248056)
    spatial_merge_size = getattr(hf_config, "spatial_merge_size", 2)
    all_position_ids = _compute_mrope_position_ids(
        input_ids,
        image_grid_thw,
        image_token_id=image_token_id,
        spatial_merge_size=spatial_merge_size,
    )

    # Step 4: Prefill — process one token at a time
    # (DeltaNet layers only support seq_len=1)
    states = _init_hybrid_states(hf_config)
    seq_len = inputs_embeds.shape[1]
    past_seq_len = 0

    print(f"\nPrompt: {prompt}")
    print(f"Image:  {image_path}")
    print("-" * 40)

    for t in range(seq_len):
        cur_embed = inputs_embeds[:, t : t + 1, :]
        cur_pos = all_position_ids[:, :, t : t + 1]
        total_seq_len = past_seq_len + 1

        feeds = {
            "inputs_embeds": cur_embed,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": cur_pos,
            **states,
        }
        outputs = _run_session(decoder_sess, feeds)
        states = _update_states(states, outputs, hf_config)
        past_seq_len = total_seq_len

    # Step 5: Autoregressive decode loop
    logits = outputs["logits"]
    next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    token_id = int(next_token[0, 0])
    generated_ids: list[int] = [token_id]

    tokenizer_stream = tokenizer.create_stream()
    print(tokenizer_stream.decode(token_id), end="", flush=True)

    max_pos = int(all_position_ids.max())
    eos_token_id = hf_config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    for _ in range(max_new_tokens - 1):
        if token_id == eos_token_id:
            break

        # Embed the new token (no image features for decode tokens)
        cur_ids = next_token.astype(np.int64)
        embed_out = _run_session(
            embed_sess,
            {
                "input_ids": cur_ids,
                "image_features": np.zeros((0, image_features.shape[-1]), dtype=np.float32),
            },
        )
        cur_embed = embed_out["inputs_embeds"]

        max_pos += 1
        pos = np.array([[[max_pos]], [[max_pos]], [[max_pos]]], dtype=np.int64)
        total_seq_len = past_seq_len + 1

        feeds = {
            "inputs_embeds": cur_embed,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": pos,
            **states,
        }
        outputs = _run_session(decoder_sess, feeds)
        states = _update_states(states, outputs, hf_config)
        past_seq_len = total_seq_len

        logits = outputs["logits"]
        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        token_id = int(next_token[0, 0])
        generated_ids.append(token_id)
        print(tokenizer_stream.decode(token_id), end="", flush=True)

    print()
    print("-" * 40)

    return tokenizer.decode(generated_ids)


# ---------------------------------------------------------------------------
# HuggingFace transformers generation (for comparison)
# ---------------------------------------------------------------------------


def generate_text_hf(model_id: str, prompt: str, max_new_tokens: int) -> str:
    """Run text-only generation with HuggingFace transformers."""
    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    print(f"[HF] Loading {model_id} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.float32,
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
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    print(f"[HF] Loading {model_id} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.float32,
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
        description="Qwen3.5-VL text generation with onnxruntime-genai.",
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
        help=("Also run with HuggingFace transformers and compare outputs."),
    )
    args = parser.parse_args()

    if args.save_to:
        build_and_export(args.model_id, args.save_to)
        return

    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = os.path.join("output", "qwen35_vl")
        if not os.path.isfile(os.path.join(model_dir, "genai_config.json")):
            build_and_export(args.model_id, model_dir)

    if args.image:
        prompt = args.prompt or "Describe this image in detail."
        print("=" * 60)
        print("ORT GenAI")
        print("=" * 60)
        generate_with_image(
            model_dir,
            args.model_id,
            prompt,
            args.image,
            args.max_new_tokens,
        )
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
        generate(model_dir, args.model_id, prompt, args.max_new_tokens)
        if args.compare_hf:
            print("\n" + "=" * 60)
            print("HuggingFace Transformers")
            print("=" * 60)
            generate_text_hf(args.model_id, prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()
