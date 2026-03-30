#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Static-cache text generation example — greedy decoding with pre-allocated KV buffers.

Unlike the dynamic-cache approach in ``text_generation.py``, static cache
pre-allocates fixed-size KV buffers of shape ``[batch, max_seq_len, kv_hidden]``
and updates them in-place via ``write_indices``.  This avoids repeated
concatenation and produces a simpler graph that is easier to optimize.

Key differences from dynamic cache:
- No ``attention_mask`` input — causal masking is handled internally.
- 3-D cache shape ``[batch, max_seq_len, kv_hidden]`` (not 4-D).
- ``write_indices`` tracks where to write the next token's KV entry.
- ``nonpad_kv_seqlen`` tracks how many valid KV entries exist.
- Outputs are ``updated_key_cache.{i}`` / ``updated_value_cache.{i}``.

Usage::

    python examples/static_cache_generation.py

    # With a different model:
    python examples/static_cache_generation.py --model Qwen/Qwen2.5-0.5B

    # Custom sequence length budget:
    python examples/static_cache_generation.py --max-seq-len 512

    # Save the ONNX model to disk without running inference:
    python examples/static_cache_generation.py --save-to output/llama-static/
"""

from __future__ import annotations

import argparse

import numpy as np
import transformers

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession
from mobius.tasks import CausalLMTask

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32
MAX_SEQ_LEN = 2048


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_model(model_id: str, *, max_seq_len: int):
    """Build an ONNX model with static KV cache."""
    task = CausalLMTask(static_cache=True, max_seq_len=max_seq_len)
    return build(model_id, task=task, dtype="f32")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(
    session: OnnxModelSession,
    tokenizer,
    prompt: str,
    *,
    num_hidden_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Greedy autoregressive generation with static KV cache."""
    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = tokens["input_ids"].astype(np.int64)
    batch_size = 1
    kv_hidden = num_kv_heads * head_dim

    # Pre-allocate cache buffers — shape [batch, max_seq_len, kv_hidden].
    # Unlike dynamic cache, these are fixed-size and updated in-place.
    cache: dict[str, np.ndarray] = {}
    for i in range(num_hidden_layers):
        cache[f"key_cache.{i}"] = np.zeros(
            (batch_size, max_seq_len, kv_hidden), dtype=np.float32
        )
        cache[f"value_cache.{i}"] = np.zeros(
            (batch_size, max_seq_len, kv_hidden), dtype=np.float32
        )

    cur_input_ids = input_ids
    # write_indices: position in the cache to write the next KV entry.
    # For prefill, this is 0 (the module writes prompt_len entries
    # starting from here); for decode steps it advances by 1 each step.
    write_indices = np.zeros((batch_size,), dtype=np.int64)

    # nonpad_kv_seqlen: number of valid (non-padding) entries in the
    # cache so far.  Starts at 0 before the first forward pass.
    nonpad_kv_seqlen = np.zeros((batch_size,), dtype=np.int64)

    generated_ids: list[int] = []

    for _step in range(max_new_tokens):
        cur_seq_len = cur_input_ids.shape[1]
        start_pos = int(write_indices[0])

        # Guard against exceeding the pre-allocated cache length.
        if start_pos + cur_seq_len > max_seq_len:
            print(
                f"\n[Stopped] Cache full: position {start_pos} + "
                f"seq_len {cur_seq_len} would exceed "
                f"max_seq_len {max_seq_len}."
            )
            break

        position_ids = np.arange(start_pos, start_pos + cur_seq_len, dtype=np.int64)[
            np.newaxis, :
        ]

        feeds = {
            "input_ids": cur_input_ids,
            "position_ids": position_ids,
            "write_indices": write_indices,
            "nonpad_kv_seqlen": nonpad_kv_seqlen,
            **cache,
        }

        outputs = session.run(feeds)

        logits = outputs["logits"]
        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        token_id = int(next_token[0, 0])
        generated_ids.append(token_id)

        # Stream the new token
        new_text = tokenizer.decode([token_id], skip_special_tokens=True)
        print(new_text, end="", flush=True)

        if token_id == tokenizer.eos_token_id:
            break

        # Feed updated caches back for the next step.
        # Output names are updated_key_cache.{i} / updated_value_cache.{i}.
        for i in range(num_hidden_layers):
            cache[f"key_cache.{i}"] = outputs[f"updated_key_cache.{i}"]
            cache[f"value_cache.{i}"] = outputs[f"updated_value_cache.{i}"]

        # Advance write position and valid-length counters
        write_indices = write_indices + cur_seq_len
        nonpad_kv_seqlen = nonpad_kv_seqlen + cur_seq_len

        # Decode one token at a time after the prefill step
        cur_input_ids = next_token.astype(np.int64)

    print()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Text generation with static KV cache.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text prompt (default: %(default)r).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum tokens to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,
        help="Static cache buffer size (default: %(default)s).",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help=("Save the ONNX model package to DIR and exit (no inference)."),
    )
    args = parser.parse_args()

    # Build the model with static cache
    print(f"Building model {args.model!r} (static cache) ...")
    pkg = build_model(args.model, max_seq_len=args.max_seq_len)
    config = pkg.config

    if args.save_to:
        pkg.save(args.save_to)
        print("Done.")
        return

    # Create inference session
    session = OnnxModelSession(pkg["model"])

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    print(f"\nPrompt: {args.prompt}")
    print("-" * 40)
    generate(
        session,
        tokenizer,
        args.prompt,
        num_hidden_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
    )
    print("-" * 40)


if __name__ == "__main__":
    main()
