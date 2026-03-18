#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Text generation example — greedy decoding with a causal LM.

Demonstrates the simplest use of ``mobius``: build an ONNX
model from a HuggingFace model ID and run autoregressive text generation.

Usage::

    python examples/text_generation.py

    # With a different model:
    python examples/text_generation.py --model Qwen/Qwen2.5-0.5B

    # Save the ONNX model to disk without running inference:
    python examples/text_generation.py --save-to output/llama/
"""

from __future__ import annotations

import argparse

import numpy as np
import transformers

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32


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
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Greedy autoregressive generation with streaming output."""
    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = tokens["input_ids"].astype(np.int64)
    batch_size = 1

    # Empty KV cache
    past_kv: dict[str, np.ndarray] = {}
    for i in range(num_hidden_layers):
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )

    cur_input_ids = input_ids
    past_seq_len = 0
    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        cur_seq_len = cur_input_ids.shape[1]
        total_seq_len = past_seq_len + cur_seq_len

        feeds = {
            "input_ids": cur_input_ids,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": np.arange(past_seq_len, total_seq_len, dtype=np.int64)[
                np.newaxis, :
            ],
            **past_kv,
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

        # Update KV cache
        for i in range(num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
            past_kv[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

        cur_input_ids = next_token.astype(np.int64)
        past_seq_len = total_seq_len

    print()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Text generation with an ONNX causal language model.",
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
        "--save-to",
        metavar="DIR",
        default=None,
        help="Save the ONNX model package to DIR and exit (no inference).",
    )
    args = parser.parse_args()

    # Build the model
    print(f"Building model {args.model!r} ...")
    pkg = build(args.model, dtype="f32")
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
        max_new_tokens=args.max_new_tokens,
    )
    print("-" * 40)


if __name__ == "__main__":
    main()
