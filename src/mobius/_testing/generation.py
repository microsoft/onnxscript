# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Autoregressive text generation using ONNX Runtime.

Provides a self-contained generation loop with KV cache management,
without depending on onnxruntime-genai.
"""

from __future__ import annotations

import numpy as np
import torch

from mobius._configs import ArchitectureConfig
from mobius._testing.ort_inference import OnnxModelSession


class OnnxGenerator:
    """Greedy autoregressive text generator backed by an ONNX model.

    Manages the KV cache, attention mask growth, and position ID
    bookkeeping for the autoregressive generation loop.

    Example::

        session = OnnxModelSession(model)
        gen = OnnxGenerator(session, config)
        output_ids = gen.generate(input_ids, max_new_tokens=20)
    """

    def __init__(
        self,
        session: OnnxModelSession,
        config: ArchitectureConfig,
    ):
        self.session = session
        self.config = config

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 20,
        eos_token_id: int | None = None,
    ) -> np.ndarray:
        """Generate tokens autoregressively using greedy decoding.

        Args:
            input_ids: [batch, seq_len] int64 prompt token IDs.
            max_new_tokens: Maximum number of new tokens to generate.
            eos_token_id: If set, stop generation when this token is produced.

        Returns:
            [batch, seq_len + generated_len] int64 array of all token IDs
            (prompt + generated).
        """
        batch_size, _prompt_len = input_ids.shape
        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        # Initialize empty past KV cache: [batch, num_kv_heads, 0, head_dim]
        past_kv = {}
        for i in range(num_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
            )

        all_ids = input_ids.copy()

        # First step: process the full prompt
        cur_input_ids = input_ids
        past_seq_len = 0

        for _step in range(max_new_tokens):
            cur_seq_len = cur_input_ids.shape[1]
            total_seq_len = past_seq_len + cur_seq_len

            attention_mask = np.ones((batch_size, total_seq_len), dtype=np.int64)
            position_ids = np.arange(past_seq_len, total_seq_len, dtype=np.int64)[
                np.newaxis, :
            ].repeat(batch_size, axis=0)

            feeds = {
                "input_ids": cur_input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                **past_kv,
            }

            outputs = self.session.run(feeds)

            # Extract logits and take argmax of last token
            logits = outputs["logits"]  # [batch, cur_seq_len, vocab]
            next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            # next_token: [batch, 1]

            all_ids = np.concatenate([all_ids, next_token], axis=1)

            # Check EOS
            if eos_token_id is not None and np.all(next_token == eos_token_id):
                break

            # Update past KV from present outputs
            for i in range(num_layers):
                past_kv[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
                past_kv[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

            # Next step: only the new token
            cur_input_ids = next_token.astype(np.int64)
            past_seq_len = total_seq_len

        return all_ids


def torch_generate_greedy(
    model,
    input_ids: np.ndarray,
    max_new_tokens: int = 20,
    eos_token_id: int | None = None,
) -> np.ndarray:
    """Greedy generation using a HuggingFace model (for reference comparison).

    Uses the same greedy argmax logic as OnnxGenerator so results are
    directly comparable (no sampling, no temperature).

    Args:
        model: HuggingFace causal LM model in eval mode.
        input_ids: [batch, seq_len] int64 numpy array.
        max_new_tokens: Maximum new tokens.
        eos_token_id: Stop token.

    Returns:
        [batch, total_len] int64 numpy array.
    """
    device = next(model.parameters()).device
    ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.ones_like(ids)

    with torch.no_grad():
        output = model.generate(
            ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
        )

    return output.cpu().numpy()
