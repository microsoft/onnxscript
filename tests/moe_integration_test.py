# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: MoE (Mixture of Experts) models.

Verifies that MoE ONNX models produce the same logits as the HuggingFace
PyTorch reference for both prefill and decode steps. Run with::

    pytest tests/moe_integration_test.py -m integration -sv

Note: This test is superseded by integration_test.py which includes MoE
models in the parametrized catalogue. It is kept for backward compatibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from mobius._testing.comparison import (
    assert_generation_match,
    assert_logits_close,
)
from mobius._testing.generation import OnnxGenerator, torch_generate_greedy
from mobius._testing.ort_inference import OnnxModelSession
from mobius._testing.torch_reference import load_torch_model, torch_forward

_MODELS = [
    pytest.param("microsoft/Phi-tiny-MoE-instruct", True, id="phi-tiny-moe"),
]


def _get_config(model_id: str, trust_remote_code: bool = False):
    """Load ArchitectureConfig for a model from HuggingFace."""
    import transformers

    from mobius._configs import ArchitectureConfig

    hf_config = transformers.AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config
    return ArchitectureConfig.from_transformers(hf_config)


def _make_prefill_feeds(config, input_ids, attention_mask, position_ids):
    feeds = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    for i in range(config.num_hidden_layers):
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )
    return feeds


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("model_id,trust_remote_code", _MODELS)
class TestMoEForwardNumerical:
    """Compare single forward pass logits between MoE ONNX and PyTorch."""

    def test_prefill_logits_match(self, model_id: str, trust_remote_code: bool):
        from mobius import build

        onnx_model = build(model_id, load_weights=True)
        torch_model, tokenizer = load_torch_model(model_id)
        config = _get_config(model_id, trust_remote_code)

        prompt = "The capital of France is"
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

        torch_logits, _ = torch_forward(torch_model, input_ids, attention_mask, position_ids)

        session = OnnxModelSession(onnx_model)
        feeds = _make_prefill_feeds(config, input_ids, attention_mask, position_ids)
        onnx_outputs = session.run(feeds)
        session.close()

        assert_logits_close(onnx_outputs["logits"], torch_logits, rtol=1e-3, atol=1e-3)

    def test_decode_step_logits_match(self, model_id: str, trust_remote_code: bool):
        from mobius import build

        onnx_model = build(model_id, load_weights=True)
        torch_model, tokenizer = load_torch_model(model_id)
        config = _get_config(model_id, trust_remote_code)

        prompt = "Hello world"
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

        torch_logits_1, torch_kv = torch_forward(
            torch_model, input_ids, attention_mask, position_ids
        )

        session = OnnxModelSession(onnx_model)
        feeds = _make_prefill_feeds(config, input_ids, attention_mask, position_ids)
        onnx_out_1 = session.run(feeds)

        next_token = np.argmax(torch_logits_1[:, -1, :], axis=-1, keepdims=True)
        decode_input_ids = next_token.astype(np.int64)
        decode_attention_mask = np.ones((1, seq_len + 1), dtype=np.int64)
        decode_position_ids = np.array([[seq_len]], dtype=np.int64)

        torch_logits_2, _ = torch_forward(
            torch_model,
            decode_input_ids,
            decode_attention_mask,
            decode_position_ids,
            past_key_values=torch_kv,
        )

        decode_feeds = {
            "input_ids": decode_input_ids,
            "attention_mask": decode_attention_mask,
            "position_ids": decode_position_ids,
        }
        for i in range(config.num_hidden_layers):
            decode_feeds[f"past_key_values.{i}.key"] = onnx_out_1[f"present.{i}.key"]
            decode_feeds[f"past_key_values.{i}.value"] = onnx_out_1[f"present.{i}.value"]
        onnx_out_2 = session.run(decode_feeds)
        session.close()

        assert_logits_close(onnx_out_2["logits"], torch_logits_2, rtol=1e-3, atol=1e-3)


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("model_id,trust_remote_code", _MODELS)
class TestMoEGeneration:
    """Compare greedy text generation between MoE ONNX and PyTorch."""

    def test_generate_tokens_match(self, model_id: str, trust_remote_code: bool):
        from mobius import build

        onnx_model = build(model_id, load_weights=True)
        torch_model, tokenizer = load_torch_model(model_id)
        config = _get_config(model_id, trust_remote_code)

        prompt = "Once upon a time"
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        max_new = 10

        session = OnnxModelSession(onnx_model)
        generator = OnnxGenerator(session, config)
        onnx_ids = generator.generate(
            input_ids,
            max_new_tokens=max_new,
            eos_token_id=tokenizer.eos_token_id,
        )
        session.close()

        torch_ids = torch_generate_greedy(
            torch_model,
            input_ids,
            max_new_tokens=max_new,
            eos_token_id=tokenizer.eos_token_id,
        )

        assert_generation_match(onnx_ids[0].tolist(), torch_ids[0].tolist())
