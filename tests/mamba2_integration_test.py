# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: Mamba2 numerical parity against HuggingFace.

Verifies that the Mamba2 ONNX model produces the same logits as the
HuggingFace Mamba2ForCausalLM when processing tokens one at a time
(single-token decode mode with SSM state carry).

Run with::

    pytest tests/mamba2_integration_test.py -m integration -sv
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import transformers
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

from mobius import build_from_module
from mobius._configs import Mamba2Config
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.mamba import Mamba2CausalLMModel
from mobius.tasks._ssm_causal_lm import SSM2CausalLMTask

# Tiny Mamba2 config satisfying: hidden_size * expand == num_heads * head_dim
# 64 * 2 = 128 == 8 * 16
_HF_CONFIG = transformers.Mamba2Config(
    hidden_size=64,
    num_hidden_layers=2,
    vocab_size=256,
    num_heads=8,
    head_dim=16,
    state_size=8,
    expand=2,
    n_groups=1,
    conv_kernel=4,
    tie_word_embeddings=False,
)


def _build_onnx_model(hf_state_dict: dict[str, torch.Tensor]):
    """Build ONNX Mamba2 model and load weights from HF state dict."""
    config = Mamba2Config(
        hidden_size=64,
        num_hidden_layers=2,
        vocab_size=256,
        num_heads=8,
        head_dim=16,
        state_size=8,
        intermediate_size=128,
        expand=2,
        n_groups=1,
        conv_kernel=4,
        tie_word_embeddings=False,
    )
    module = Mamba2CausalLMModel(config)
    pkg = build_from_module(module, config, task=SSM2CausalLMTask())
    sd = module.preprocess_weights(hf_state_dict)
    pkg.apply_weights(sd)
    return pkg, config


def _run_onnx_steps(
    session: OnnxModelSession,
    tokens: list[int],
    config: Mamba2Config,
) -> np.ndarray:
    """Run ONNX model one token at a time, returning last-step logits."""
    d_inner = config.intermediate_size
    conv_dim = d_inner + 2 * config.n_groups * config.state_size
    conv_state_len = config.conv_kernel - 1

    # Initialize zero states
    conv_states = [
        np.zeros((1, conv_dim, conv_state_len), dtype=np.float32)
        for _ in range(config.num_hidden_layers)
    ]
    ssm_states = [
        np.zeros(
            (1, config.num_heads, config.head_dim, config.state_size),
            dtype=np.float32,
        )
        for _ in range(config.num_hidden_layers)
    ]

    for tok in tokens:
        feeds: dict[str, np.ndarray] = {
            "input_ids": np.array([[tok]], dtype=np.int64),
        }
        for i in range(config.num_hidden_layers):
            feeds[f"past_states.{i}.conv_state"] = conv_states[i]
            feeds[f"past_states.{i}.ssm_state"] = ssm_states[i]

        out = session.run(feeds)
        for i in range(config.num_hidden_layers):
            conv_states[i] = out[f"present.{i}.conv_state"]
            ssm_states[i] = out[f"present.{i}.ssm_state"]

    return out["logits"]


def _run_hf_steps(
    model: transformers.Mamba2ForCausalLM,
    tokens: list[int],
) -> np.ndarray:
    """Run HF Mamba2 one token at a time, returning last-step logits."""
    cache = Mamba2Cache(_HF_CONFIG, batch_size=1)
    for pos, tok in enumerate(tokens):
        with torch.no_grad():
            hf_out = model(
                torch.tensor([[tok]]),
                cache_params=cache,
                cache_position=torch.tensor([pos]),
            )
        cache = hf_out.cache_params
    return hf_out.logits.numpy()


@pytest.mark.integration
class TestMamba2Forward:
    """Numerical parity: ONNX Mamba2 vs HuggingFace Mamba2ForCausalLM."""

    def test_step_by_step_logits_match(self):
        """Process tokens one at a time, compare final logits."""
        torch.manual_seed(42)
        hf_model = transformers.Mamba2ForCausalLM(_HF_CONFIG)
        hf_model.eval()

        sd = {k: v.clone() for k, v in hf_model.state_dict().items()}
        pkg, config = _build_onnx_model(sd)
        session = OnnxModelSession(pkg["model"])

        tokens = [1, 2, 3, 5, 10]
        onnx_logits = _run_onnx_steps(session, tokens, config)
        hf_logits = _run_hf_steps(hf_model, tokens)

        np.testing.assert_allclose(
            onnx_logits,
            hf_logits,
            atol=1e-3,
            rtol=0,
            err_msg="Mamba2 step-by-step logits mismatch",
        )
        session.close()

    def test_single_token_logits_match(self):
        """Single token from zero state should match."""
        torch.manual_seed(42)
        hf_model = transformers.Mamba2ForCausalLM(_HF_CONFIG)
        hf_model.eval()

        sd = {k: v.clone() for k, v in hf_model.state_dict().items()}
        pkg, config = _build_onnx_model(sd)
        session = OnnxModelSession(pkg["model"])

        tokens = [42]
        onnx_logits = _run_onnx_steps(session, tokens, config)
        hf_logits = _run_hf_steps(hf_model, tokens)

        np.testing.assert_allclose(
            onnx_logits,
            hf_logits,
            atol=1e-3,
            rtol=0,
            err_msg="Mamba2 single-token logits mismatch",
        )
        session.close()
