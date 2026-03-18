# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: T5 and BART encoder-decoder seq2seq models.

Verifies that the T5 and BART ONNX encoder and decoder produce the same
outputs as the HuggingFace PyTorch reference models.

Run with::

    pytest tests/seq2seq_integration_test.py -m integration -sv
"""

from __future__ import annotations

import numpy as np
import pytest
import transformers

from mobius import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._testing.comparison import assert_logits_close
from mobius._testing.ort_inference import OnnxModelSession
from mobius._testing.torch_reference import (
    torch_seq2seq_decoder_forward,
    torch_seq2seq_encoder_forward,
)
from mobius._weight_loading import _download_weights
from mobius.models.bart import BartForConditionalGeneration
from mobius.tasks import Seq2SeqTask

# =========================================================================
# BART integration tests (facebook/bart-base, ~139M params)
# =========================================================================

_BART_MODEL_ID = "facebook/bart-base"


def _load_bart_config():
    hf_config = transformers.AutoConfig.from_pretrained(_BART_MODEL_ID)
    return ArchitectureConfig.from_transformers(hf_config)


def _build_bart_package(config):
    module = BartForConditionalGeneration(config)
    pkg = build_from_module(module, config, task=Seq2SeqTask())

    state_dict = _download_weights(_BART_MODEL_ID)
    state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)
    return pkg


def _load_torch_bart():
    model = transformers.BartForConditionalGeneration.from_pretrained(_BART_MODEL_ID)
    model.eval()
    return model


def _make_decoder_feeds(
    config,
    decoder_input_ids: np.ndarray,
    encoder_hidden_states: np.ndarray,
    attention_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create decoder feed dict with empty KV cache for prefill."""
    num_layers = getattr(config, "num_decoder_layers", config.num_hidden_layers)
    feeds: dict[str, np.ndarray] = {
        "input_ids": decoder_input_ids,
        "encoder_hidden_states": encoder_hidden_states,
        "attention_mask": attention_mask,
    }
    for i in range(num_layers):
        for prefix in ("self", "cross"):
            feeds[f"past_key_values.{i}.{prefix}.key"] = np.zeros(
                (1, config.num_attention_heads, 0, config.head_dim),
                dtype=np.float32,
            )
            feeds[f"past_key_values.{i}.{prefix}.value"] = np.zeros(
                (1, config.num_attention_heads, 0, config.head_dim),
                dtype=np.float32,
            )
    return feeds


@pytest.mark.integration
@pytest.mark.integration_fast
class TestBartEncoderForward:
    """Compare BART encoder output between ONNX and PyTorch."""

    def test_encoder_hidden_states_match(self):
        """Encoder forward: input_ids → encoder_hidden_states."""
        config = _load_bart_config()
        pkg = _build_bart_package(config)
        torch_model = _load_torch_bart()

        input_ids = np.array([[0, 31414, 232, 328, 740, 2]], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)

        torch_enc = torch_seq2seq_encoder_forward(torch_model, input_ids, attention_mask)

        encoder_session = OnnxModelSession(pkg["encoder"])
        onnx_enc = encoder_session.run(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        encoder_session.close()

        assert_logits_close(
            onnx_enc["last_hidden_state"],
            torch_enc,
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.integration
@pytest.mark.integration_fast
class TestBartDecoderForward:
    """Compare BART decoder output between ONNX and PyTorch."""

    def test_decoder_prefill_logits_match(self):
        """Decoder prefill: decoder_input_ids + encoder output → logits."""
        config = _load_bart_config()
        pkg = _build_bart_package(config)
        torch_model = _load_torch_bart()

        # Encoder pass
        input_ids = np.array([[0, 31414, 232, 328, 740, 2]], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)
        encoder_hidden_states = torch_seq2seq_encoder_forward(
            torch_model, input_ids, attention_mask
        )

        # Decoder prefill
        decoder_input_ids = np.array([[2, 0, 31414]], dtype=np.int64)

        torch_logits, _ = torch_seq2seq_decoder_forward(
            torch_model,
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask,
        )

        decoder_session = OnnxModelSession(pkg["decoder"])
        feeds = _make_decoder_feeds(
            config,
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask,
        )
        onnx_out = decoder_session.run(feeds)
        decoder_session.close()

        assert_logits_close(
            onnx_out["logits"],
            torch_logits,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_decoder_decode_step_logits_match(self):
        """Decoder decode: single token + KV cache → logits."""
        config = _load_bart_config()
        pkg = _build_bart_package(config)
        torch_model = _load_torch_bart()

        # Encoder
        input_ids = np.array([[0, 31414, 232, 328, 740, 2]], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)
        encoder_hidden_states = torch_seq2seq_encoder_forward(
            torch_model, input_ids, attention_mask
        )

        # Prefill
        decoder_input_ids = np.array([[2, 0, 31414]], dtype=np.int64)
        torch_logits_1, torch_kv = torch_seq2seq_decoder_forward(
            torch_model,
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask,
        )

        decoder_session = OnnxModelSession(pkg["decoder"])
        feeds = _make_decoder_feeds(
            config,
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask,
        )
        onnx_out_1 = decoder_session.run(feeds)

        # Decode step
        next_token = np.argmax(torch_logits_1[:, -1, :], axis=-1, keepdims=True)
        decode_ids = next_token.astype(np.int64)

        torch_logits_2, _ = torch_seq2seq_decoder_forward(
            torch_model,
            decode_ids,
            encoder_hidden_states,
            attention_mask,
            past_key_values=torch_kv,
        )

        num_layers = getattr(config, "num_decoder_layers", config.num_hidden_layers)
        decode_feeds: dict[str, np.ndarray] = {
            "input_ids": decode_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": attention_mask,
        }
        for i in range(num_layers):
            decode_feeds[f"past_key_values.{i}.self.key"] = onnx_out_1[f"present.{i}.self.key"]
            decode_feeds[f"past_key_values.{i}.self.value"] = onnx_out_1[
                f"present.{i}.self.value"
            ]
            decode_feeds[f"past_key_values.{i}.cross.key"] = onnx_out_1[
                f"present.{i}.cross.key"
            ]
            decode_feeds[f"past_key_values.{i}.cross.value"] = onnx_out_1[
                f"present.{i}.cross.value"
            ]
        onnx_out_2 = decoder_session.run(decode_feeds)
        decoder_session.close()

        assert_logits_close(
            onnx_out_2["logits"],
            torch_logits_2,
            rtol=1e-2,
            atol=1e-2,
        )


# =========================================================================
# T5 integration tests (google-t5/t5-small, ~60M params)
# =========================================================================

_T5_MODEL_ID = "google-t5/t5-small"


def _load_t5_config():
    hf_config = transformers.AutoConfig.from_pretrained(_T5_MODEL_ID)
    return ArchitectureConfig.from_transformers(hf_config)


def _build_t5_package(config):
    from mobius.models.t5 import T5ForConditionalGeneration

    module = T5ForConditionalGeneration(config)
    pkg = build_from_module(module, config, task=Seq2SeqTask())

    state_dict = _download_weights(_T5_MODEL_ID)
    state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)
    return pkg


def _load_torch_t5():
    model = transformers.T5ForConditionalGeneration.from_pretrained(_T5_MODEL_ID)
    model.eval()
    return model


@pytest.mark.integration
class TestT5EncoderForward:
    """Compare T5 encoder output between ONNX and PyTorch."""

    def test_encoder_hidden_states_match(self):
        """Encoder forward: input_ids → encoder_hidden_states."""
        config = _load_t5_config()
        pkg = _build_t5_package(config)
        torch_model = _load_torch_t5()

        # "translate English to German: Hello world"
        input_ids = np.array([[13959, 1566, 12, 2968, 10, 8774, 296, 1]], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)

        torch_enc = torch_seq2seq_encoder_forward(torch_model, input_ids, attention_mask)

        encoder_session = OnnxModelSession(pkg["encoder"])
        onnx_enc = encoder_session.run(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        encoder_session.close()

        assert_logits_close(
            onnx_enc["last_hidden_state"],
            torch_enc,
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.integration
class TestT5DecoderForward:
    """Compare T5 decoder output between ONNX and PyTorch."""

    def test_decoder_prefill_logits_match(self):
        """Decoder prefill: decoder_input_ids + encoder output → logits."""
        config = _load_t5_config()
        pkg = _build_t5_package(config)
        torch_model = _load_torch_t5()

        input_ids = np.array([[13959, 1566, 12, 2968, 10, 8774, 296, 1]], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)
        encoder_hidden_states = torch_seq2seq_encoder_forward(
            torch_model, input_ids, attention_mask
        )

        decoder_input_ids = np.array([[0, 1, 2]], dtype=np.int64)

        torch_logits, _ = torch_seq2seq_decoder_forward(
            torch_model,
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask,
        )

        decoder_session = OnnxModelSession(pkg["decoder"])
        feeds = _make_decoder_feeds(
            config,
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask,
        )
        onnx_out = decoder_session.run(feeds)
        decoder_session.close()

        assert_logits_close(
            onnx_out["logits"],
            torch_logits,
            rtol=1e-3,
            atol=1e-3,
        )


# =========================================================================
# mT5 integration tests (google/mt5-small, ~300M params)
# Verifies gated FFN (wi_0 + wi_1) and multilingual tokenizer handling.
# =========================================================================

_MT5_MODEL_ID = "google/mt5-small"


def _load_mt5_config():
    hf_config = transformers.AutoConfig.from_pretrained(_MT5_MODEL_ID)
    return ArchitectureConfig.from_transformers(hf_config)


def _build_mt5_package(config):
    from mobius.models.t5 import T5ForConditionalGeneration

    module = T5ForConditionalGeneration(config)
    pkg = build_from_module(module, config, task=Seq2SeqTask())

    # mt5-small only has pytorch_model.bin (no safetensors), so load
    # weights via transformers and extract the state_dict directly.
    torch_model = transformers.MT5ForConditionalGeneration.from_pretrained(_MT5_MODEL_ID)
    state_dict = dict(torch_model.state_dict())
    state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)
    return pkg


def _load_torch_mt5():
    model = transformers.MT5ForConditionalGeneration.from_pretrained(_MT5_MODEL_ID)
    model.eval()
    return model


@pytest.mark.integration
class TestMT5EncoderForward:
    """Compare mT5 encoder output between ONNX and PyTorch."""

    def test_encoder_hidden_states_match(self):
        """Encoder forward: input_ids → encoder_hidden_states (gated FFN)."""
        config = _load_mt5_config()
        pkg = _build_mt5_package(config)
        torch_model = _load_torch_mt5()

        # mT5 uses SentencePiece; these are example token IDs
        input_ids = np.array([[259, 1738, 267, 3, 9, 1]], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)

        torch_enc = torch_seq2seq_encoder_forward(torch_model, input_ids, attention_mask)

        encoder_session = OnnxModelSession(pkg["encoder"])
        onnx_enc = encoder_session.run(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        encoder_session.close()

        assert_logits_close(
            onnx_enc["last_hidden_state"],
            torch_enc,
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.integration
class TestMT5DecoderForward:
    """Compare mT5 decoder output between ONNX and PyTorch."""

    def test_decoder_prefill_logits_match(self):
        """Decoder prefill: decoder_input_ids + encoder output → logits."""
        config = _load_mt5_config()
        pkg = _build_mt5_package(config)
        torch_model = _load_torch_mt5()

        input_ids = np.array([[259, 1738, 267, 3, 9, 1]], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)
        encoder_hidden_states = torch_seq2seq_encoder_forward(
            torch_model, input_ids, attention_mask
        )

        decoder_input_ids = np.array([[0, 259, 1738]], dtype=np.int64)

        torch_logits, _ = torch_seq2seq_decoder_forward(
            torch_model,
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask,
        )

        decoder_session = OnnxModelSession(pkg["decoder"])
        feeds = _make_decoder_feeds(
            config,
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask,
        )
        onnx_out = decoder_session.run(feeds)
        decoder_session.close()

        assert_logits_close(
            onnx_out["logits"],
            torch_logits,
            rtol=1e-3,
            atol=1e-3,
        )


# =========================================================================
# FLAN-T5 integration tests (google/flan-t5-small, ~77M params)
# Verifies gated FFN with model_type=t5 (same class, different config).
# =========================================================================

_FLAN_T5_MODEL_ID = "google/flan-t5-small"


def _load_flan_t5_config():
    hf_config = transformers.AutoConfig.from_pretrained(_FLAN_T5_MODEL_ID)
    return ArchitectureConfig.from_transformers(hf_config)


def _build_flan_t5_package(config):
    from mobius.models.t5 import T5ForConditionalGeneration

    module = T5ForConditionalGeneration(config)
    pkg = build_from_module(module, config, task=Seq2SeqTask())

    state_dict = _download_weights(_FLAN_T5_MODEL_ID)
    state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)
    return pkg


def _load_torch_flan_t5():
    model = transformers.T5ForConditionalGeneration.from_pretrained(_FLAN_T5_MODEL_ID)
    model.eval()
    return model


@pytest.mark.integration
class TestFlanT5EncoderForward:
    """Compare FLAN-T5 encoder output between ONNX and PyTorch."""

    def test_encoder_hidden_states_match(self):
        """Encoder forward: gated-gelu FFN variant of T5."""
        config = _load_flan_t5_config()
        pkg = _build_flan_t5_package(config)
        torch_model = _load_torch_flan_t5()

        # "translate English to German: Hello world"
        input_ids = np.array([[13959, 1566, 12, 2968, 10, 8774, 296, 1]], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)

        torch_enc = torch_seq2seq_encoder_forward(torch_model, input_ids, attention_mask)

        encoder_session = OnnxModelSession(pkg["encoder"])
        onnx_enc = encoder_session.run(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        encoder_session.close()

        assert_logits_close(
            onnx_enc["last_hidden_state"],
            torch_enc,
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.integration
class TestFlanT5DecoderForward:
    """Compare FLAN-T5 decoder output between ONNX and PyTorch."""

    def test_decoder_prefill_logits_match(self):
        """Decoder prefill: gated-gelu FFN with tie_word_embeddings=True."""
        config = _load_flan_t5_config()
        pkg = _build_flan_t5_package(config)
        torch_model = _load_torch_flan_t5()

        input_ids = np.array([[13959, 1566, 12, 2968, 10, 8774, 296, 1]], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)
        encoder_hidden_states = torch_seq2seq_encoder_forward(
            torch_model, input_ids, attention_mask
        )

        decoder_input_ids = np.array([[0, 1, 2]], dtype=np.int64)

        torch_logits, _ = torch_seq2seq_decoder_forward(
            torch_model,
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask,
        )

        decoder_session = OnnxModelSession(pkg["decoder"])
        feeds = _make_decoder_feeds(
            config,
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask,
        )
        onnx_out = decoder_session.run(feeds)
        decoder_session.close()

        assert_logits_close(
            onnx_out["logits"],
            torch_logits,
            rtol=1e-3,
            atol=1e-3,
        )
