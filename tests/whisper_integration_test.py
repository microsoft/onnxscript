# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: Whisper encoder-decoder speech-to-text.

Verifies that the Whisper ONNX encoder and decoder produce the same outputs
as the HuggingFace PyTorch reference model.

Run with::

    pytest tests/whisper_integration_test.py -m integration -sv
"""

from __future__ import annotations

import pathlib

import librosa
import numpy as np
import pytest
import transformers

from mobius import build_from_module
from mobius._configs import WhisperConfig
from mobius._testing.comparison import assert_logits_close
from mobius._testing.ort_inference import OnnxModelSession
from mobius._testing.torch_reference import (
    torch_whisper_decoder_forward,
    torch_whisper_encoder_forward,
)
from mobius._weight_loading import _download_weights
from mobius.models.whisper import WhisperForConditionalGeneration
from mobius.tasks import SpeechToTextTask

_MODEL_ID = "openai/whisper-tiny"


def _load_whisper_config():
    """Load WhisperConfig from HuggingFace (full model, no layer overrides)."""
    hf_config = transformers.AutoConfig.from_pretrained(_MODEL_ID)
    return WhisperConfig.from_transformers(hf_config)


def _build_whisper_package(config):
    """Build Whisper ONNX package (encoder + decoder) with weights."""
    module = WhisperForConditionalGeneration(config)
    pkg = build_from_module(module, config, task=SpeechToTextTask())

    state_dict = _download_weights(_MODEL_ID)
    state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)

    return pkg


def _load_torch_whisper():
    """Load full HuggingFace Whisper-tiny model."""
    processor = transformers.AutoProcessor.from_pretrained(_MODEL_ID)
    model = transformers.WhisperForConditionalGeneration.from_pretrained(_MODEL_ID)
    model.eval()

    return model, processor


@pytest.mark.integration
@pytest.mark.integration_fast
class TestWhisperEncoderForward:
    """Compare Whisper encoder output between ONNX and PyTorch."""

    def test_encoder_hidden_states_match(self):
        """Encoder forward: random mel features → encoder_hidden_states."""
        config = _load_whisper_config()
        pkg = _build_whisper_package(config)
        torch_model, _processor = _load_torch_whisper()

        # Random mel features: [batch=1, num_mel_bins, max_source_positions * 2]
        # HF Whisper encoder expects fixed length = max_source_positions * 2
        audio_len = config.max_source_positions * 2
        rng = np.random.default_rng(42)
        input_features = rng.standard_normal((1, config.num_mel_bins, audio_len)).astype(
            np.float32
        )

        # PyTorch encoder
        torch_enc = torch_whisper_encoder_forward(torch_model, input_features)

        # ONNX encoder
        encoder_session = OnnxModelSession(pkg["encoder"])
        onnx_enc = encoder_session.run({"input_features": input_features})
        encoder_session.close()

        assert_logits_close(
            onnx_enc["encoder_hidden_states"],
            torch_enc,
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.integration
@pytest.mark.integration_fast
class TestWhisperDecoderForward:
    """Compare Whisper decoder output between ONNX and PyTorch."""

    def test_decoder_prefill_logits_match(self):
        """Decoder prefill: decoder_input_ids + encoder output → logits."""
        config = _load_whisper_config()
        pkg = _build_whisper_package(config)
        torch_model, _processor = _load_torch_whisper()

        # Get encoder hidden states from PyTorch (shared reference)
        audio_len = config.max_source_positions * 2
        rng = np.random.default_rng(42)
        input_features = rng.standard_normal((1, config.num_mel_bins, audio_len)).astype(
            np.float32
        )
        encoder_hidden_states = torch_whisper_encoder_forward(torch_model, input_features)

        # Decoder inputs
        decoder_input_ids = np.array([[50258, 50259, 50360]], dtype=np.int64)
        seq_len = decoder_input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

        # PyTorch decoder
        torch_logits, _ = torch_whisper_decoder_forward(
            torch_model,
            decoder_input_ids,
            encoder_hidden_states,
        )

        # ONNX decoder
        decoder_session = OnnxModelSession(pkg["decoder"])
        feeds: dict[str, np.ndarray] = {
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
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
        onnx_out = decoder_session.run(feeds)
        decoder_session.close()

        assert_logits_close(
            onnx_out["logits"],
            torch_logits,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_decoder_decode_step_logits_match(self):
        """Decoder decode step: single token + KV cache → logits."""
        config = _load_whisper_config()
        pkg = _build_whisper_package(config)
        torch_model, _processor = _load_torch_whisper()

        # Encoder hidden states
        audio_len = config.max_source_positions * 2
        rng = np.random.default_rng(42)
        input_features = rng.standard_normal((1, config.num_mel_bins, audio_len)).astype(
            np.float32
        )
        encoder_hidden_states = torch_whisper_encoder_forward(torch_model, input_features)

        # Prefill
        decoder_input_ids = np.array([[50258, 50259, 50360]], dtype=np.int64)
        seq_len = decoder_input_ids.shape[1]

        torch_logits_1, torch_kv = torch_whisper_decoder_forward(
            torch_model,
            decoder_input_ids,
            encoder_hidden_states,
        )

        decoder_session = OnnxModelSession(pkg["decoder"])
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]
        feeds: dict[str, np.ndarray] = {
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
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
        onnx_out_1 = decoder_session.run(feeds)

        # Decode step
        next_token = np.argmax(torch_logits_1[:, -1, :], axis=-1, keepdims=True)
        decode_ids = next_token.astype(np.int64)
        decode_pos = np.array([[seq_len]], dtype=np.int64)

        torch_logits_2, _ = torch_whisper_decoder_forward(
            torch_model,
            decode_ids,
            encoder_hidden_states,
            past_key_values=torch_kv,
        )

        decode_feeds: dict[str, np.ndarray] = {
            "decoder_input_ids": decode_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "position_ids": decode_pos,
        }
        for i in range(config.num_hidden_layers):
            decode_feeds[f"past_key_values.{i}.key"] = onnx_out_1[f"present.{i}.key"]
            decode_feeds[f"past_key_values.{i}.value"] = onnx_out_1[f"present.{i}.value"]
        onnx_out_2 = decoder_session.run(decode_feeds)
        decoder_session.close()

        assert_logits_close(
            onnx_out_2["logits"],
            torch_logits_2,
            rtol=1e-2,
            atol=1e-2,
        )


_TESTDATA_DIR = pathlib.Path(__file__).parent.parent / "testdata"


@pytest.mark.integration
@pytest.mark.integration_fast
class TestWhisperEndToEnd:
    """End-to-end generation test using a real audio file."""

    def test_greedy_generation_matches_hf(self):
        """Greedy decode from audio file produces same tokens as HF."""
        config = _load_whisper_config()
        pkg = _build_whisper_package(config)
        torch_model, processor = _load_torch_whisper()

        audio_path = _TESTDATA_DIR / "652-129742-0006.flac"
        audio, _sr = librosa.load(str(audio_path), sr=16000)

        # Process through HF processor to get mel features
        inputs = processor(audio, sampling_rate=16000, return_tensors="np")
        input_features = inputs.input_features  # [1, 80, 3000]

        # Shared encoder output (from torch, for identical starting point)
        torch_enc = torch_whisper_encoder_forward(torch_model, input_features)

        # --- HF manual greedy decode ---
        decoder_start_token_id = config.decoder_start_token_id or 50258
        hf_generated = [decoder_start_token_id]
        hf_past_kv = None
        for step in range(20):
            ids = (
                np.array([[hf_generated[-1]]], dtype=np.int64)
                if step > 0
                else np.array([hf_generated], dtype=np.int64)
            )
            hf_logits, hf_past_kv = torch_whisper_decoder_forward(
                torch_model,
                ids,
                torch_enc,
                past_key_values=hf_past_kv,
            )
            next_tok = int(np.argmax(hf_logits[0, -1, :]))
            hf_generated.append(next_tok)

        # --- ONNX greedy decode ---
        encoder_session = OnnxModelSession(pkg["encoder"])
        enc_out = encoder_session.run({"input_features": input_features})
        encoder_hidden_states = enc_out["encoder_hidden_states"]
        encoder_session.close()

        decoder_session = OnnxModelSession(pkg["decoder"])
        onnx_generated = [decoder_start_token_id]
        past_kv: dict[str, np.ndarray] = {}
        for i in range(config.num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )

        for step in range(20):
            if step == 0:
                ids = np.array([onnx_generated], dtype=np.int64)
                pos = np.array([[0]], dtype=np.int64)
            else:
                ids = np.array([[onnx_generated[-1]]], dtype=np.int64)
                pos = np.array([[len(onnx_generated) - 1]], dtype=np.int64)

            feeds = {
                "decoder_input_ids": ids,
                "encoder_hidden_states": encoder_hidden_states,
                "position_ids": pos,
                **past_kv,
            }
            out = decoder_session.run(feeds)
            next_token = int(np.argmax(out["logits"][0, -1, :]))
            onnx_generated.append(next_token)

            for i in range(config.num_hidden_layers):
                past_kv[f"past_key_values.{i}.key"] = out[f"present.{i}.key"]
                past_kv[f"past_key_values.{i}.value"] = out[f"present.{i}.value"]

        decoder_session.close()

        # Decode tokens to text for readable output
        tokenizer = processor.tokenizer
        onnx_text = tokenizer.decode(onnx_generated, skip_special_tokens=True)
        hf_text = tokenizer.decode(hf_generated, skip_special_tokens=True)
        print(f"\nONNX text: {onnx_text}")
        print(f"HF text:   {hf_text}")
        print(f"ONNX tokens: {onnx_generated}")
        print(f"HF tokens:   {hf_generated}")

        assert onnx_generated == hf_generated, (
            f"Token mismatch!\nONNX text: {onnx_text}\nHF text:   {hf_text}"
        )
