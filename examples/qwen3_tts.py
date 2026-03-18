#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-TTS full multi-head text-to-speech with 4 ONNX models.

Builds four ONNX models (embedding, talker, code predictor, speaker encoder)
and runs the full multi-head TTS pipeline:

    text → embedding → talker (code group 0) → code predictor (groups 1-15)
    → 16-group audio codes → codec decoder → waveform

Architecture:
    1. **Embedding**: text_ids → text_embeds, codec_ids → codec_embeds
    2. **Talker**: inputs_embeds → logits (first code group) + last_hidden_state
    3. **Code Predictor**: talker_hidden + codec_ids → logits (code groups 1-15)
    4. **Speaker Encoder**: mel spectrogram → speaker embedding (voice cloning)

Prerequisites::

    pip install mobius-ai[transformers] sounddevice soundfile

Usage::

    # Interactive mode: type text, hear audio
    python examples/qwen3_tts.py

    # Use a specific model
    python examples/qwen3_tts.py --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

    # Single-shot mode
    python examples/qwen3_tts.py --text "Hello, world!"

    # Save the ONNX models
    python examples/qwen3_tts.py --save-to output/qwen3-tts/
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import transformers

from mobius import build
from mobius._model_package import ModelPackage
from mobius._testing.ort_inference import OnnxModelSession

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
CODEC_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
MAX_AUDIO_FRAMES = 2000


def build_or_load(
    model_id: str,
    *,
    dtype: str = "f32",
    cache_dir: str = CACHE_DIR,
) -> ModelPackage:
    """Build ONNX models, caching to *cache_dir* for fast reloads."""
    # Derive a cache subdirectory from the model name
    model_name = model_id.rsplit("/", 1)[-1]
    model_cache = os.path.join(cache_dir, model_name)

    if os.path.isdir(model_cache) and any(
        f.endswith(".onnx") for root, _, files in os.walk(model_cache) for f in files
    ):
        print(f"Loading cached ONNX models from {model_cache} ...")
        pkg = ModelPackage.load(model_cache)
        # Reconstruct config from HF (fast, only downloads config.json).
        # TTS models have nested talker_config; codec models don't.
        # Config reconstruction is best-effort — not all model types
        # support it (e.g. codec tokenizers use a custom config).
        from mobius._config_resolver import _config_from_hf, _try_load_config_json

        parent_config = _try_load_config_json(model_id)
        if parent_config is not None:
            hf_config = parent_config
            try:
                if hasattr(hf_config, "talker_config"):
                    hf_config = hf_config.talker_config
                    pkg.config = _config_from_hf(hf_config, parent_config=parent_config)
                else:
                    pkg.config = _config_from_hf(hf_config)
            except (AttributeError, TypeError):
                pass  # Config not reconstructable from cache
        return pkg

    # Build from scratch and save to cache
    print(f"Building ONNX models from {model_id!r} ...")
    pkg = build(model_id, dtype=dtype, load_weights=True)
    print(f"Caching to {model_cache} ...")
    pkg.save(model_cache)
    return pkg


# Codec special tokens (talker_config, not text vocab)
CODEC_BOS_ID = 2149  # Start of codec sequence
CODEC_EOS_ID = 2150  # End of codec sequence
CODEC_PAD_ID = 2148  # Padding token
CODEC_VOCAB_SIZE = 3072  # Talker vocabulary size (2048 codebook + special)

# Suppress tokens: block generation of special tokens except EOS.
# Only codebook tokens (0-2047) and EOS (2150) should be generated.
# All other special tokens (2048-3071) are only used in prefill.
SUPPRESS_TOKENS = [
    i for i in range(CODEC_VOCAB_SIZE - 1024, CODEC_VOCAB_SIZE) if i != CODEC_EOS_ID
]

# Sampling parameters (from generation_config.json)
TEMPERATURE = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.05
SUBTALKER_TEMPERATURE = 0.9
SUBTALKER_TOP_K = 50


def _make_dummy_cp_inputs(config) -> dict[str, np.ndarray]:
    """Create minimal dummy inputs for a single code predictor run.

    Used to extract the constant ``stacked_codec_embedding`` output.
    """
    tts = config.tts
    cp = tts.code_predictor if tts else None
    # inputs_embeds is in talker_hidden space (projected internally)
    talker_hidden = config.hidden_size
    cp_kv_heads = cp.num_key_value_heads if cp else 8
    cp_head_dim = cp.head_dim if cp else 128
    cp_num_layers = cp.num_hidden_layers if cp else 5

    feeds: dict[str, np.ndarray] = {
        "inputs_embeds": np.zeros((1, 1, talker_hidden), dtype=np.float32),
        "step_index": np.int64(0),
        "attention_mask": np.ones((1, 1), dtype=np.int64),
        "position_ids": np.array([[0]], dtype=np.int64),
    }
    for i in range(cp_num_layers):
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, cp_kv_heads, 0, cp_head_dim), dtype=np.float32
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, cp_kv_heads, 0, cp_head_dim), dtype=np.float32
        )
    return feeds


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def apply_suppress_tokens(
    logits: np.ndarray,
    suppress_tokens: list[int],
) -> np.ndarray:
    """Suppress tokens by setting their logits to -inf.

    Used to prevent the model from generating special tokens
    (BOS, PAD, think tokens, language IDs) during codec generation.
    Only codebook tokens (0-2047) and EOS should be generated.
    """
    logits = logits.copy()
    for token_id in suppress_tokens:
        if 0 <= token_id < len(logits):
            logits[token_id] = -np.inf
    return logits


def apply_repetition_penalty(
    logits: np.ndarray,
    generated_ids: list[int],
    penalty: float,
) -> np.ndarray:
    """Apply repetition penalty to logits for previously generated tokens.

    For each token in generated_ids:
      - If logits[token] > 0, divide by penalty
      - If logits[token] < 0, multiply by penalty
    This reduces the probability of repeating tokens.
    """
    if penalty == 1.0 or not generated_ids:  # noqa: RUF069
        return logits
    logits = logits.copy()
    unique_ids = set(generated_ids)
    for token_id in unique_ids:
        if 0 <= token_id < len(logits):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
    return logits


def sample_top_k(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 50,
) -> int:
    """Sample from top-k logits with temperature scaling.

    Args:
        logits: 1-D array of raw logits (vocab_size,).
        temperature: Temperature for softmax. Lower = more greedy.
        top_k: Number of top candidates to keep.

    Returns:
        Sampled token ID.
    """
    logits = logits.astype(np.float64)
    if temperature > 0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0 and top_k < len(logits):
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_k_indices] = logits[top_k_indices]
        logits = mask

    # Softmax
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# Audio playback
# ---------------------------------------------------------------------------


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    """Play audio through the default output device."""
    import sounddevice as sd

    sd.play(audio, samplerate=sample_rate)
    sd.wait()


# ---------------------------------------------------------------------------
# TTS Generation (4-model pipeline)
# ---------------------------------------------------------------------------


def generate_codes(
    sessions: dict[str, OnnxModelSession],
    tokenizer,
    text: str,
    config,
    *,
    cp_codec_weights: np.ndarray,
    speaker: str = "",
    language: str = "Auto",
    instruct: str = "",
    max_frames: int = MAX_AUDIO_FRAMES,
) -> list[list[int]]:
    r"""Generate multi-group audio codes using the 4-model TTS pipeline.

    Follows the HuggingFace Qwen3-TTS generation flow:
      1. **Prefill**: [instruct] + role prefix (3 text tokens)
         + codec think/language tags paired with tts_pad/tts_bos
         + first text token + codec_bos
      2. **Generation loop**: each step feeds (codec_sum + trailing_text)
         to the talker, samples code_0, then runs code predictor for
         codes 1-15.

    The text is formatted as:
      ``<|im_start|>assistant\\n{text}<|im_end|>\\n<|im_start|>assistant\\n``

    Args:
        sessions: Dict of ORT sessions for the 4 models.
        tokenizer: HuggingFace tokenizer.
        text: Text to synthesize.
        config: ArchitectureConfig from the model package.
        speaker: Speaker name (must match talker_config.spk_id keys).
            Empty string means no speaker embedding.
        language: Language tag (default: Auto for auto-detection).
        instruct: Voice instruction (e.g. 'Speak slowly.').
        max_frames: Maximum number of audio frames to generate.

    Returns:
        List of frames, each a list of ``num_code_groups`` codec IDs.
    """
    batch_size = 1
    hidden_size = config.hidden_size  # noqa: F841
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    tts = config.tts
    num_code_groups = tts.num_code_groups if tts else 16
    cp = tts.code_predictor if tts else None
    cp_num_layers = cp.num_hidden_layers if cp else 5
    cp_kv_heads = cp.num_key_value_heads if cp else 8
    cp_head_dim = cp.head_dim if cp else 128

    # --- Codec think/language token IDs (from talker_config) ---
    CODEC_NOTHINK_ID = 2155  # noqa: N806
    CODEC_THINK_ID = 2154  # noqa: N806
    CODEC_THINK_BOS_ID = 2156  # noqa: N806
    CODEC_THINK_EOS_ID = 2157  # noqa: N806
    # Language IDs from talker_config.codec_language_id
    LANGUAGE_IDS = {  # noqa: N806
        "chinese": 2055,
        "english": 2050,
        "german": 2053,
        "italian": 2070,
        "portuguese": 2071,
        "spanish": 2054,
        "japanese": 2058,
        "korean": 2064,
        "french": 2061,
        "russian": 2069,
    }
    # Speaker IDs from talker_config.spk_id (CustomVoice model)
    SPEAKER_IDS = {  # noqa: N806
        "serena": 3066,
        "vivian": 3065,
        "uncle_fu": 3010,
        "ryan": 3061,
        "aiden": 2861,
        "ono_anna": 2873,
        "sohee": 2864,
        "eric": 2875,
        "dylan": 2878,
    }
    # Text-side special tokens
    TTS_BOS_TOKEN_ID = 151672  # noqa: N806
    TTS_EOS_TOKEN_ID = 151673  # noqa: N806
    TTS_PAD_TOKEN_ID = 151671  # noqa: N806

    # --- Step 1: Tokenize text in HF format ---
    # Format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int64)
    # text_ids layout: [<|im_start|>, assistant, \n, ...text...,
    #                   <|im_end|>, \n, <|im_start|>, assistant, \n]
    # Tokens [0:3]   = role prefix
    # Token  [3]     = first text character
    # Tokens [4:-5]  = remaining text
    # Tokens [-5:]   = end role (discarded)

    # --- Step 2: Precompute embeddings ---
    # Text embeds for all tokens (projected through text_projection)
    embed_out = sessions["embedding"].run(
        {
            "text_ids": text_ids,
            "codec_ids": np.array([[0]], dtype=np.int64),  # dummy
        }
    )
    all_text_embeds = embed_out["text_embeds"]  # (1, text_len, hidden)

    # TTS special token embeds (bos, eos, pad — projected text embeddings)
    tts_special_ids = np.array(
        [[TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID]], dtype=np.int64
    )
    special_out = sessions["embedding"].run(
        {
            "text_ids": tts_special_ids,
            "codec_ids": np.array([[0]], dtype=np.int64),
        }
    )
    special_embeds = special_out["text_embeds"]  # (1, 3, hidden)
    tts_bos_embed = special_embeds[:, 0:1, :]  # (1, 1, hidden)
    tts_eos_embed = special_embeds[:, 1:2, :]
    tts_pad_embed = special_embeds[:, 2:3, :]

    # --- Step 2b: Optional instruct embedding ---
    # Instruct is formatted as: <|im_start|>user\n{instruct}<|im_end|>\n
    # Its projected text embedding is prepended before the role prefix.
    instruct_embed = None
    if instruct:
        instruct_prompt = f"<|im_start|>user\n{instruct}<|im_end|>\n"
        instruct_ids = tokenizer(instruct_prompt, return_tensors="np")["input_ids"].astype(
            np.int64
        )
        instruct_out = sessions["embedding"].run(
            {
                "text_ids": instruct_ids,
                "codec_ids": np.array([[0]], dtype=np.int64),
            }
        )
        instruct_embed = instruct_out["text_embeds"]  # (1, I, hidden)

    # Role prefix: first 3 text tokens (pure text, no codec overlay)
    role_embed = all_text_embeds[:, :3, :]  # (1, 3, hidden)

    # --- Step 3: Build codec prefill tokens ---
    # Language determines which think tokens to use:
    #   Auto: [nothink, think_bos, think_eos]
    #   Specified: [think, think_bos, language_id, think_eos]
    # Then optionally: [speaker_id]
    # Then: [pad, bos]
    lang_lower = language.lower()
    if lang_lower == "auto" or lang_lower not in LANGUAGE_IDS:
        codec_prefill_ids = [
            CODEC_NOTHINK_ID,
            CODEC_THINK_BOS_ID,
            CODEC_THINK_EOS_ID,
        ]
    else:
        codec_prefill_ids = [
            CODEC_THINK_ID,
            CODEC_THINK_BOS_ID,
            LANGUAGE_IDS[lang_lower],
            CODEC_THINK_EOS_ID,
        ]

    # Speaker embedding: inserted between think tokens and [pad, bos]
    speaker_embed = None
    spk_lower = speaker.lower() if speaker else ""
    if spk_lower and spk_lower in SPEAKER_IDS:
        spk_id = SPEAKER_IDS[spk_lower]
        spk_out = sessions["embedding"].run(
            {
                "text_ids": np.array([[0]], dtype=np.int64),
                "codec_ids": np.array([[spk_id]], dtype=np.int64),
            }
        )
        speaker_embed = spk_out["codec_embeds"]  # (1, 1, hidden)

    codec_prefill_ids.extend([CODEC_PAD_ID, CODEC_BOS_ID])

    # Embed codec prefill tokens
    codec_prefill_np = np.array([codec_prefill_ids], dtype=np.int64)
    codec_out = sessions["embedding"].run(
        {
            "text_ids": np.array([[0]], dtype=np.int64),
            "codec_ids": codec_prefill_np,
        }
    )
    codec_prefill_embeds = codec_out["codec_embeds"]  # (1, N, hidden)

    # Insert speaker embedding between think tokens and [pad, bos]
    if speaker_embed is not None:
        # Split: [think...think_eos] + [speaker] + [pad, bos]
        think_part = codec_prefill_embeds[:, :-2, :]
        tail_part = codec_prefill_embeds[:, -2:, :]
        codec_prefill_embeds = np.concatenate([think_part, speaker_embed, tail_part], axis=1)

    num_codec_prefill = codec_prefill_embeds.shape[1]

    # Text side of codec prefill: tts_pad * (N-2) + tts_bos
    # Paired with codec_prefill[:-1] (all except last which is bos)
    text_pad_repeated = np.tile(
        tts_pad_embed, (1, num_codec_prefill - 2, 1)
    )  # (1, N-2, hidden)
    text_side = np.concatenate([text_pad_repeated, tts_bos_embed], axis=1)  # (1, N-1, hidden)
    codec_side = codec_prefill_embeds[:, :-1, :]  # (1, N-1, hidden)
    codec_text_pairs = text_side + codec_side  # (1, N-1, hidden)

    # First text token + last codec token (bos)
    first_text_embed = all_text_embeds[:, 3:4, :]  # (1, 1, hidden)
    codec_bos_embed = codec_prefill_embeds[:, -1:, :]  # (1, 1, hidden)
    first_text_codec = first_text_embed + codec_bos_embed  # (1, 1, hidden)

    # Full prefill: [instruct] + role(3) + codec_text_pairs(N-1)
    #               + first_text_codec(1)
    prefill_parts = []
    if instruct_embed is not None:
        prefill_parts.append(instruct_embed)
    prefill_parts.extend([role_embed, codec_text_pairs, first_text_codec])
    prefill_embeds = np.concatenate(prefill_parts, axis=1)
    prefill_len = prefill_embeds.shape[1]

    # Trailing text: remaining text tokens + tts_eos
    # Tokens [4:-5] = remaining text after first character
    remaining_text = all_text_embeds[:, 4:-5, :]  # (1, M, hidden)
    trailing_text = np.concatenate([remaining_text, tts_eos_embed], axis=1)  # (1, M+1, hidden)
    trailing_len = trailing_text.shape[1]

    # --- Step 4: Prefill the talker with the full prefill sequence ---
    talker_past_kv: dict[str, np.ndarray] = {}
    for i in range(num_layers):
        talker_past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )
        talker_past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )

    # MRoPE 3D positions: all 3 dims get same positions for TTS
    prefill_pos = np.arange(prefill_len, dtype=np.int64).reshape(1, -1)
    prefill_position_ids = np.stack([prefill_pos, prefill_pos, prefill_pos])

    talker_out = sessions["talker"].run(
        {
            "inputs_embeds": prefill_embeds,
            "attention_mask": np.ones((batch_size, prefill_len), dtype=np.int64),
            "position_ids": prefill_position_ids,
            **talker_past_kv,
        }
    )

    # Sample first code_0 from prefill output (last position)
    talker_logits = talker_out["logits"]
    last_hidden = talker_out["last_hidden_state"]
    for i in range(num_layers):
        talker_past_kv[f"past_key_values.{i}.key"] = talker_out[f"present.{i}.key"]
        talker_past_kv[f"past_key_values.{i}.value"] = talker_out[f"present.{i}.value"]
    talker_seq = prefill_len

    # --- Step 5: Generation loop ---
    all_frames: list[list[int]] = []
    generated_code0s: list[int] = []

    for frame_idx in range(max_frames):
        # 5a. Sample code_0 from talker logits
        code0_logits = apply_suppress_tokens(talker_logits[0, -1, :], SUPPRESS_TOKENS)
        code0_logits = apply_repetition_penalty(
            code0_logits, generated_code0s, REPETITION_PENALTY
        )
        code_0 = sample_top_k(code0_logits, temperature=TEMPERATURE, top_k=TOP_K)

        if code_0 == CODEC_EOS_ID:
            break

        generated_code0s.append(code_0)

        # 5b. Run code predictor for remaining code groups (1 to 15)
        # HF flow: prefill with 2 tokens [talker_hidden, talker_embed(code_0)]
        # then generate 14 more tokens using CP_embed[step-1](code_i).
        codes = [code_0]
        cp_past_kv: dict[str, np.ndarray] = {}
        for i in range(cp_num_layers):
            cp_past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, cp_kv_heads, 0, cp_head_dim), dtype=np.float32
            )
            cp_past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, cp_kv_heads, 0, cp_head_dim), dtype=np.float32
            )

        cp_seq = 0
        for step in range(num_code_groups - 1):
            if step == 0:
                # Prefill: concat [talker_hidden, talker_codec_embed(code_0)]
                # as 2 tokens, matching HF's code predictor prefill.
                talker_h = last_hidden[:, -1:, :]  # (1, 1, hidden)
                # talker.codec_embedding(code_0) from embedding model
                code0_talker_embed = sessions["embedding"].run(
                    {
                        "text_ids": np.array([[0]], dtype=np.int64),
                        "codec_ids": np.array([[code_0]], dtype=np.int64),
                    }
                )["codec_embeds"]  # (1, 1, hidden)
                cp_inputs_embeds = np.concatenate(
                    [talker_h, code0_talker_embed], axis=1
                )  # (1, 2, hidden)
                cp_input_len = 2
            else:
                # Generation: CP_embed[step-1](code_i) — 1 token
                # cp_codec_weights: (15, vocab, hidden)
                code_i = codes[-1]
                embed = cp_codec_weights[step - 1, code_i, :]  # (hidden,)
                cp_inputs_embeds = embed.reshape(1, 1, -1)  # (1, 1, hidden)
                cp_input_len = 1

            cp_total_seq = cp_seq + cp_input_len
            cp_pos = np.arange(cp_seq, cp_total_seq, dtype=np.int64).reshape(1, -1)

            cp_out = sessions["code_predictor"].run(
                {
                    "inputs_embeds": cp_inputs_embeds.astype(np.float32),
                    "step_index": np.int64(step),
                    "attention_mask": np.ones((batch_size, cp_total_seq), dtype=np.int64),
                    "position_ids": cp_pos,
                    **cp_past_kv,
                }
            )

            next_code = sample_top_k(
                cp_out["logits"][0, -1, :],
                temperature=SUBTALKER_TEMPERATURE,
                top_k=SUBTALKER_TOP_K,
            )
            codes.append(next_code)

            for i in range(cp_num_layers):
                cp_past_kv[f"past_key_values.{i}.key"] = cp_out[f"present.{i}.key"]
                cp_past_kv[f"past_key_values.{i}.value"] = cp_out[f"present.{i}.value"]
            cp_seq = cp_total_seq

        all_frames.append(codes)

        if frame_idx % 20 == 0:
            print(
                f"  Generated {frame_idx} frames...",
                end="\r",
                flush=True,
            )

        # 5c. Prepare next talker input: codec_sum + trailing_text
        # HF codec_sum = talker.embed(code_0) + Σ CP.embed[i](code_{i+1})
        # Code 0 embedding from talker's codec_embedding
        code0_embed = sessions["embedding"].run(
            {
                "text_ids": np.array([[0]], dtype=np.int64),
                "codec_ids": np.array([[code_0]], dtype=np.int64),
            }
        )["codec_embeds"]  # (1, 1, hidden)

        # Sum: talker.embed(code_0) + CP.embed[i](code_{i+1}) for i=0..14
        codec_sum = code0_embed
        for i in range(num_code_groups - 1):
            # codes[i+1] is the code generated at code predictor step i
            cp_embed = cp_codec_weights[i, codes[i + 1], :]  # (hidden,)
            codec_sum = codec_sum + cp_embed.reshape(1, 1, -1)

        # Add trailing text (one per step) or tts_pad if text exhausted
        if frame_idx < trailing_len:
            text_embed = trailing_text[:, frame_idx : frame_idx + 1, :]
        else:
            text_embed = tts_pad_embed
        inputs_embeds = codec_sum + text_embed

        # Run talker for next step
        total_seq = talker_seq + 1
        pos = np.array([[talker_seq]], dtype=np.int64)
        position_ids = np.stack([pos, pos, pos])

        talker_out = sessions["talker"].run(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": np.ones((batch_size, total_seq), dtype=np.int64),
                "position_ids": position_ids,
                **talker_past_kv,
            }
        )

        talker_logits = talker_out["logits"]
        last_hidden = talker_out["last_hidden_state"]
        for i in range(num_layers):
            talker_past_kv[f"past_key_values.{i}.key"] = talker_out[f"present.{i}.key"]
            talker_past_kv[f"past_key_values.{i}.value"] = talker_out[f"present.{i}.value"]
        talker_seq = total_seq

    print(f"  Generated {len(all_frames)} audio frames ({num_code_groups} codes each).    ")
    return all_frames


def decode_codes_to_audio(
    codec_session: OnnxModelSession,
    frames: list[list[int]],
    sample_rate: int = 24000,
) -> tuple[np.ndarray, int]:
    """Decode multi-group audio codes to waveform using the ONNX codec.

    Args:
        codec_session: ORT session for the ONNX codec decoder.
        frames: List of frames, each containing 16 codec IDs.
        sample_rate: Output sample rate (default: 24000).

    Returns:
        (waveform, sample_rate)
    """
    # Reshape to (1, num_code_groups, num_frames) for codec decoder
    codes_array = np.array(frames).T  # (num_groups, num_frames)
    codes = codes_array[np.newaxis, :, :].astype(np.int64)
    out = codec_session.run({"codes": codes})
    waveform = out["waveform"].squeeze()
    return waveform, sample_rate


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

BANNER = """\
╔═══════════════════════════════════════════════════╗
║         🔊 Qwen3-TTS Multi-Head CLI               ║
╠═══════════════════════════════════════════════════╣
║  Type text and press Enter to generate speech.    ║
║  Commands:                                        ║
║    /speaker <name>  — change speaker              ║
║    /language <lang> — change language             ║
║    /instruct <text> — set voice instruction       ║
║    /quit            — exit                        ║
╚═══════════════════════════════════════════════════╝"""


def interactive_loop(
    sessions: dict[str, OnnxModelSession],
    tokenizer,
    config,
    *,
    cp_codec_weights: np.ndarray,
    codec_session: OnnxModelSession,
    speaker: str,
    language: str,
    instruct: str,
    save_dir: str | None,
):
    """Run the interactive TTS loop."""
    print(BANNER)
    print(f"  Speaker: {speaker} | Language: {language}")
    if instruct:
        print(f"  Instruct: {instruct}")
    print()

    utterance_count = 0

    while True:
        try:
            text = input("💬 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not text:
            continue

        # Handle commands
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "/quit":
                print("Goodbye! 👋")
                break
            elif cmd == "/speaker" and len(parts) > 1:
                speaker = parts[1]
                print(f"  Speaker set to: {speaker}")
            elif cmd == "/language" and len(parts) > 1:
                language = parts[1]
                print(f"  Language set to: {language}")
            elif cmd == "/instruct" and len(parts) > 1:
                instruct = parts[1]
                print(f"  Instruct set to: {instruct}")
            else:
                print("  Unknown command. Type /quit to exit.")
            continue

        # Generate audio codes
        print("  🔄 Generating...", flush=True)
        try:
            frames = generate_codes(
                sessions,
                tokenizer,
                text,
                config,
                cp_codec_weights=cp_codec_weights,
                speaker=speaker,
                language=language,
                instruct=instruct,
            )
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

        if not frames:
            print("  ⚠ No audio frames generated.")
            continue

        # Decode to audio and play
        audio, sr = decode_codes_to_audio(codec_session, frames)
        duration = len(audio) / sr
        print(f"  ✅ Playing {duration:.1f}s of audio...")
        if save_dir:
            import soundfile as sf

            utterance_count += 1
            path = os.path.join(save_dir, f"utterance_{utterance_count:03d}.wav")
            sf.write(path, audio, sr)
            print(f"  📁 Saved to {path}")
        play_audio(audio, sr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS multi-head text-to-speech with ONNX.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--speaker",
        default="",
        help="Speaker name (e.g. Vivian, for CustomVoice models).",
    )
    parser.add_argument(
        "--language",
        default="Auto",
        help="Language (default: %(default)s for auto-detection).",
    )
    parser.add_argument(
        "--instruct",
        default="",
        help="Voice instruction (e.g. 'Speak slowly.').",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Save generated audio to directory.",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help="Save the ONNX models to DIR and exit.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Text to synthesize (single-shot mode, no interactive loop).",
    )
    args = parser.parse_args()

    if args.save_to:
        # Build without caching, save to specified dir, and exit
        print(f"Building ONNX models from {args.model!r} ...")
        pkg = build(args.model, dtype="f32", load_weights=True)
        pkg.save(args.save_to)
        print(f"Saved to {args.save_to}")
        return

    # Build (or load from cache)
    pkg = build_or_load(args.model)
    config = pkg.config

    # Create ORT sessions for each model
    print("Creating inference sessions ...")
    sessions = {name: OnnxModelSession(model) for name, model in pkg.items()}

    # Extract CP codec embedding weights for generation loop lookups.
    # The stacked_codec_embedding is exposed as a graph output via Identity;
    # we only need the weight once (it's constant).
    cp_codec_weights = sessions["code_predictor"].run(_make_dummy_cp_inputs(config))[
        "codec_embeddings"
    ]  # (15, vocab, talker_hidden)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load codec model for audio decoding
    codec_pkg = build_or_load(CODEC_ID)
    codec_session = OnnxModelSession(codec_pkg["decoder"])
    print("Codec decoder loaded.")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    print("Ready.\n")

    if args.text:
        # Single-shot mode: generate one utterance and exit
        print(f"  🔄 Generating '{args.text}' ...")
        frames = generate_codes(
            sessions,
            tokenizer,
            args.text,
            config,
            cp_codec_weights=cp_codec_weights,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
        )
        if not frames:
            print("  ⚠ No audio frames generated.")
            return
        audio, sr = decode_codes_to_audio(codec_session, frames)
        duration = len(audio) / sr
        print(f"  ✅ {duration:.1f}s of audio ({len(frames)} frames)")
        if args.save_dir:
            import soundfile as sf

            path = os.path.join(args.save_dir, "output.wav")
            sf.write(path, audio, sr)
            print(f"  📁 Saved to {path}")
        play_audio(audio, sr)
        return

    interactive_loop(
        sessions,
        tokenizer,
        config,
        cp_codec_weights=cp_codec_weights,
        codec_session=codec_session,
        speaker=args.speaker,
        language=args.language,
        instruct=args.instruct,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
