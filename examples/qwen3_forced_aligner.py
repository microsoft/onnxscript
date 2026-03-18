#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-ForcedAligner — speech-text alignment with ONNX models.

Builds three ONNX models (audio encoder, embedding, decoder) and
runs forced alignment to produce per-token timestamps.

The ForcedAligner uses the same architecture as Qwen3-ASR but with a
classification head (``classify_num=5000``) instead of a language model
head. The classifier output is interpreted as alignment probabilities.

Prerequisites::

    pip install mobius-ai[transformers] torchaudio

Usage::

    # Align mic recording with known text
    python examples/qwen3_forced_aligner.py --text "Hello, how are you?"

    # Align an audio file
    python examples/qwen3_forced_aligner.py --audio speech.wav --text "The quick brown fox."

    # Save ONNX models to disk
    python examples/qwen3_forced_aligner.py --save-to output/aligner/
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import transformers

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession

# Reuse audio helpers from the ASR example
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from qwen3_asr import (
    ASSISTANT_ID,
    AUDIO_END_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    AUDIO_TOKEN_ID,
    IM_END,
    IM_START,
    NEWLINE_ID,
    SAMPLE_RATE,
    SYSTEM_ID,
    USER_ID,
    compute_mel_spectrogram,
    load_audio_file,
    record_until_enter,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"

# Token IDs (from config)
TIMESTAMP_TOKEN_ID = 151705
# Each class index maps to 80 ms of audio
TIMESTAMP_SEGMENT_TIME_MS = 80.0


# ---------------------------------------------------------------------------
# Alignment inference
# ---------------------------------------------------------------------------


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.mmm."""
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:06.3f}"


def tokenize_text(text: str) -> list[str]:
    """Split text into words for alignment.

    CJK characters are treated as individual words. Latin words
    are split on whitespace/punctuation and stripped of non-alphanumeric
    characters.
    """
    import unicodedata

    def is_cjk(ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0xF900 <= code <= 0xFAFF
        )

    def is_kept(ch: str) -> bool:
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        return cat.startswith(("L", "N"))

    tokens: list[str] = []
    current: list[str] = []

    def flush():
        nonlocal current
        if current:
            cleaned = "".join(current)
            if cleaned:
                tokens.append(cleaned)
            current = []

    for ch in text:
        if is_cjk(ch):
            flush()
            tokens.append(ch)
        elif is_kept(ch):
            current.append(ch)
        else:
            flush()

    flush()
    return tokens


def run_alignment(
    sessions: dict[str, OnnxModelSession],
    tokenizer,
    audio: np.ndarray,
    text: str,
    config,
) -> list[dict]:
    """Run forced alignment: audio + text → timestamped words.

    The aligner uses ``<timestamp>`` tokens interleaved between words.
    Each word gets two ``<timestamp>`` markers (start and end).  The
    classifier output at those positions is an index into a time grid
    where each class maps to ``TIMESTAMP_SEGMENT_TIME_MS`` milliseconds.

    Returns a list of dicts with 'text', 'start_time', 'end_time' keys
    (times in seconds).
    """
    batch_size = 1

    # Step 1: Tokenize text into words and build alignment prompt
    word_list = tokenize_text(text)
    # Format: word0<timestamp><timestamp>word1<timestamp><timestamp>...
    # Audio tokens (<|audio_start|>, <|audio_pad|>*N, <|audio_end|>)
    # are added explicitly in prompt_ids below — not via the tokenizer.
    alignment_text = "<timestamp><timestamp>".join(word_list)
    alignment_text += "<timestamp><timestamp>"

    # Step 2: Compute mel spectrogram and run audio encoder
    mel = compute_mel_spectrogram(audio)
    audio_out = sessions["audio_encoder"].run({"input_features": mel})
    audio_features = audio_out["audio_features"]
    num_audio_tokens = audio_features.shape[1]
    audio_features_2d = audio_features.reshape(-1, audio_features.shape[-1])

    # Step 3: Build full prompt IDs
    # system header + user header + <audio_start> + audio_pad*N + <audio_end>
    # + alignment_text_tokens + user footer + assistant header
    text_ids = tokenizer.encode(alignment_text, add_special_tokens=False)
    prompt_ids = (
        [
            IM_START,
            SYSTEM_ID,
            NEWLINE_ID,
            IM_END,
            NEWLINE_ID,
            IM_START,
            USER_ID,
            NEWLINE_ID,
            AUDIO_START_TOKEN_ID,
        ]
        + [AUDIO_TOKEN_ID] * num_audio_tokens
        + [AUDIO_END_TOKEN_ID]
        + text_ids
        + [IM_END, NEWLINE_ID, IM_START, ASSISTANT_ID, NEWLINE_ID]
    )
    input_ids = np.array([prompt_ids], dtype=np.int64)

    # Step 4: Run embedding model
    embed_out = sessions["embedding"].run(
        {"input_ids": input_ids, "audio_features": audio_features_2d}
    )
    inputs_embeds = embed_out["inputs_embeds"]

    # Step 5: Run decoder (single forward pass)
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    past_kv = {}
    for i in range(num_layers):
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )

    seq_len = inputs_embeds.shape[1]
    pos = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]
    position_ids = np.stack([pos, pos, pos])  # MRoPE: (3, 1, seq_len)

    out = sessions["decoder"].run(
        {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": position_ids,
            **past_kv,
        }
    )

    # Step 6: Extract timestamps from <timestamp> token positions
    logits = out["logits"]  # (1, seq_len, classify_num)
    output_ids = np.argmax(logits[0], axis=-1)  # (seq_len,)

    # Find positions where input_id == TIMESTAMP_TOKEN_ID
    timestamp_mask = input_ids[0] == TIMESTAMP_TOKEN_ID
    timestamp_class_ids = output_ids[timestamp_mask]

    # Convert class indices to milliseconds, then seconds
    timestamp_ms = timestamp_class_ids.astype(np.float64) * TIMESTAMP_SEGMENT_TIME_MS

    # Each word gets 2 timestamps: start_time, end_time
    segments = []
    for i, word in enumerate(word_list):
        start_ms = float(timestamp_ms[i * 2]) if i * 2 < len(timestamp_ms) else 0.0
        end_ms = float(timestamp_ms[i * 2 + 1]) if i * 2 + 1 < len(timestamp_ms) else 0.0
        segments.append(
            {
                "text": word,
                "start_time": round(start_ms / 1000.0, 3),
                "end_time": round(end_ms / 1000.0, 3),
            }
        )

    return segments


def print_alignment(segments: list[dict]) -> None:
    """Pretty-print alignment results."""
    print("\n┌─────────────────┬─────────────────┬────────────────────┐")
    print("│  Start          │  End            │  Text              │")
    print("├─────────────────┼─────────────────┼────────────────────┤")
    for seg in segments:
        st = format_timestamp(seg["start_time"])
        et = format_timestamp(seg["end_time"])
        txt = seg["text"]
        print(f"│  {st:<14} │  {et:<14} │  {txt:<17} │")
    print("└─────────────────┴─────────────────┴────────────────────┘")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ForcedAligner: speech-text alignment with ONNX.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Reference text to align against.",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to audio file. If omitted, records from mic.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help="Save ONNX models to DIR and exit.",
    )
    args = parser.parse_args()

    # Build 3 ONNX models (auto-detected from model_type)
    print(f"Building ONNX models from {args.model!r} ...")
    pkg = build(args.model, dtype="f32", load_weights=not args.save_to)
    config = pkg.config

    if args.save_to:
        pkg.save(args.save_to, check_weights=False)
        print(f"Saved to {args.save_to}")
        return

    print("Creating inference sessions ...")
    sessions = {name: OnnxModelSession(model) for name, model in pkg.items()}

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Ready.\n")

    if args.audio:
        audio = load_audio_file(args.audio)
        print(f"Audio: {len(audio) / SAMPLE_RATE:.1f}s")
    else:
        audio = record_until_enter()
        if len(audio) < SAMPLE_RATE * 0.3:
            print("No audio recorded.")
            return

    print(f"Text: {args.text!r}")
    segments = run_alignment(sessions, tokenizer, audio, args.text, config)
    print_alignment(segments)


if __name__ == "__main__":
    main()
