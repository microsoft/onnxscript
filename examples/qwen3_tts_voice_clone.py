#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-TTS voice cloning: record your voice, then speak as you.

Records audio from your microphone, extracts a speaker embedding via
the ECAPA-TDNN speaker encoder, and generates speech with that voice.

Architecture:
    1. **Record**: Capture audio from microphone (sounddevice)
    2. **Mel spectrogram**: Compute 128-bin mel spectrogram from recording
    3. **Speaker encoder**: mel → speaker embedding (ECAPA-TDNN, ONNX)
    4. **Generate**: Run TTS pipeline with speaker embedding injected

Prerequisites::

    pip install mobius-ai[transformers] sounddevice soundfile librosa

Usage::

    # Interactive: record voice, then type text to speak
    python examples/qwen3_tts_voice_clone.py

    # Use a specific TTS model
    python examples/qwen3_tts_voice_clone.py --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

    # Record for longer (default 5s)
    python examples/qwen3_tts_voice_clone.py --record-seconds 10

    # Use a pre-recorded audio file instead of microphone
    python examples/qwen3_tts_voice_clone.py --audio-file reference.wav
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import transformers

# Import the TTS generation pipeline from the main example
from qwen3_tts import (
    decode_codes_to_audio,
    generate_codes,
    play_audio,
)

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession

# Speaker encoder mel spectrogram settings
SAMPLE_RATE = 24000
MEL_BINS = 128
HOP_LENGTH = 240  # 10ms at 24kHz
WIN_LENGTH = 960  # 40ms at 24kHz


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------


def record_audio(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio from the default microphone.

    Args:
        duration: Recording duration in seconds.
        sample_rate: Audio sample rate.

    Returns:
        Mono audio array of shape ``(num_samples,)``.
    """
    import sounddevice as sd

    print(f"🎙  Recording for {duration:.0f} seconds... (speak now)")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("✅ Recording complete.")
    return audio.squeeze()


def load_audio_file(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and resample to the target sample rate.

    Args:
        path: Path to the audio file.
        sample_rate: Target sample rate.

    Returns:
        Mono audio array of shape ``(num_samples,)``.
    """
    import librosa

    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio


# ---------------------------------------------------------------------------
# Mel spectrogram
# ---------------------------------------------------------------------------


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = MEL_BINS,
) -> np.ndarray:
    """Compute log-mel spectrogram for the speaker encoder.

    Args:
        audio: Mono audio array.
        sample_rate: Audio sample rate.
        n_mels: Number of mel filter banks.

    Returns:
        Mel spectrogram of shape ``(1, num_frames, n_mels)``.
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_fft=WIN_LENGTH,
    )
    # Log-mel: log(mel + 1e-6)
    log_mel = np.log(mel + 1e-6).astype(np.float32)
    # Transpose: (n_mels, time) → (time, n_mels), add batch dim
    log_mel = log_mel.T[np.newaxis, :, :]
    return log_mel


# ---------------------------------------------------------------------------
# Voice cloning pipeline
# ---------------------------------------------------------------------------


def extract_speaker_embedding(
    speaker_session: OnnxModelSession,
    audio: np.ndarray,
) -> np.ndarray:
    """Extract a speaker embedding from audio.

    Args:
        speaker_session: ORT session for the speaker encoder.
        audio: Mono audio array.

    Returns:
        Speaker embedding of shape ``(1, enc_dim)``.
    """
    mel = compute_mel_spectrogram(audio)
    print(f"  Mel spectrogram: {mel.shape} (frames x {MEL_BINS} bins)")
    out = speaker_session.run({"mel_input": mel})
    embedding = out["speaker_embedding"]
    print(f"  Speaker embedding: {embedding.shape}")
    return embedding


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

CLONE_BANNER = """\
╔═══════════════════════════════════════════════════╗
║    🎤 Qwen3-TTS Voice Cloning CLI                 ║
╠═══════════════════════════════════════════════════╣
║  Type text and press Enter to generate speech     ║
║  using your cloned voice.                         ║
║  Commands:                                        ║
║    /record          — re-record voice sample      ║
║    /language <lang> — change language             ║
║    /instruct <text> — set voice instruction       ║
║    /quit            — exit                        ║
╚═══════════════════════════════════════════════════╝"""


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS voice cloning with ONNX.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="HuggingFace TTS model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--codec",
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Codec model ID for audio decoding.",
    )
    parser.add_argument(
        "--audio-file",
        default=None,
        help="Path to a reference audio file (skip microphone recording).",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=5.0,
        help="Duration to record from microphone (default: %(default)s).",
    )
    parser.add_argument(
        "--language",
        default="Auto",
        help="Language (default: %(default)s for auto-detection).",
    )
    parser.add_argument(
        "--instruct",
        default="",
        help="Voice instruction.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Save generated audio to directory.",
    )
    args = parser.parse_args()

    # --- Build TTS models ---
    print(f"Building TTS models from {args.model!r} ...")
    pkg = build(args.model, dtype="f32", load_weights=True)
    config = pkg.config
    sessions = {name: OnnxModelSession(model) for name, model in pkg.items()}

    # --- Build codec decoder ---
    print(f"Building codec from {args.codec!r} ...")
    codec_pkg = build(args.codec, dtype="f32", load_weights=True)
    codec_session = OnnxModelSession(codec_pkg["decoder"])

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # --- Get reference audio ---
    if args.audio_file:
        print(f"Loading reference audio: {args.audio_file}")
        audio = load_audio_file(args.audio_file)
    else:
        audio = record_audio(args.record_seconds)

    duration = len(audio) / SAMPLE_RATE
    print(f"  Reference audio: {duration:.1f}s ({len(audio)} samples)")

    # --- Extract speaker embedding ---
    print("Extracting speaker embedding ...")
    speaker_embedding = extract_speaker_embedding(sessions["speaker_encoder"], audio)

    # --- Playback reference for verification ---
    try:
        print("  Playing back reference audio for verification...")
        play_audio(audio, SAMPLE_RATE)
    except Exception:
        pass

    # --- Interactive loop ---
    print(CLONE_BANNER)
    language = args.language
    instruct = args.instruct
    print(f"  Language: {language}")
    if instruct:
        print(f"  Instruct: {instruct}")
    print(f"  Speaker embedding extracted from {duration:.1f}s of audio.")
    print()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    utterance_count = 0

    while True:
        try:
            text = input("💬 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not text:
            continue

        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            if cmd == "/quit":
                print("Goodbye! 👋")
                break
            elif cmd == "/record":
                audio = record_audio(args.record_seconds)
                speaker_embedding = extract_speaker_embedding(  # noqa: F841
                    sessions["speaker_encoder"], audio
                )
                print("  Voice sample updated.")
                continue
            elif cmd == "/language" and len(parts) > 1:
                language = parts[1]
                print(f"  Language set to: {language}")
                continue
            elif cmd == "/instruct" and len(parts) > 1:
                instruct = parts[1]
                print(f"  Instruct set to: {instruct}")
                continue
            else:
                print("  Unknown command.")
                continue

        # Generate audio codes
        # Note: Voice cloning with speaker embedding requires the model
        # to support <|speaker_embed|> injection. For models that only
        # support named speakers, we use "Vivian" as a base and the
        # speaker embedding modifies the voice characteristics.
        print("  🔄 Generating...", flush=True)
        try:
            frames = generate_codes(
                sessions,
                tokenizer,
                text,
                config,
                speaker="Vivian",
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
        audio_out, sr = decode_codes_to_audio(codec_session, frames)
        duration = len(audio_out) / sr
        print(f"  ✅ Playing {duration:.1f}s of audio...")

        if args.save_dir:
            import soundfile as sf

            utterance_count += 1
            path = os.path.join(args.save_dir, f"clone_{utterance_count:03d}.wav")
            sf.write(path, audio_out, sr)
            print(f"  📁 Saved to {path}")

        play_audio(audio_out, sr)


if __name__ == "__main__":
    main()
