#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

r"""Phi-4 Multimodal generation -- text, vision, audio, and combined.

Demonstrates building 4 separate ONNX models from
``microsoft/Phi-4-multimodal-instruct`` and running autoregressive
generation across all supported modalities:

    - **Text-only**: Standard language model generation
    - **Vision**: Image understanding (describe, answer questions)
    - **Audio**: Speech transcription / understanding
    - **Vision + Audio**: Combined multimodal reasoning

The model is split into 4 ONNX graphs, each running in its own
ONNX Runtime session:

    - **Vision**:    ``pixel_values`` + ``image_sizes`` → ``image_features``
      (SigLIP encoder + projection)
    - **Speech**:    ``audio_features`` + ``audio_sizes`` +
      ``audio_projection_mode`` → ``audio_features``
      (Conformer encoder + mode-selected projection)
    - **Embedding**: ``input_ids`` + ``image_features`` +
      ``audio_features`` → ``inputs_embeds``
    - **Decoder**:   ``inputs_embeds`` + ``attention_mask`` +
      ``position_ids`` + KV cache → ``logits`` + present KV

During prefill the pipeline chains all four sessions.  During decode
only the embedding and decoder sessions are used (no vision/speech).

Prerequisites::

    pip install mobius-ai[transformers] torchaudio

Usage::

    # Run all modality demos:
    python examples/phi4mm_multimodal.py

    # Text-only generation:
    python examples/phi4mm_multimodal.py --mode text

    # Vision (image + text):
    python examples/phi4mm_multimodal.py --mode vision \
        --image testdata/pipeline-cat-chonk.jpeg

    # Audio (speech + text):
    python examples/phi4mm_multimodal.py --mode audio \
        --audio testdata/652-129742-0006.flac

    # Combined vision + audio:
    python examples/phi4mm_multimodal.py --mode vision-audio \
        --image testdata/pipeline-cat-chonk.jpeg \
        --audio testdata/652-129742-0006.flac

    # Compare ONNX output with HuggingFace transformers:
    python examples/phi4mm_multimodal.py --mode text --compare-hf

    # Save all 4 ONNX models to disk without running inference:
    python examples/phi4mm_multimodal.py --save-to output/phi4mm/

    # Build without downloading weights (graph skeleton only):
    python examples/phi4mm_multimodal.py \
        --save-to output/phi4mm/ --no-weights
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

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
MAX_NEW_TOKENS = 64

# Special token IDs for Phi-4 multimodal
IMAGE_TOKEN_ID = 200010  # <|endoftext10|> — image placeholder
AUDIO_TOKEN_ID = 200011  # <|endoftext11|> — audio placeholder
EOS_TOKEN_IDS = {199999, 200020}  # <|endoftext|> and <|end|>

# Audio preprocessing defaults (80-dim mel filterbank at 16kHz)
AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MELS = 80

# SigLIP single-crop produces ~256 tokens after spatial merge
NUM_IMAGE_TOKENS = 256

# Conformer 3-stage stride-2 convolution time reduction factor
AUDIO_COMPRESSION_FACTOR = 8


# ---------------------------------------------------------------------------
# Input preprocessing — each function returns feeds for a *single* session
# ---------------------------------------------------------------------------


def prepare_vision_feeds(
    processor,
    image_path: str,
) -> dict[str, np.ndarray]:
    """Prepare feeds for the **vision** session.

    Returns:
        ``{"pixel_values": [1, 3, H, W], "image_sizes": [1, 2]}``
        ready for the vision ONNX model.
    """
    pixel_values = _load_image(image_path, processor)
    image_sizes = np.array([[pixel_values.shape[-2], pixel_values.shape[-1]]], dtype=np.int64)
    return {"pixel_values": pixel_values, "image_sizes": image_sizes}


def prepare_speech_feeds(
    audio_path: str,
    audio_projection_mode: int = 0,
) -> dict[str, np.ndarray]:
    """Prepare feeds for the **speech** session.

    Computes an 80-dim mel spectrogram and transposes to
    ``(1, time, n_mels)`` layout.

    Args:
        audio_path: Path to audio file.
        audio_projection_mode: 0=speech branch, 1=vision branch
            (for combined vision+audio mode).

    Returns:
        Feeds dict for the speech ONNX model.
    """
    audio_features = _load_audio(audio_path)
    # (1, n_mels, time) → (1, time, n_mels)
    audio_transposed = audio_features.transpose(0, 2, 1)
    return {
        "audio_embeds": audio_transposed,
        "audio_sizes": np.array([audio_transposed.shape[1]], dtype=np.int64),
        "audio_projection_mode": np.array(audio_projection_mode, dtype=np.int64),
    }


def prepare_embedding_feeds(
    input_ids: np.ndarray,
    image_features: np.ndarray,
    audio_features: np.ndarray,
) -> dict[str, np.ndarray]:
    """Prepare feeds for the **embedding** session.

    Args:
        input_ids: ``[batch, seq_len]`` INT64 token ids (with
            image/audio placeholder tokens already inserted).
        image_features: ``[num_image_tokens, hidden_size]`` float32,
            or ``[0, hidden_size]`` when no image is present.
        audio_features: ``[num_speech_tokens, hidden_size]`` float32,
            or ``[0, hidden_size]`` when no audio is present.

    Returns:
        Feeds dict for the embedding ONNX model.
    """
    return {
        "input_ids": input_ids,
        "image_features": image_features,
        "audio_features": audio_features,
    }


def prepare_decoder_feeds(
    inputs_embeds: np.ndarray,
    past_seq_len: int,
    past_kv: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Prepare feeds for the **decoder** session.

    Constructs the attention mask and position ids from the current
    and past sequence lengths.

    Args:
        inputs_embeds: ``[batch, cur_seq_len, hidden_size]`` float32.
        past_seq_len: Number of tokens already in the KV cache.
        past_kv: Dict of ``past_key_values.{i}.key/value`` arrays.

    Returns:
        Complete feeds dict for the decoder ONNX model.
    """
    batch_size = inputs_embeds.shape[0]
    cur_seq_len = inputs_embeds.shape[1]
    total_seq_len = past_seq_len + cur_seq_len

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
        "position_ids": np.arange(past_seq_len, total_seq_len, dtype=np.int64)[np.newaxis, :],
        **past_kv,
    }


# ---------------------------------------------------------------------------
# Token-id construction helpers
# ---------------------------------------------------------------------------


def _build_input_ids_text(
    tokenizer,
    prompt: str,
) -> np.ndarray:
    """Tokenize a text-only prompt.

    Returns ``[1, seq_len]`` INT64 array.
    """
    tokens = tokenizer(prompt, return_tensors="np")
    return tokens["input_ids"].astype(np.int64)


def _build_input_ids_vision(
    tokenizer,
    prompt: str,
    num_image_tokens: int = NUM_IMAGE_TOKENS,
) -> np.ndarray:
    """Tokenize a prompt and insert image placeholder tokens after BOS.

    Returns ``[1, seq_len]`` INT64 array.
    """
    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = tokens["input_ids"].astype(np.int64)
    img_block = np.full((1, num_image_tokens), IMAGE_TOKEN_ID, dtype=np.int64)
    return np.concatenate([input_ids[:, :1], img_block, input_ids[:, 1:]], axis=1)


def _build_input_ids_audio(
    tokenizer,
    prompt: str,
    num_audio_tokens: int,
) -> np.ndarray:
    """Tokenize a prompt and insert audio placeholder tokens after BOS.

    Returns ``[1, seq_len]`` INT64 array.
    """
    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = tokens["input_ids"].astype(np.int64)
    audio_block = np.full((1, num_audio_tokens), AUDIO_TOKEN_ID, dtype=np.int64)
    return np.concatenate([input_ids[:, :1], audio_block, input_ids[:, 1:]], axis=1)


def _build_input_ids_vision_audio(
    tokenizer,
    prompt: str,
    num_image_tokens: int = NUM_IMAGE_TOKENS,
    num_audio_tokens: int = 1,
) -> np.ndarray:
    """Tokenize a prompt and insert image + audio placeholders.

    Layout: ``BOS + image_tokens + audio_tokens + rest_of_prompt``.
    Returns ``[1, seq_len]`` INT64 array.
    """
    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = tokens["input_ids"].astype(np.int64)
    img_block = np.full((1, num_image_tokens), IMAGE_TOKEN_ID, dtype=np.int64)
    audio_block = np.full((1, num_audio_tokens), AUDIO_TOKEN_ID, dtype=np.int64)
    return np.concatenate(
        [
            input_ids[:, :1],
            img_block,
            audio_block,
            input_ids[:, 1:],
        ],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Image and audio loading helpers
# ---------------------------------------------------------------------------


def _load_image(image_path: str, processor) -> np.ndarray:
    """Load and preprocess an image using the HuggingFace processor.

    Returns:
        ``pixel_values`` as ``[1, 3, H, W]`` float32 numpy array.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    processed = processor.image_processor(images=img, return_tensors="np")
    return processed["pixel_values"].astype(np.float32)


def _load_audio(
    audio_path: str,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> np.ndarray:
    """Load audio and compute 80-dim mel spectrogram.

    Returns:
        Mel spectrogram as ``[1, n_mels, time_frames]`` float32 array.
    """
    import torchaudio

    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    audio = waveform.mean(dim=0).numpy().astype(np.float32)

    fe = transformers.WhisperFeatureExtractor(
        feature_size=AUDIO_N_MELS,
        sampling_rate=sample_rate,
    )
    out = fe(
        audio,
        sampling_rate=sample_rate,
        return_tensors="np",
        padding=False,
    )
    return out["input_features"].astype(np.float32)


def _empty_features(hidden_size: int) -> np.ndarray:
    """Return a ``[0, hidden_size]`` float32 array (no tokens)."""
    return np.zeros((0, hidden_size), dtype=np.float32)


def _add_empty_kv_cache(
    feeds: dict[str, np.ndarray],
    config,
) -> None:
    """Add empty KV cache entries to the feeds dict."""
    for i in range(config.num_hidden_layers):
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Generation loop — chains 4 sessions
# ---------------------------------------------------------------------------


def generate(
    vision_session: OnnxModelSession,
    speech_session: OnnxModelSession,
    embedding_session: OnnxModelSession,
    decoder_session: OnnxModelSession,
    tokenizer,
    input_ids: np.ndarray,
    image_features: np.ndarray,
    audio_features: np.ndarray,
    config,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Greedy autoregressive generation with KV cache and streaming.

    Chains four ONNX sessions:

    **Prefill** (first step):
      1. *vision*  → ``image_features``   (run once)
      2. *speech*  → ``audio_features``   (run once)
      3. *embedding* → ``inputs_embeds``
      4. *decoder*   → ``logits`` + KV cache

    **Decode** (subsequent tokens):
      1. *embedding* → ``inputs_embeds``
      2. *decoder*   → ``logits`` + updated KV cache

    Args:
        vision_session:   ONNX session for the SigLIP vision encoder.
        speech_session:   ONNX session for the Conformer speech encoder.
        embedding_session: ONNX session that fuses text + multimodal
            features into ``inputs_embeds``.
        decoder_session:  ONNX session for the causal LM decoder.
        tokenizer: HuggingFace tokenizer for decoding generated ids.
        input_ids: ``[1, seq_len]`` INT64 token ids (placeholders
            already inserted for image / audio tokens).
        image_features: ``[num_image_tokens, hidden_size]`` float32 or
            ``[0, hidden_size]`` when no image is present.
        audio_features: ``[num_speech_tokens, hidden_size]`` float32
            or ``[0, hidden_size]`` when no audio is present.
        config: Architecture config for model dimensions.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        The generated text (excluding the prompt).
    """
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size

    # Empty features for decode steps (no image/audio on later tokens)
    zero_image = _empty_features(hidden_size)
    zero_audio = _empty_features(hidden_size)

    # Initialize empty KV cache
    past_kv: dict[str, np.ndarray] = {}
    _add_empty_kv_cache(past_kv, config)

    cur_input_ids = input_ids
    past_seq_len = 0
    generated_ids: list[int] = []

    for step in range(max_new_tokens):
        # --- Embedding session ---
        embed_feeds = prepare_embedding_feeds(
            cur_input_ids,
            image_features if step == 0 else zero_image,
            audio_features if step == 0 else zero_audio,
        )
        embed_out = embedding_session.run(embed_feeds)
        inputs_embeds = embed_out["inputs_embeds"]

        # --- Decoder session ---
        decoder_feeds = prepare_decoder_feeds(inputs_embeds, past_seq_len, past_kv)
        outputs = decoder_session.run(decoder_feeds)

        logits = outputs["logits"]
        next_token = int(np.argmax(logits[:, -1, :]))
        generated_ids.append(next_token)

        new_text = tokenizer.decode([next_token], skip_special_tokens=True)
        print(new_text, end="", flush=True)

        if next_token in EOS_TOKEN_IDS:
            break

        # Update KV cache
        for i in range(num_layers):
            past_kv[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
            past_kv[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

        past_seq_len += cur_input_ids.shape[1]
        cur_input_ids = np.array([[next_token]], dtype=np.int64)

    print()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# HuggingFace transformers comparison
# ---------------------------------------------------------------------------


def generate_hf(
    model_id: str,
    prompt: str,
    *,
    image_path: str | None = None,
    audio_path: str | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Run generation with HuggingFace transformers for comparison.

    Loads the model with ``trust_remote_code=True`` and runs the same
    prompt through HF's generation pipeline.  Supports all modality
    combinations.
    """
    import torch

    print(f"[HF] Loading {model_id} ...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Build inputs based on available modalities
    kwargs: dict = {}

    if image_path is not None:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        img_inputs = processor.image_processor(images=image, return_tensors="pt")
        kwargs["pixel_values"] = img_inputs["pixel_values"].to(dtype=torch.float32)

    if audio_path is not None:
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        if sr != AUDIO_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, AUDIO_SAMPLE_RATE)
        audio = waveform.mean(dim=0)
        fe = transformers.WhisperFeatureExtractor(
            feature_size=AUDIO_N_MELS,
            sampling_rate=AUDIO_SAMPLE_RATE,
        )
        mel = fe(
            audio.numpy(),
            sampling_rate=AUDIO_SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        # audio_features: (1, time, n_mels)
        kwargs["audio_features"] = (
            mel["input_features"].permute(0, 2, 1).to(dtype=torch.float32)
        )

    tokens = processor.tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"]

    # Insert placeholder tokens if needed.
    # When both modalities are present, insert image before audio
    # to match _build_input_ids_vision_audio: [BOS, img, audio, text].
    if image_path is not None and audio_path is not None:
        num_img_tokens = NUM_IMAGE_TOKENS
        if "audio_features" in kwargs:
            num_audio_tokens = max(
                1,
                kwargs["audio_features"].shape[1] // AUDIO_COMPRESSION_FACTOR,
            )
        else:
            num_audio_tokens = 1
        img_block = torch.full((1, num_img_tokens), IMAGE_TOKEN_ID, dtype=torch.long)
        audio_block = torch.full((1, num_audio_tokens), AUDIO_TOKEN_ID, dtype=torch.long)
        input_ids = torch.cat(
            [input_ids[:, :1], img_block, audio_block, input_ids[:, 1:]],
            dim=1,
        )
    elif image_path is not None:
        num_img_tokens = NUM_IMAGE_TOKENS
        img_block = torch.full((1, num_img_tokens), IMAGE_TOKEN_ID, dtype=torch.long)
        input_ids = torch.cat([input_ids[:, :1], img_block, input_ids[:, 1:]], dim=1)
    elif audio_path is not None:
        if "audio_features" in kwargs:
            num_audio_tokens = max(
                1,
                kwargs["audio_features"].shape[1] // AUDIO_COMPRESSION_FACTOR,
            )
        else:
            num_audio_tokens = 1
        audio_block = torch.full((1, num_audio_tokens), AUDIO_TOKEN_ID, dtype=torch.long)
        input_ids = torch.cat([input_ids[:, :1], audio_block, input_ids[:, 1:]], dim=1)

    attention_mask = torch.ones_like(input_ids)

    print(f"[HF] Generating (max {max_new_tokens} tokens) ...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )

    gen_ids = output_ids[:, input_ids.shape[1] :]
    output = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return output


# ---------------------------------------------------------------------------
# Modality demo functions
# ---------------------------------------------------------------------------


def demo_text_only(
    vision_session: OnnxModelSession,
    speech_session: OnnxModelSession,
    embedding_session: OnnxModelSession,
    decoder_session: OnnxModelSession,
    tokenizer,
    config,
    prompt: str = "The capital of France is",
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Text-only generation demo."""
    print("\n" + "=" * 60)
    print("📝 TEXT-ONLY GENERATION")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print("-" * 40)

    hidden_size = config.hidden_size
    input_ids = _build_input_ids_text(tokenizer, prompt)

    result = generate(
        vision_session,
        speech_session,
        embedding_session,
        decoder_session,
        tokenizer,
        input_ids,
        image_features=_empty_features(hidden_size),
        audio_features=_empty_features(hidden_size),
        config=config,
        max_new_tokens=max_new_tokens,
    )

    print("-" * 40)
    return result


def demo_vision(
    vision_session: OnnxModelSession,
    speech_session: OnnxModelSession,
    embedding_session: OnnxModelSession,
    decoder_session: OnnxModelSession,
    tokenizer,
    processor,
    config,
    prompt: str = "Describe what you see in this image.",
    image_path: str = "testdata/pipeline-cat-chonk.jpeg",
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Vision (image + text) generation demo."""
    print("\n" + "=" * 60)
    print("🖼️  VISION GENERATION")
    print("=" * 60)
    print(f"Image:  {image_path}")
    print(f"Prompt: {prompt}")
    print("-" * 40)

    hidden_size = config.hidden_size

    # Run vision encoder to get image features
    vision_feeds = prepare_vision_feeds(processor, image_path)
    vision_out = vision_session.run(vision_feeds)
    image_features = vision_out["image_features"]
    # Squeeze batch dim: [1, num_tokens, hidden] → [num_tokens, hidden]
    if image_features.ndim == 3:
        image_features = image_features[0]

    num_image_tokens = image_features.shape[0]
    input_ids = _build_input_ids_vision(tokenizer, prompt, num_image_tokens)

    result = generate(
        vision_session,
        speech_session,
        embedding_session,
        decoder_session,
        tokenizer,
        input_ids,
        image_features=image_features,
        audio_features=_empty_features(hidden_size),
        config=config,
        max_new_tokens=max_new_tokens,
    )

    print("-" * 40)
    return result


def demo_audio(
    vision_session: OnnxModelSession,
    speech_session: OnnxModelSession,
    embedding_session: OnnxModelSession,
    decoder_session: OnnxModelSession,
    tokenizer,
    config,
    prompt: str = "Transcribe the following audio.",
    audio_path: str = "testdata/652-129742-0006.flac",
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Audio (speech + text) generation demo."""
    print("\n" + "=" * 60)
    print("🔊 AUDIO GENERATION")
    print("=" * 60)
    print(f"Audio:  {audio_path}")
    print(f"Prompt: {prompt}")
    print("-" * 40)

    hidden_size = config.hidden_size

    # Run speech encoder to get audio features (mode=0: speech branch)
    speech_feeds = prepare_speech_feeds(audio_path, audio_projection_mode=0)
    speech_out = speech_session.run(speech_feeds)
    audio_feats = speech_out["audio_features"]
    # Squeeze batch dim if present
    if audio_feats.ndim == 3:
        audio_feats = audio_feats[0]

    num_audio_tokens = audio_feats.shape[0]
    input_ids = _build_input_ids_audio(tokenizer, prompt, num_audio_tokens)

    result = generate(
        vision_session,
        speech_session,
        embedding_session,
        decoder_session,
        tokenizer,
        input_ids,
        image_features=_empty_features(hidden_size),
        audio_features=audio_feats,
        config=config,
        max_new_tokens=max_new_tokens,
    )

    print("-" * 40)
    return result


def demo_vision_audio(
    vision_session: OnnxModelSession,
    speech_session: OnnxModelSession,
    embedding_session: OnnxModelSession,
    decoder_session: OnnxModelSession,
    tokenizer,
    processor,
    config,
    prompt: str = "Describe the image and transcribe the audio.",
    image_path: str = "testdata/pipeline-cat-chonk.jpeg",
    audio_path: str = "testdata/652-129742-0006.flac",
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Combined vision + audio generation demo."""
    print("\n" + "=" * 60)
    print("🖼️ + 🔊 VISION + AUDIO GENERATION")
    print("=" * 60)
    print(f"Image:  {image_path}")
    print(f"Audio:  {audio_path}")
    print(f"Prompt: {prompt}")
    print("-" * 40)

    # Run vision encoder
    vision_feeds = prepare_vision_feeds(processor, image_path)
    vision_out = vision_session.run(vision_feeds)
    image_features = vision_out["image_features"]
    if image_features.ndim == 3:
        image_features = image_features[0]

    # Run speech encoder — use vision branch (mode=1) for combined mode
    speech_feeds = prepare_speech_feeds(audio_path, audio_projection_mode=1)
    speech_out = speech_session.run(speech_feeds)
    audio_feats = speech_out["audio_features"]
    if audio_feats.ndim == 3:
        audio_feats = audio_feats[0]

    num_image_tokens = image_features.shape[0]
    num_audio_tokens = audio_feats.shape[0]
    input_ids = _build_input_ids_vision_audio(
        tokenizer, prompt, num_image_tokens, num_audio_tokens
    )

    result = generate(
        vision_session,
        speech_session,
        embedding_session,
        decoder_session,
        tokenizer,
        input_ids,
        image_features=image_features,
        audio_features=audio_feats,
        config=config,
        max_new_tokens=max_new_tokens,
    )

    print("-" * 40)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phi-4 Multimodal generation with ONNX — text, vision, audio, and combined."
        ),
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "vision", "audio", "vision-audio", "all"],
        default="all",
        help=(
            "Which modality demo to run (default: %(default)s). "
            "'all' runs all four demos in sequence."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt (default depends on --mode).",
    )
    parser.add_argument(
        "--image",
        default="testdata/pipeline-cat-chonk.jpeg",
        help="Path to image file (default: %(default)s).",
    )
    parser.add_argument(
        "--audio",
        default="testdata/652-129742-0006.flac",
        help="Path to audio file (default: %(default)s).",
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
        help=("Save all 4 ONNX models to DIR and exit (no inference)."),
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Build graph skeleton only (no weight download).",
    )
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help=("Also run with HuggingFace transformers and compare outputs."),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Build the 4-model ONNX package
    # ------------------------------------------------------------------
    load_weights = not args.no_weights
    print(f"Building ONNX models from {args.model_id!r} ...")
    pkg = build(args.model_id, dtype="f32", load_weights=load_weights)
    config = pkg.config
    print(f"Package components: {list(pkg.keys())}")

    if args.save_to:
        pkg.save(args.save_to, check_weights=load_weights)
        print(f"Saved to {args.save_to}")
        return

    # ------------------------------------------------------------------
    # Step 2: Create 4 inference sessions and tokenizer
    # ------------------------------------------------------------------
    print("Creating ONNX Runtime sessions ...")
    vision_session = OnnxModelSession(pkg["vision"])
    speech_session = OnnxModelSession(pkg["speech"])
    embedding_session = OnnxModelSession(pkg["embedding"])
    decoder_session = OnnxModelSession(pkg["model"])

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True
    )
    processor = transformers.AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=True
    )

    # ------------------------------------------------------------------
    # Step 3: Run selected modality demos
    # ------------------------------------------------------------------
    modes = ["text", "vision", "audio", "vision-audio"] if args.mode == "all" else [args.mode]
    max_tokens = args.max_new_tokens

    for mode in modes:
        onnx_result: str | None = None

        if mode == "text":
            prompt = args.prompt or "The capital of France is"
            onnx_result = demo_text_only(
                vision_session,
                speech_session,
                embedding_session,
                decoder_session,
                tokenizer,
                config,
                prompt,
                max_new_tokens=max_tokens,
            )

        elif mode == "vision":
            prompt = args.prompt or "Describe what you see in this image."
            onnx_result = demo_vision(
                vision_session,
                speech_session,
                embedding_session,
                decoder_session,
                tokenizer,
                processor,
                config,
                prompt,
                args.image,
                max_new_tokens=max_tokens,
            )

        elif mode == "audio":
            prompt = args.prompt or "Transcribe the following audio."
            onnx_result = demo_audio(
                vision_session,
                speech_session,
                embedding_session,
                decoder_session,
                tokenizer,
                config,
                prompt,
                args.audio,
                max_new_tokens=max_tokens,
            )

        elif mode == "vision-audio":
            prompt = args.prompt or "Describe the image and transcribe the audio."
            onnx_result = demo_vision_audio(
                vision_session,
                speech_session,
                embedding_session,
                decoder_session,
                tokenizer,
                processor,
                config,
                prompt,
                args.image,
                args.audio,
                max_new_tokens=max_tokens,
            )

        # Optional HuggingFace comparison
        if args.compare_hf and onnx_result is not None:
            print("\n" + "=" * 60)
            print(f"🤗 HUGGINGFACE COMPARISON ({mode})")
            print("=" * 60)

            hf_result = generate_hf(
                args.model_id,
                prompt,
                image_path=(args.image if mode in ("vision", "vision-audio") else None),
                audio_path=(args.audio if mode in ("audio", "vision-audio") else None),
                max_new_tokens=max_tokens,
            )

            print(f"\n[HF]  : {hf_result}")
            print(f"[ONNX]: {onnx_result}")

            if hf_result.strip() == onnx_result.strip():
                print("✅ Outputs match!")
            else:
                print("⚠️  Outputs differ (may be due to numerical precision).")

    print("\nDone.")


if __name__ == "__main__":
    main()
