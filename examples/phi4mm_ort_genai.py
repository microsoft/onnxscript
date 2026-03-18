#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

r"""Phi-4 Multimodal generation with onnxruntime-genai.

Builds the 4-model ONNX package (vision, speech, embedding, decoder),
saves it in the layout expected by onnxruntime-genai, and runs text
generation with image and/or audio input.

The model is split into 4 ONNX graphs:

    - **Vision**  (``vision/model.onnx``): SigLIP encoder + projection
    - **Speech**  (``speech/model.onnx``): Conformer encoder + projection
    - **Embedding** (``embedding/model.onnx``): token embed + InputMixer
    - **Decoder** (``model/model.onnx``): LoRA text decoder + lm_head

Requirements::

    pip install mobius-ai[ort-genai] torchaudio

Usage::

    # Build, export, and run text-only generation:
    python examples/phi4mm_ort_genai.py

    # With an image:
    python examples/phi4mm_ort_genai.py \
        --image testdata/pipeline-cat-chonk.jpeg

    # With audio:
    python examples/phi4mm_ort_genai.py \
        --audio testdata/652-129742-0006.flac

    # Combined image + audio:
    python examples/phi4mm_ort_genai.py \
        --image testdata/pipeline-cat-chonk.jpeg \
        --audio testdata/652-129742-0006.flac

    # Compare with HuggingFace transformers:
    python examples/phi4mm_ort_genai.py \
        --image testdata/pipeline-cat-chonk.jpeg --compare-hf

    # Use a pre-built model directory:
    python examples/phi4mm_ort_genai.py --model-dir output/phi4mm/

    # Build and save (skip inference):
    python examples/phi4mm_ort_genai.py --save-to output/phi4mm/
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import onnxruntime_genai as og

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
MAX_NEW_TOKENS = 64

# Special token IDs
IMAGE_TOKEN_ID = 200010  # <|endoftext10|>
AUDIO_TOKEN_ID = 200011  # <|endoftext11|>
EOS_TOKEN_IDS = [199999, 200020]  # <|endoftext|> and <|end|>

# Audio preprocessing
AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MELS = 80

# Image/audio token counts
NUM_IMAGE_TOKENS = 256
AUDIO_COMPRESSION_FACTOR = 8


# ---------------------------------------------------------------------------
# Model export
# ---------------------------------------------------------------------------


def _write_genai_config(config, output_dir: str) -> None:
    """Write genai_config.json for the Phi4MM 4-model split."""
    genai_config = {
        "model": {
            "type": "phi4mm",
            "vocab_size": config.vocab_size,
            "context_length": min(config.max_position_embeddings or 131072, 131072),
            "bos_token_id": config.bos_token_id or 199999,
            "eos_token_id": EOS_TOKEN_IDS,
            "pad_token_id": config.pad_token_id or 199999,
            "image_token_id": IMAGE_TOKEN_ID,
            "decoder": {
                "filename": "model/model.onnx",
                "hidden_size": config.hidden_size,
                "head_size": config.head_dim,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "num_hidden_layers": config.num_hidden_layers,
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
            },
            "embedding": {
                "filename": "embedding/model.onnx",
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "inputs": {
                    "input_ids": "input_ids",
                    "image_features": "image_features",
                    "audio_features": "audio_features",
                },
                "outputs": {
                    "inputs_embeds": "inputs_embeds",
                },
            },
            "vision": {
                "filename": "vision/model.onnx",
                "config_filename": "vision_processor.json",
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "inputs": {
                    "pixel_values": "pixel_values",
                    "image_sizes": "image_sizes",
                },
                "outputs": {
                    "image_features": "image_features",
                },
            },
            "speech": {
                "filename": "speech/model.onnx",
                "config_filename": "speech_processor.json",
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "inputs": {
                    "audio_embeds": "audio_embeds",
                    "audio_sizes": "audio_sizes",
                    "audio_projection_mode": "audio_projection_mode",
                },
                "outputs": {
                    "audio_features": "audio_features",
                },
            },
        },
        "search": {
            "do_sample": False,
            "early_stopping": True,
            "max_length": 4096,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }
    path = os.path.join(output_dir, "genai_config.json")
    with open(path, "w") as f:
        json.dump(genai_config, f, indent=4)


def _copy_tokenizer(model_id: str, output_dir: str) -> None:
    """Copy tokenizer files from the HuggingFace cache."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.save_pretrained(output_dir)


def build_and_export(model_id: str, output_dir: str) -> None:
    """Build the 4-model ONNX package and save for onnxruntime-genai."""
    from mobius import build

    print(f"Building {model_id!r} ...")
    pkg = build(model_id, dtype="f32", load_weights=True)
    print(f"Package components: {list(pkg.keys())}")

    print(f"Saving to {output_dir} ...")
    pkg.save(output_dir)
    _write_genai_config(pkg.config, output_dir)
    _copy_tokenizer(model_id, output_dir)
    print("Export complete.")


# ---------------------------------------------------------------------------
# Audio preprocessing helper
# ---------------------------------------------------------------------------


def _load_audio_features(audio_path: str) -> np.ndarray:
    """Load audio and compute mel spectrogram features.

    Returns:
        Mel spectrogram as ``[1, time, n_mels]`` float32 array.
    """
    import torchaudio
    import transformers

    waveform, sr = torchaudio.load(audio_path)
    if sr != AUDIO_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, AUDIO_SAMPLE_RATE)
    audio = waveform.mean(dim=0).numpy().astype(np.float32)

    fe = transformers.WhisperFeatureExtractor(
        feature_size=AUDIO_N_MELS,
        sampling_rate=AUDIO_SAMPLE_RATE,
    )
    out = fe(
        audio,
        sampling_rate=AUDIO_SAMPLE_RATE,
        return_tensors="np",
        padding=False,
    )
    # (1, n_mels, time) → (1, time, n_mels)
    return out["input_features"].astype(np.float32).transpose(0, 2, 1)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_text(
    model_dir: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Run text-only generation with onnxruntime-genai."""
    print(f"Loading model from {model_dir} ...")
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)

    input_ids = tokenizer.encode(prompt)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(input_ids) + max_new_tokens)

    generator = og.Generator(model, params)
    generator.append_tokens(input_ids)

    print(f"\nPrompt: {prompt}")
    print("-" * 40)

    tokenizer_stream = tokenizer.create_stream()
    generated_tokens = []
    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        generated_tokens.append(token)
        print(tokenizer_stream.decode(token), end="", flush=True)
        if len(generated_tokens) >= max_new_tokens:
            break

    print("\n" + "-" * 40)
    del generator
    return tokenizer.decode(generated_tokens)


def generate_multimodal(
    model_dir: str,
    prompt: str,
    max_new_tokens: int,
    *,
    image_path: str | None = None,
    audio_path: str | None = None,
) -> None:
    """Run multimodal generation (text + image + audio).

    Uses the ORT GenAI ``PhiMultiModalProcessor`` for image
    preprocessing. Audio features are computed externally and
    injected via ``NamedTensors``.
    """
    print(f"Loading model from {model_dir} ...")
    model = og.Model(model_dir)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)

    # Prepare images for the processor (if provided)
    images = None
    if image_path is not None:
        images = og.Images.open(image_path)

    # Process text + image through the ORT GenAI processor
    if images is not None:
        inputs = processor(prompt, images=images)
    else:
        inputs = processor(prompt)

    # If audio is provided, compute mel features and inject
    if audio_path is not None:
        audio_features = _load_audio_features(audio_path)
        audio_tensor = og.OrtValue.ortvalue_from_numpy(audio_features)
        inputs["audio_features"] = audio_tensor

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=4096)

    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    desc_parts = []
    if image_path:
        desc_parts.append(f"Image: {image_path}")
    if audio_path:
        desc_parts.append(f"Audio: {audio_path}")
    print(f"\nPrompt: {prompt}")
    for part in desc_parts:
        print(part)
    print("-" * 40)

    tokenizer_stream = tokenizer.create_stream()
    tokens_generated = 0
    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(token), end="", flush=True)
        tokens_generated += 1
        if tokens_generated >= max_new_tokens:
            break

    print("\n" + "-" * 40)
    del generator


# ---------------------------------------------------------------------------
# HuggingFace comparison
# ---------------------------------------------------------------------------


def generate_hf(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    *,
    image_path: str | None = None,
    audio_path: str | None = None,
) -> str:
    """Run generation with HuggingFace transformers for comparison."""
    import torch
    import transformers

    print(f"[HF] Loading {model_id} ...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    hf_processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    kwargs: dict = {}

    if image_path is not None:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        img_inputs = hf_processor.image_processor(images=image, return_tensors="pt")
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
        kwargs["audio_features"] = (
            mel["input_features"].permute(0, 2, 1).to(dtype=torch.float32)
        )

    tokens = hf_processor.tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"]

    # Insert placeholder tokens matching the ONNX pipeline layout.
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
    return hf_processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=("Phi-4 Multimodal generation with onnxruntime-genai."),
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=("Pre-built model directory. Skip export if provided."),
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help="Export the model to DIR and exit (no inference).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to an image file for vision input.",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to an audio file for speech input.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt (default depends on modality).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum tokens to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help="Also run with HuggingFace and compare outputs.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    if args.save_to:
        build_and_export(args.model_id, args.save_to)
        return

    # ------------------------------------------------------------------
    # Resolve model directory
    # ------------------------------------------------------------------
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = os.path.join("output", "phi4mm")
        config_path = os.path.join(model_dir, "genai_config.json")
        if not os.path.isfile(config_path):
            build_and_export(args.model_id, model_dir)

    # ------------------------------------------------------------------
    # Determine prompt based on modalities
    # ------------------------------------------------------------------
    has_image = args.image is not None
    has_audio = args.audio is not None

    if args.prompt:
        prompt = args.prompt
    elif has_image and has_audio:
        prompt = "Describe the image and transcribe the audio."
    elif has_image:
        prompt = "Describe what you see in this image."
    elif has_audio:
        prompt = "Transcribe the following audio."
    else:
        prompt = "The capital of France is"

    # ------------------------------------------------------------------
    # Run generation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("ORT GenAI")
    print("=" * 60)

    if has_image or has_audio:
        generate_multimodal(
            model_dir,
            prompt,
            args.max_new_tokens,
            image_path=args.image,
            audio_path=args.audio,
        )
    else:
        generate_text(model_dir, prompt, args.max_new_tokens)

    # ------------------------------------------------------------------
    # Optional HuggingFace comparison
    # ------------------------------------------------------------------
    if args.compare_hf:
        print("\n" + "=" * 60)
        print("HuggingFace Transformers")
        print("=" * 60)
        hf_result = generate_hf(
            args.model_id,
            prompt,
            args.max_new_tokens,
            image_path=args.image,
            audio_path=args.audio,
        )
        print(f"[HF] Output: {hf_result}")

    print("\nDone.")


if __name__ == "__main__":
    main()
