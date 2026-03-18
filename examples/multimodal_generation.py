#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Multimodal generation example — image captioning with Gemma 3.

Demonstrates building an ONNX multimodal model with ``mobius.build``
and running greedy-decoded inference with streaming text output.

Usage::

    python examples/multimodal_generation.py

    # With a custom image URL:
    python examples/multimodal_generation.py --image "https://example.com/photo.jpg"

    # With a local image file:
    python examples/multimodal_generation.py --image ./my_photo.jpg

    # Save the ONNX model to disk without running inference:
    python examples/multimodal_generation.py --save-to output/gemma3/
"""

from __future__ import annotations

import argparse
import urllib.request
from io import BytesIO

import numpy as np
import transformers
from PIL import Image

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models import Gemma3MultiModalModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemma-3-4b-pt"
DEFAULT_IMAGE_URL = "testdata/pipeline-cat-chonk.jpeg"
DEFAULT_PROMPT = "What is shown in this image?"
MAX_NEW_TOKENS = 64


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class MultimodalPipeline:
    """Simple pipeline for ONNX multimodal generation.

    Example::

        pipe = MultimodalPipeline("google/gemma-3-4b-pt")
        result = pipe(
            "https://example.com/cat.jpg",
            text="What is shown in this image?",
        )
        print(result)
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        print(f"Building model {model_id!r} ...")
        # Gemma 3 multimodal models report model_type="gemma3" which maps
        # to the text-only class by default.  Explicitly request the 3-model
        # multimodal split (decoder, vision, embedding).
        pkg = build(
            model_id,
            task="vision-language",
            module_class=Gemma3MultiModalModel,
        )
        self._config = pkg.config
        # VisionLanguageTask produces 3 separate models
        self._decoder = OnnxModelSession(pkg["decoder"])
        self._vision = OnnxModelSession(pkg["vision"])
        self._embedding = OnnxModelSession(pkg["embedding"])

        # Tokenizer and image processor
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._image_processor = transformers.AutoProcessor.from_pretrained(
            model_id
        ).image_processor
        print("Model ready.")

    def __call__(
        self,
        image: str,
        *,
        text: str = DEFAULT_PROMPT,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate text for an image + text prompt with streaming output."""
        max_new_tokens = max_new_tokens or self.max_new_tokens

        pixel_values = _load_image(image, self._image_processor)
        input_ids = _prepare_input_ids(text, self._tokenizer, self._config)

        generated_text = ""
        prev_text = ""
        for token_ids in _generate_tokens(
            self._decoder,
            self._vision,
            self._embedding,
            self._config,
            input_ids,
            pixel_values,
            max_new_tokens=max_new_tokens,
            eos_token_id=self._tokenizer.eos_token_id,
        ):
            generated_text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
            new_chars = generated_text[len(prev_text) :]
            if new_chars:
                print(new_chars, end="", flush=True)
            prev_text = generated_text

        print()
        return generated_text


# ---------------------------------------------------------------------------
# Image loading and preprocessing
# ---------------------------------------------------------------------------


def _load_image(source: str, image_processor) -> np.ndarray:
    """Load an image from a URL or file path and preprocess it.

    Returns:
        pixel_values as ``[1, 3, H, W]`` float32 numpy array.
    """
    if source.startswith(("http://", "https://")):
        print(f"Downloading image from {source} …")
        with urllib.request.urlopen(source) as resp:
            data = resp.read()
        img = Image.open(BytesIO(data)).convert("RGB")
    else:
        img = Image.open(source).convert("RGB")

    processed = image_processor(images=img, return_tensors="pt")
    return processed["pixel_values"].numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def _prepare_input_ids(text: str, tokenizer, config) -> np.ndarray:
    """Tokenize text and insert image placeholder tokens.

    Returns:
        ``[1, seq_len]`` int64 numpy array.
    """
    image_token_id = config.image_token_id
    mm_tokens = config.mm_tokens_per_image or 256

    tokens = tokenizer(text, return_tensors="np", add_special_tokens=True)
    ids = tokens["input_ids"].astype(np.int64)

    img_block = np.full((1, mm_tokens), image_token_id, dtype=np.int64)

    boi_token_id = tokenizer.convert_tokens_to_ids("<start_of_image>")
    boi_positions = np.where(ids[0] == boi_token_id)[0]

    if len(boi_positions) > 0:
        pos = boi_positions[0] + 1
        ids = np.concatenate([ids[:, :pos], img_block, ids[:, pos:]], axis=1)
    else:
        ids = np.concatenate([ids[:, :1], img_block, ids[:, 1:]], axis=1)

    return ids


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------


def _generate_tokens(
    decoder,
    vision,
    embedding,
    config,
    input_ids: np.ndarray,
    pixel_values: np.ndarray,
    max_new_tokens: int = 64,
    eos_token_id: int | None = None,
):
    """Yield growing list of generated token IDs (for streaming decode).

    Uses the 3-model VisionLanguageTask pipeline:
      1. Vision model: pixel_values → image_features
      2. Embedding model: input_ids + image_features → inputs_embeds
      3. Decoder model: inputs_embeds + attention_mask + position_ids + KV cache → logits
    """
    batch_size = 1
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    # Step 1: Run vision encoder once to get image features
    vision_out = vision.run({"pixel_values": pixel_values})
    # Vision output is [batch, num_tokens, hidden] but embedding expects
    # [num_tokens, hidden], so squeeze the batch dimension.
    image_features = vision_out["image_features"][0]

    # Initialize empty KV cache
    past_kv: dict[str, np.ndarray] = {}
    for i in range(num_layers):
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )

    cur_input_ids = input_ids
    past_seq_len = 0
    # Zero image features for decode steps (no image on subsequent tokens)
    zero_image_features = np.zeros_like(image_features)
    generated_ids: list[int] = []

    for step in range(max_new_tokens):
        cur_seq_len = cur_input_ids.shape[1]
        total_seq_len = past_seq_len + cur_seq_len

        # Step 2: Run embedding model to fuse text + image
        embed_out = embedding.run(
            {
                "input_ids": cur_input_ids,
                "image_features": image_features if step == 0 else zero_image_features,
            }
        )
        inputs_embeds = embed_out["inputs_embeds"]

        # Step 3: Run decoder with inputs_embeds
        feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": np.arange(past_seq_len, total_seq_len, dtype=np.int64)[
                np.newaxis, :
            ],
            **past_kv,
        }

        outputs = decoder.run(feeds)

        logits = outputs["logits"]
        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        token_id = int(next_token[0, 0])
        generated_ids.append(token_id)

        yield generated_ids

        if eos_token_id is not None and token_id == eos_token_id:
            break

        for i in range(num_layers):
            past_kv[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
            past_kv[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

        cur_input_ids = next_token.astype(np.int64)
        past_seq_len = total_seq_len


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Image captioning with an ONNX multimodal model.",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE_URL,
        help="URL or local path to an image (default: HuggingFace cat image).",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_PROMPT,
        help="Text prompt (default: %(default)r).",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
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
        help="Save the ONNX model package to DIR and exit (no inference).",
    )
    args = parser.parse_args()

    if args.save_to:
        print(f"Building model {args.model!r} ...")
        pkg = build(
            args.model,
            task="vision-language",
            module_class=Gemma3MultiModalModel,
        )
        pkg.save(args.save_to)
        print("Done.")
        return

    pipe = MultimodalPipeline(args.model, max_new_tokens=args.max_new_tokens)

    print()
    print("=" * 60)
    print(f"Prompt: {args.text}")
    print("=" * 60)
    pipe(args.image, text=args.text)
    print("=" * 60)


if __name__ == "__main__":
    main()
