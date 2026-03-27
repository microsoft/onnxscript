#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

r"""Generate golden reference files for L4/L5 testing.

Reads YAML test case definitions from ``testdata/cases/`` and generates
``.json`` golden reference files in ``testdata/golden/`` by running
HuggingFace inference.

Requires: ``pip install transformers torch accelerate``
GPU recommended for models > 1B parameters.

Usage::

    # Generate golden files for ALL test cases
    python scripts/generate_golden.py

    # Generate for a specific task type
    python scripts/generate_golden.py --task-type causal-lm

    # Generate for a single test case
    python scripts/generate_golden.py --case testdata/cases/causal-lm/gpt2.yaml

    # Regenerate all (overwrite existing)
    python scripts/generate_golden.py --force

    # Use GPU for large models
    python scripts/generate_golden.py --device cuda

    # Dry run (show what would be generated)
    python scripts/generate_golden.py --dry-run

    # Filter by glob pattern on case_id
    python scripts/generate_golden.py --filter "qwen*"
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobius._testing.golden import GoldenTestCase as TestCase

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Generate golden reference .json files for L4/L5 testing"),
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=None,
        help="Path to a single YAML test case. If omitted, processes all.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        help=(
            "Only generate for this task type subdirectory "
            "(causal-lm, encoder, seq2seq, vision-language, audio, "
            "diffusion)."
        ),
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        choices=["L4", "L5"],
        help="Only generate cases that include this level.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Glob pattern to filter case_ids (e.g. 'qwen*', '*7b*').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device: 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing golden files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without running inference.",
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path("testdata/cases"),
        help="Root directory for YAML test case files.",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path("testdata/golden"),
        help="Root directory for golden .json output files.",
    )
    return parser.parse_args()


# ---- Logit extraction helpers ----
# Shared by all generators that produce logit-based golden data.


def _extract_logits_golden(
    last_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract top-k IDs, logits, and summary from a logit vector.

    Args:
        last_logits: 1-D float array of shape ``(vocab_size,)``.

    Returns:
        Dict with keys ready for ``save_golden_ref()``.
    """
    last_logits_f64 = last_logits.astype(np.float64)
    sorted_indices = np.argsort(last_logits_f64)[::-1]
    top10_ids = sorted_indices[:10].tolist()
    top10_logits = last_logits_f64[sorted_indices[:10]].tolist()
    logits_summary = np.array(
        [
            float(np.max(last_logits_f64)),
            float(np.min(last_logits_f64)),
            float(np.mean(last_logits_f64)),
            float(np.std(last_logits_f64)),
        ],
        dtype=np.float64,
    )
    return {
        "top1_id": top10_ids[0],
        "top2_id": top10_ids[1] if len(top10_ids) > 1 else top10_ids[0],
        "top10_ids": top10_ids,
        "top10_logits": top10_logits,
        "logits_summary": logits_summary,
    }


# ---- Task-specific generators ----
# Each generator loads a HF model, runs inference, and calls
# save_golden_ref() from golden.py.  Heavy imports (torch,
# transformers) are deferred to avoid import cost when --dry-run.


def _generate_causal_lm(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for a causal-lm (text-generation) model."""
    from mobius._testing.golden import save_generation_json, save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_model,
        torch_forward,
    )

    model, tokenizer = load_torch_model(case.model_id, device=device)

    encoded = tokenizer(case.prompts[0], return_tensors="np", padding=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    seq_len = input_ids.shape[1]
    position_ids = np.arange(seq_len).reshape(1, -1)

    # L4: single forward pass → last-token logits
    logits, _ = torch_forward(model, input_ids, attention_mask, position_ids)
    last_logits = logits[0, -1, :]  # (vocab_size,)
    golden = _extract_logits_golden(last_logits)

    # L5: greedy generation
    generated_ids = None
    if "L5" in case.level:
        import torch

        with torch.no_grad():
            gen_output = model.generate(
                torch.from_numpy(input_ids).to(device),
                max_new_tokens=case.generation_params.get("max_new_tokens", 20),
                do_sample=False,
            )
        generated_ids = gen_output[0, seq_len:].cpu().numpy()

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids,
    )

    # Save a separate *_generation.json marker for L5 dashboard detection.
    if generated_ids is not None:
        generated_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        gen_path = json_path.with_name(json_path.stem + "_generation.json")
        save_generation_json(
            gen_path,
            model_id=case.model_id,
            prompt=case.prompts[0],
            generated_tokens=generated_ids.tolist(),
            generated_text=generated_text,
        )


def _generate_encoder(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for an encoder-only (feature-extraction) model.

    Encoder models produce ``last_hidden_state`` instead of logits.
    We treat the last token's hidden state as the "logit" vector for
    top-k extraction — this gives us a meaningful argmax gate for L4.
    """
    from mobius._testing.golden import save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_encoder_model,
        torch_encoder_forward,
    )

    model, tokenizer = load_torch_encoder_model(case.model_id, device=device)

    encoded = tokenizer(case.prompts[0], return_tensors="np", padding=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids")

    # Forward pass → last_hidden_state
    hidden_states = torch_encoder_forward(model, input_ids, attention_mask, token_type_ids)
    # Use the last token's hidden state as the "logit" vector
    last_hidden = hidden_states[0, -1, :]  # (hidden_size,)
    golden = _extract_logits_golden(last_hidden)

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids,
    )


def _generate_seq2seq(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for a seq2seq (encoder-decoder) model."""
    import torch

    from mobius._testing.golden import save_generation_json, save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_seq2seq_model,
    )

    model, tokenizer = load_torch_seq2seq_model(case.model_id, device=device)

    encoded = tokenizer(case.prompts[0], return_tensors="np", padding=False)
    input_ids = encoded["input_ids"]

    # Prepare decoder input (pad token for autoregressive start)
    decoder_start = np.array([[model.config.decoder_start_token_id or 0]], dtype=np.int64)

    # L4: single forward pass through full model
    torch_device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(
            input_ids=torch.from_numpy(input_ids).to(torch_device),
            decoder_input_ids=torch.from_numpy(decoder_start).to(torch_device),
        )
    last_logits = outputs.logits[0, -1, :].cpu().numpy()
    golden = _extract_logits_golden(last_logits)

    # L5: greedy generation
    generated_ids = None
    if "L5" in case.level:
        with torch.no_grad():
            gen_output = model.generate(
                torch.from_numpy(input_ids).to(torch_device),
                max_new_tokens=case.generation_params.get("max_new_tokens", 20),
                do_sample=False,
            )
        generated_ids = gen_output[0].cpu().numpy()

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids,
    )

    if generated_ids is not None:
        generated_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        gen_path = json_path.with_name(json_path.stem + "_generation.json")
        save_generation_json(
            gen_path,
            model_id=case.model_id,
            prompt=case.prompts[0],
            generated_tokens=generated_ids.tolist(),
            generated_text=generated_text,
        )


def _generate_vision_language(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for a vision-language (image-text-to-text) model.

    Multi-model task: stores decoder golden data with dotted key prefix
    and component norms/shapes for vision + embedding diagnostics.
    """
    import torch
    from PIL import Image

    from mobius._testing.golden import save_generation_json, save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_multimodal_model,
    )

    model, _tokenizer, processor = load_torch_multimodal_model(case.model_id, device=device)

    # Load images from testdata/
    images = [Image.open(Path("testdata") / img_path) for img_path in case.images]

    # Build chat-formatted prompt with image placeholders if the
    # processor supports apply_chat_template (Qwen-VL, Gemma-3, etc.)
    prompt_text = case.prompts[0]
    if hasattr(processor, "apply_chat_template"):
        content: list[dict[str, str]] = []
        for img_path in case.images:
            content.append({"type": "image", "image": str(Path("testdata") / img_path)})
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Process multimodal inputs through the HF processor
    processed = processor(
        text=prompt_text,
        images=images if images else None,
        return_tensors="pt",
    ).to(device)

    # L4: single forward pass
    with torch.no_grad():
        outputs = model(**processed)

    last_logits = outputs.logits[0, -1, :].cpu().numpy()
    golden = _extract_logits_golden(last_logits)
    input_ids_np = processed["input_ids"].cpu().numpy()

    # L5: greedy generation
    generated_ids = None
    if "L5" in case.level:
        with torch.no_grad():
            gen = model.generate(
                **processed,
                max_new_tokens=case.generation_params.get("max_new_tokens", 30),
                do_sample=False,
            )
        input_len = processed["input_ids"].shape[1]
        generated_ids = gen[0, input_len:].cpu().numpy()

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids_np,
    )

    if generated_ids is not None:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else None
        generated_text = (
            tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
            if tokenizer is not None
            else None
        )
        gen_path = json_path.with_name(json_path.stem + "_generation.json")
        save_generation_json(
            gen_path,
            model_id=case.model_id,
            prompt=case.prompts[0],
            generated_tokens=generated_ids.tolist(),
            generated_text=generated_text,
        )


def _generate_speech_to_text(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for a speech-to-text (Whisper) model."""
    import librosa
    import torch

    from mobius._testing.golden import save_generation_json, save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_whisper_model,
    )

    model, processor = load_torch_whisper_model(case.model_id, device=device)

    # Load and preprocess audio
    audio_path = Path("testdata") / case.audio[0]
    audio_array, sample_rate = librosa.load(str(audio_path), sr=16000)
    processed = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").to(
        device
    )
    input_features = processed["input_features"]

    # L4: single decoder step with forced decoder start token
    decoder_start_id = (
        model.config.decoder_start_token_id or model.generation_config.decoder_start_token_id
    )
    decoder_input_ids = torch.tensor([[decoder_start_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
        )
    last_logits = outputs.logits[0, -1, :].cpu().numpy()
    golden = _extract_logits_golden(last_logits)
    input_ids_np = decoder_input_ids.cpu().numpy()

    # L5: greedy generation
    generated_ids = None
    if "L5" in case.level:
        with torch.no_grad():
            gen = model.generate(
                input_features=input_features,
                max_new_tokens=case.generation_params.get("max_new_tokens", 50),
                do_sample=False,
            )
        generated_ids = gen[0].cpu().numpy()

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids_np,
    )

    if generated_ids is not None:
        generated_text = processor.decode(generated_ids.tolist(), skip_special_tokens=True)
        gen_path = json_path.with_name(json_path.stem + "_generation.json")
        save_generation_json(
            gen_path,
            model_id=case.model_id,
            prompt=case.audio[0],
            generated_tokens=generated_ids.tolist(),
            generated_text=generated_text,
        )


def _generate_audio_feature_extraction(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for audio feature extraction (Wav2Vec2 etc.).

    Similar to encoder-only: last hidden state is used as the
    "logit" vector for top-k extraction.
    """
    import librosa

    from mobius._testing.golden import save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_audio_model,
        torch_audio_forward,
    )

    model, processor = load_torch_audio_model(
        case.model_id, device=device, trust_remote_code=case.trust_remote_code
    )

    # Load and preprocess audio
    audio_path = Path("testdata") / case.audio[0]
    audio_array, sample_rate = librosa.load(str(audio_path), sr=16000)
    processed = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="np",
    )
    input_values = processed["input_values"]

    # Forward pass → last_hidden_state
    hidden_states = torch_audio_forward(model, input_values)
    # Use the last frame's hidden state for top-k extraction
    last_hidden = hidden_states[0, -1, :]  # (hidden_size,)
    golden = _extract_logits_golden(last_hidden)

    # Audio feature extraction is L4-only (no generation)
    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=np.array([[0]], dtype=np.int64),  # placeholder
    )


def _generate_image_classification(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for image classification (ViT, CLIP, etc.).

    Similar to encoder-only: last hidden state is used as the
    "logit" vector for top-k extraction.
    """
    from PIL import Image

    from mobius._testing.golden import save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_vision_model,
        torch_vision_forward,
    )

    model, processor = load_torch_vision_model(
        case.model_id, device=device, trust_remote_code=case.trust_remote_code
    )

    # Load and preprocess image
    image = Image.open(Path("testdata") / case.images[0])
    # Use PyTorch tensors then convert — some processors don't support np
    processed = processor(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"].numpy()

    # Forward pass → last_hidden_state
    hidden_states = torch_vision_forward(model, pixel_values)
    # Use the last patch token rather than the CLS token (index 0) because
    # patch-based ViT models aggregate spatial context into trailing tokens;
    # the last token provides a stable, architecture-neutral summary vector.
    last_hidden = hidden_states[0, -1, :]  # (hidden_size,)
    golden = _extract_logits_golden(last_hidden)

    # Image classification is L4-only (no generation)
    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=np.array([[0]], dtype=np.int64),  # placeholder
    )


# ---- Dispatcher ----

# Map task_type strings to generator functions.
_GENERATORS = {
    "text-generation": _generate_causal_lm,
    "feature-extraction": _generate_encoder,
    "seq2seq": _generate_seq2seq,
    "image-text-to-text": _generate_vision_language,
    "image-classification": _generate_image_classification,
    "speech-to-text": _generate_speech_to_text,
    "audio-feature-extraction": _generate_audio_feature_extraction,
}


def generate_golden_for_case(case: TestCase, json_path: Path, device: str) -> bool:
    """Generate golden reference data for one test case.

    Returns True on success, False on failure (logged to stderr).
    """
    generator = _GENERATORS.get(case.task_type)
    if generator is None:
        print(
            f"  SKIP: unsupported task_type={case.task_type!r}",
            file=sys.stderr,
        )
        return False

    try:
        generator(case, json_path, device)
    except Exception as exc:
        print(
            f"  ERROR: {case.case_id}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False
    else:
        return True


# ---- Main ----


def main() -> int:
    """Entry point.  Returns 0 on success, 1 if any cases failed."""
    args = parse_args()

    if args.device.startswith("cuda"):
        import torch

        # Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on
        # systems where the cuDNN library version doesn't match the
        # CUDA toolkit bundled with PyTorch.
        torch.backends.cudnn.enabled = False

    from mobius._testing.golden import (
        discover_test_cases,
        golden_path_for_case,
        has_golden,
        load_test_case,
    )

    golden_dir: Path = args.golden_dir

    # Collect test cases.
    if args.case is not None:
        cases = [load_test_case(args.case)]
    else:
        cases = discover_test_cases(
            task_type=args.task_type,
            level=args.level,
            root=args.cases_dir,
        )

    # Apply glob filter on case_id.
    if args.filter:
        cases = [c for c in cases if fnmatch.fnmatch(c.case_id, args.filter)]

    if not cases:
        print("No test cases found matching the given filters.")
        return 0

    print(f"Found {len(cases)} test case(s).")

    succeeded = 0
    skipped = 0
    failed = 0
    failed_ids: list[str] = []

    for case in cases:
        json_path = golden_path_for_case(case, golden_dir=golden_dir)
        label = f"{case.yaml_path.parent.name}/{case.case_id}"

        if case.skip_reason:
            print(f"  SKIP: {label} — {case.skip_reason}")
            skipped += 1
            continue

        if has_golden(case, golden_dir=golden_dir) and not args.force:
            print(f"  EXISTS: {label} (use --force to overwrite)")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  DRY-RUN: {label} → {json_path}")
            skipped += 1
            continue

        print(f"  GENERATING: {label} ...")
        start = time.time()
        ok = generate_golden_for_case(case, json_path, args.device)
        elapsed = time.time() - start

        if ok:
            print(f"  SAVED: {json_path} ({elapsed:.1f}s)")
            succeeded += 1
        else:
            failed += 1
            failed_ids.append(label)

    # Summary
    print(f"\nDone: {succeeded} saved, {skipped} skipped, {failed} failed.")
    if failed_ids:
        print("Failed cases:")
        for fid in failed_ids:
            print(f"  - {fid}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
