# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Auto-export pipeline for onnxruntime-genai.

Chains: build() → apply weights → generate genai_config.json → save ONNX
models → copy tokenizer files. Produces a directory that onnxruntime-genai
can load directly.

This module imports from the core library at call time (not module load)
to keep the integration layer lightweight.

Example::

    from mobius.integrations.ort_genai import auto_export

    auto_export("Qwen/Qwen3-0.6B", "/output/qwen3")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any

logger = logging.getLogger(__name__)

# ORT-GenAI model type overrides for model types whose ORT-GenAI name
# differs from the HuggingFace model_type.
_ORT_GENAI_MODEL_TYPE: dict[str, str] = {
    "llama": "llama",
    "qwen2": "qwen2",
    "qwen3": "qwen2",
    "phi3": "phi3",
    "phi": "phi",
    "phi4mm": "phi4mm",
    "phi4_multimodal": "phi4mm",
    "gemma": "gemma",
    "gemma2": "gemma",
    "mistral": "mistral",
}


def _resolve_ort_genai_model_type(model_type: str) -> str:
    """Map HuggingFace model_type to ORT-GenAI model type string."""
    return _ORT_GENAI_MODEL_TYPE.get(model_type, model_type)


def _copy_tokenizer_files(
    model_id: str,
    output_dir: str,
) -> list[str]:
    """Download and copy tokenizer files from HuggingFace Hub.

    Returns list of copied filenames.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",  # SentencePiece
        "added_tokens.json",
        "merges.txt",  # BPE
        "vocab.json",  # BPE
    ]
    copied: list[str] = []
    for filename in tokenizer_files:
        try:
            src = hf_hub_download(model_id, filename)
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            copied.append(filename)
        except (EntryNotFoundError, OSError):
            continue
    return copied


def _write_processor_config(
    config: Any,
    output_dir: str,
) -> str | None:
    """Write a minimal processor_config.json for VLM models.

    Returns the path if written, None otherwise.
    """
    vision = getattr(config, "vision", None)
    if vision is None:
        return None

    processor: dict[str, Any] = {
        "image_size": getattr(vision, "image_size", 448),
        "patch_size": getattr(vision, "patch_size", 14),
    }
    path = os.path.join(output_dir, "processor_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(processor, f, indent=4)
    return path


def auto_export(
    model_id: str,
    output_dir: str,
    *,
    dtype: str | None = None,
    task: str | None = None,
    external_data: str = "onnx",
    trust_remote_code: bool = False,
    context_length: int = 4096,
    progress_bar: bool = True,
) -> dict[str, str]:
    """Build and export a model for onnxruntime-genai.

    This is the main entry point for producing ORT-GenAI-ready model
    directories. It:

    1. Builds the ONNX graph(s) via :func:`build`
    2. Downloads and applies HuggingFace weights
    3. Generates ``genai_config.json``
    4. Saves ONNX model(s) with external data
    5. Copies tokenizer files from HuggingFace Hub

    Args:
        model_id: HuggingFace model repository ID.
        output_dir: Directory to write all output files.
        dtype: Override model dtype (``"f32"``, ``"f16"``, ``"bf16"``).
        task: Override model task (auto-detected if ``None``).
        external_data: External data format (``"onnx"`` or
            ``"safetensors"``).
        trust_remote_code: Trust remote code for HuggingFace config.
        context_length: Maximum context length for genai_config.json.
        progress_bar: Show progress bar during save.

    Returns:
        Dict mapping output artifact names to file paths, e.g.::

            {
                "genai_config": "/output/genai_config.json",
                "model": "/output/model.onnx",
                "tokenizer": "/output/tokenizer.json",
            }
    """
    import transformers

    from mobius._builder import build
    from mobius.integrations.ort_genai.genai_config import (
        GenaiConfigGenerator,
    )

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load HF config for metadata (model_type, token IDs)
    hf_config = transformers.AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    model_type = hf_config.model_type
    ort_model_type = _resolve_ort_genai_model_type(model_type)

    # 2. Build ONNX graph(s) with weights
    logger.info("Building ONNX model for %s", model_id)
    pkg = build(
        model_id,
        task=task,
        dtype=dtype,
        load_weights=True,
        trust_remote_code=trust_remote_code,
    )
    config = getattr(pkg, "config", None)
    if config is None:
        raise ValueError(
            f"Model package for '{model_id}' has no config attribute. "
            "auto_export requires a config to generate genai_config.json. "
            "Diffusion models are not yet supported."
        )

    # 3. Detect multimodal capabilities from the model package
    is_vlm = "vision" in pkg and "embedding" in pkg
    has_speech = "speech" in pkg

    # Auto-detect phi4mm: HF model_type may be "phi" but model has speech
    if ort_model_type == "phi" and has_speech:
        ort_model_type = "phi4mm"

    # 4. Generate genai_config.json
    logger.info("Generating genai_config.json")
    generator = GenaiConfigGenerator.from_config(
        config,
        ort_model_type,
        context_length=context_length,
        bos_token_id=getattr(hf_config, "bos_token_id", None),
        eos_token_id=getattr(hf_config, "eos_token_id", None),
        pad_token_id=getattr(hf_config, "pad_token_id", None),
    )
    if is_vlm:
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is not None:
            # Phi4MM uses different vision inputs than Qwen2.5-VL
            vision_kwargs: dict[str, Any] = {}
            if has_speech:
                vision_kwargs["spatial_merge_size"] = None
                vision_kwargs["config_filename"] = "vision_processor.json"
                vision_kwargs["input_names"] = {
                    "pixel_values": "pixel_values",
                    "image_sizes": "image_sizes",
                }
            generator.with_vision(image_token_id=image_token_id, **vision_kwargs)

    if has_speech:
        audio_config = getattr(config, "audio", None)
        audio_token_id = (
            getattr(audio_config, "token_id", None) if audio_config is not None else None
        )
        generator.with_speech(audio_token_id=audio_token_id)

    genai_path = generator.write(output_dir)

    # 5. Save ONNX models
    logger.info("Saving ONNX models to %s", output_dir)
    pkg.save(
        output_dir,
        external_data=external_data,
        progress_bar=progress_bar,
    )

    # 6. Copy tokenizer files
    logger.info("Copying tokenizer files")
    tokenizer_files = _copy_tokenizer_files(model_id, output_dir)

    # 7. Write processor_config.json for VLMs
    processor_path = _write_processor_config(config, output_dir)

    # Build result manifest
    result: dict[str, str] = {"genai_config": genai_path}
    if len(pkg) == 1:
        result["model"] = os.path.join(output_dir, "model.onnx")
    else:
        for name in pkg:
            result[name] = os.path.join(output_dir, name, "model.onnx")
    for tf in tokenizer_files:
        result[tf] = os.path.join(output_dir, tf)
    if processor_path:
        result["processor_config"] = processor_path

    logger.info("Export complete: %d artifacts", len(result))
    return result
