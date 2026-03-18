# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Diffusers pipeline building support.

This module handles building ONNX models from HuggingFace diffusers
pipelines (Flux, Stable Diffusion 3, VAEs, etc.).
"""

from __future__ import annotations

__all__ = [
    "build_diffusers_pipeline",
]

import json
import logging

import onnx_ir as ir
import torch
import tqdm

from mobius._builder import build_from_module, resolve_dtype
from mobius._model_package import ModelPackage
from mobius._weight_loading import _parallel_download, apply_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diffusers pipeline support
# ---------------------------------------------------------------------------

#: Mapping of diffusers ``_class_name`` to (module_class, config_class, task_name).
#: Each entry maps a diffusers component class to the mobius
#: module class, config parser, and task used to build the ONNX graph.
_DIFFUSERS_CLASS_MAP: dict[str, tuple[type, type, str]] = {}


def _init_diffusers_class_map() -> None:
    """Lazily populate the diffusers class map on first use."""
    if _DIFFUSERS_CLASS_MAP:
        return

    from mobius._diffusers_configs import (
        CogVideoXConfig,
        QwenImageConfig,
        QwenImageVAEConfig,
        VAEConfig,
    )
    from mobius.models.cogvideox import (
        CogVideoXTransformer3DModel,
    )
    from mobius.models.dit import DiTConfig, DiTTransformer2DModel
    from mobius.models.flux_sd3 import (
        FluxConfig,
        FluxTransformer2DModel,
        SD3Config,
        SD3Transformer2DModel,
    )
    from mobius.models.hunyuan_dit import HunyuanDiT2DModel, HunyuanDiTConfig
    from mobius.models.qwen_image import QwenImageTransformer2DModel
    from mobius.models.qwen_image_vae import AutoencoderKLQwenImageModel
    from mobius.models.vae import AutoencoderKLModel
    from mobius.models.video_vae import VideoAutoencoderModel, VideoVAEConfig

    _DIFFUSERS_CLASS_MAP.update(
        {
            "DiTTransformer2DModel": (DiTTransformer2DModel, DiTConfig, "denoising"),
            "HunyuanDiT2DModel": (HunyuanDiT2DModel, HunyuanDiTConfig, "denoising"),
            "PixArtTransformer2DModel": (DiTTransformer2DModel, DiTConfig, "denoising"),
            "FluxTransformer2DModel": (FluxTransformer2DModel, FluxConfig, "denoising"),
            "SD3Transformer2DModel": (SD3Transformer2DModel, SD3Config, "denoising"),
            "QwenImageTransformer2DModel": (
                QwenImageTransformer2DModel,
                QwenImageConfig,
                "denoising",
            ),
            "AutoencoderKL": (AutoencoderKLModel, VAEConfig, "vae"),
            "AutoencoderKLQwenImage": (
                AutoencoderKLQwenImageModel,
                QwenImageVAEConfig,
                "qwen-image-vae",
            ),
            "AutoencoderKLCogVideoX": (
                VideoAutoencoderModel,
                VideoVAEConfig,
                "vae",
            ),
            "CogVideoXTransformer3DModel": (
                CogVideoXTransformer3DModel,
                CogVideoXConfig,
                "video-denoising",
            ),
        }
    )


def _load_diffusers_pipeline_index(model_id: str) -> dict | None:
    """Try to load a diffusers ``model_index.json`` from HuggingFace.

    Returns the parsed JSON dict, or ``None`` if not found.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id=model_id, filename="model_index.json")
    except (OSError, ValueError) as e:
        logger.debug("Failed to download model_index.json for %s: %s", model_id, e)
        return None

    with open(path) as f:
        return json.load(f)


def _download_diffusers_component_weights(
    model_id: str, component_name: str
) -> dict[str, torch.Tensor]:
    """Download weights for a specific component of a diffusers pipeline.

    Diffusers pipelines store weights in subdirectories using either
    ``diffusion_pytorch_model.safetensors`` (standard) or ``model.safetensors``
    as the weight filename. Sharded weights use a corresponding
    ``.index.json`` file.
    """
    import safetensors.torch
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    prefix = f"{component_name}/"
    # Diffusers uses two naming conventions for weight files
    weight_basenames = ["diffusion_pytorch_model", "model"]

    all_files = None
    for basename in weight_basenames:
        try:
            index_path = hf_hub_download(
                repo_id=model_id,
                filename=f"{prefix}{basename}.safetensors.index.json",
            )
            with open(index_path) as f:
                index = json.load(f)
            all_files = sorted(set(index["weight_map"].values()))
            break
        except EntryNotFoundError:
            continue

    if all_files is None:
        # No index file found — try single-file weights
        for basename in weight_basenames:
            try:
                hf_hub_download(
                    repo_id=model_id,
                    filename=f"{prefix}{basename}.safetensors",
                )
                all_files = [f"{basename}.safetensors"]
                break
            except EntryNotFoundError:
                continue

    if all_files is None:
        raise FileNotFoundError(
            f"Could not find weight files for component '{component_name}' "
            f"in '{model_id}'. Tried diffusion_pytorch_model.safetensors "
            f"and model.safetensors."
        )

    paths = _parallel_download(
        model_id,
        [f"{prefix}{f}" for f in all_files],
        desc=f"{component_name} weights",
    )

    state_dict: dict[str, torch.Tensor] = {}
    for path in tqdm.tqdm(paths, desc=f"Loading {component_name} weights"):
        state_dict.update(safetensors.torch.load_file(path))
    return state_dict


def _load_diffusers_component_config(model_id: str, component_name: str) -> dict:
    """Load the config.json for a specific diffusers pipeline component."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=model_id, filename=f"{component_name}/config.json")
    with open(path) as f:
        return json.load(f)


def build_diffusers_pipeline(
    model_id: str,
    *,
    dtype: str | ir.DataType | None = None,
    load_weights: bool = True,
) -> ModelPackage:
    """Build ONNX models for all supported components in a diffusers pipeline.

    Parses the pipeline's ``model_index.json`` and builds each neural network
    component (transformer, VAE, etc.) as a separate ONNX model in the
    returned :class:`ModelPackage`.

    Components that are not neural networks (schedulers, tokenizers) or that
    don't have a registered ONNX model class are skipped.

    Args:
        model_id: HuggingFace model repository ID for a diffusers pipeline.
        dtype: Override the model dtype.
        load_weights: Whether to download and apply weights.

    Returns:
        A :class:`ModelPackage` containing the built component model(s).

    Raises:
        ValueError: If the model does not have a ``model_index.json``.
    """
    _init_diffusers_class_map()

    pipeline_index = _load_diffusers_pipeline_index(model_id)
    if pipeline_index is None:
        raise ValueError(
            f"'{model_id}' does not appear to be a diffusers pipeline "
            f"(no model_index.json found)."
        )

    if dtype is not None and isinstance(dtype, str):
        dtype = resolve_dtype(dtype)

    package = ModelPackage({})

    for component_name, component_info in pipeline_index.items():
        if component_name.startswith("_"):
            continue
        if not isinstance(component_info, list) or len(component_info) != 2:
            continue

        library, class_name = component_info
        if class_name not in _DIFFUSERS_CLASS_MAP:
            logger.info(
                "Skipping diffusers component '%s' (class '%s' from '%s' is not registered).",
                component_name,
                class_name,
                library,
            )
            continue

        module_class, config_class, task_name = _DIFFUSERS_CLASS_MAP[class_name]
        logger.info(
            "Building diffusers component '%s' (%s)...",
            component_name,
            class_name,
        )

        component_config_dict = _load_diffusers_component_config(model_id, component_name)
        config = config_class.from_diffusers(component_config_dict)

        if dtype is not None and hasattr(config, "dtype"):
            import dataclasses

            config = dataclasses.replace(config, dtype=dtype)

        model_module = module_class(config)

        sub_pkg = build_from_module(model_module, config, task_name)

        # Flatten sub-package into the top-level package
        if len(sub_pkg) == 1 and "model" in sub_pkg:
            sub_pkg["model"].graph.name = f"{model_id}/{component_name}"
            package[component_name] = sub_pkg["model"]
        else:
            for sub_name, sub_model in sub_pkg.items():
                sub_model.graph.name = f"{model_id}/{component_name}_{sub_name}"
                package[f"{component_name}_{sub_name}"] = sub_model

        if load_weights:
            state_dict = _download_diffusers_component_weights(model_id, component_name)
            if hasattr(model_module, "preprocess_weights"):
                state_dict = model_module.preprocess_weights(state_dict)
            for model in sub_pkg.values():
                apply_weights(model, state_dict)

    if not package:
        raise ValueError(
            f"No supported neural network components found in '{model_id}'. "
            f"Supported diffusers classes: {sorted(_DIFFUSERS_CLASS_MAP)}."
        )

    return package
