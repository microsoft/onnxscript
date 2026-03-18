# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""GGUF → ONNX build pipeline.

Converts ``.gguf`` model files to ONNX using the standard build
pipeline.  Two modes:

- **Dequantized** (default): All quantized tensors are dequantized to
  float.  Simple, but loses the compression benefit of quantization.
- **Quantized** (``keep_quantized=True``): Linear-layer weights that
  use supported GGUF types (Q4_0, Q4_1, Q8_0) are repacked into
  MatMulNBits format.  Other tensors (norms, embeddings) are
  dequantized as usual.
"""

from __future__ import annotations

__all__ = ["build_from_gguf"]

import logging
from collections import Counter
from pathlib import Path

import tqdm

from mobius._model_package import ModelPackage

logger = logging.getLogger(__name__)


def build_from_gguf(
    gguf_path: str | Path,
    *,
    task: str | None = None,
    dtype: str | None = None,
    keep_quantized: bool = False,
) -> ModelPackage:
    """Build an ONNX :class:`ModelPackage` from a GGUF file.

    1. Parse GGUF metadata → :class:`ArchitectureConfig`
    2. Look up the model class and task from the registry
    3. Build the ONNX graph (standard ``build_from_module`` pipeline)
    4. Map GGUF tensor names → HuggingFace names
    5. Apply architecture-specific tensor processors
    6. Run ``preprocess_weights()`` (HF → ONNX name mapping)
    7. Apply weights to the ONNX model

    When *keep_quantized* is ``True``, supported quantized tensors are
    repacked into MatMulNBits format instead of dequantized.

    Args:
        gguf_path: Path to the ``.gguf`` file.
        task: Override the model task (e.g. ``"text-generation"``).
            When ``None``, the task is auto-detected from the
            model type.
        dtype: Override model dtype (e.g. ``"f16"``). When ``None``,
            defaults to float32.
        keep_quantized: When ``True``, preserve quantization for
            supported GGUF types (Q4_0, Q4_1, Q8_0) by repacking
            linear-layer weights into MatMulNBits format.

    Returns:
        A :class:`ModelPackage` containing the built model(s).

    Raises:
        ImportError: If the ``gguf`` package is not installed.
        FileNotFoundError: If the GGUF file does not exist.
        KeyError: If the GGUF architecture is not in the registry.
    """
    import dataclasses

    from mobius._builder import (
        build_from_module,
        resolve_dtype,
    )
    from mobius._config_resolver import (
        _default_task_for_model,
    )
    from mobius._registry import registry
    from mobius.integrations.gguf._config_mapping import (
        GGUF_ARCH_TO_MODEL_TYPE,
        gguf_to_config,
    )
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.integrations.gguf._tensor_processors import (
        process_tensors,
    )

    # 1. Parse GGUF file
    gguf_model = GGUFModel(gguf_path)
    gguf_arch = gguf_model.architecture
    logger.info("Loaded GGUF file: %s (arch=%s)", gguf_path, gguf_arch)

    # 2. Extract config from GGUF metadata
    config = gguf_to_config(gguf_model)
    model_type = getattr(config, "_gguf_model_type", None)
    if model_type is None:
        model_type = GGUF_ARCH_TO_MODEL_TYPE.get(gguf_arch, gguf_arch)

    if dtype is not None:
        resolved = resolve_dtype(dtype)
        if resolved is not None:
            config = dataclasses.replace(config, dtype=resolved)

    # 3. Quantized path: detect dominant type and set config
    if keep_quantized:
        from mobius._configs import QuantizationConfig

        bits, is_sym = _detect_quant_params(gguf_model, gguf_arch)
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=bits,
                group_size=32,
                quant_method="gguf",
                sym=is_sym,
            ),
        )
        logger.info("Quantized mode: bits=%d, symmetric=%s", bits, is_sym)

    # 4. Look up module class and resolve task
    module_class = registry.get(model_type)
    if task is None:
        task = _default_task_for_model(model_type)

    # 5. Build ONNX graph
    module = module_class(config)
    pkg = build_from_module(module, config, task)
    logger.info(
        "Built ONNX graph for %s (%d components)",
        model_type,
        len(pkg),
    )

    # 6. Load tensors from GGUF → state_dict
    if keep_quantized:
        state_dict = _load_quantized_state_dict(gguf_model, gguf_arch, module, config)
    else:
        state_dict = _load_dequantized_state_dict(gguf_model, gguf_arch)

    logger.info(
        "Mapped %d state_dict entries from GGUF tensors",
        len(state_dict),
    )

    # 7. Apply architecture-specific tensor processors.
    # For the quantized path, only float tensors go through
    # process_tensors; quantized Q/K tensors were permuted in
    # _load_quantized_state_dict already.
    if keep_quantized:
        float_keys = {
            k
            for k in state_dict
            if not (
                k.endswith((".scales", ".zero_points")) or _is_quantized_weight(k, state_dict)
            )
        }
        float_dict = {k: state_dict[k] for k in float_keys}
        quant_dict = {k: state_dict[k] for k in state_dict if k not in float_keys}
        float_dict = process_tensors(float_dict, config)
        state_dict = {**float_dict, **quant_dict}
    else:
        state_dict = process_tensors(state_dict, config)

    # 7b. Normalize GGUF-specific weight shapes to match HF conventions.
    # This converts GGUF tensor quirks (stacked experts, 1D gates, 2D
    # conv weights, suffix artifacts) into the shapes that HF models
    # produce, so preprocess_weights only needs to handle HF→ONNX.
    state_dict = _normalize_gguf_weights(state_dict)

    # 8. Run model-specific preprocess_weights (HF → ONNX names)
    if hasattr(module, "preprocess_weights"):
        state_dict = module.preprocess_weights(state_dict)

    # 9. Apply weights to ONNX model
    prefix_map = getattr(module, "weight_prefix_map", None)
    pkg.apply_weights(state_dict, prefix_map=prefix_map)

    return pkg


def _is_quantized_weight(key: str, state_dict: dict) -> bool:
    """Check if a .weight key has a matching .scales companion."""
    if not key.endswith(".weight"):
        return False
    stem = key[: -len(".weight")]
    return f"{stem}.scales" in state_dict


def _normalize_gguf_weights(
    state_dict: dict,
) -> dict:
    """Normalize GGUF-specific weight shapes to match HF conventions.

    GGUF tensor mapping + dequantization produces weights that differ
    from HuggingFace in several ways. This function converts them so
    that ``preprocess_weights`` only needs to handle HF→ONNX mapping.

    Transforms applied:

    - **Stacked expert weights**: GGUF provides separate 3D tensors
      ``experts.{gate,up,down}_proj.weight`` with shape
      ``[num_experts, out, in]``. These are unpacked into per-expert
      ``experts.{i}.{proj}.weight`` tensors, matching the HF
      ``experts.down_proj`` format that ``preprocess_weights`` expects.
    - **1D shared_expert_gate**: GGUF stores as ``[hidden]``; HF/ONNX
      ``Linear(hidden, 1)`` expects ``[1, hidden]``.
    - **2D conv1d**: GGUF stores as ``[channels, kernel]``; depthwise
      ``Conv1d`` expects ``[channels, 1, kernel]``.
    - **dt_bias suffix**: GGUF ``ssm_dt.bias`` maps to
      ``dt_bias.bias`` after suffix splitting, but the model parameter
      is just ``dt_bias`` (an ``nn.Parameter``, not a module bias).
    """
    import torch

    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        # Stacked expert weights [num_experts, out, in] → per-expert
        unpacked = False
        for proj in ("gate_proj", "up_proj", "down_proj"):
            suffix = f".mlp.experts.{proj}.weight"
            if key.endswith(suffix) and value.dim() == 3:
                prefix = key[: -len(suffix)]
                for i in range(value.shape[0]):
                    result[f"{prefix}.mlp.experts.{i}.{proj}.weight"] = value[i]
                unpacked = True
                break
        if unpacked:
            continue

        # 1D shared_expert_gate → [1, hidden]
        if key.endswith(".mlp.shared_expert_gate.weight") and value.dim() == 1:
            result[key] = value.unsqueeze(0)
            continue

        # 2D conv1d → [channels, 1, kernel]
        if key.endswith(".conv1d.weight") and value.dim() == 2:
            result[key] = value.unsqueeze(1)
            continue

        # dt_bias.bias → dt_bias (nn.Parameter, not module bias)
        if key.endswith(".dt_bias.bias"):
            result[key[: -len(".bias")]] = value
            continue

        result[key] = value

    return result


def _detect_quant_params(gguf_model, gguf_arch: str) -> tuple[int, bool]:
    """Detect bits and symmetry from dominant GGUF quant type.

    Scans block-level (projection) tensors and returns the
    bit-width and symmetry flag of the most common repackable type.

    Returns:
        ``(bits, is_symmetric)`` tuple.

    Raises:
        ValueError: If no repackable quantized tensors are found.
    """
    from gguf import GGMLQuantizationType

    from mobius.integrations.gguf._repacker import can_repack
    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
    )

    q4_types = {
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q4_1,
    }
    symmetric_types = {
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q8_0,
    }

    counts: Counter = Counter()
    for name, _raw, qtype, _shape in gguf_model.tensor_items_raw():
        hf_name = map_gguf_to_hf_names(name, gguf_arch)
        if hf_name is None:
            continue
        if can_repack(qtype.value if hasattr(qtype, "value") else qtype):
            counts[qtype] += 1

    if not counts:
        raise ValueError(
            "No repackable quantized tensors found in GGUF file. "
            "Use keep_quantized=False for dequantized import."
        )

    dominant = counts.most_common(1)[0][0]
    bits = 4 if dominant in q4_types else 8
    is_sym = dominant in symmetric_types
    logger.info(
        "Dominant GGUF quant type: %s (%d tensors, bits=%d)",
        dominant,
        counts[dominant],
        bits,
    )
    return bits, is_sym


def _load_dequantized_state_dict(
    gguf_model,
    gguf_arch: str,
) -> dict:
    """Load all tensors dequantized to float (Phase 1 path)."""
    import numpy as np
    import torch

    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
    )

    state_dict = {}
    for gguf_name, np_array in tqdm.tqdm(
        gguf_model.tensor_items(),
        desc="Dequantizing tensors",
        total=len(gguf_model._tensor_index),
    ):
        hf_name = map_gguf_to_hf_names(gguf_name, gguf_arch)
        if hf_name is not None:
            # F32/F16 tensors are mmap'd read-only views; make
            # writable so PyTorch can mutate if needed.
            if not np_array.flags.writeable:
                np_array = np.array(np_array)
            state_dict[hf_name] = torch.from_numpy(np_array)
        else:
            logger.warning("Unmapped GGUF tensor: %s (skipped)", gguf_name)
    return state_dict


def _load_quantized_state_dict(
    gguf_model,
    gguf_arch: str,
    module,
    config,
) -> dict:
    """Load tensors, repacking supported quantized types.

    Projection weights (Q/K/V/O and MLP) that use repackable GGUF
    types are converted to MatMulNBits format.  All other tensors
    (norms, embeddings, unsupported quant types) are dequantized.

    For llama-family models, quantized Q/K weights receive the
    row-level reverse-permutation that ``process_tensors`` would
    normally apply.
    """
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType, dequantize

    from mobius.components import QuantizedLinear
    from mobius.integrations.gguf._repacker import (
        can_repack,
        repack_gguf_tensor,
    )
    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
    )
    from mobius.integrations.gguf._tensor_processors import (
        _reverse_permute,
    )

    # Collect module paths that use QuantizedLinear so we know
    # which .weight parameters should receive repacked data.
    quantized_stems = set()
    for mod_name, mod in module.named_modules():
        if isinstance(mod, QuantizedLinear):
            quantized_stems.add(mod_name)

    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)

    state_dict: dict[str, torch.Tensor] = {}
    n_repacked = 0

    for gguf_name, raw, qtype, np_shape in tqdm.tqdm(
        gguf_model.tensor_items_raw(),
        desc="Repacking tensors",
        total=len(gguf_model._tensor_index),
    ):
        hf_name = map_gguf_to_hf_names(gguf_name, gguf_arch)
        if hf_name is None:
            logger.warning("Unmapped GGUF tensor: %s (skipped)", gguf_name)
            continue

        # Determine the int value of the quant type for can_repack
        qtype_val = qtype.value if hasattr(qtype, "value") else qtype

        # Repack if the tensor is repackable AND its target ONNX
        # parameter is from a QuantizedLinear module.
        stem = hf_name[: -len(".weight")] if hf_name.endswith(".weight") else None
        should_repack = stem is not None and stem in quantized_stems and can_repack(qtype_val)

        if should_repack:
            shape_2d = (int(np_shape[0]), int(np_shape[1]))
            repacked = repack_gguf_tensor(
                raw.ravel().view(np.uint8),
                qtype_val,
                shape_2d,
            )
            w = torch.from_numpy(repacked.weight)
            s = torch.from_numpy(repacked.scales)

            # Apply Q/K row permutation to quantized tensors
            # (same transform as _process_llama, on all arrays)
            if _needs_qk_permute(hf_name, num_heads, num_kv_heads):
                n_head = (
                    num_heads
                    if ".q_proj." in hf_name or ".qkv_proj." in hf_name
                    else num_kv_heads
                )
                w = _reverse_permute(w, n_head)
                s = _reverse_permute(s, n_head)

            state_dict[hf_name] = w
            state_dict[f"{stem}.scales"] = s
            if repacked.zero_points is not None:
                zp = torch.from_numpy(repacked.zero_points)
                if _needs_qk_permute(hf_name, num_heads, num_kv_heads):
                    zp = _reverse_permute(zp, n_head)
                state_dict[f"{stem}.zero_points"] = zp
            n_repacked += 1
        else:
            # Dequantize to float
            if qtype in (
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F16,
            ):
                arr = gguf_model.get_tensor(gguf_name)
                # F32/F16 tensors are mmap'd read-only views
                if not arr.flags.writeable:
                    arr = np.array(arr)
            else:
                arr = dequantize(raw, qtype).reshape(np_shape)
            state_dict[hf_name] = torch.from_numpy(arr)

    logger.info(
        "Loaded %d tensors (%d repacked as MatMulNBits, %d dequantized)",
        len(state_dict),
        n_repacked,
        len(state_dict) - n_repacked,
    )
    return state_dict


def _needs_qk_permute(
    hf_name: str,
    num_heads: int | None,
    num_kv_heads: int | None,
) -> bool:
    """Check if this tensor needs Q/K reverse-permutation.

    Name-based gating (``.q_proj.``, ``.k_proj.``, ``.qkv_proj.``) is
    sufficient without a model_type guard because non-llama architectures
    use different projection names (``.query_key_value.``, ``.c_attn.``,
    etc.) that won't match these patterns.
    """
    if num_heads is None or num_kv_heads is None:
        return False
    return (
        ".q_proj." in hf_name or ".k_proj." in hf_name or ".qkv_proj." in hf_name
    ) and hf_name.endswith(".weight")
