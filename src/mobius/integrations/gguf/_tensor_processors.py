# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Architecture-specific tensor processors for GGUF import.

GGUF files store tensors in different layouts or conventions than
HuggingFace checkpoints. This module applies the necessary transforms
after GGUF→HF name mapping but before ``preprocess_weights()``.

Known transforms (mirrored from HF's ``modeling_gguf_pytorch_utils.py``):

- **Llama/Mistral** — Q/K weight reverse-permutation. GGUF stores
  Q/K weights interleaved as ``(n_head, dim//2, 2, ...)``; HF uses
  standard ``(n_head * head_dim, hidden_size)`` layout.
- **Gemma2/Gemma3** — Norm weight offset. GGUF stores
  ``w_gguf = w_hf - 1``; we restore by adding 1.
- **Nemotron** — Same norm weight offset as Gemma.
- **GPT-2** — Weight transpose for attn and FFN projections.
- **Mamba** — ``conv1d.weight`` needs an extra dimension;
  ``A_log`` needs ``log(-x)`` transform.

Usage::

    from mobius.integrations.gguf._tensor_processors import (
        process_tensors,
    )

    state_dict = process_tensors(state_dict, config)
"""

from __future__ import annotations

__all__ = ["process_tensors", "_reverse_permute"]

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def process_tensors(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Apply architecture-specific tensor transformations.

    Dispatches to an architecture-specific processor based on
    ``config.model_type``. If no processor is registered for the
    model type, the state dict is returned unchanged.

    Args:
        state_dict: HuggingFace-named state dict from GGUF import.
        config: The :class:`ArchitectureConfig` for this model.
            Must have ``model_type``, ``num_attention_heads``, and
            ``num_key_value_heads`` attributes.

    Returns:
        The transformed state dict (modified in-place).
    """
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        return state_dict

    processor = _PROCESSORS.get(model_type)
    if processor is None:
        return state_dict

    return processor(state_dict, config)


# ---------------------------------------------------------------------------
# Processor implementations
# ---------------------------------------------------------------------------


def _process_llama(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Reverse-permute Q/K weights for Llama/Mistral.

    Reference: ``LlamaTensorProcessor`` in HF's
    ``modeling_gguf_pytorch_utils.py``.
    Reference: https://github.com/ggerganov/llama.cpp/blob/
    a38b884c6c4b0c256583acfaaabdf556c62fabea/convert_hf_to_gguf.py#L1402
    """
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_heads is None or num_kv_heads is None:
        logger.warning(
            "Cannot reverse-permute Q/K weights: "
            "num_attention_heads or num_key_value_heads not in config"
        )
        return state_dict

    for name, tensor in state_dict.items():
        if ".q_proj." in name and name.endswith(".weight"):
            state_dict[name] = _reverse_permute(tensor, num_heads)
        elif ".k_proj." in name and name.endswith(".weight"):
            state_dict[name] = _reverse_permute(tensor, num_kv_heads)

    return state_dict


def _reverse_permute(
    weights: torch.Tensor,
    n_head: int,
) -> torch.Tensor:
    """Reverse the Q/K weight permutation applied by llama.cpp.

    llama.cpp interleaves head dimensions as:
        ``(n_head, dim, 2, ...) -> swapaxes(2,1) -> reshape``

    We reverse this to get standard HF layout.

    Args:
        weights: The Q or K projection weight tensor.
        n_head: Number of heads for this tensor — ``num_attention_heads``
            for Q weights, ``num_key_value_heads`` for K weights.
    """
    dim = weights.shape[0] // n_head // 2
    w = weights.reshape(n_head, 2, dim, *weights.shape[1:])
    return w.swapaxes(1, 2).reshape(weights.shape)


def _process_gemma(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Restore Gemma2/3 norm weights.

    GGUF stores ``w_gguf = w_hf - 1`` for all norm weights.
    Reference: ``Gemma2TensorProcessor`` in HF's
    ``modeling_gguf_pytorch_utils.py``.
    """
    for name in list(state_dict):
        if "norm" in name and name.endswith(".weight"):
            state_dict[name] = state_dict[name] + 1
    return state_dict


def _process_nemotron(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Restore Nemotron norm weights (same offset as Gemma).

    Reference: ``NemotronTensorProcessor`` in HF's
    ``modeling_gguf_pytorch_utils.py``.
    """
    for name in list(state_dict):
        if "norm" in name and name.endswith(".weight"):
            state_dict[name] = state_dict[name] + 1
    return state_dict


def _process_gpt2(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Transpose GPT-2 attention and FFN weights.

    GGUF stores these transposed relative to HF convention.
    Reference: ``GPT2TensorProcessor`` in HF's
    ``modeling_gguf_pytorch_utils.py``.
    """
    for name, tensor in list(state_dict.items()):
        needs_transpose = (
            ".c_attn." in name or ".c_proj." in name or ".c_fc." in name
        ) and name.endswith(".weight")
        if needs_transpose:
            state_dict[name] = tensor.T
    return state_dict


def _process_mamba(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """Fix Mamba tensor shapes and transforms.

    - ``conv1d.weight``: unsqueeze dim 1 (GGUF is 2D, HF is 3D)
    - ``A_log``: GGUF stores ``-exp(A_log)``; restore with ``log(-x)``

    Reference: ``MambaTensorProcessor`` in HF's
    ``modeling_gguf_pytorch_utils.py``.
    """
    for name, tensor in list(state_dict.items()):
        if "conv1d" in name and name.endswith(".weight"):
            if tensor.dim() == 2:
                state_dict[name] = tensor.unsqueeze(1)
        elif "A_log" in name:
            state_dict[name] = torch.from_numpy(np.log(-tensor.numpy()))
    return state_dict


# Map model_type → processor function.
# Architectures not listed here need no tensor transforms.
_PROCESSORS: dict[str, Any] = {
    "llama": _process_llama,
    "mistral": _process_llama,
    "qwen2": _process_llama,
    "qwen3": _process_llama,
    "gemma": _process_gemma,
    "gemma2": _process_gemma,
    "gemma3": _process_gemma,
    "nemotron": _process_nemotron,
    "gpt2": _process_gpt2,
    "mamba": _process_mamba,
}
