# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

# SECURITY: Do NOT use torch.load() or pickle deserialization anywhere in this
# module.  Only safetensors is permitted for weight loading to prevent arbitrary
# code execution from untrusted weight files.

"""Weight loading and application for ONNX models.

This module handles downloading model weights from HuggingFace Hub and
applying them to ONNX IR models. All weight loading uses the safetensors
format exclusively — no ``torch.load`` or pickle deserialization is used,
eliminating arbitrary code execution risks from untrusted weight files.
"""

from __future__ import annotations

__all__ = [
    "apply_weights",
]

import concurrent.futures
import json
import logging

import onnx_ir as ir
import torch
import tqdm
from onnx_ir import tensor_adapters

logger = logging.getLogger(__name__)


def _assign_weight(
    initializer: ir.Value,
    tensor: torch.Tensor,
    name: str,
) -> None:
    """Assign a weight tensor to an initializer with shape/dtype handling.

    This is the single source of truth for weight assignment logic:

    * **Shape mismatch error** — raises :class:`ValueError` when the
      tensor shape differs from the initializer shape.
    * **Lazy dtype cast** — when the tensor dtype differs from the
      initializer's declared ONNX type, wraps the tensor in
      ``ir.LazyTensor`` so the cast happens at serialization time,
      avoiding eager memory allocation.
    """
    # Raise on shape mismatch (initializers always have concrete int dims).
    init_shape = initializer.shape
    if init_shape is not None:
        expected = list(init_shape)
        actual = list(tensor.shape)
        if expected != actual:
            raise ValueError(
                f"Weight shape mismatch for '{name}': model expects {expected}, got {actual}"
            )

    onnx_dtype = initializer.dtype
    assert onnx_dtype is not None
    target_dtype = tensor_adapters.to_torch_dtype(onnx_dtype)

    if tensor.dtype != target_dtype:

        def tensor_func(t=tensor, dt=target_dtype, n=name) -> tensor_adapters.TorchTensor:
            return tensor_adapters.TorchTensor(t.to(dt), name=n)

        ir_tensor = ir.LazyTensor(
            tensor_func,
            dtype=onnx_dtype,
            shape=ir.Shape(tensor.shape),
            name=name,
        )
    else:
        ir_tensor = tensor_adapters.TorchTensor(tensor, name)
    initializer.const_value = ir_tensor


def apply_weights(model: ir.Model, state_dict: dict[str, torch.Tensor]) -> None:
    """Apply weights from a state dict to an ONNX model.

    When the weight dtype differs from the initializer's declared type,
    ``ir.LazyTensor`` is used to lazily cast the tensor at serialization
    time, avoiding eager memory allocation.

    Args:
        model: The ONNX IR model.
        state_dict: Mapping of parameter names to torch tensors.
    """
    for name, tensor in state_dict.items():
        if name not in model.graph.initializers:
            logger.warning(
                "Weight '%s' not found in the model. Skipped applying.",
                name,
            )
            continue

        _assign_weight(model.graph.initializers[name], tensor, name)


def _parallel_download(
    model_id: str, filenames: list[str], *, desc: str = "files"
) -> list[str]:
    """Download files from HuggingFace Hub in parallel.

    Uses a thread pool to download multiple safetensors shards
    concurrently, similar to how ``transformers`` handles sharded
    checkpoints.

    Args:
        model_id: HuggingFace model identifier.
        filenames: List of filenames to download.
        desc: Description for the progress bar.

    Returns:
        List of local file paths in the same order as *filenames*.
    """
    from huggingface_hub import hf_hub_download

    if len(filenames) <= 1:
        # No benefit from parallelism for a single file
        return [hf_hub_download(repo_id=model_id, filename=f) for f in filenames]

    print(f"Downloading {len(filenames)} {desc} files (parallel)...")
    path_map: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(hf_hub_download, repo_id=model_id, filename=f): f
            for f in filenames
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Downloading {desc}",
        ):
            fname = futures[future]
            path_map[fname] = future.result()

    # Return paths in original order
    return [path_map[f] for f in filenames]


def _download_weights(model_id: str) -> dict[str, torch.Tensor]:
    """Download weights from HuggingFace and return as a state dict.

    Uses parallel downloads when multiple safetensors shards exist.
    """
    import safetensors.torch
    from huggingface_hub import hf_hub_download

    try:
        index_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
        )
        with open(index_path) as f:
            index = json.load(f)
        all_files = sorted(set(index["weight_map"].values()))
    except Exception as e:
        if "Entry Not Found" in str(e):
            all_files = ["model.safetensors"]
        else:
            raise

    paths = _parallel_download(model_id, all_files, desc="safetensors")

    state_dict: dict[str, torch.Tensor] = {}
    for path in tqdm.tqdm(paths, desc="Loading weights"):
        state_dict.update(safetensors.torch.load_file(path))
    return state_dict
