# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Model building API.

This module provides the core functions for constructing ONNX models from
``onnxscript.nn.Module`` instances:

- :func:`build_from_module` — Build from a module instance and config.
- :func:`build` — Build from a HuggingFace model ID.
- :func:`resolve_dtype` — Resolve dtype strings to ``ir.DataType``.
"""

from __future__ import annotations

__all__ = [
    "DTYPE_MAP",
    "build",
    "build_from_module",
    "resolve_dtype",
]

import logging

import onnx_ir as ir
import onnx_shape_inference
import onnxscript.optimizer._constant_folding  # TODO(justinchuby): Expose the FoldConstantsPass from onnxscript
import torch
from onnx_ir import tensor_adapters
from onnx_ir.passes import common as common_passes
from onnxscript import nn

from mobius._configs import (
    BaseModelConfig,
)
from mobius._model_package import ModelPackage
from mobius._registry import registry
from mobius._weight_loading import _download_weights
from mobius.tasks import ModelTask, get_task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public build API
# ---------------------------------------------------------------------------


class SymbolicShapeInferencePass(ir.passes.InPlacePass):
    """ONNX IR pass that applies symbolic shape inference to all nodes."""

    def __init__(self, policy: onnx_shape_inference.ShapeMergePolicy = "refine"):
        super().__init__()
        self.policy = policy

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        onnx_shape_inference.infer_symbolic_shapes(model, policy=self.policy)
        return ir.passes.PassResult(model, modified=True)


class CleanupMetadataPass(ir.passes.InPlacePass):
    """ONNX IR pass that removes redundant metadata from all nodes."""

    def __init__(self):
        self.keys_to_remove = ["pkg.onnxscript.shape_inference_error"]

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        for node in model.graph.all_nodes():
            for key in self.keys_to_remove:
                if key in node.metadata_props:
                    modified = True
                    del node.metadata_props[key]
        return ir.passes.PassResult(model, modified=modified)


_DEFAULT_PASSES = [
    common_passes.IdentityEliminationPass(),
    common_passes.LiftConstantsToInitializersPass(),
    common_passes.DeduplicateInitializersPass(),
    common_passes.CommonSubexpressionEliminationPass(),
    common_passes.RemoveUnusedNodesPass(),
    common_passes.RemoveUnusedOpsetsPass(),
    SymbolicShapeInferencePass(),
    onnxscript.optimizer._constant_folding.FoldConstantsPass(
        shape_inference=False, input_size_limit=8192, output_size_limit=512 * 512
    ),
    CleanupMetadataPass(),
]


def _optimize(model: ir.Model) -> None:
    """Apply default optimization passes to a model in-place."""
    pass_ = ir.passes.PassManager(_DEFAULT_PASSES, steps=2)
    pass_(model)


# Mapping of short dtype names to ONNX IR dtypes
DTYPE_MAP: dict[str, ir.DataType] = {
    "f32": ir.DataType.FLOAT,
    "float32": ir.DataType.FLOAT,
    "f16": ir.DataType.FLOAT16,
    "float16": ir.DataType.FLOAT16,
    "bf16": ir.DataType.BFLOAT16,
    "bfloat16": ir.DataType.BFLOAT16,
}


def resolve_dtype(dtype: str | ir.DataType | None) -> ir.DataType | None:
    """Resolve a dtype string to an ``ir.DataType``.

    Args:
        dtype: A dtype string (e.g. ``"f16"``), an ``ir.DataType``, or ``None``.

    Returns:
        The resolved ``ir.DataType``, or ``None`` if *dtype* is ``None``.

    Raises:
        ValueError: If the dtype string is not recognised.
    """
    if dtype is None or isinstance(dtype, ir.DataType):
        return dtype
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{dtype}'. Available: {sorted(DTYPE_MAP)}")
    return DTYPE_MAP[dtype]


def _cast_module_dtype(module: nn.Module, dtype: ir.DataType) -> None:
    """Cast all FLOAT parameters in a module to the target dtype before graph building.

    This must be called **before** tracing/building the graph so that ONNX
    type inference propagates the correct dtype through all intermediate
    values. For parameters with precomputed ``const_value`` (e.g. RoPE
    caches), the underlying data is also cast.

    Only recasts parameters that are currently FLOAT (float32). Integer
    parameters and non-float types are left unchanged.
    """
    if dtype == ir.DataType.FLOAT:
        return
    torch_dtype = tensor_adapters.to_torch_dtype(dtype)
    for param in module.parameters():
        if param.dtype != ir.DataType.FLOAT:
            continue
        param.type = ir.TensorType(dtype)
        if param.const_value is not None:
            cast_tensor = torch.from_numpy(param.const_value.numpy()).to(torch_dtype)
            param.const_value = tensor_adapters.TorchTensor(cast_tensor)


def build_from_module(
    module: nn.Module,
    config: BaseModelConfig,
    task: str | ModelTask = "text-generation",
) -> ModelPackage:
    """Build an ONNX :class:`ModelPackage` from a module instance and config.

    Use this when you have a custom :class:`onnxscript.nn.Module` or want
    full control over module construction. The model dtype is determined
    by ``config.dtype``.

    Args:
        module: An ``onnxscript.nn.Module`` instance. Its ``forward()``
            signature must be compatible with the task.
        config: Architecture configuration. The ``dtype`` field controls
            the target precision for model weights. If the config has a
            ``validate()`` method, it is called before graph construction
            to catch invalid fields early.
        task: The model task. Either a task name string
            (e.g. ``"text-generation"``) or a :class:`ModelTask` instance.

    Returns:
        A :class:`ModelPackage` containing the built model(s).

    Raises:
        ValueError: If config validation fails (e.g. non-positive
            ``hidden_size``, ``num_attention_heads``, etc.).

    Example::

        from onnxscript import nn
        from mobius import ArchitectureConfig, build_from_module

        class MyModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                # ... define layers ...

            def forward(self, op, input_ids, attention_mask,
                        position_ids, past_key_values):
                # ... build graph ...
                return logits, present_key_values

        config = ArchitectureConfig(vocab_size=32000, hidden_size=4096, ...)
        pkg = build_from_module(MyModel(config), config)
    """
    if hasattr(config, "validate"):
        config.validate()
    _cast_module_dtype(module, getattr(config, "dtype", ir.DataType.FLOAT))
    resolved_task = get_task(task)
    pkg = resolved_task.build(module, config)
    for model in pkg.values():
        _optimize(model)
    return pkg


def build(
    model_id: str,
    task: str | ModelTask | None = None,
    *,
    module_class: type[nn.Module] | None = None,
    dtype: str | ir.DataType | None = None,
    load_weights: bool = True,
    trust_remote_code: bool = False,
) -> ModelPackage:
    """Build an ONNX :class:`ModelPackage` from a HuggingFace model ID.

    This is the main entry point for building models. It downloads the
    model configuration (and optionally weights) from HuggingFace Hub,
    selects the appropriate module class, and builds the ONNX graph(s).

    For single-component models (e.g. CausalLM), the package contains one
    ``"model"`` entry.  For multi-component models (e.g. encoder-decoder),
    it contains separate entries (``"encoder"``, ``"decoder"``).  For
    diffusers pipelines, each neural-network component gets its own entry.

    The model dtype is auto-detected from the HuggingFace config's
    ``torch_dtype`` field unless overridden by *dtype*.

    Args:
        model_id: HuggingFace model repository ID
            (e.g. ``"meta-llama/Llama-3-8B"``).
        task: The model task. Either a task name string
            (e.g. ``"text-generation"``) or a :class:`ModelTask` instance.
            When ``None``, the task is auto-detected from the model type.
        module_class: Custom module class to use instead of the auto-detected
            one. The class must accept an :class:`ArchitectureConfig` as its
            constructor argument and have a ``forward()`` method compatible
            with the task.
        dtype: Override the model dtype. Accepts short names (``"f32"``,
            ``"f16"``, ``"bf16"``) or :class:`ir.DataType` values.
            When ``None``, the dtype is auto-detected from the HuggingFace
            config.
        load_weights: Whether to download and apply weights from HuggingFace.
        trust_remote_code: Whether to trust remote code when loading the
            HuggingFace config.

    Returns:
        A :class:`ModelPackage` containing the built model(s).

    Example::

        from mobius import build

        # Auto-detect architecture and task
        pkg = build("meta-llama/Llama-3-8B")

        # Save all components
        pkg.save("/output/llama/")

        # Access individual models
        model = pkg["model"]
    """
    import dataclasses

    import transformers

    from mobius._config_resolver import (
        _config_from_hf,
        _default_task_for_model,
        _try_load_config_json,
    )
    from mobius._diffusers_builder import build_diffusers_pipeline

    try:
        hf_config = transformers.AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    except (ValueError, OSError):
        # AutoConfig failed — the model_type may not be in transformers.
        # Try loading config.json directly if the model is in our registry.
        hf_config = _try_load_config_json(model_id)
        if hf_config is None or hf_config.model_type not in registry:
            # Not a model we support — try diffusers pipeline
            return build_diffusers_pipeline(
                model_id,
                dtype=dtype,
                load_weights=load_weights,
            )

    model_type = hf_config.model_type
    parent_config = hf_config
    if hasattr(hf_config, "talker_config"):
        hf_config = hf_config.talker_config
    elif hasattr(hf_config, "thinker_config"):
        thinker = hf_config.thinker_config
        if hasattr(thinker, "text_config"):
            hf_config = thinker.text_config
    elif hasattr(hf_config, "decoder_config") and model_type == "qwen3_tts_tokenizer_12hz":
        # Codec tokenizer: use decoder_config as the primary config source
        dc = hf_config.decoder_config
        if isinstance(dc, dict):
            dc = type("DC", (), {**dc, "model_type": model_type})()
        else:
            dc.model_type = model_type
        hf_config = dc
    elif hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config

    if module_class is None:
        if model_type in registry:
            module_class = registry.get(model_type)
        else:
            from mobius._registry import _detect_fallback_registration

            fallback = _detect_fallback_registration(parent_config)
            if fallback is not None:
                module_class = fallback.module_class
                logger.warning(
                    "Model type '%s' is not registered. Auto-detected as compatible with %s.",
                    model_type,
                    module_class.__name__,
                )
                if task is None and fallback.task is not None:
                    task = fallback.task
            else:
                # No compatible fallback — raise the original error
                registry.get(model_type)  # raises KeyError

    config = _config_from_hf(hf_config, parent_config=parent_config, module_class=module_class)

    if dtype is not None:
        dtype = resolve_dtype(dtype)
        config = dataclasses.replace(config, dtype=dtype)

    if task is None:
        task = _default_task_for_model(model_type)

    model_module = module_class(config)
    pkg = build_from_module(model_module, config, task)

    # Set graph names
    for name, model in pkg.items():
        model.graph.name = f"{model_id}/{name}"

    if load_weights:
        state_dict = _download_weights(model_id)
        if hasattr(model_module, "preprocess_weights"):
            state_dict = model_module.preprocess_weights(state_dict)
        prefix_map = getattr(model_module, "weight_prefix_map", None)
        pkg.apply_weights(state_dict, prefix_map=prefix_map)

    return pkg
