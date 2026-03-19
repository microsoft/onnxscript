# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Causal language model tasks with internal and static KV cache."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.components._attention import StaticCacheState
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_hybrid_cache_inputs,
    _make_kv_cache_inputs,
    _make_model,
    _register_hybrid_cache_outputs,
    _register_kv_cache_outputs,
    _register_linear_attention_functions,
)


class CausalLMTask(ModelTask):
    """Causal language model with KV cache for text generation.

    Inputs:
        - input_ids: [batch, sequence_len] INT64
        - attention_mask: [batch, total_seq_len] INT64
        - position_ids: [batch, sequence_len] INT64
        - past_key_values.{i}.key: [batch, num_kv_heads, past_seq_len, head_dim] FLOAT
        - past_key_values.{i}.value: [batch, num_kv_heads, past_seq_len, head_dim] FLOAT

    Outputs:
        - logits: FLOAT
        - present.{i}.key: FLOAT
        - present.{i}.value: FLOAT

    The module's ``forward()`` must accept
    ``(op, input_ids, attention_mask, position_ids, past_key_values)``
    and return ``(logits, list_of_(key, value)_tuples)``.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [input_ids, attention_mask, position_ids]

        kv_inputs, past_key_values = _make_kv_cache_inputs(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
            # MLA attention has separate key/value head dims
            key_head_dim=((config.qk_nope_head_dim or 0) + (config.qk_rope_head_dim or 0))
            or None,
            value_head_dim=config.v_head_dim or None,
        )
        graph_inputs.extend(kv_inputs)

        graph, builder = _make_graph(graph_inputs)
        op = builder.op

        logits, present_key_values = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_kv_cache_outputs(graph, present_key_values, past_key_values=past_key_values)

        return ModelPackage({"model": _make_model(graph)}, config=config)


class HybridCausalLMTask(ModelTask):
    """Causal LM with hybrid KV cache + DeltaNet recurrent states.

    For models with mixed ``"full_attention"`` and ``"linear_attention"``
    layers (e.g. Qwen3.5).  Full-attention layers use standard KV cache;
    linear-attention (DeltaNet) layers carry ``conv_state`` and
    ``recurrent_state`` tensors instead.

    Inputs (per layer):
        Full attention:
          - past_key_values.{i}.key: [batch, num_kv_heads, past_seq_len, head_dim]
          - past_key_values.{i}.value: [batch, num_kv_heads, past_seq_len, head_dim]
        Linear attention:
          - past_key_values.{i}.conv_state: [batch, conv_dim, kernel_size-1]
          - past_key_values.{i}.recurrent_state: [batch, num_v_heads, k_dim, v_dim]

    Outputs:
        - logits: FLOAT
        - present.{i}.{key|value|conv_state|recurrent_state}: FLOAT
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [input_ids, attention_mask, position_ids]

        cache_inputs, past_key_values = _make_hybrid_cache_inputs(
            config,
            config.dtype,
            batch,
            past_seq_len,
        )
        graph_inputs.extend(cache_inputs)

        graph, builder = _make_graph(graph_inputs)
        op = builder.op

        logits, present_key_values = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_hybrid_cache_outputs(
            graph,
            present_key_values,
            config.layer_types or [],
            past_key_values=past_key_values,
        )

        model = _make_model(graph)
        _register_linear_attention_functions(model, config)
        return ModelPackage({"model": model}, config=config)


def _make_static_cache_inputs(
    num_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype: ir.DataType,
    batch: ir.SymbolicDim,
    max_seq_len: int,
) -> tuple[list[ir.Value], list[StaticCacheState]]:
    """Create static KV cache inputs for ``num_layers`` layers.

    Returns:
        ``(flat_inputs, static_caches)`` where *flat_inputs* is a flat
        list suitable for extending ``graph_inputs``, and
        *static_caches* is a list of :class:`StaticCacheState` tuples
        for passing to the module via ``past_key_values``.
    """
    kv_hidden = num_key_value_heads * head_dim
    flat: list[ir.Value] = []
    cache_pairs: list[tuple[ir.Value, ir.Value]] = []

    for i in range(num_layers):
        key_cache = ir.Value(
            name=f"key_cache.{i}",
            shape=ir.Shape([batch, max_seq_len, kv_hidden]),
            type=ir.TensorType(dtype),
        )
        value_cache = ir.Value(
            name=f"value_cache.{i}",
            shape=ir.Shape([batch, max_seq_len, kv_hidden]),
            type=ir.TensorType(dtype),
        )
        flat.extend([key_cache, value_cache])
        cache_pairs.append((key_cache, value_cache))

    # Shared inputs across all layers
    write_indices = ir.Value(
        name="write_indices",
        shape=ir.Shape([batch]),
        type=ir.TensorType(ir.DataType.INT64),
    )
    nonpad_kv_seqlen = ir.Value(
        name="nonpad_kv_seqlen",
        shape=ir.Shape([batch]),
        type=ir.TensorType(ir.DataType.INT64),
    )
    flat.extend([write_indices, nonpad_kv_seqlen])

    # Build StaticCacheState for each layer (shared indices)
    static_caches: list[StaticCacheState] = []
    for key_cache, value_cache in cache_pairs:
        static_caches.append(
            StaticCacheState(
                key_cache=key_cache,
                value_cache=value_cache,
                write_indices=write_indices,
                nonpad_kv_seqlen=nonpad_kv_seqlen,
            )
        )

    return flat, static_caches


def _register_static_cache_outputs(
    graph: ir.Graph,
    present_key_values: list[tuple[ir.Value, ir.Value]],
    dtype: ir.DataType,
    batch: ir.SymbolicDim,
    max_seq_len: int,
    kv_hidden: int,
) -> None:
    """Name and register static cache outputs on the graph."""
    for i, (updated_key, updated_value) in enumerate(present_key_values):
        updated_key.name = f"updated_key_cache.{i}"
        updated_value.name = f"updated_value_cache.{i}"
        updated_key.shape = ir.Shape([batch, max_seq_len, kv_hidden])
        updated_key.type = ir.TensorType(dtype)
        updated_value.shape = ir.Shape([batch, max_seq_len, kv_hidden])
        updated_value.type = ir.TensorType(dtype)
        graph.outputs.append(updated_key)
        graph.outputs.append(updated_value)


def _validate_static_cache_support(module: nn.Module) -> None:
    """Check that the module's decoder layers support StaticCacheState.

    Only :class:`DecoderLayer` and :class:`MoEDecoderLayer` have the
    ``isinstance(StaticCacheState)`` dispatch in ``forward()``.  Custom
    decoder layers will silently unpack the NamedTuple as a regular
    ``(key, value)`` tuple, producing wrong results.

    NOTE: The following models are NOT yet supported in static cache
    mode and will raise TypeError from this check:

    - **Gemma2**: ``Gemma2Attention`` overrides ``forward()`` and calls
      ``op.Attention`` directly with ``attn_logit_softcapping``,
      bypassing ``_apply_attention()``.  Needs Attention refactoring to
      support softcap in the shared path.

    - **GPT-2**: Uses learned positional embeddings (not RoPE).
      ``_GPT2TextModel.forward()`` unconditionally calls
      ``create_attention_bias()``, so ``attention_mask=None`` would
      fail.  Needs position embedding adaptation.

    - **Falcon (ALiBi)**: The ALiBi variant uses ``is_causal=0`` with a
      position-dependent bias that encodes both causal masking and
      distance-based attention decay.  This is fundamentally
      incompatible with the ``is_causal=1`` static cache pattern.

    Raises:
        TypeError: If any decoder layer is not a supported type.
    """
    from mobius.components._decoder import DecoderLayer
    from mobius.models.moe import MoEDecoderLayer

    for name, child in module.named_modules():
        if not isinstance(child, nn.ModuleList):
            continue
        for i, layer in enumerate(child):
            if not isinstance(layer, nn.Module):
                continue
            # Check modules that look like decoder layers: they have an
            # attention sub-module named either "self_attn" (standard) or
            # "attn" (GPT-2 style).
            if not hasattr(layer, "self_attn") and not hasattr(layer, "attn"):
                continue
            if not isinstance(layer, (DecoderLayer, MoEDecoderLayer)):
                raise TypeError(
                    f"StaticCacheCausalLMTask requires decoder layers that "
                    f"inherit from DecoderLayer or MoEDecoderLayer, but "
                    f"{name}[{i}] is {type(layer).__name__}. Either use a "
                    f"compatible model or add StaticCacheState dispatch to "
                    f"{type(layer).__name__}.forward()."
                )


class StaticCacheCausalLMTask(ModelTask):
    """Causal LM with statically managed KV cache.

    Uses opset-24 TensorScatter + Attention for static cache management.
    The caller provides pre-allocated cache buffers and receives updated
    caches as outputs.

    Compatible models:
        Models using the base :class:`DecoderLayer` (Llama, Qwen2, Mistral,
        etc.) work out of the box.  Custom decoder layers
        (Qwen35DecoderLayer, Gemma2DecoderLayer, etc.) require their own
        ``StaticCacheState`` dispatch to use this task.

    Inputs:
        - input_ids: [batch, seq_len] INT64
        - position_ids: [batch, seq_len] INT64
        - key_cache.{i}: [batch, max_seq_len, kv_hidden] FLOAT per layer
        - value_cache.{i}: [batch, max_seq_len, kv_hidden] FLOAT per layer
        - write_indices: [batch] INT64
        - nonpad_kv_seqlen: [batch] INT64

    Outputs:
        - logits: FLOAT
        - updated_key_cache.{i}: [batch, max_seq_len, kv_hidden] FLOAT
        - updated_value_cache.{i}: [batch, max_seq_len, kv_hidden] FLOAT

    Note:
        No ``attention_mask`` input — causal masking is handled by the
        Attention op's ``is_causal=1`` attribute, and padding is handled
        via ``nonpad_kv_seqlen``.

    The module's ``forward()`` must accept
    ``(op, input_ids, attention_mask, position_ids, past_key_values)``
    and return ``(logits, list_of_(key, value)_tuples)``.  The
    ``past_key_values`` entries will be :class:`StaticCacheState` tuples.
    """

    def __init__(self, max_seq_len: int | None = None):
        self._max_seq_len = max_seq_len

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        max_seq_len = self._max_seq_len
        if max_seq_len is None:
            max_seq_len = getattr(config, "max_position_embeddings", None)
        if max_seq_len is None or max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be a positive integer. Either pass it to "
                "StaticCacheCausalLMTask(max_seq_len=...) or ensure "
                "config.max_position_embeddings is set."
            )

        # Validate that the module's decoder layers support static cache.
        # Only DecoderLayer has the isinstance(StaticCacheState) dispatch;
        # custom layers will silently unpack the NamedTuple as a regular
        # tuple, producing wrong results.
        _validate_static_cache_support(module)

        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        # seq_len is intentionally dynamic: static cache supports both
        # prefill (seq_len=N tokens) and single-token decode (seq_len=1)
        # via the start-position semantics of write_indices.
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [input_ids, position_ids]

        cache_inputs, static_caches = _make_static_cache_inputs(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            max_seq_len,
        )
        graph_inputs.extend(cache_inputs)

        graph, builder = _make_graph(graph_inputs)
        op = builder.op

        # StaticCacheState objects flow through past_key_values;
        # DecoderLayer dispatches them to Attention's static_cache.
        # attention_mask=None skips create_attention_bias() — causal
        # masking is handled by is_causal=1 on the Attention op.
        # See _apply_attention() TODO(titaiwang) for future attn_mask
        # and sliding window support.
        logits, present_key_values = module(
            op,
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=static_caches,
        )

        logits.name = "logits"
        graph.outputs.append(logits)

        kv_hidden = config.num_key_value_heads * config.head_dim
        _register_static_cache_outputs(
            graph,
            present_key_values,
            config.dtype,
            batch,
            max_seq_len,
            kv_hidden,
        )

        return ModelPackage({"model": _make_model(graph)}, config=config)
