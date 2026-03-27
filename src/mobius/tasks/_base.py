# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Base class for model tasks and shared graph construction helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal.builder import GraphBuilder

import mobius
from mobius._configs import BaseModelConfig
from mobius._constants import OPSET_VERSION
from mobius._model_package import ModelPackage

_FUNCTIONS_DOMAIN = "pkg.mobius"


class LinearAttentionDims(NamedTuple):
    """Dimension sizes for linear attention (DeltaNet) layers."""

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    key_dim: int  # = head_k_dim * num_k_heads
    value_dim: int  # = head_v_dim * num_v_heads
    conv_dim: int  # = key_dim * 2 + value_dim
    conv_kernel: int


def linear_attention_dims(config: BaseModelConfig) -> LinearAttentionDims:
    """Compute dimension sizes for linear attention from config.

    Raises ``TypeError`` if any required config field is ``None``.
    """
    num_k_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    head_k_dim = config.linear_key_head_dim
    head_v_dim = config.linear_value_head_dim
    conv_kernel = config.linear_conv_kernel_dim
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    conv_dim = key_dim * 2 + value_dim
    return LinearAttentionDims(
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        key_dim=key_dim,
        value_dim=value_dim,
        conv_dim=conv_dim,
        conv_kernel=conv_kernel,
    )


def _make_graph(
    inputs: list[ir.Value],
    name: str = "main_graph",
) -> tuple[ir.Graph, GraphBuilder]:
    """Create an empty graph and its builder.

    Returns:
        ``(graph, builder)`` — call ``builder.op`` to get the op handle.
    """
    graph = ir.Graph(
        inputs,
        [],
        nodes=[],
        name=name,
        opset_imports={"": OPSET_VERSION},
    )
    return graph, GraphBuilder(graph)


def _make_model(graph: ir.Graph) -> ir.Model:
    """Create an ``ir.Model`` with standard producer metadata."""
    model = ir.Model(graph, ir_version=11)
    model.producer_name = "mobius"
    model.producer_version = mobius.__version__
    return model


def _make_kv_cache_inputs(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: ir.DataType,
    batch: ir.SymbolicDim,
    past_seq_len: ir.SymbolicDim,
    *,
    prefix: str = "past_key_values",
    key_head_dim: int | None = None,
    value_head_dim: int | None = None,
) -> tuple[list[ir.Value], list[tuple[ir.Value, ir.Value]]]:
    """Create KV cache input values for ``num_layers`` layers.

    Args:
        key_head_dim: Head dim for keys. Defaults to ``head_dim``.
            For MLA attention, this is ``qk_nope_head_dim + qk_rope_head_dim``.
        value_head_dim: Head dim for values. Defaults to ``head_dim``.
            For MLA attention, this is ``v_head_dim``.

    Returns:
        ``(flat_inputs, kv_pairs)`` where *flat_inputs* is a flat list
        suitable for extending ``graph_inputs`` and *kv_pairs* is a list
        of ``(key, value)`` tuples for passing to the module.
    """
    k_dim = key_head_dim if key_head_dim is not None else head_dim
    v_dim = value_head_dim if value_head_dim is not None else head_dim
    flat: list[ir.Value] = []
    pairs: list[tuple[ir.Value, ir.Value]] = []
    for i in range(num_layers):
        past_key = ir.Value(
            name=f"{prefix}.{i}.key",
            shape=ir.Shape([batch, num_kv_heads, past_seq_len, k_dim]),
            type=ir.TensorType(dtype),
        )
        past_value = ir.Value(
            name=f"{prefix}.{i}.value",
            shape=ir.Shape([batch, num_kv_heads, past_seq_len, v_dim]),
            type=ir.TensorType(dtype),
        )
        flat.extend([past_key, past_value])
        pairs.append((past_key, past_value))
    return flat, pairs


def _register_kv_cache_outputs(
    graph: ir.Graph,
    present_key_values: list[tuple[ir.Value, ir.Value]],
    *,
    past_key_values: list[tuple[ir.Value, ir.Value]] | None = None,
    prefix: str = "present",
) -> None:
    """Name and register KV cache outputs on the graph.

    When ``past_key_values`` is provided, output shapes are derived from
    the corresponding past inputs (replacing the sequence-length dimension
    with a ``total_sequence_len`` symbolic dim).
    """
    total_seq_len = ir.SymbolicDim("total_sequence_len")
    for i, (present_key, present_value) in enumerate(present_key_values):
        present_key.name = f"{prefix}.{i}.key"
        present_value.name = f"{prefix}.{i}.value"
        if past_key_values is not None:
            past_k, past_v = past_key_values[i]
            if past_k.shape is not None and len(past_k.shape) >= 3:
                present_key.shape = ir.Shape(
                    [*past_k.shape[:-2], total_seq_len, past_k.shape[-1]]
                )
            if past_k.type is not None:
                present_key.type = past_k.type
            if past_v.shape is not None and len(past_v.shape) >= 3:
                present_value.shape = ir.Shape(
                    [*past_v.shape[:-2], total_seq_len, past_v.shape[-1]]
                )
            if past_v.type is not None:
                present_value.type = past_v.type
        graph.outputs.append(present_key)
        graph.outputs.append(present_value)


class ModelTask(ABC):
    """Abstract base defining how to wire a module into an ONNX graph.

    Subclass this to support new model tasks (e.g. feature extraction,
    sequence classification). Each task defines its own graph I/O contract.
    """

    @abstractmethod
    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        """Build a :class:`ModelPackage` for this task.

        Single-component tasks return a package with one ``"model"`` entry.
        Multi-component tasks (e.g. encoder-decoder) return a package with
        separate entries for each component.

        Args:
            module: The onnxscript.nn.Module to wire into the graph.
            config: Architecture configuration.

        Returns:
            A :class:`ModelPackage` containing the built model(s).
        """
        ...


def _make_hybrid_cache_inputs(
    config: BaseModelConfig,
    dtype: ir.DataType,
    batch: ir.SymbolicDim,
    past_seq_len: ir.SymbolicDim,
    *,
    prefix: str = "past_key_values",
) -> tuple[list[ir.Value], list[tuple[ir.Value, ir.Value]]]:
    """Create cache inputs for hybrid models with mixed layer types.

    Supported layer types:
        ``"full_attention"`` — standard KV cache (key + value).
        ``"linear_attention"`` (DeltaNet) — conv_state + recurrent_state.
        ``"mamba"`` — conv_state + ssm_state (Mamba SSM carry).

    Returns:
        ``(flat_inputs, state_pairs)`` — same shape as
        :func:`_make_kv_cache_inputs`.
    """
    layer_types = config.layer_types or []
    flat: list[ir.Value] = []
    pairs: list[tuple[ir.Value, ir.Value]] = []

    # DeltaNet dimensions from config (computed once via shared helper)
    has_linear = "linear_attention" in layer_types
    if has_linear:
        dims = linear_attention_dims(config)

    # Mamba SSM dimensions from config (Jamba-style)
    mamba_expand = getattr(config, "mamba_expand", 2)
    mamba_d_inner = config.hidden_size * mamba_expand
    mamba_d_conv = getattr(config, "mamba_d_conv", 4)
    mamba_d_state = getattr(config, "mamba_d_state", 16)

    # Mamba2/SSD dimensions from config (Bamba-style).
    # Defaults are 0 so a missing field produces a clear shape error
    # rather than silently using model-specific values.
    mamba2_n_heads = getattr(config, "mamba_n_heads", 0)
    mamba2_d_head = getattr(config, "mamba_d_head", 0)
    mamba2_d_state = getattr(config, "mamba_d_state", 0)
    mamba2_n_groups = getattr(config, "mamba_n_groups", 1)
    mamba2_d_inner = config.hidden_size * mamba_expand
    mamba2_conv_dim = mamba2_d_inner + 2 * mamba2_n_groups * mamba2_d_state

    for i in range(config.num_hidden_layers):
        ltype = layer_types[i] if i < len(layer_types) else "full_attention"

        if ltype == "linear_attention":
            conv_state = ir.Value(
                name=f"{prefix}.{i}.conv_state",
                shape=ir.Shape([batch, dims.conv_dim, dims.conv_kernel - 1]),
                type=ir.TensorType(dtype),
            )
            rec_state = ir.Value(
                name=f"{prefix}.{i}.recurrent_state",
                shape=ir.Shape([batch, dims.num_v_heads, dims.head_k_dim, dims.head_v_dim]),
                type=ir.TensorType(dtype),
            )
            flat.extend([conv_state, rec_state])
            pairs.append((conv_state, rec_state))
        elif ltype == "mamba":
            conv_state = ir.Value(
                name=f"{prefix}.{i}.conv_state",
                shape=ir.Shape([batch, mamba_d_inner, mamba_d_conv - 1]),
                type=ir.TensorType(dtype),
            )
            ssm_state = ir.Value(
                name=f"{prefix}.{i}.ssm_state",
                shape=ir.Shape([batch, mamba_d_inner, mamba_d_state]),
                type=ir.TensorType(dtype),
            )
            flat.extend([conv_state, ssm_state])
            pairs.append((conv_state, ssm_state))
        elif ltype == "mamba2":
            conv_state = ir.Value(
                name=f"{prefix}.{i}.conv_state",
                shape=ir.Shape([batch, mamba2_conv_dim, mamba_d_conv - 1]),
                type=ir.TensorType(dtype),
            )
            ssm_state = ir.Value(
                name=f"{prefix}.{i}.ssm_state",
                shape=ir.Shape([batch, mamba2_n_heads, mamba2_d_head, mamba2_d_state]),
                type=ir.TensorType(dtype),
            )
            flat.extend([conv_state, ssm_state])
            pairs.append((conv_state, ssm_state))
        else:
            past_key = ir.Value(
                name=f"{prefix}.{i}.key",
                shape=ir.Shape(
                    [batch, config.num_key_value_heads, past_seq_len, config.head_dim]
                ),
                type=ir.TensorType(dtype),
            )
            past_value = ir.Value(
                name=f"{prefix}.{i}.value",
                shape=ir.Shape(
                    [batch, config.num_key_value_heads, past_seq_len, config.head_dim]
                ),
                type=ir.TensorType(dtype),
            )
            flat.extend([past_key, past_value])
            pairs.append((past_key, past_value))

    return flat, pairs


def _register_hybrid_cache_outputs(
    graph: ir.Graph,
    present_key_values: list[tuple[ir.Value, ir.Value]],
    layer_types: list[str],
    *,
    past_key_values: list[tuple[ir.Value, ir.Value]] | None = None,
    prefix: str = "present",
) -> None:
    """Name and register hybrid cache outputs on the graph.

    Uses ``.key``/``.value`` for full attention layers,
    ``.conv_state``/``.recurrent_state`` for linear attention layers,
    and ``.conv_state``/``.ssm_state`` for mamba/mamba2 layers.

    When ``past_key_values`` is provided, output shapes are derived
    from the corresponding past inputs.
    """
    total_seq_len = ir.SymbolicDim("total_sequence_len")
    for i, (state_a, state_b) in enumerate(present_key_values):
        ltype = layer_types[i] if i < len(layer_types) else "full_attention"
        if ltype == "linear_attention":
            state_a.name = f"{prefix}.{i}.conv_state"
            state_b.name = f"{prefix}.{i}.recurrent_state"
        elif ltype in ("mamba", "mamba2"):
            state_a.name = f"{prefix}.{i}.conv_state"
            state_b.name = f"{prefix}.{i}.ssm_state"
        else:
            state_a.name = f"{prefix}.{i}.key"
            state_b.name = f"{prefix}.{i}.value"
        if past_key_values is not None:
            past_a, past_b = past_key_values[i]
            if ltype == "full_attention":
                # KV cache: replace seq_len dim with total_seq_len
                if past_a.shape is not None and len(past_a.shape) >= 3:
                    state_a.shape = ir.Shape(
                        [*past_a.shape[:-2], total_seq_len, past_a.shape[-1]]
                    )
                if past_b.shape is not None and len(past_b.shape) >= 3:
                    state_b.shape = ir.Shape(
                        [*past_b.shape[:-2], total_seq_len, past_b.shape[-1]]
                    )
            else:
                # Recurrent/conv states have fixed shape
                if past_a.shape is not None:
                    state_a.shape = past_a.shape
                if past_b.shape is not None:
                    state_b.shape = past_b.shape
            if past_a.type is not None:
                state_a.type = past_a.type
            if past_b.type is not None:
                state_b.type = past_b.type
        graph.outputs.append(state_a)
        graph.outputs.append(state_b)


def _register_linear_attention_functions(
    model: ir.Model,
    config: BaseModelConfig,
) -> None:
    """Register CausalConvWithState and LinearAttention functions.

    Only registers functions when the model has ``linear_attention`` layers.
    Adds the ``pkg.mobius`` opset import to the graph.
    """
    layer_types = getattr(config, "layer_types", None) or []
    if "linear_attention" not in layer_types:
        return

    from mobius.functions import (
        causal_conv_nd_with_state,
        linear_attention,
    )

    dims = linear_attention_dims(config)

    conv_func = causal_conv_nd_with_state(
        kernel_size=dims.conv_kernel,
        channels=dims.conv_dim,
        ndim=1,
        activation="silu",
    )
    attn_func = linear_attention(
        q_num_heads=dims.num_k_heads,
        kv_num_heads=dims.num_v_heads,
        update_rule="gated_delta",
        scale=1.0 / (dims.head_k_dim**0.5),
        stash_type=config.dtype,
    )

    model.functions[conv_func.identifier()] = conv_func
    model.functions[attn_func.identifier()] = attn_func
    model.graph.opset_imports[_FUNCTIONS_DOMAIN] = 1
