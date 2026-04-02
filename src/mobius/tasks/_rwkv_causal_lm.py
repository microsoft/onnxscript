# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""RWKV causal language model task with WKV recurrent state carry.

Unlike CausalLMTask (for transformers), RWKV models:
    - Do NOT use attention_mask or position_ids
    - Carry 5 state tensors per layer instead of KV cache
    - Still produce input_ids → logits

State per layer:
    past_states.{i}.shift_attn:  (batch, hidden_size)          — time-shift for attention
    past_states.{i}.wkv_num:     (batch, attention_hidden_size) — WKV numerator accumulator
    past_states.{i}.wkv_den:     (batch, attention_hidden_size) — WKV denominator accumulator
    past_states.{i}.wkv_max:     (batch, attention_hidden_size) — WKV max (numerical stability)
    past_states.{i}.shift_ffn:   (batch, hidden_size)          — time-shift for feed-forward
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig, RwkvConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_model,
)


class RwkvCausalLMTask(ModelTask):
    """Causal language model task for RWKV linear-RNN models.

    Each layer carries five state tensors that are updated token-by-token.
    This differs from transformer KV cache (which grows with sequence length) —
    RWKV state is always fixed-size.

    Inputs:
        - input_ids: [batch, 1] INT64 (single token per step)
        - past_states.{i}.shift_attn: [batch, hidden_size]
        - past_states.{i}.wkv_num:    [batch, attention_hidden_size]
        - past_states.{i}.wkv_den:    [batch, attention_hidden_size]
        - past_states.{i}.wkv_max:    [batch, attention_hidden_size]
        - past_states.{i}.shift_ffn:  [batch, hidden_size]

    Outputs:
        - logits: [batch, 1, vocab_size] FLOAT
        - present.{i}.shift_attn: [batch, hidden_size]
        - present.{i}.wkv_num:    [batch, attention_hidden_size]
        - present.{i}.wkv_den:    [batch, attention_hidden_size]
        - present.{i}.wkv_max:    [batch, attention_hidden_size]
        - present.{i}.shift_ffn:  [batch, hidden_size]

    The module's ``forward()`` must accept
    ``(op, input_ids, past_states)``
    and return ``(logits, list_of_5_tuple_states)``.
    """

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        assert isinstance(config, RwkvConfig), (
            f"RwkvCausalLMTask requires RwkvConfig, got {type(config).__name__}"
        )

        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [input_ids]

        hidden_size = config.hidden_size
        attn_size = config.attention_hidden_size

        # Create per-layer WKV state inputs
        past_states: list[tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value]] = []
        for i in range(config.num_hidden_layers):
            shift_attn = ir.Value(
                name=f"past_states.{i}.shift_attn",
                shape=ir.Shape([batch, hidden_size]),
                type=ir.TensorType(config.dtype),
            )
            wkv_num = ir.Value(
                name=f"past_states.{i}.wkv_num",
                shape=ir.Shape([batch, attn_size]),
                type=ir.TensorType(config.dtype),
            )
            wkv_den = ir.Value(
                name=f"past_states.{i}.wkv_den",
                shape=ir.Shape([batch, attn_size]),
                type=ir.TensorType(config.dtype),
            )
            wkv_max = ir.Value(
                name=f"past_states.{i}.wkv_max",
                shape=ir.Shape([batch, attn_size]),
                type=ir.TensorType(config.dtype),
            )
            shift_ffn = ir.Value(
                name=f"past_states.{i}.shift_ffn",
                shape=ir.Shape([batch, hidden_size]),
                type=ir.TensorType(config.dtype),
            )
            graph_inputs.extend([shift_attn, wkv_num, wkv_den, wkv_max, shift_ffn])
            past_states.append((shift_attn, wkv_num, wkv_den, wkv_max, shift_ffn))

        graph, builder = _make_graph(graph_inputs)
        op = builder.op

        logits, present_states = module(
            op,
            input_ids=input_ids,
            past_states=past_states,
        )

        # Register outputs
        logits.name = "logits"
        graph.outputs.append(logits)

        for i, (shift_attn, wkv_num, wkv_den, wkv_max, shift_ffn) in enumerate(present_states):
            shift_attn.name = f"present.{i}.shift_attn"
            wkv_num.name = f"present.{i}.wkv_num"
            wkv_den.name = f"present.{i}.wkv_den"
            wkv_max.name = f"present.{i}.wkv_max"
            shift_ffn.name = f"present.{i}.shift_ffn"
            graph.outputs.extend([shift_attn, wkv_num, wkv_den, wkv_max, shift_ffn])

        return ModelPackage({"model": _make_model(graph)}, config=config)
