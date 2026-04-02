# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""RWKV-6 causal language model task with WKV-6 matrix recurrent state.

RWKV-6 differs from RWKV-4 in its recurrent state shape:
  - RWKV-4: 5 scalar/vector states per layer (num, den, max, shift_attn, shift_ffn)
  - RWKV-6: 3 states per layer — two shift vectors + one full-rank matrix state

State per layer:
    past_states.{i}.shift_attn: (batch, hidden_size)                         — float32 or model dtype
    past_states.{i}.wkv_state:  (batch, num_heads, head_size, head_size)     — always float32
    past_states.{i}.shift_ffn:  (batch, hidden_size)                         — float32 or model dtype
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig, Rwkv6Config
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_model,
)


class Rwkv6CausalLMTask(ModelTask):
    """Causal language model task for RWKV-6 (Eagle/Finch) models.

    Each layer carries three state tensors:
    - shift_attn: hidden vector from the previous token (time mixing shift).
    - wkv_state:  WKV-6 matrix state; has shape (B, Nh, S, S) and is always float32.
    - shift_ffn:  hidden vector from the previous token (channel mixing shift).

    Inputs:
        input_ids:                    [batch, 1]                               INT64
        past_states.{i}.shift_attn:   [batch, hidden_size]                     model dtype
        past_states.{i}.wkv_state:    [batch, num_heads, head_size, head_size] FLOAT32
        past_states.{i}.shift_ffn:    [batch, hidden_size]                     model dtype

    Outputs:
        logits:                       [batch, 1, vocab_size]                   FLOAT
        present.{i}.shift_attn:       [batch, hidden_size]                     model dtype
        present.{i}.wkv_state:        [batch, num_heads, head_size, head_size] FLOAT32
        present.{i}.shift_ffn:        [batch, hidden_size]                     model dtype
    """

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        assert isinstance(config, Rwkv6Config), (
            f"Rwkv6CausalLMTask requires Rwkv6Config, got {type(config).__name__}"
        )

        batch = ir.SymbolicDim("batch")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, 1]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [input_ids]

        hidden_size = config.hidden_size
        num_heads = config.attention_hidden_size // config.head_size
        head_size = config.head_size

        # Per-layer state inputs
        past_states: list[tuple[ir.Value, ir.Value, ir.Value]] = []
        for i in range(config.num_hidden_layers):
            shift_attn = ir.Value(
                name=f"past_states.{i}.shift_attn",
                shape=ir.Shape([batch, hidden_size]),
                type=ir.TensorType(config.dtype),
            )
            # wkv_state is always float32 regardless of model dtype
            wkv_state = ir.Value(
                name=f"past_states.{i}.wkv_state",
                shape=ir.Shape([batch, num_heads, head_size, head_size]),
                type=ir.TensorType(ir.DataType.FLOAT),
            )
            shift_ffn = ir.Value(
                name=f"past_states.{i}.shift_ffn",
                shape=ir.Shape([batch, hidden_size]),
                type=ir.TensorType(config.dtype),
            )
            graph_inputs.extend([shift_attn, wkv_state, shift_ffn])
            past_states.append((shift_attn, wkv_state, shift_ffn))

        graph, builder = _make_graph(graph_inputs)
        op = builder.op

        logits, present_states = module(
            op,
            input_ids=input_ids,
            past_states=past_states,
        )

        logits.name = "logits"
        graph.outputs.append(logits)

        for i, (shift_attn, wkv_state, shift_ffn) in enumerate(present_states):
            shift_attn.name = f"present.{i}.shift_attn"
            wkv_state.name = f"present.{i}.wkv_state"
            shift_ffn.name = f"present.{i}.shift_ffn"
            graph.outputs.extend([shift_attn, wkv_state, shift_ffn])

        return ModelPackage({"model": _make_model(graph)}, config=config)
