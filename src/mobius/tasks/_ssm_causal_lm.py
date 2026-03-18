# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""SSM causal language model task with conv_state + ssm_state carry.

Unlike CausalLMTask (for transformers), SSM models:
    - Do NOT use attention_mask or position_ids
    - Carry conv_state + ssm_state per layer instead of KV cache
    - Still produce input_ids → logits

State per layer:
    past_states.{i}.conv_state:  (batch, d_inner, conv_kernel - 1)
    past_states.{i}.ssm_state:   (batch, d_inner, state_size)
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig, Mamba2Config, MambaConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_model,
)


class SSMCausalLMTask(ModelTask):
    """Causal language model with SSM state carry for Mamba-style models.

    Inputs:
        - input_ids: [batch, sequence_len] INT64
        - past_states.{i}.conv_state: [batch, d_inner, conv_kernel-1] FLOAT
        - past_states.{i}.ssm_state: [batch, d_inner, state_size] FLOAT

    Outputs:
        - logits: FLOAT
        - present.{i}.conv_state: [batch, d_inner, conv_kernel-1] FLOAT
        - present.{i}.ssm_state: [batch, d_inner, state_size] FLOAT

    The module's ``forward()`` must accept
    ``(op, input_ids, past_states)``
    and return ``(logits, list_of_(conv_state, ssm_state)_tuples)``.
    """

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        assert isinstance(config, MambaConfig), (
            f"SSMCausalLMTask requires MambaConfig, got {type(config).__name__}"
        )

        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [input_ids]

        # Create SSM state inputs for each layer
        d_inner = config.intermediate_size
        conv_state_len = config.conv_kernel - 1
        state_size = config.state_size

        past_states: list[tuple[ir.Value, ir.Value]] = []
        for i in range(config.num_hidden_layers):
            conv_state = ir.Value(
                name=f"past_states.{i}.conv_state",
                shape=ir.Shape([batch, d_inner, conv_state_len]),
                type=ir.TensorType(config.dtype),
            )
            ssm_state = ir.Value(
                name=f"past_states.{i}.ssm_state",
                shape=ir.Shape([batch, d_inner, state_size]),
                type=ir.TensorType(config.dtype),
            )
            graph_inputs.extend([conv_state, ssm_state])
            past_states.append((conv_state, ssm_state))

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

        for i, (conv_state, ssm_state) in enumerate(present_states):
            conv_state.name = f"present.{i}.conv_state"
            ssm_state.name = f"present.{i}.ssm_state"
            graph.outputs.append(conv_state)
            graph.outputs.append(ssm_state)

        return ModelPackage({"model": _make_model(graph)}, config=config)


class SSM2CausalLMTask(ModelTask):
    """Causal language model with Mamba2/SSD state carry.

    Like SSMCausalLMTask but with 4D SSM state for Mamba2 multi-head
    architecture and wider conv_dim.

    Inputs:
        - input_ids: [batch, sequence_len] INT64
        - past_states.{i}.conv_state: [batch, conv_dim, conv_kernel-1]
        - past_states.{i}.ssm_state: [batch, num_heads, head_dim, state_size]

    Outputs:
        - logits: FLOAT
        - present.{i}.conv_state: [batch, conv_dim, conv_kernel-1]
        - present.{i}.ssm_state: [batch, num_heads, head_dim, state_size]
    """

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        assert isinstance(config, Mamba2Config), (
            f"SSM2CausalLMTask requires Mamba2Config, got {type(config).__name__}"
        )

        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [input_ids]

        # Mamba2 state dimensions
        d_inner = config.intermediate_size
        n_groups = config.n_groups
        state_size = config.state_size
        conv_dim = d_inner + 2 * n_groups * state_size
        conv_state_len = config.conv_kernel - 1
        num_heads = config.num_heads
        head_dim = config.head_dim

        past_states: list[tuple[ir.Value, ir.Value]] = []
        for i in range(config.num_hidden_layers):
            conv_state = ir.Value(
                name=f"past_states.{i}.conv_state",
                shape=ir.Shape([batch, conv_dim, conv_state_len]),
                type=ir.TensorType(config.dtype),
            )
            # 4D SSM state for multi-head Mamba2
            ssm_state = ir.Value(
                name=f"past_states.{i}.ssm_state",
                shape=ir.Shape([batch, num_heads, head_dim, state_size]),
                type=ir.TensorType(config.dtype),
            )
            graph_inputs.extend([conv_state, ssm_state])
            past_states.append((conv_state, ssm_state))

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

        for i, (conv_state, ssm_state) in enumerate(present_states):
            conv_state.name = f"present.{i}.conv_state"
            ssm_state.name = f"present.{i}.ssm_state"
            graph.outputs.append(conv_state)
            graph.outputs.append(ssm_state)

        return ModelPackage({"model": _make_model(graph)}, config=config)
