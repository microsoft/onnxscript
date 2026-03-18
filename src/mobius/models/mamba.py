# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Mamba and Mamba2 (Selective State Space) causal language models.

Mamba1: pure SSM with selective scan (S6). Each layer is:
    RMSNorm -> MambaBlock -> residual add

Mamba2/SSD: pure SSM with multi-head structured state space duality.
Each layer is:
    RMSNorm -> Mamba2Block -> residual add

Unlike transformers, these models carry per-layer ``conv_state`` and
``ssm_state`` instead of KV cache, and require no attention mask or
position IDs.

State per layer:
    Mamba1:
        conv_state:  (batch, d_inner, conv_kernel - 1)
        ssm_state:   (batch, d_inner, state_size)
    Mamba2:
        conv_state:  (batch, conv_dim, conv_kernel - 1)
        ssm_state:   (batch, num_heads, head_dim, state_size)

HuggingFace references: ``MambaForCausalLM``, ``Mamba2ForCausalLM``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import Mamba2Config, MambaConfig
from mobius.components import (
    Embedding,
    Linear,
    Mamba2Block,
    MambaBlock,
    RMSNorm,
)

if TYPE_CHECKING:
    import onnx_ir as ir


class MambaBackbone(nn.Module):
    """Mamba backbone: embedding -> N x (norm + MambaBlock + residual) -> final norm.

    HuggingFace reference: ``MambaModel``.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        d_inner = config.intermediate_size

        self.embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                _MambaResidualBlock(
                    d_model=config.hidden_size,
                    d_inner=d_inner,
                    d_state=config.state_size,
                    dt_rank=config.time_step_rank,
                    conv_kernel=config.conv_kernel,
                    layer_norm_epsilon=config.layer_norm_epsilon,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        past_states: list[tuple] | None = None,
    ):
        """Forward pass.

        Args:
            op: ONNX op builder.
            input_ids: (batch, seq_len) token IDs.
            past_states: Per-layer (conv_state, ssm_state) tuples.

        Returns:
            hidden_states: (batch, seq_len, hidden_size)
            present_states: List of (conv_state, ssm_state) tuples.
        """
        hidden_states = self.embeddings(op, input_ids)

        present_states = []
        past = past_states or [None] * len(self.layers)
        for layer, past_state in zip(self.layers, past):
            hidden_states, new_conv_state, new_ssm_state = layer(op, hidden_states, past_state)
            present_states.append((new_conv_state, new_ssm_state))

        hidden_states = self.norm_f(op, hidden_states)
        return hidden_states, present_states


class _MambaResidualBlock(nn.Module):
    """Single Mamba layer with pre-norm and residual connection.

    Structure: norm → MambaBlock → residual add.
    HuggingFace reference: ``MambaBlock`` (the residual wrapper, not
    the inner mixer which is our ``MambaBlock`` component).
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int,
        dt_rank: int,
        conv_kernel: int,
        layer_norm_epsilon: float,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=layer_norm_epsilon)
        self.mixer = MambaBlock(
            d_model=d_model,
            d_inner=d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
            conv_kernel=conv_kernel,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        past_state: tuple | None = None,
    ):
        """Forward pass with residual connection.

        Args:
            op: ONNX op builder.
            hidden_states: (batch, 1, d_model)
            past_state: (conv_state, ssm_state) or None.

        Returns:
            output: (batch, 1, d_model)
            new_conv_state: (batch, d_inner, conv_kernel - 1)
            new_ssm_state: (batch, d_inner, d_state)
        """
        residual = hidden_states
        normed = self.norm(op, hidden_states)

        if past_state is not None:
            conv_state, ssm_state = past_state
        else:
            conv_state = None
            ssm_state = None

        output, new_conv_state, new_ssm_state = self.mixer(op, normed, conv_state, ssm_state)

        # Residual connection
        output = op.Add(output, residual)
        return output, new_conv_state, new_ssm_state


class MambaCausalLMModel(nn.Module):
    """Mamba causal language model with SSM backbone and LM head.

    Uses selective state space layers instead of transformer attention.
    Carries conv_state + ssm_state per layer (no KV cache).

    Inputs: input_ids, past_states (conv + ssm per layer).
    Outputs: logits, present_states.

    HuggingFace reference: ``MambaForCausalLM``.
    """

    default_task: str = "ssm-text-generation"
    category: str = "Text Generation"
    config_class: type = MambaConfig

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.model = MambaBackbone(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        past_states: list[tuple] | None = None,
    ):
        """Forward pass.

        Args:
            op: ONNX op builder.
            input_ids: (batch, seq_len)
            past_states: Per-layer (conv_state, ssm_state) tuples.

        Returns:
            logits: (batch, seq_len, vocab_size)
            present_states: List of (conv_state, ssm_state) tuples.
        """
        hidden_states, present_states = self.model(op, input_ids, past_states=past_states)
        logits = self.lm_head(op, hidden_states)
        return logits, present_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace MambaForCausalLM weights to ONNX parameter names.

        HF naming:
            model.embeddings.weight → model.embeddings.weight
            model.layers.{i}.norm.weight → model.layers.{i}.norm.weight
            model.layers.{i}.mixer.{param} → model.layers.{i}.mixer.{param}
            model.norm_f.weight → model.norm_f.weight
            lm_head.weight → lm_head.weight

        The HF and ONNX models both use "model." prefix (aligned).
        HF stores SSM params directly on mixer (e.g.
        ``mixer.A_log``) while our MambaBlock nests them under
        ``mixer.ssm`` (e.g. ``mixer.ssm.A_log``).
        """
        renames = {}
        # SSM params that HF stores flat on mixer but we nest under mixer.ssm
        _ssm_params = ("A_log", "D", "x_proj.weight", "dt_proj.weight", "dt_proj.bias")

        for key in list(state_dict):
            new_key = key

            # mixer.{ssm_param} → mixer.ssm.{ssm_param}
            for param in _ssm_params:
                old_seg = f".mixer.{param}"
                new_seg = f".mixer.ssm.{param}"
                if new_key.endswith(old_seg):
                    new_key = new_key.replace(old_seg, new_seg)
                    break

            if new_key != key:
                renames[key] = new_key

        for old_key, new_key in renames.items():
            state_dict[new_key] = state_dict.pop(old_key)

        # Tied embeddings
        if self.config.tie_word_embeddings:
            if "lm_head.weight" in state_dict:
                state_dict["model.embeddings.weight"] = state_dict["lm_head.weight"]
            elif "model.embeddings.weight" in state_dict:
                state_dict["lm_head.weight"] = state_dict["model.embeddings.weight"]

        return state_dict


# ---------------------------------------------------------------------------
# Mamba2 / SSD
# ---------------------------------------------------------------------------


class _Mamba2ResidualBlock(nn.Module):
    """Single Mamba2 layer with pre-norm and residual connection.

    Structure: norm -> Mamba2Block -> residual add.
    HuggingFace reference: ``Mamba2Block`` (the residual wrapper).
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        num_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        conv_kernel: int,
        layer_norm_epsilon: float,
        use_conv_bias: bool = True,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=layer_norm_epsilon)
        self.mixer = Mamba2Block(
            d_model=d_model,
            d_inner=d_inner,
            num_heads=num_heads,
            d_head=d_head,
            d_state=d_state,
            n_groups=n_groups,
            conv_kernel=conv_kernel,
            conv_bias=use_conv_bias,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        past_state: tuple | None = None,
    ):
        """Forward pass with residual connection.

        Args:
            op: ONNX op builder.
            hidden_states: (batch, 1, d_model)
            past_state: (conv_state, ssm_state) or None.

        Returns:
            output: (batch, 1, d_model)
            new_conv_state: (batch, conv_dim, conv_kernel - 1)
            new_ssm_state: (batch, num_heads, d_head, d_state)
        """
        residual = hidden_states
        normed = self.norm(op, hidden_states)

        if past_state is not None:
            conv_state, ssm_state = past_state
        else:
            conv_state = None
            ssm_state = None

        output, new_conv_state, new_ssm_state = self.mixer(op, normed, conv_state, ssm_state)

        # Residual connection
        output = op.Add(output, residual)
        return output, new_conv_state, new_ssm_state


class Mamba2Backbone(nn.Module):
    """Mamba2 backbone: embedding -> N x (norm + Mamba2Block + residual) -> norm.

    HuggingFace reference: ``Mamba2Model``.
    """

    def __init__(self, config: Mamba2Config):
        super().__init__()
        d_inner = config.intermediate_size

        self.embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                _Mamba2ResidualBlock(
                    d_model=config.hidden_size,
                    d_inner=d_inner,
                    num_heads=config.num_heads,
                    d_head=config.head_dim,
                    d_state=config.state_size,
                    n_groups=config.n_groups,
                    conv_kernel=config.conv_kernel,
                    layer_norm_epsilon=config.layer_norm_epsilon,
                    use_conv_bias=config.use_conv_bias,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        past_states: list[tuple] | None = None,
    ):
        """Forward pass.

        Args:
            op: ONNX op builder.
            input_ids: (batch, seq_len) token IDs.
            past_states: Per-layer (conv_state, ssm_state) tuples.

        Returns:
            hidden_states: (batch, seq_len, hidden_size)
            present_states: List of (conv_state, ssm_state) tuples.
        """
        hidden_states = self.embeddings(op, input_ids)

        present_states = []
        past = past_states or [None] * len(self.layers)
        for layer, past_state in zip(self.layers, past):
            hidden_states, new_conv_state, new_ssm_state = layer(op, hidden_states, past_state)
            present_states.append((new_conv_state, new_ssm_state))

        hidden_states = self.norm_f(op, hidden_states)
        return hidden_states, present_states


class Mamba2CausalLMModel(nn.Module):
    """Mamba2/SSD causal language model with multi-head SSM.

    Pure Mamba2 model — all layers are Mamba2Block (no attention).
    Carries conv_state + ssm_state per layer (no KV cache).

    State shapes (per layer):
        conv_state:  (batch, conv_dim, conv_kernel - 1)
        ssm_state:   (batch, num_heads, head_dim, state_size)

    HuggingFace reference: ``Mamba2ForCausalLM``.
    """

    default_task: str = "ssm2-text-generation"
    category: str = "Text Generation"
    config_class: type = Mamba2Config

    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        self.backbone = Mamba2Backbone(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        past_states: list[tuple] | None = None,
    ):
        """Forward pass.

        Args:
            op: ONNX op builder.
            input_ids: (batch, seq_len)
            past_states: Per-layer (conv_state, ssm_state) tuples.

        Returns:
            logits: (batch, seq_len, vocab_size)
            present_states: List of (conv_state, ssm_state) tuples.
        """
        hidden_states, present_states = self.backbone(op, input_ids, past_states=past_states)
        logits = self.lm_head(op, hidden_states)
        return logits, present_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace Mamba2ForCausalLM weights to ONNX names.

        HF naming:
            backbone.layers.{i}.norm.weight (same)
            backbone.layers.{i}.mixer.{A_log,D,dt_bias}
                -> backbone.layers.{i}.mixer.ssm.{A_log,D,dt_bias}
            backbone.layers.{i}.mixer.{in_proj,conv1d,norm,out_proj}
                (same)
            backbone.embeddings.weight (same)
            backbone.norm_f.weight (same)
            lm_head.weight (same)
        """
        renames = {}
        _ssm_params = ("A_log", "D", "dt_bias")

        for key in list(state_dict):
            new_key = key
            # mixer.{ssm_param} -> mixer.ssm.{ssm_param}
            for param in _ssm_params:
                old_seg = f".mixer.{param}"
                new_seg = f".mixer.ssm.{param}"
                if new_key.endswith(old_seg):
                    new_key = new_key.replace(old_seg, new_seg)
                    break

            if new_key != key:
                renames[key] = new_key

        for old_key, new_key in renames.items():
            state_dict[new_key] = state_dict.pop(old_key)

        # Tied embeddings (Mamba2 uses self.backbone, not self.model)
        if self.config.tie_word_embeddings:
            if "lm_head.weight" in state_dict:
                state_dict["backbone.embeddings.weight"] = state_dict["lm_head.weight"]
            elif "backbone.embeddings.weight" in state_dict:
                state_dict["lm_head.weight"] = state_dict["backbone.embeddings.weight"]

        return state_dict
