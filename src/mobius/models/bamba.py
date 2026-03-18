# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bamba hybrid Mamba2/SSD + Attention causal language model.

Bamba interleaves Mamba2 (SSD) layers with Transformer attention layers.
Unlike Jamba which uses Mamba1 + MoE, Bamba uses the multi-head Mamba2
architecture with dense MLPs only.

Layer type selection:
    ``attn_layer_indices`` specifies which layers use attention.
    All other layers use Mamba2/SSD.

Mamba2 vs Mamba1 key differences:
    - Multi-head SSM: ``n_heads x d_head`` instead of flat ``d_inner``
    - ``in_proj → [gate, xBC, dt]`` instead of ``[x, z]``
    - Conv1D on wider ``xBC`` (``d_inner + 2*n_groups*d_state``)
    - GatedRMSNorm instead of SiLU gating
    - ``dt`` directly from in_proj (no rank reduction), just bias

State per layer:
    Mamba2: conv_state (batch, conv_dim, d_conv-1)
            ssm_state (batch, num_heads, d_head, d_state)
    Attention: standard KV cache (key + value)

HuggingFace reference: ``BambaForCausalLM``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import BambaConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    Linear,
    Mamba2Block,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


class BambaMambaDecoderLayer(nn.Module):
    """Bamba Mamba2 layer: RMSNorm → Mamba2Block → residual → RMSNorm → MLP.

    Uses multi-head Mamba2/SSD with GatedRMSNorm.

    Args:
        config: Bamba architecture config.
    """

    def __init__(self, config: BambaConfig):
        super().__init__()
        d_inner = config.hidden_size * config.mamba_expand

        self.mamba = Mamba2Block(
            d_model=config.hidden_size,
            d_inner=d_inner,
            num_heads=config.mamba_n_heads,
            d_head=config.mamba_d_head,
            d_state=config.mamba_d_state,
            n_groups=config.mamba_n_groups,
            conv_kernel=config.mamba_d_conv,
            conv_bias=config.mamba_conv_bias,
            proj_bias=config.mamba_proj_bias,
            eps=config.rms_norm_eps,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = MLP(config)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (conv_state, ssm_state)).

        attention_bias and position_embeddings are unused by Mamba layers
        but accepted for uniform interface with attention layers.
        """
        del attention_bias, position_embeddings  # unused

        # Mamba2 path with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        conv_state, ssm_state = past_key_value if past_key_value is not None else (None, None)
        mamba_out, new_conv_state, new_ssm_state = self.mamba(
            op, hidden_states, conv_state, ssm_state
        )
        hidden_states = op.Add(residual, mamba_out)

        # MLP path with pre-norm
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, (new_conv_state, new_ssm_state)


class BambaAttentionDecoderLayer(nn.Module):
    """Bamba attention layer: RMSNorm → Attention → residual → RMSNorm → MLP.

    Standard transformer decoder layer with GQA attention.

    Args:
        config: Bamba architecture config.
    """

    def __init__(self, config: BambaConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = MLP(config)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (key, value))."""
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        attn_output, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, attn_output)

        # MLP path with pre-norm
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class _BambaTextModel(nn.Module):
    """Bamba text backbone: embedding → N x (Mamba2|Attention) → norm.

    Mamba2 and Attention layers are selected based on ``layer_types``.
    All layers use dense MLPs (no MoE).
    """

    def __init__(self, config: BambaConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        layer_types = config.layer_types or []
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            if ltype == "mamba2":
                self.layers.append(BambaMambaDecoderLayer(config))
            else:
                self.layers.append(BambaAttentionDecoderLayer(config))

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)

        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.final_layernorm(op, hidden_states)
        return hidden_states, present_key_values


class BambaCausalLMModel(nn.Module):
    """Bamba hybrid Mamba2+Attention causal language model.

    Uses ``HybridCausalLMTask`` with mixed ``"mamba2"`` and
    ``"full_attention"`` layer types for the cache.

    HuggingFace reference: ``BambaForCausalLM``.
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid SSM+Attention"
    config_class: type = BambaConfig

    def __init__(self, config: BambaConfig):
        super().__init__()
        self.config = config
        self.model = _BambaTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace BambaForCausalLM weights to ONNX parameters.

        Handles:
        1. Weight tying (embed_tokens ↔ lm_head)
        2. Mamba2 SSM params: A_log, D, dt_bias stay under mamba.ssm
        3. Norm rename: mamba.norm → mamba.norm (matches HF naming)
        4. MLP rename: feed_forward → feed_forward (matches HF naming)
        """
        if self.config.tie_word_embeddings:
            tie_word_embeddings(state_dict)

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_bamba_weight(key)
            new_state_dict[new_key] = value

        return new_state_dict


def _rename_bamba_weight(key: str) -> str:
    """Rename a single HF weight key to match ONNX module structure.

    HF BambaForCausalLM weight naming:
        model.layers.N.mamba.{in_proj, conv1d, out_proj, norm, A_log, D, dt_bias}
        model.layers.N.self_attn.{q,k,v,o}_proj
        model.layers.N.feed_forward.{gate,up,down}_proj
        model.layers.N.{input_layernorm, pre_ff_layernorm}
        model.{embed_tokens, final_layernorm}
        lm_head

    ONNX parameter naming:
        Same as HF, except SSM params are nested under mamba.ssm:
        model.layers.N.mamba.ssm.{A_log, D, dt_bias}
    """
    # SSM params: nest A_log, D, dt_bias under mamba.ssm
    ssm_params = (".mamba.A_log", ".mamba.D", ".mamba.dt_bias")
    for param in ssm_params:
        if key.endswith(param):
            # e.g. "model.layers.0.mamba.A_log" → "model.layers.0.mamba.ssm.A_log"
            return key.replace(".mamba.", ".mamba.ssm.")

    return key
