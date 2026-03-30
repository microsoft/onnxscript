# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""NanoChat causal language model.

Standard Llama-family decoder with a non-gated FCMLP (``fc1 → relu² → fc2``)
and parameter-free RMSNorm (no learnable weight).  QK-norm is also enabled.

HuggingFace NanoChat applies RoPE *then* QK-norm (reversed from standard
Llama ordering).  ``NanoChatAttention`` replicates this ordering.

HuggingFace NanoChat also applies ``norm`` *both before and after* the
transformer layers (unlike the standard single post-layers norm).
``NanoChatCausalLMModel`` replicates this by manually running the model
components and applying ``self.model.norm`` at both stages.

HuggingFace uses ``fc1`` / ``fc2`` for the MLP projections; our FCMLP
uses ``up_proj`` / ``down_proj``, so ``preprocess_weights`` renames them.
All norm weights are initialised to ones because HF NanoChatRMSNorm has
no learnable parameters.

Replicates HuggingFace ``NanoChatForCausalLM``.
"""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import torch
from onnxscript._internal import builder

from mobius._configs import NanoChatConfig
from mobius._weight_utils import rename_mlp_projections
from mobius.components import FCMLP, Attention, Linear
from mobius.components._attention import _apply_attention
from mobius.components._common import create_padding_mask
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models.base import CausalLMModel, TextModel


class NanoChatAttention(Attention):
    """Attention variant that applies RoPE before QK-norm.

    HF NanoChatAttention ordering: q/k/v_proj → RoPE → QK-norm → attention.
    The standard ``Attention`` component applies QK-norm → RoPE.

    This subclass overrides ``forward`` to replicate HF's ordering exactly.
    All parameters (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm) are
    inherited from ``Attention`` unchanged.
    """

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
        static_cache: object | None = None,
    ):
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        # RoPE FIRST — NanoChat applies rotation before normalizing QK magnitudes
        if position_embeddings is not None:
            query_states = apply_rotary_pos_emb(
                op,
                x=query_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )
            key_states = apply_rotary_pos_emb(
                op,
                x=key_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_key_value_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )

        # QK-norm AFTER RoPE (per-head): reshape [B, S, H*D] → [B, S, H, D] → norm → back
        if self.q_norm is not None and self.k_norm is not None:
            query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
            key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
            query_states = self.q_norm(op, query_states)
            key_states = self.k_norm(op, key_states)
            query_states = op.Reshape(query_states, [0, 0, -1])
            key_states = op.Reshape(key_states, [0, 0, -1])

        attn_output, present_key, present_value = _apply_attention(
            op,
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling,
            static_cache=static_cache,
        )
        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)


class NanoChatTextModel(TextModel):
    """TextModel subclass that applies the final norm both before and after layers.

    HF NanoChatModel.forward calls ``self.norm`` twice: once immediately after
    embedding (pre-layer norm) and once after all transformer layers (standard
    post-layer norm).  Overriding ``TextModel.forward`` here keeps the ONNX
    parameter namespace correct (``model.*`` paths stay intact).
    """

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)

        if attention_mask is not None:
            padding_mask = create_padding_mask(
                op,
                input_ids=hidden_states if input_ids is None else input_ids,
                attention_mask=attention_mask,
            )
        else:
            padding_mask = None

        # PRE-LAYER NORM — applied to embeddings before the transformer stack.
        # HF NanoChatModel.forward does: hidden = self.norm(hidden); for layer in layers: ...
        hidden_states = self.norm(op, hidden_states)

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=padding_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        # POST-LAYER NORM (standard final norm)
        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class NanoChatCausalLMModel(CausalLMModel):
    """NanoChat model: FCMLP with relu2 + parameter-free RMSNorm + QK-norm.

    HF NanoChat applies RoPE then QK-norm (reversed from standard Llama).
    We replace every layer's ``self_attn`` with ``NanoChatAttention`` which
    implements the correct RoPE → QK-norm ordering.

    HF NanoChat also applies ``norm`` before AND after the transformer layers
    (via ``NanoChatTextModel``).

    NanoChat also applies final logit soft-capping:
    ``logits = tanh(logits / cap) * cap`` where ``cap = 15.0`` by default.
    """

    config_class: type = NanoChatConfig

    def __init__(self, config: NanoChatConfig):
        # Enable QK-norm (per-head, not full)
        config = dataclasses.replace(config, attn_qk_norm=True)
        # Build all submodules manually so we can use NanoChatTextModel
        # (which applies pre-layer norm) instead of the standard TextModel.
        from onnxscript import nn as oxnn

        oxnn.Module.__init__(self)
        self.config = config
        self.model = NanoChatTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self._final_logit_softcapping = getattr(config, "final_logit_softcapping", 0.0)
        # Replace gated MLP → non-gated FCMLP and standard Attention → NanoChatAttention
        for layer in self.model.layers:
            layer.mlp = FCMLP(
                config.hidden_size,
                config.intermediate_size,
                activation=config.hidden_act,
                bias=config.mlp_bias,
            )
            # Swap to NanoChatAttention (RoPE → QK-norm ordering)
            layer.self_attn = NanoChatAttention(config)

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

        # Final logit soft-capping: tanh(logits/cap) * cap
        if self._final_logit_softcapping > 0.0:
            logits = op.Div(logits, self._final_logit_softcapping)
            logits = op.Tanh(logits)
            logits = op.Mul(logits, self._final_logit_softcapping)

        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HF fc1/fc2 → FCMLP up_proj/down_proj.

        Also injects ones for all norm weights since HF NanoChatRMSNorm
        is parameter-free while our ONNX RMSNorm expects a weight tensor.
        """
        new_state_dict = {}
        for name, tensor in state_dict.items():
            name = rename_mlp_projections(name, "fc1", "fc2")  # fc1 → up_proj, fc2 → down_proj
            new_state_dict[name] = tensor

        # Inject ones for all norm weights (parameter-free in HF)
        hidden_size = next(iter(state_dict.values())).shape[-1]
        head_dim = self.config.head_dim
        ones_hidden = torch.ones(hidden_size)
        ones_head = torch.ones(head_dim)
        for i in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{i}"
            new_state_dict.setdefault(f"{prefix}.input_layernorm.weight", ones_hidden)
            new_state_dict.setdefault(f"{prefix}.post_attention_layernorm.weight", ones_hidden)
            new_state_dict.setdefault(f"{prefix}.self_attn.q_norm.weight", ones_head)
            new_state_dict.setdefault(f"{prefix}.self_attn.k_norm.weight", ones_head)
        new_state_dict.setdefault("model.norm.weight", ones_hidden)
        return super().preprocess_weights(new_state_dict)
