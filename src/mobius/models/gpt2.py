# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""GPT-2 model with absolute positional embeddings and pre-norm LayerNorm.

Replicates HuggingFace's ``GPT2LMHeadModel``. Conv1D weights are transposed
during ``preprocess_weights`` and fused QKV projections are split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    FCMLP,
    Embedding,
    LayerNorm,
    Linear,
    create_attention_bias,
)
from mobius.components._attention import Attention
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class GPT2CausalLMModel(CausalLMModel):
    """GPT-2 causal language model.

    Differences from Llama-style:
    - Pre-norm with LayerNorm (not RMSNorm)
    - Absolute positional embeddings (not RoPE)
    - Combined QKV projection in HF weights (split during preprocess)
    - Conv1D weights are transposed vs Linear
    - Tied word embeddings (lm_head = wte)

    Replicates HuggingFace's ``GPT2LMHeadModel``.
    """

    default_task = "text-generation"
    category = "causal-lm"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.transformer = _GPT2TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.transformer(
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
        """Map HF GPT-2 weight names to ours.

        Most names match HF directly after attribute alignment. Only
        Conv1D transposes, QKV splitting, output proj rename, and weight
        tying remain.
        """
        new_state_dict = {}
        hidden = self.config.hidden_size
        for name, tensor in state_dict.items():
            # Combined QKV (c_attn): transpose Conv1D and split
            if "attn.c_attn.weight" in name:
                tensor = tensor.t()  # Conv1D [in, out] → Linear [out, in]
                q, k, v = tensor.split(hidden, dim=0)
                prefix = name.replace("attn.c_attn.weight", "attn.")
                new_state_dict[f"{prefix}q_proj.weight"] = q
                new_state_dict[f"{prefix}k_proj.weight"] = k
                new_state_dict[f"{prefix}v_proj.weight"] = v
                continue
            if "attn.c_attn.bias" in name:
                q, k, v = tensor.split(hidden, dim=0)
                prefix = name.replace("attn.c_attn.bias", "attn.")
                new_state_dict[f"{prefix}q_proj.bias"] = q
                new_state_dict[f"{prefix}k_proj.bias"] = k
                new_state_dict[f"{prefix}v_proj.bias"] = v
                continue

            # Output proj: c_proj → o_proj (transpose Conv1D weight)
            if "attn.c_proj.weight" in name:
                new_state_dict[name.replace("c_proj.", "o_proj.")] = tensor.t()
                continue
            if "attn.c_proj.bias" in name:
                new_state_dict[name.replace("c_proj.", "o_proj.")] = tensor
                continue

            # MLP Conv1D weights need transposing + rename to FCMLP naming
            # c_fc → up_proj, c_proj → down_proj (only in mlp.* context)
            if name.endswith(".weight") and ".mlp.c_fc." in name:
                new_state_dict[name.replace(".c_fc.", ".up_proj.")] = tensor.t()
                continue
            if name.endswith(".weight") and ".mlp.c_proj." in name:
                new_state_dict[name.replace(".c_proj.", ".down_proj.")] = tensor.t()
                continue
            # MLP biases also need renaming
            if name.endswith(".bias") and ".mlp.c_fc." in name:
                new_state_dict[name.replace(".c_fc.", ".up_proj.")] = tensor
                continue
            if name.endswith(".bias") and ".mlp.c_proj." in name:
                new_state_dict[name.replace(".c_proj.", ".down_proj.")] = tensor
                continue

            new_state_dict[name] = tensor

        # Tied embeddings: lm_head uses the same weights as wte
        if (
            "lm_head.weight" not in new_state_dict
            and "transformer.wte.weight" in new_state_dict
        ):
            new_state_dict["lm_head.weight"] = new_state_dict["transformer.wte.weight"]
        return new_state_dict


class _GPT2TextModel(nn.Module):
    """GPT-2 text backbone with absolute positional embeddings.

    Attribute names match HF GPT-2 naming (wte, wpe, h, ln_f).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.wte = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.wpe = Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList(
            [_GPT2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_f = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.wte(op, input_ids)

        position_embeds = self.wpe(op, position_ids)
        hidden_states = op.Add(hidden_states, position_embeds)

        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.h)
        for layer, past_kv in zip(self.h, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.ln_f(op, hidden_states)
        return hidden_states, present_key_values


class _GPT2DecoderLayer(nn.Module):
    """GPT-2 pre-norm decoder layer with LayerNorm.

    Structure: norm → attn → residual → norm → mlp → residual.
    Attribute names match HF GPT-2 naming (ln_1, ln_2, attn).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act,
            bias=True,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        past_key_value: tuple | None = None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(op, hidden_states)
        hidden_states, present_kv = self.attn(
            op, hidden_states, attention_bias, past_key_value=past_key_value
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.ln_2(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv
