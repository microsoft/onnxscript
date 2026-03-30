# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""GPT-J and CodeGen causal LMs with single-norm parallel residual.

Both GPT-J and CodeGen share the same ``transformer.h.N`` weight layout and
the same parallel residual computation:

    ln_out = ln_1(hidden)
    out = hidden + attn(ln_out) + mlp(ln_out)

The only difference between the two is that CodeGen uses a fused ``qkv_proj``
weight while GPT-J uses separate ``q_proj``, ``k_proj``, ``v_proj`` weights.

Both models are MHA only (no GQA), so ``num_key_value_heads`` is forced to
match ``num_attention_heads``.

Replicates HuggingFace's ``GPTJForCausalLM`` and ``CodeGenForCausalLM``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import rename_mlp_projections, split_codegen_qkv
from mobius.components import (
    FCMLP,
    Embedding,
    LayerNorm,
    Linear,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._attention import Attention
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class _GPTJDecoderLayer(nn.Module):
    """GPT-J / CodeGen decoder layer with single-norm parallel residual.

    A single LayerNorm is applied to the hidden states, then both attention
    and MLP receive the same normalized output. Their results are summed
    together with the residual:

        ln_out = ln_1(hidden)
        out = hidden + attn(ln_out) + mlp(ln_out)

    Attribute names match HF GPT-J / CodeGen naming (``ln_1``, ``attn``).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config)  # 'attn' matches HF attribute name
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act or "gelu",
            bias=config.mlp_bias,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ) -> tuple[ir.Value, tuple]:
        residual = hidden_states

        # Single norm feeds both attention and MLP
        ln_out = self.ln_1(op, hidden_states)  # (B, S, H)
        attn_out, present_kv = self.attn(
            op, ln_out, attention_bias, position_embeddings, past_key_value
        )
        mlp_out = self.mlp(op, ln_out)  # same ln_out as attention

        # Parallel residual: both branches added in one step
        hidden_states = op.Add(residual, op.Add(attn_out, mlp_out))
        return hidden_states, present_kv


class _GPTJTextModel(nn.Module):
    """GPT-J / CodeGen backbone with absolute position embedding + RoPE.

    Attribute names match HF GPT-J / CodeGen naming:
    - ``wte`` for token embedding
    - ``h`` for the decoder layer list
    - ``ln_f`` for the final LayerNorm
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.wte = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.h = nn.ModuleList(
            [_GPTJDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_f = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
        hidden_states = self.wte(op, input_ids)  # (B, S, H)
        position_embeddings = self.rotary_emb(op, position_ids)
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
                op, hidden_states, attention_bias, position_embeddings, past_kv
            )
            present_key_values.append(present_kv)

        hidden_states = self.ln_f(op, hidden_states)
        return hidden_states, present_key_values


class GPTJCausalLMModel(CausalLMModel):
    """GPT-J causal language model.

    Uses single-norm parallel residual: one LayerNorm feeds both the
    attention and MLP branches whose outputs are summed with the residual.
    GPT-J weights use separate ``q_proj``, ``k_proj``, ``v_proj`` projections
    (no fused QKV).

    GPT-J is MHA only (no GQA), so ``num_key_value_heads`` is forced to
    match ``num_attention_heads``.

    Replicates HuggingFace's ``GPTJForCausalLM``.
    """

    default_task: str = "text-generation"
    category: str = "Text Generation"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        # GPT-J is MHA only — override KV heads to prevent shape mismatches.
        # GPT-J uses interleaved RoPE (rotate_every_two style).
        config = dataclasses.replace(
            config,
            num_key_value_heads=config.num_attention_heads,
            rope_interleave=True,
        )
        self.config = config
        self.transformer = _GPTJTextModel(config)
        # GPT-J LM head has a bias term
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.transformer(
            op, input_ids, attention_mask, position_ids, past_key_values
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF GPT-J weight names to our ONNX attribute names.

        Most paths match directly (transformer.h, ln_1, attn.q/k/v_proj,
        lm_head, transformer.ln_f, transformer.wte). Only three renames:

        1. Output proj: ``attn.out_proj.*`` → ``attn.o_proj.*``
        2. MLP up:   ``mlp.fc_in.*``  → ``mlp.up_proj.*``
        3. MLP down: ``mlp.fc_out.*`` → ``mlp.down_proj.*``

        Also handles weight tying if ``lm_head.weight`` is absent.
        """
        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            key = key.replace(".attn.out_proj.", ".attn.o_proj.")
            key = rename_mlp_projections(key, "fc_in", "fc_out")
            new_state_dict[key] = value

        # Weight tying fallback
        if (
            "lm_head.weight" not in new_state_dict
            and "transformer.wte.weight" in new_state_dict
        ):
            new_state_dict["lm_head.weight"] = new_state_dict["transformer.wte.weight"]

        return super().preprocess_weights(new_state_dict)


class CodeGenCausalLMModel(CausalLMModel):
    """CodeGen causal language model.

    Identical architecture to GPT-J (single-norm parallel residual) but
    uses a fused ``qkv_proj`` weight that is split during
    ``preprocess_weights``.

    CodeGen is MHA only (no GQA), so ``num_key_value_heads`` is forced to
    match ``num_attention_heads``.

    Replicates HuggingFace's ``CodeGenForCausalLM``.
    """

    default_task: str = "text-generation"
    category: str = "Text Generation"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        # CodeGen is MHA only — override KV heads to prevent shape mismatches.
        # CodeGen uses interleaved RoPE (rotate_every_two style, same as GPT-J).
        config = dataclasses.replace(
            config,
            num_key_value_heads=config.num_attention_heads,
            rope_interleave=True,
        )
        self.config = config
        self.transformer = _GPTJTextModel(config)
        # CodeGen LM head has a bias term
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.transformer(
            op, input_ids, attention_mask, position_ids, past_key_values
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF CodeGen weight names to our ONNX attribute names.

        CodeGen uses a fused ``qkv_proj`` weight (no bias). All other
        transforms match GPT-J:

        1. Fused QKV: ``attn.qkv_proj.weight`` → split into
           ``attn.{q,k,v}_proj.weight``
        2. Output proj: ``attn.out_proj.*`` → ``attn.o_proj.*``
        3. MLP up:   ``mlp.fc_in.*``  → ``mlp.up_proj.*``
        4. MLP down: ``mlp.fc_out.*`` → ``mlp.down_proj.*``
        """
        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if ".attn.qkv_proj.weight" in key:
                # CodeGen fused QKV uses a model-parallel interleaved layout:
                # [q_mp0, v_mp0, k_mp0, q_mp1, v_mp1, k_mp1, ...]
                q, k, v = split_codegen_qkv(
                    value,
                    self.config.num_attention_heads,
                    self.config.head_dim,
                )
                prefix = key[: key.index(".attn.qkv_proj.weight")]
                new_state_dict[f"{prefix}.attn.q_proj.weight"] = q
                new_state_dict[f"{prefix}.attn.k_proj.weight"] = k
                new_state_dict[f"{prefix}.attn.v_proj.weight"] = v
                continue

            key = key.replace(".attn.out_proj.", ".attn.o_proj.")
            key = rename_mlp_projections(key, "fc_in", "fc_out")
            new_state_dict[key] = value

        # Weight tying fallback
        if (
            "lm_head.weight" not in new_state_dict
            and "transformer.wte.weight" in new_state_dict
        ):
            new_state_dict["lm_head.weight"] = new_state_dict["transformer.wte.weight"]

        return super().preprocess_weights(new_state_dict)
