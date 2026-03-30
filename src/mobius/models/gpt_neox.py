# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""GPT-NeoX / Pythia causal LM with parallel residual connections.

Architecture: dual LayerNorm parallel residual — both attention and MLP receive
their own LayerNorm of the *original* hidden states and their outputs are summed
with the residual in a single addition.

Replicates HuggingFace's ``GPTNeoXForCausalLM`` (``use_parallel_residual=True``).
Attribute names are aligned with HF so that ``preprocess_weights`` only needs to
split fused QKV and rename a few MLP/attention projection keys.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import rename_mlp_projections, split_interleaved_qkv_weights
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


class _GPTNeoXDecoderLayer(nn.Module):
    """GPT-NeoX decoder layer with dual-LN parallel residual.

    Both attention and MLP receive separate LayerNorms applied to the
    *original* hidden states (before the residual addition), then their
    outputs are summed together with the residual:

        out = hidden + attn(input_layernorm(hidden))
                     + mlp(post_attention_layernorm(hidden))

    Attribute names match HF ``GPTNeoXLayer`` naming so that only
    minimal renames are needed in ``preprocess_weights``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        # Two separate LayerNorms — both applied to the ORIGINAL hidden states
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Attention(config)  # 'attention' matches HF attribute name
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

        # Dual norm: each branch normalizes the original hidden states independently
        attn_input = self.input_layernorm(op, hidden_states)  # (B, S, H)
        mlp_input = self.post_attention_layernorm(op, hidden_states)  # (B, S, H)

        attn_out, present_kv = self.attention(
            op, attn_input, attention_bias, position_embeddings, past_key_value
        )
        mlp_out = self.mlp(op, mlp_input)

        # Parallel residual: both branches added in one step
        hidden_states = op.Add(residual, op.Add(attn_out, mlp_out))
        return hidden_states, present_kv


class _GPTNeoXTextModel(nn.Module):
    """GPT-NeoX backbone with RoPE positional embeddings.

    Attribute names match HF ``GPTNeoXModel``:
    - ``embed_in`` for the token embedding
    - ``layers`` for the list of decoder layers
    - ``final_layer_norm`` for the output LayerNorm
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_in = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [_GPTNeoXDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
        hidden_states = self.embed_in(op, input_ids)  # (B, S, H)
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
                op, hidden_states, attention_bias, position_embeddings, past_kv
            )
            present_key_values.append(present_kv)

        hidden_states = self.final_layer_norm(op, hidden_states)
        return hidden_states, present_key_values


class GPTNeoXCausalLMModel(CausalLMModel):
    """GPT-NeoX / Pythia causal language model.

    Uses dual LayerNorm parallel residual connections — unlike standard
    sequential (Llama-style) or single-norm parallel (GPT-J style),
    each layer applies two separate norms to the original hidden states,
    one for attention and one for the MLP branch.

    Attribute names align with HF ``GPTNeoXForCausalLM``:
    - ``gpt_neox`` for the backbone
    - ``embed_out`` for the LM head (not ``lm_head``)

    GPT-NeoX only supports MHA (no GQA), so ``num_key_value_heads`` is
    forced to equal ``num_attention_heads`` to prevent weight shape
    mismatches.

    Replicates HuggingFace's ``GPTNeoXForCausalLM``.
    """

    default_task: str = "text-generation"
    category: str = "Text Generation"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        # GPT-NeoX is MHA only — override KV heads to prevent shape mismatches
        config = dataclasses.replace(config, num_key_value_heads=config.num_attention_heads)
        self.config = config
        self.gpt_neox = _GPTNeoXTextModel(config)
        self.embed_out = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.gpt_neox(
            op, input_ids, attention_mask, position_ids, past_key_values
        )
        logits = self.embed_out(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF GPT-NeoX weight names to our ONNX attribute names.

        Most paths match directly (gpt_neox.*, embed_out, input_layernorm,
        post_attention_layernorm). Only four transforms are needed:

        1. Split fused QKV:  ``attention.query_key_value.*``
           → ``attention.{q,k,v}_proj.*``
        2. Output proj rename: ``attention.dense.*``
           → ``attention.o_proj.*``
        3. MLP up rename: ``mlp.dense_h_to_4h.*`` → ``mlp.up_proj.*``
        4. MLP down rename: ``mlp.dense_4h_to_h.*`` → ``mlp.down_proj.*``
        """
        # gpt_neox_japanese uses the same architecture but a different top-level
        # module prefix; normalize it so the rest of the renames apply uniformly
        state_dict = {
            k.replace("gpt_neox_japanese.", "gpt_neox."): v for k, v in state_dict.items()
        }

        # Split per-head interleaved QKV: [h0_q, h0_k, h0_v, h1_q, ...]
        state_dict = split_interleaved_qkv_weights(
            state_dict,
            fused_key="attention.query_key_value",
            num_heads=self.config.num_attention_heads,
            kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
        )

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            key = key.replace(".attention.dense.", ".attention.o_proj.")
            # GPT-NeoX-Japanese stores the output projection bias as a separate
            # parameter (not as part of the Linear layer)
            key = key.replace(".attention.dense_bias", ".attention.o_proj.bias")
            key = rename_mlp_projections(key, "dense_h_to_4h", "dense_4h_to_h")
            new_state_dict[key] = value

        return super().preprocess_weights(new_state_dict)


class _GPTNeoXJapaneseDecoderLayer(nn.Module):
    """GPT-NeoX-Japanese decoder layer with sequential pre-norm.

    Unlike standard GPT-NeoX (which uses parallel residual), GPT-NeoX-Japanese
    uses the standard sequential pre-norm architecture:

        after_attn = hidden + attn(input_layernorm(hidden))
        out = after_attn + mlp(post_attention_layernorm(after_attn))

    Attribute names match HF ``GPTNeoXJapaneseLayer`` naming (same as GPT-NeoX).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Attention(config)
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

        # Sequential pre-norm: normalize then attend, then residual
        attn_input = self.input_layernorm(op, hidden_states)  # (B, S, H)
        attn_out, present_kv = self.attention(
            op, attn_input, attention_bias, position_embeddings, past_key_value
        )
        hidden_states = op.Add(residual, attn_out)  # first residual

        # Normalize post-attention hidden states for MLP
        residual = hidden_states
        mlp_input = self.post_attention_layernorm(op, hidden_states)  # (B, S, H)
        mlp_out = self.mlp(op, mlp_input)
        hidden_states = op.Add(residual, mlp_out)  # second residual

        return hidden_states, present_kv


class _GPTNeoXJapaneseTextModel(nn.Module):
    """GPT-NeoX-Japanese backbone.

    Identical structure to ``_GPTNeoXTextModel`` but uses
    ``_GPTNeoXJapaneseDecoderLayer`` (sequential residual).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_in = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [_GPTNeoXJapaneseDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
        hidden_states = self.embed_in(op, input_ids)
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
                op, hidden_states, attention_bias, position_embeddings, past_kv
            )
            present_key_values.append(present_kv)

        hidden_states = self.final_layer_norm(op, hidden_states)
        return hidden_states, present_key_values


class GPTNeoXJapaneseCausalLMModel(GPTNeoXCausalLMModel):
    """GPT-NeoX-Japanese causal language model.

    GPT-NeoX-Japanese uses sequential pre-norm rather than GPT-NeoX's
    parallel residual.  The backbone, embedding, and LM-head attribute names
    are otherwise identical to ``GPTNeoXCausalLMModel``.

    Additional weight-name differences vs GPT-NeoX:
    - ``gpt_neox_japanese.*`` top-level prefix (normalized in ``preprocess_weights``)
    - ``attention.dense_bias`` (separate bias tensor, not part of the Linear)
      is renamed to ``attention.o_proj.bias`` during preprocessing
    - ``mlp.dense_h_to_4h.*`` / ``mlp.dense_4h_to_h.*`` → up/down proj
      (same renaming as GPT-NeoX)
    - ``intermediate_multiple_size`` in HF config determines MLP width:
      set ``intermediate_size = intermediate_multiple_size * hidden_size``
      in the test config to match

    Replicates HuggingFace's ``GPTNeoXJapaneseForCausalLM``.
    """

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        config = dataclasses.replace(config, num_key_value_heads=config.num_attention_heads)
        self.config = config
        # Use sequential backbone instead of parallel-residual
        self.gpt_neox = _GPTNeoXJapaneseTextModel(config)
        self.embed_out = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF GPT-NeoX-Japanese weight names to ONNX attribute names.

        Extends the parent GPT-NeoX preprocess_weights with:
        1. ``gpt_neox_japanese.*`` → ``gpt_neox.*`` prefix normalization (done
           in parent)
        2. ``attention.dense_bias`` → ``attention.o_proj.bias`` rename (done in
           parent)
        3. Zero-fill ``attention.o_proj.bias`` for layers that lack a
           ``dense_bias`` in HF (the first layer always lacks it).

        In HF GPT-NeoX-Japanese, ``GPTNeoXJapaneseAttention`` only carries a
        ``dense_bias`` parameter when ``use_bias=True``, which is set for every
        layer *except* the first one (``i != 0``).  Our ONNX graph has
        ``o_proj.bias`` for all layers (``attn_o_bias=True``), so we provide
        explicit zeros for the first layer to match HF's implicit zero bias.
        """
        new_state_dict = super().preprocess_weights(state_dict)

        # Provide zero o_proj bias for any layer that HF did not supply a
        # dense_bias for (always at least layer 0).
        hidden_size = self.config.hidden_size
        for key in list(new_state_dict.keys()):
            if "attention.o_proj.weight" in key:
                bias_key = key.replace(".weight", ".bias")
                if bias_key not in new_state_dict:
                    new_state_dict[bias_key] = torch.zeros(hidden_size)

        return new_state_dict
