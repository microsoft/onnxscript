# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""DistilBERT encoder-only model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import FCMLP
from mobius.components._common import Embedding, LayerNorm
from mobius.components._encoder import EncoderAttention

if TYPE_CHECKING:
    import onnx_ir as ir


class _DistilBertEmbeddings(nn.Module):
    """DistilBERT embeddings: word + position (no token_type)."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value):
        word_embeds = self.word_embeddings(op, input_ids)

        seq_len = op.Shape(input_ids, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=0),
            seq_len,
            op.Constant(value_int=1),
        )
        position_ids = op.Cast(position_ids, to=7)
        position_ids = op.Unsqueeze(position_ids, [0])
        pos_embeds = self.position_embeddings(op, position_ids)

        embeddings = op.Add(word_embeds, pos_embeds)
        return self.LayerNorm(op, embeddings)


class _DistilBertEncoderLayer(nn.Module):
    """DistilBERT encoder layer: pre-norm attention + pre-norm FFN."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.attention = EncoderAttention(config.hidden_size, config.num_attention_heads)
        self.sa_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act,
        )
        self.output_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # Self-attention with post-norm
        attn_output = self.attention(op, hidden_states)
        hidden_states = self.sa_layer_norm(op, op.Add(hidden_states, attn_output))

        # FFN with post-norm
        ffn_output = self.ffn(op, hidden_states)
        hidden_states = self.output_layer_norm(op, op.Add(hidden_states, ffn_output))

        return hidden_states


class DistilBertModel(nn.Module):
    """DistilBERT encoder-only model for feature extraction.

    DistilBERT is a distilled version of BERT with:
    - No token type embeddings
    - Different weight naming (transformer.layer instead of encoder.layer)
    - Pre-norm style attention (sa_layer_norm, output_layer_norm)
    """

    default_task = "feature-extraction"
    category = "encoder-only"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embeddings = _DistilBertEmbeddings(config)
        self.transformer = _DistilBertEncoder(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None = None,
        token_type_ids: ir.Value | None = None,
    ):
        hidden_states = self.embeddings(op, input_ids)
        hidden_states = self.transformer(op, hidden_states)
        return hidden_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_distilbert_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


class _DistilBertEncoder(nn.Module):
    """Stack of DistilBERT encoder layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [_DistilBertEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        for layer in self.layer:
            hidden_states = layer(op, hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------

_DISTILBERT_RENAMES = {
    "attention.q_lin.": "attention.q_proj.",
    "attention.k_lin.": "attention.k_proj.",
    "attention.v_lin.": "attention.v_proj.",
    "attention.out_lin.": "attention.out_proj.",
}


def _rename_distilbert_weight(name: str) -> str | None:
    """Rename a HF DistilBERT weight to our naming convention."""
    # Strip "distilbert." prefix
    if name.startswith("distilbert."):
        name = name[len("distilbert.") :]

    # Skip vocab projector / MLM head
    if name.startswith(("vocab_", "cls.")):
        return None

    # Embeddings pass through
    if name.startswith("embeddings."):
        return name

    # Transformer layers
    if name.startswith("transformer.layer."):
        parts = name.split(".", 3)  # transformer, layer, idx, remainder
        if len(parts) < 4:
            return None
        layer_idx = parts[2]
        remainder = parts[3]

        for old, new in _DISTILBERT_RENAMES.items():
            if remainder.startswith(old):
                suffix = remainder[len(old) :]
                return f"transformer.layer.{layer_idx}.{new}{suffix}"

        # ffn, sa_layer_norm, output_layer_norm pass through
        # FFN: lin1 → up_proj, lin2 → down_proj (FCMLP naming)
        new_name = name.replace("ffn.lin1.", "ffn.up_proj.")
        new_name = new_name.replace("ffn.lin2.", "ffn.down_proj.")
        return new_name

    return None
