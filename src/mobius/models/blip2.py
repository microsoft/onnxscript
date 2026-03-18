# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""BLIP-2 multimodal model (vision + text) — 3-model split.

Splits the BLIP-2 architecture into three ONNX models for
onnxruntime-genai:

- **decoder**: text decoder taking ``inputs_embeds``
- **vision**: ViT vision encoder + Q-Former + language projection
- **embedding**: token embedding + image feature fusion

Architecture per forward pass:
    pixel_values → ViT → visual features
    visual features → Q-Former (cross-attention with learned queries)
    → language_projection → image features (in text embedding space)
    input_ids + image_features → embedding fusion → inputs_embeds
    inputs_embeds → LLM decoder → logits

HuggingFace weight names:
- ``vision_model.*``
- ``qformer.embeddings.query_tokens``
- ``qformer.encoder.layer.{i}.*``
- ``qformer.layernorm.*``
- ``language_projection.*``
- ``language_model.model.* / language_model.lm_head.*``

Reference: BLIP-2 (https://arxiv.org/abs/2301.12597)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import vlm_decoder_weights, vlm_embedding_weights
from mobius.components import (
    Embedding,
    Linear,
    QFormer,
    VisionModel,
)
from mobius.models.base import TextModel

if TYPE_CHECKING:
    import onnx_ir as ir


class _Blip2DecoderModel(nn.Module):
    """BLIP-2 text decoder taking inputs_embeds."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_decoder_weights(state_dict, tie=self.config.tie_word_embeddings)


class _Blip2VisionEncoderModel(nn.Module):
    """BLIP-2 vision encoder: ViT + Q-Former + language projection.

    Produces image features projected to the text embedding space:
        pixel_values → ViT → visual features
        → Q-Former (cross-attend with learned queries)
        → language_projection → [batch, num_query_tokens, text_hidden_size]
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        qformer_hidden = config.qformer_hidden_size or 768
        # ViT vision encoder
        self.vision_model = VisionModel(config)
        # Q-Former: learned queries cross-attend to visual features
        self.qformer = QFormer(
            num_query_tokens=config.num_query_tokens or 32,
            num_layers=config.qformer_num_hidden_layers or 12,
            hidden_size=qformer_hidden,
            num_attention_heads=config.qformer_num_attention_heads or 12,
            intermediate_size=config.qformer_intermediate_size or 3072,
            encoder_hidden_size=config.vision.hidden_size,
        )
        # Project Q-Former output to text embedding dimension
        self.language_projection = Linear(qformer_hidden, config.hidden_size)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # ViT: [batch, C, H, W] → [batch, num_patches, vision_hidden_size]
        vision_features = self.vision_model(op, pixel_values)
        # Q-Former: cross-attend → [batch, num_query_tokens, qformer_hidden_size]
        query_output = self.qformer(op, vision_features)
        # Project to text dim: [batch, num_query_tokens, text_hidden_size]
        return self.language_projection(op, query_output)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Vision model weights (ViT)
            if key.startswith("vision_model."):
                renamed[key] = value
            # Language projection (Q-Former output → text dim)
            elif key.startswith("language_projection."):
                renamed[key] = value
            # Q-Former weights
            elif key.startswith("qformer."):
                new_key = _rename_qformer_weight(key)
                if new_key is not None:
                    renamed[new_key] = value
        return renamed


class _Blip2EmbeddingModel(nn.Module):
    """BLIP-2 embedding: token lookup + image feature fusion.

    Replaces image token positions with projected Q-Former features.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.image_token_id = config.image_token_id or 0

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value, image_features: ir.Value):
        text_embeds = self.embed_tokens(op, input_ids)

        image_mask = op.Equal(
            input_ids,
            op.Constant(value_int=self.image_token_id),
        )
        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        # Build indices to scatter image features into text positions
        mask_int = op.Cast(image_mask, to=7)
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        gathered = op.Gather(image_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_embedding_weights(state_dict)


class Blip2Model(nn.Module):
    """BLIP-2 vision-language model (3-model split).

    Builds three separate ONNX models:
    - decoder: text decoder taking inputs_embeds
    - vision_encoder: ViT + Q-Former + language projection
    - embedding: token embedding + image feature fusion

    HuggingFace reference: ``Blip2ForConditionalGeneration``.
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _Blip2DecoderModel(config)
        self.vision_encoder = _Blip2VisionEncoderModel(config)
        self.embedding = _Blip2EmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "Blip2Model uses VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.config.tie_word_embeddings:
            embed_key = "language_model.model.embed_tokens.weight"
            head_key = "language_model.lm_head.weight"
            if head_key not in state_dict and embed_key in state_dict:
                state_dict[head_key] = state_dict[embed_key]
        return state_dict


# ---------------------------------------------------------------------------
# Q-Former weight name mapping
# ---------------------------------------------------------------------------

_QFORMER_LAYER_PATTERN = re.compile(r"^qformer\.encoder\.layer\.(\d+)\.(.+)$")

# Maps HF Q-Former sub-layer names to our component attribute names
_QFORMER_LAYER_RENAMES: dict[str, str] = {
    # Self-attention
    "attention.attention.query": "self_attn.q_proj",
    "attention.attention.key": "self_attn.k_proj",
    "attention.attention.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.out_proj",
    "attention.output.LayerNorm": "self_attn_layernorm",
    # Cross-attention
    "crossattention.attention.query": "cross_attn.q_proj",
    "crossattention.attention.key": "cross_attn.k_proj",
    "crossattention.attention.value": "cross_attn.v_proj",
    "crossattention.output.dense": "cross_attn.out_proj",
    "crossattention.output.LayerNorm": "cross_attn_layernorm",
    # FFN
    "intermediate_query.dense": "mlp.up_proj",
    "output_query.dense": "mlp.down_proj",
    "output_query.LayerNorm": "mlp_layernorm",
}


def _rename_qformer_weight(name: str) -> str | None:
    """Rename a HuggingFace BLIP-2 Q-Former weight to our naming convention.

    HF: ``qformer.encoder.layer.{i}.attention.attention.query.weight``
    Ours: ``qformer.layers.{i}.self_attn.q_proj.weight``
    """
    # Query tokens embedding
    if name == "qformer.embeddings.query_tokens":
        return "qformer.query_tokens"

    # Final layer norm
    if name.startswith("qformer.layernorm."):
        return name  # Already matches: qformer.layernorm.weight/bias

    # Encoder layers
    m = _QFORMER_LAYER_PATTERN.match(name)
    if m:
        layer_idx, suffix = m.group(1), m.group(2)
        for old, new in _QFORMER_LAYER_RENAMES.items():
            if suffix.startswith(old):
                remainder = suffix[len(old) :]
                return f"qformer.layers.{layer_idx}.{new}{remainder}"

    return None
