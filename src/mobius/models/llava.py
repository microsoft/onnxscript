# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""LLaVA multimodal model (vision + text) — 3-model split.

Splits the LLaVA architecture into three ONNX models for
onnxruntime-genai:

- **decoder**: text decoder taking ``inputs_embeds``
- **vision**: CLIP/SigLIP vision tower + MLP projector
- **embedding**: token embedding + image feature fusion

Used by: llava, llava_next, llava_onevision, molmo, paligemma, pixtral,
video_llava, and other models with the CLIP+MLP+LLM pattern.

HuggingFace weight names:
- ``vision_tower.vision_model.*``
- ``multi_modal_projector.linear_1/2.*``
- ``language_model.model.* / language_model.lm_head.*``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import vlm_decoder_weights, vlm_embedding_weights
from mobius.components import (
    Embedding,
    Linear,
    MLPMultiModalProjector,
    VisionModel,
)
from mobius.models.base import TextModel

if TYPE_CHECKING:
    import onnx_ir as ir


class _LLaVADecoderModel(nn.Module):
    """LLaVA text decoder taking inputs_embeds."""

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


class _LLaVAVisionEncoderModel(nn.Module):
    """LLaVA vision encoder: CLIP/SigLIP + MLP projector."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)
        self.multi_modal_projector = MLPMultiModalProjector(
            vision_hidden_size=config.vision.hidden_size,
            text_hidden_size=config.hidden_size,
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        vision_features = self.vision_tower(op, pixel_values)
        return self.multi_modal_projector(op, vision_features)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        renamed: dict[str, torch.Tensor] = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("vision_tower.", "multi_modal_projector."))
        }
        return renamed


class _LLaVAEmbeddingModel(nn.Module):
    """LLaVA embedding: token lookup + image feature fusion."""

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


class LLaVAModel(nn.Module):
    """LLaVA vision-language model (3-model split).

    Builds three separate ONNX models:
    - decoder: text decoder taking inputs_embeds
    - vision_encoder: CLIP/SigLIP + MLP projector
    - embedding: token embedding + image feature fusion
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _LLaVADecoderModel(config)
        self.vision_encoder = _LLaVAVisionEncoderModel(config)
        self.embedding = _LLaVAEmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "LLaVAModel uses VisionLanguageTask which calls "
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
