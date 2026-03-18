# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Gemma3 multimodal model (vision + text) — 3-model split.

Splits the Gemma3 architecture into three ONNX models:

- **decoder**: Gemma3 text decoder taking ``inputs_embeds``
- **vision**: SigLIP vision encoder + Gemma3 multimodal projector
- **embedding**: token embedding + image feature fusion
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import vlm_decoder_weights, vlm_embedding_weights
from mobius.components import (
    Gemma3MultiModalProjector,
    Linear,
    OffsetRMSNorm,
    VisionModel,
)
from mobius.models.gemma3_text import (
    Gemma3TextModel,
    Gemma3TextScaledWordEmbedding,
)

if TYPE_CHECKING:
    import onnx_ir as ir


class _Gemma3DecoderModel(nn.Module):
    """Gemma3 text decoder taking inputs_embeds."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = Gemma3TextModel(config)
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


class _Gemma3VisionEncoderModel(nn.Module):
    """Gemma3 vision encoder: SigLIP + Gemma3MultiModalProjector."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)
        patches_per_image = config.vision.image_size // config.vision.patch_size
        self.multi_modal_projector = Gemma3MultiModalProjector(
            vision_hidden_size=config.vision.hidden_size,
            text_hidden_size=config.hidden_size,
            patches_per_image=patches_per_image,
            tokens_per_image=config.vision.mm_tokens_per_image or patches_per_image**2,
            norm=OffsetRMSNorm(
                config.vision.hidden_size,
                eps=config.vision.norm_eps,
            ),
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


class _Gemma3EmbeddingModel(nn.Module):
    """Gemma3 embedding: scaled token lookup + image feature fusion.

    Uses Gemma3TextScaledWordEmbedding (multiply by sqrt(hidden_size))
    to match the text decoder's embedding layer.  Image token positions
    are replaced with projected vision features from the vision encoder.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        embed_scale = float(__import__("numpy").float16(config.hidden_size**0.5))
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=embed_scale,
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


class Gemma3MultiModalModel(nn.Module):
    """Gemma 3 vision-language model (3-model split).

    Builds three separate ONNX models:
    - decoder: Gemma3 text decoder taking inputs_embeds
    - vision_encoder: SigLIP + Gemma3MultiModalProjector
    - embedding: token embedding + image feature fusion
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _Gemma3DecoderModel(config)
        self.vision_encoder = _Gemma3VisionEncoderModel(config)
        self.embedding = _Gemma3EmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "Gemma3MultiModalModel uses VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HuggingFace weight keys to match ONNX initializer names.

        HF checkpoint layout → ONNX initializer names:
          ``language_model.`` → ``decoder.`` (text decoder weights)
          ``vision_tower.``   → ``vision_encoder.vision_tower.``
          ``multi_modal_projector.`` → ``vision_encoder.multi_modal_projector.``

        The embedding weight ``language_model.model.embed_tokens.weight``
        is duplicated into ``embedding.embed_tokens.weight`` so that both
        the decoder and the embedding sub-model receive it.
        """
        if self.config.tie_word_embeddings:
            embed_key = "language_model.model.embed_tokens.weight"
            head_key = "language_model.lm_head.weight"
            if head_key not in state_dict and embed_key in state_dict:
                state_dict[head_key] = state_dict[embed_key]

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("language_model."):
                new_key = "decoder." + key[len("language_model.") :]
                renamed[new_key] = value
                # Duplicate embed_tokens for the embedding sub-model
                if key == "language_model.model.embed_tokens.weight":
                    renamed["embedding.embed_tokens.weight"] = value
            elif key.startswith("vision_tower."):
                renamed["vision_encoder." + key] = value
            elif key.startswith("multi_modal_projector."):
                renamed["vision_encoder." + key] = value
            else:
                renamed[key] = value
        return renamed
