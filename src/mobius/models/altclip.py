# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""AltCLIP contrastive model (BAAI/AltCLIP).

AltCLIP replaces CLIP's text encoder with XLM-RoBERTa for multilingual
support while keeping the CLIP vision encoder unchanged.

Architecture:
  text_model: XLM-RoBERTa encoder → LayerNorm → Linear → text_projection
  vision_model: CLIP ViT encoder → visual_projection

HuggingFace class: ``AltCLIPModel``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import LayerNorm, Linear
from mobius.models.bert import BertModel
from mobius.models.clip import CLIPVisionModel, _rename_clip_vision_weight

if TYPE_CHECKING:
    from onnxscript import ir


class _AltCLIPTextEncoder(nn.Module):
    """AltCLIP text encoder: XLM-RoBERTa → LayerNorm → Linear → projection.

    AltCLIP pools via the CLS token (index 0), applies a pre-LayerNorm,
    then a learned linear transformation, and finally projects to the
    shared embedding space.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.roberta = BertModel(config)
        self.pre_LN = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.transformation = Linear(
            config.hidden_size, config.hidden_size, bias=True
        )
        projection_dim = getattr(config, "projection_dim", config.hidden_size)
        self.text_projection = Linear(
            config.hidden_size, projection_dim, bias=False
        )

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
    ) -> ir.Value:
        # Encode text with XLM-RoBERTa
        hidden_states = self.roberta(
            op, input_ids, attention_mask, token_type_ids=input_ids
        )
        # CLS pooling (first token)
        cls_token = op.Gather(hidden_states, op.Constant(value_int=0), axis=1)
        # LayerNorm + linear transformation
        cls_token = self.pre_LN(op, cls_token)
        cls_token = self.transformation(op, cls_token)
        # Project to shared embedding space
        return self.text_projection(op, cls_token)


class _AltCLIPVisionEncoder(nn.Module):
    """AltCLIP vision encoder: CLIP ViT → visual_projection.

    Identical to the standard CLIP vision encoder. Pools via CLS token
    (index 0) and projects to the shared embedding space.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.vision_model = CLIPVisionModel(config)
        projection_dim = getattr(config, "projection_dim", config.hidden_size)
        self.visual_projection = Linear(
            config.hidden_size, projection_dim, bias=False
        )

    def forward(
        self, op: builder.OpBuilder, pixel_values: ir.Value
    ) -> ir.Value:
        hidden_states = self.vision_model(op, pixel_values=pixel_values)
        cls_token = op.Gather(
            hidden_states, op.Constant(value_int=0), axis=1
        )
        return self.visual_projection(op, cls_token)


class AltCLIPModel(nn.Module):
    """Top-level AltCLIP contrastive model (model_type: ``altclip``).

    Dual-encoder contrastive model mapping text and images to a shared
    embedding space. Text encoder is XLM-RoBERTa (multilingual), vision
    encoder is CLIP ViT.

    Replicates HuggingFace's ``AltCLIPModel``.
    """

    default_task = "contrastive"
    category = "Multimodal"
    modality = "vision"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.text_encoder = _AltCLIPTextEncoder(config)
        self.modality_encoder = _AltCLIPVisionEncoder(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_sd: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            new_name = _rename_altclip_weight(name)
            if new_name is not None:
                new_sd[new_name] = tensor
        return new_sd


def _rename_altclip_weight(name: str) -> str | None:
    """Map HuggingFace AltCLIP weight names to our module layout.

    HF layout:
      text_model.roberta.*         → text_encoder.roberta.*
      text_model.pre_LN.*          → text_encoder.pre_LN.*
      text_model.transformation.*  → text_encoder.transformation.*
      text_projection.*            → text_encoder.text_projection.*
      vision_model.*               → modality_encoder.vision_model.*
      visual_projection.*          → modality_encoder.visual_projection.*
      logit_scale                  → (skipped)
    """
    if name.startswith("logit_scale"):
        return None
    if name.startswith("text_projection."):
        return f"text_encoder.{name}"
    if name.startswith("visual_projection."):
        return f"modality_encoder.{name}"
    if name.startswith("text_model.roberta."):
        # Strip text_model. prefix; BertModel handles roberta. prefix
        inner = name[len("text_model."):]
        return f"text_encoder.{inner}"
    if name.startswith("text_model.pre_LN."):
        return f"text_encoder.{name[len('text_model.'):]}"
    if name.startswith("text_model.transformation."):
        return f"text_encoder.{name[len('text_model.'):]}"
    if name.startswith("vision_model."):
        inner = _rename_clip_vision_weight(name)
        return f"modality_encoder.vision_model.{inner}" if inner else None
    # Skip pooler and other unused weights
    return None
