# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""BLIP vision model for image feature extraction.

BLIP's vision encoder is architecturally identical to ViT (pre-norm encoder
with Conv2d patch embedding, CLS token, and learned position embeddings).
The only differences are in HuggingFace weight naming: BLIP uses fused QKV
projections and different layer component names. These are handled entirely
in preprocess_weights.
"""

from __future__ import annotations

import re

import torch

from mobius.models.vit import ViTModel

_LAYER_PATTERN = re.compile(r"^encoder\.layers\.(\d+)\.(.+)$")


class BlipVisionModel(ViTModel):
    """BLIP vision encoder for image feature extraction.

    Reuses ViTModel architecture; only weight renaming differs.
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name, new_tensors = _rename_blip_weight(name, tensor)
            if new_tensors is not None:
                new_state_dict.update(zip(new_name, new_tensors))
        return new_state_dict


def _rename_blip_weight(
    name: str, tensor: torch.Tensor
) -> tuple[list[str], list[torch.Tensor]] | tuple[None, None]:
    """Rename HF BLIP vision weight and split fused QKV if needed."""
    # Strip prefixes
    for prefix in ("vision_model.", "blip.vision_model.", "model.vision_model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    # Skip text model, decoder, and projection weights
    if name.startswith(
        (
            "text_model.",
            "text_decoder.",
            "text_projection.",
            "visual_projection.",
            "logit_scale",
        )
    ):
        return None, None

    # Embeddings
    if name == "embeddings.class_embedding":
        return ["embeddings.cls_token"], [tensor]
    if name == "embeddings.position_embedding":
        return ["embeddings.position_embeddings"], [tensor]
    if name.startswith("embeddings.patch_embedding."):
        suffix = name[len("embeddings.patch_embedding.") :]
        return [f"embeddings.patch_embeddings.projection.{suffix}"], [tensor]

    # Post-layernorm
    if name.startswith("post_layernorm."):
        suffix = name[len("post_layernorm.") :]
        return [f"layernorm.{suffix}"], [tensor]

    # Encoder layers
    m = _LAYER_PATTERN.match(name)
    if m:
        layer_idx, suffix = m.group(1), m.group(2)
        base = f"encoder.layer.{layer_idx}"

        # Fused QKV → split into separate Q, K, V
        if suffix == "self_attn.qkv.weight":
            q, k, v = tensor.chunk(3, dim=0)
            return (
                [
                    f"{base}.self_attn.q_proj.weight",
                    f"{base}.self_attn.k_proj.weight",
                    f"{base}.self_attn.v_proj.weight",
                ],
                [q, k, v],
            )
        if suffix == "self_attn.qkv.bias":
            q, k, v = tensor.chunk(3, dim=0)
            return (
                [
                    f"{base}.self_attn.q_proj.bias",
                    f"{base}.self_attn.k_proj.bias",
                    f"{base}.self_attn.v_proj.bias",
                ],
                [q, k, v],
            )

        # Output projection rename
        if suffix.startswith("self_attn.projection."):
            rest = suffix[len("self_attn.projection.") :]
            return [f"{base}.self_attn.out_proj.{rest}"], [tensor]

        # LayerNorm rename
        if suffix.startswith("layer_norm1."):
            rest = suffix[len("layer_norm1.") :]
            return [f"{base}.layernorm_before.{rest}"], [tensor]
        if suffix.startswith("layer_norm2."):
            rest = suffix[len("layer_norm2.") :]
            return [f"{base}.layernorm_after.{rest}"], [tensor]

        # MLP rename: fc1 → up_proj, fc2 → down_proj
        if suffix.startswith("mlp.fc1."):
            rest = suffix[len("mlp.fc1.") :]
            return [f"{base}.mlp.up_proj.{rest}"], [tensor]
        if suffix.startswith("mlp.fc2."):
            rest = suffix[len("mlp.fc2.") :]
            return [f"{base}.mlp.down_proj.{rest}"], [tensor]

    return None, None
