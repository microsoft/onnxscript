# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""LayoutLMv3 model for document understanding.

LayoutLMv3 uses the same BERT-style encoder architecture. The key
difference is the embedding layer which adds 2D spatial position
embeddings from bounding box coordinates. For text-only feature
extraction, the model is architecturally identical to BERT.
"""

from __future__ import annotations

import torch

from mobius.models.bert import BertModel

# Old BERT checkpoints use gamma/beta instead of weight/bias
_PARAM_RENAMES = {"gamma": "weight", "beta": "bias"}


class LayoutLMv3Model(BertModel):
    """LayoutLMv3 document understanding model.

    Reuses BertModel architecture; only weight renaming differs.
    Spatial (bbox) position embeddings are not yet supported.
    """

    # TODO(feature): Add bbox (bounding box) input for spatial position
    # embeddings. LayoutLMv3 uses 2D spatial coordinates per token for
    # document layout understanding. Requires adding bbox inputs to the
    # task and wiring the spatial embedding layer in forward().
    # Prerequisites: Extend ImageClassificationTask (or create a new
    # DocumentUnderstandingTask) to accept bbox [B, seq_len, 4] input.
    # Add x/y position embedding tables and combine with text embeddings.
    # See HF LayoutLMv3Embeddings for the spatial embedding math.
    # Complexity: M — new input, 4 embedding tables (x1, y1, x2, y2).

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HF LayoutLMv3 weights.

        After BERT attribute alignment, only prefix stripping and
        skipping unsupported layers remain.
        """
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_layoutlmv3_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


def _rename_layoutlmv3_weight(name: str) -> str | None:
    """Rename HF LayoutLMv3 weight to our naming convention."""
    # Strip layoutlmv3. prefix
    if name.startswith("layoutlmv3."):
        name = name[len("layoutlmv3.") :]

    # Skip pooler, cls heads, visual-specific, spatial embeddings
    if name.startswith(
        (
            "pooler.",
            "cls.",
            "classifier.",
            "patch_embed.",
            "cls_token",
            "pos_embed",
            "norm.",
            "LayerNorm.",
            "embeddings.x_position_embeddings.",
            "embeddings.y_position_embeddings.",
            "embeddings.h_position_embeddings.",
            "embeddings.w_position_embeddings.",
        )
    ):
        return None

    # Rename gamma/beta to weight/bias (old BERT compat)
    parts = name.rsplit(".", 1)
    if len(parts) == 2 and parts[1] in _PARAM_RENAMES:
        name = f"{parts[0]}.{_PARAM_RENAMES[parts[1]]}"

    return name
