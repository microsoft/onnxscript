# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Longformer encoder model.

Longformer uses sliding window + global attention for efficient long-document
processing. For ONNX inference, we use full bidirectional attention (which is
a strict superset of Longformer's attention pattern) and drop the separate
global Q/K/V projections.

This produces mathematically equivalent output for short sequences and valid
(though not identical to sparse attention) output for long sequences.

HuggingFace class: ``LongformerModel``
"""

from __future__ import annotations

import torch

from mobius.models.bert import BertModel, _rename_bert_weight


class LongformerModel(BertModel):
    """Longformer encoder using full bidirectional attention.

    Extends BertModel to handle Longformer's weight naming:
    - Strips ``longformer.`` prefix from HF weights
    - Drops ``*_global`` Q/K/V projections (not needed for full attention)
    - Otherwise identical to BERT encoder

    Replicates HuggingFace's ``LongformerModel`` for inference.
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_sd: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            new_name = _rename_longformer_weight(name)
            if new_name is not None:
                new_sd[new_name] = tensor
        return new_sd


def _rename_longformer_weight(name: str) -> str | None:
    """Map HuggingFace Longformer weight names to BertModel layout.

    Strips ``longformer.`` prefix and drops global attention projections
    (``query_global``, ``key_global``, ``value_global``) which are unused
    when using full bidirectional attention.
    """
    # Strip model prefix
    if name.startswith("longformer."):
        name = name[len("longformer."):]

    # Drop global attention projections — not needed for full attention
    if "_global." in name:
        return None

    # Drop pooler (not part of feature extraction output)
    if name.startswith("pooler."):
        return None

    return _rename_bert_weight(name)
