# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""TrOCR decoder model for OCR.

TrOCR is an encoder-decoder OCR model. The decoder has the same architecture
as BART (post-norm self-attention + cross-attention + FFN). The encoder is
typically a separate ViT model composed via VisionEncoderDecoderModel.

This module provides the decoder, which maps to BartForConditionalGeneration
with custom weight renaming for TrOCR-specific names.
"""

from __future__ import annotations

import torch

from mobius.models.bart import BartForConditionalGeneration, _rename_bart_weight


class TrOCRForConditionalGeneration(BartForConditionalGeneration):
    """TrOCR decoder model.

    Architecturally identical to BART; only weight renaming differs.
    The vision encoder is a separate model (e.g. ViT, DeiT, BEiT).
    """

    # TODO(feature): Add a VisionEncoderDecoderTask that pairs this decoder
    # with a vision encoder (ViT/DeiT/BEiT) for full TrOCR inference.
    # Currently only the decoder half is supported as a seq2seq model.
    # Prerequisites: Create a VisionEncoderDecoderTask in tasks/ that
    # produces 2 models (vision encoder + text decoder) with cross-attention.
    # The encoder output feeds into the decoder's encoder_hidden_states.
    # Complexity: M — similar to VisionLanguageTask but for encoder-decoder.

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_trocr_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor

        # Tied lm_head
        if "decoder.lm_head.weight" not in new_state_dict:
            embed = new_state_dict.get("decoder.embed_tokens.weight")
            if embed is not None:
                new_state_dict["decoder.lm_head.weight"] = embed

        return new_state_dict


def _rename_trocr_weight(name: str) -> str | None:
    """Rename HF TrOCR weight to our naming convention."""
    # TrOCR uses output_projection instead of lm_head
    if name == "output_projection.weight":
        return "decoder.lm_head.weight"

    # Everything else follows BART naming
    return _rename_bart_weight(name)
