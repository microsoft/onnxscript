# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from mobius.models.base import CausalLMModel


class ChatGLMCausalLMModel(CausalLMModel):
    """ChatGLM model with partial rotary (0.5 factor) and MLP name remapping.

    ChatGLM uses interleaved RoPE and has a different MLP attribute naming
    convention (dense_4h_to_h instead of down_proj).
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = super().preprocess_weights(state_dict)
        for key in list(state_dict.keys()):
            # Map ChatGLM MLP attribute names
            if "dense_4h_to_h" in key:
                new_key = key.replace("dense_4h_to_h", "down_proj")
                state_dict[new_key] = state_dict.pop(key)
            elif "dense_h_to_4h" in key:
                new_key = key.replace("dense_h_to_4h", "up_proj")
                state_dict[new_key] = state_dict.pop(key)
            # Map ChatGLM attention attribute names
            if "self_attention" in key:
                new_key = key.replace("self_attention", "self_attn")
                state_dict[new_key] = state_dict.pop(key)
        return state_dict
