# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phi-3 / Phi-3.5 causal language model.

Replicates HuggingFace's ``Phi3ForCausalLM``. The main difference from
the base CausalLMModel is that HuggingFace fuses QKV and gate-up projections
into single tensors, so ``preprocess_weights`` splits them.
"""

from __future__ import annotations

import torch

from mobius._weight_utils import split_fused_qkv, split_gate_up_proj
from mobius.models.base import CausalLMModel


class Phi3CausalLMModel(CausalLMModel):
    """Phi-3 / Phi-3.5 model with SuRoPE and fused QKV/gate-up weight splitting.

    Replicates HuggingFace's ``Phi3ForCausalLM``.
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = super().preprocess_weights(state_dict)
        for key in list(state_dict.keys()):
            if "qkv_proj" in key:
                q, k, v = split_fused_qkv(
                    state_dict.pop(key),
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
                state_dict[key.replace("qkv_proj", "q_proj")] = q
                state_dict[key.replace("qkv_proj", "k_proj")] = k
                state_dict[key.replace("qkv_proj", "v_proj")] = v
            elif "gate_up_proj" in key:
                gate, up = split_gate_up_proj(
                    state_dict.pop(key),
                    self.config.intermediate_size,
                )
                state_dict[key.replace("gate_up_proj", "gate_proj")] = gate
                state_dict[key.replace("gate_up_proj", "up_proj")] = up
        return state_dict
