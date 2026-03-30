# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""HunYuan V1 Dense causal language model.

Extends the base CausalLMModel with:
- QK-norm enabled unconditionally (HF hardcodes ``query_layernorm`` /
  ``key_layernorm`` in every attention layer).
- Weight renames: HF ``query_layernorm`` → ``q_norm``,
  ``key_layernorm`` → ``k_norm`` to match our Attention component.

Replicates ``tencent/Hunyuan-A13B-Instruct`` (dense variant).
"""

from __future__ import annotations

import dataclasses

import torch

from mobius._configs import ArchitectureConfig
from mobius.models.base import CausalLMModel


class HunYuanV1DenseCausalLMModel(CausalLMModel):
    """HunYuan V1 Dense text model.

    HF HunYuanV1Dense unconditionally applies per-head RMSNorm on Q/K
    projections (``query_layernorm`` / ``key_layernorm``).  We enable
    ``attn_qk_norm=True`` in the config and rename the HF weight keys.
    """

    def __init__(self, config: ArchitectureConfig):
        # HunYuanV1Dense always uses QK-norm but the HF config does not
        # expose a ``use_qk_norm`` flag — force it on.
        config = dataclasses.replace(config, attn_qk_norm=True)
        super().__init__(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = super().preprocess_weights(state_dict)
        # HF uses .query_layernorm. / .key_layernorm.;
        # ONNX Attention uses .q_norm. / .k_norm.
        return {
            k.replace(".query_layernorm.", ".q_norm.").replace(
                ".key_layernorm.", ".k_norm."
            ): v
            for k, v in state_dict.items()
        }
