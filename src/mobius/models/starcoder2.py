# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""StarCoder2 causal language model.

StarCoder2 uses LayerNorm (not RMSNorm) and a non-gated two-layer MLP
(``c_fc`` → activation → ``c_proj``) rather than the SwiGLU three-matrix
MLP used by Llama-family models.

Replicates HuggingFace's ``Starcoder2ForCausalLM``.
"""

from __future__ import annotations

import torch
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import rename_mlp_projections
from mobius.components import FCMLP, DecoderLayer, LayerNorm
from mobius.models.base import LayerNormCausalLMModel, LayerNormTextModel


class _StarCoder2DecoderLayer(DecoderLayer):
    """DecoderLayer variant for StarCoder2.

    Replaces the default SwiGLU ``MLP`` with a non-gated ``FCMLP``
    (two linear projections with GELU activation), matching HF
    ``Starcoder2MLP`` (``c_fc`` / ``c_proj``).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config, norm_class=LayerNorm)
        # Override the gated MLP with a non-gated two-layer MLP.
        # Bias is handled by mlp_bias config field (default False for tests).
        bias = getattr(config, "mlp_bias", False)
        self.mlp = FCMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=config.hidden_act,
            bias=bias,
        )


class _StarCoder2TextModel(LayerNormTextModel):
    """StarCoder2 text backbone using non-gated MLP and LayerNorm."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Replace LayerNorm layers (from super) with StarCoder2-specific decoder layers
        # that use FCMLP instead of the gated MLP.
        self.layers = nn.ModuleList(
            [_StarCoder2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # Final norm is already LayerNorm from LayerNormTextModel.


class StarCoder2CausalLMModel(LayerNormCausalLMModel):
    """StarCoder2 causal language model.

    Differences from the base ``CausalLMModel``:

    - Uses ``LayerNorm`` (mean-centering + std-normalisation) instead of RMSNorm.
    - Uses a non-gated two-layer MLP (``c_fc`` / ``c_proj`` in HF naming,
      mapped to ``up_proj`` / ``down_proj`` in ONNX).

    Replicates HuggingFace's ``Starcoder2ForCausalLM``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.model = _StarCoder2TextModel(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HF StarCoder2 MLP weight keys to match our attribute names.

        HF StarCoder2 uses ``mlp.c_fc`` / ``mlp.c_proj``; our ``FCMLP`` uses
        ``mlp.up_proj`` / ``mlp.down_proj``.
        """
        renamed = {}
        for key, value in state_dict.items():
            key = rename_mlp_projections(key, "c_fc", "c_proj")
            renamed[key] = value
        return super().preprocess_weights(renamed)
