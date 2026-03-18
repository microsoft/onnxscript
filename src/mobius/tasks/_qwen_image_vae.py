# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""QwenImage 3D VAE task for encoder/decoder with 5D (video) inputs.

Builds a ModelPackage with separate "encoder" and "decoder" ONNX graphs:
- "encoder": sample (B, 3, T, H, W) → latent_dist (B, 2*z_dim, T', H', W')
- "decoder": latent_sample (B, z_dim, T', H', W') → sample (B, 3, T, H, W)
"""

from __future__ import annotations

import onnx_ir as ir

from mobius._diffusers_configs import QwenImageVAEConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class QwenImageVAETask(ModelTask):
    """Build 3D causal VAE encoder and decoder ONNX graphs."""

    name = "qwen-image-vae"

    def build(
        self,
        module,
        config: QwenImageVAEConfig,
    ) -> ModelPackage:
        pkg = ModelPackage()
        pkg["encoder"] = self._build_encoder_graph(module, config)
        pkg["decoder"] = self._build_decoder_graph(module, config)
        return pkg

    def _build_encoder_graph(
        self,
        module,
        config: QwenImageVAEConfig,
    ) -> ir.Model:
        sample = ir.Value(
            name="sample",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(("batch", 3, "frames", "height", "width")),
        )

        graph, builder = _make_graph([sample], name="vae_encoder")
        op = builder.op

        hidden_states = module.encoder(op, sample)
        hidden_states = module.quant_conv(op, hidden_states)

        hidden_states.name = "latent_dist"
        graph.outputs.append(hidden_states)

        return _make_model(graph)

    def _build_decoder_graph(
        self,
        module,
        config: QwenImageVAEConfig,
    ) -> ir.Model:
        latent_sample = ir.Value(
            name="latent_sample",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(("batch", config.z_dim, "frames", "height", "width")),
        )

        graph, builder = _make_graph([latent_sample], name="vae_decoder")
        op = builder.op

        hidden_states = module.post_quant_conv(op, latent_sample)
        hidden_states = module.decoder(op, hidden_states)

        hidden_states.name = "sample"
        graph.outputs.append(hidden_states)

        return _make_model(graph)
