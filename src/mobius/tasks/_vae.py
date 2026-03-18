# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""VAE encode/decode task for diffusers autoencoders.

Builds a ModelPackage with separate "encoder" and "decoder" ONNX graphs:
- "encoder": sample (image) → latent_dist (mean + logvar)
- "decoder": latent_sample → sample (reconstructed image)
"""

from __future__ import annotations

import onnx_ir as ir

from mobius._diffusers_configs import VAEConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class VAETask(ModelTask):
    """Build VAE encoder and decoder ONNX graphs."""

    name = "vae"

    def build(
        self,
        module,
        config: VAEConfig,
    ) -> ModelPackage:
        pkg = ModelPackage()
        pkg["encoder"] = self._build_encoder_graph(module, config)
        pkg["decoder"] = self._build_decoder_graph(module, config)
        return pkg

    def _build_encoder_graph(
        self,
        module,
        config: VAEConfig,
    ) -> ir.Model:
        sample = ir.Value(
            name="sample",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(("batch", config.in_channels, "height", "width")),
        )

        graph, builder = _make_graph([sample], name="vae_encoder")
        op = builder.op

        hidden_states = module.encoder(op, sample=sample)
        if module.quant_conv is not None:
            hidden_states = module.quant_conv(op, hidden_states)

        hidden_states.name = "latent_dist"
        graph.outputs.append(hidden_states)

        return _make_model(graph)

    def _build_decoder_graph(
        self,
        module,
        config: VAEConfig,
    ) -> ir.Model:
        latent_sample = ir.Value(
            name="latent_sample",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(("batch", config.latent_channels, "height", "width")),
        )

        graph, builder = _make_graph([latent_sample], name="vae_decoder")
        op = builder.op

        hidden_states = latent_sample
        if module.post_quant_conv is not None:
            hidden_states = module.post_quant_conv(op, hidden_states)
        hidden_states = module.decoder(op, latent_sample=hidden_states)

        hidden_states.name = "sample"
        graph.outputs.append(hidden_states)

        return _make_model(graph)
