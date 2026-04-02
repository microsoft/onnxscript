# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ZoeDepth model for metric depth estimation.

ZoeDepth combines a BEiT ViT-Large backbone with a DPT neck and two heads:
  1. A relative depth head (DPT-style) that produces relative depth.
  2. A metric depth head (attractor-based) that calibrates relative depth to
     metric scale using learnable bin centers and attractor updates.

Reference: ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth
(Bhat et al., 2023). HuggingFace: ``model_type=zoedepth``.
"""

from __future__ import annotations

import re

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig, ZoeDepthConfig
from mobius.components import (
    FCMLP,
    Conv2d,
    Conv2dNoBias,
    EncoderAttention,
    LayerNorm,
    Linear,
)
from mobius.models.depth_anything import (
    _Conv2dPatchEmbed,
    _FeatureFusionLayer,
    _ReassembleLayer,
)

# ---------------------------------------------------------------------------
# BEiT ViT backbone with layer scale
# ---------------------------------------------------------------------------


class _BEiTEncoderLayer(nn.Module):
    """BEiT encoder layer with layer-scale on attention and FFN residuals.

    Identical to the ViT encoder layer but multiplies the attention and FFN
    outputs by per-channel learnable scalars (lambda_1, lambda_2) before
    adding them to the residual.  This is the ``LayerScale`` mechanism
    introduced in CaiT and adopted by BEiT.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.layernorm_before = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = EncoderAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            bias=True,
        )
        # Learnable per-channel scale applied to attn output before residual
        self.layer_scale_1 = nn.Parameter(
            [config.hidden_size],
            data=ir.tensor(np.ones(config.hidden_size, dtype=np.float32)),
        )
        self.layernorm_after = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act,
        )
        # Learnable per-channel scale applied to FFN output before residual
        self.layer_scale_2 = nn.Parameter(
            [config.hidden_size],
            data=ir.tensor(np.ones(config.hidden_size, dtype=np.float32)),
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # Pre-norm self-attention with layer scale
        residual = hidden_states
        hidden_states = self.layernorm_before(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states)
        # Scale attn output: (B, S, H) * (H,) → (B, S, H)
        hidden_states = op.Mul(hidden_states, self.layer_scale_1)
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm FFN with layer scale
        residual = hidden_states
        hidden_states = self.layernorm_after(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        # Scale FFN output: (B, S, H) * (H,) → (B, S, H)
        hidden_states = op.Mul(hidden_states, self.layer_scale_2)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class _ZoeDepthViTBackbone(nn.Module):
    """BEiT ViT backbone for ZoeDepth.

    Extracts hidden states at ``config.backbone_out_indices`` (1-indexed).
    BEiT uses ``use_absolute_position_embeddings=False``, so no additive
    position embedding is added to patch tokens.  Layer-scale is applied
    inside each ``_BEiTEncoderLayer``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size

        # Patch + CLS token embeddings (no absolute position embedding)
        self.patch_embeddings = _Conv2dPatchEmbed(num_channels, hidden_size, patch_size)
        self.cls_token = nn.Parameter(
            [1, 1, hidden_size],
            data=ir.tensor(np.zeros((1, 1, hidden_size), dtype=np.float32)),
        )

        self.encoder = nn.ModuleList(
            [_BEiTEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 1-indexed layer indices at which to extract hidden states
        self.out_indices: list[int] = config.backbone_out_indices or []

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value) -> list[ir.Value]:
        # Embed patches: (B, C, H, W) → (B, S, hidden)
        patch_embeds = self.patch_embeddings(op, pixel_values)
        batch_size = op.Shape(patch_embeds, start=0, end=1)

        # Prepend CLS token: (1, 1, hidden) → (B, 1, hidden)
        cls_tokens = op.Expand(
            self.cls_token,
            op.Concat(batch_size, op.Constant(value_ints=[1, 1]), axis=0),
        )
        # Concat along sequence dim: (B, S+1, hidden)
        hidden_states = op.Concat(cls_tokens, patch_embeds, axis=1)

        # Run encoder, collect hidden states at out_indices
        feature_maps: list[ir.Value] = []
        for i, layer in enumerate(self.encoder):
            hidden_states = layer(op, hidden_states)
            # out_indices are 1-indexed: index 1 = after layer 0
            if (i + 1) in self.out_indices:
                normed = self.layernorm(op, hidden_states)
                feature_maps.append(normed)

        return feature_maps  # each: (B, S+1, hidden)


# ---------------------------------------------------------------------------
# DPT Neck with readout projection (for BEiT backbone)
# ---------------------------------------------------------------------------


class _ZoeDepthNeck(nn.Module):
    """DPT neck for ZoeDepth: reassemble, readout project, and fuse.

    Differs from the Depth Anything neck in that it uses
    ``readout_type="project"``: the CLS token representation is projected
    with each patch token before spatial upsampling/downsampling.
    """

    def __init__(self, config: ZoeDepthConfig):
        super().__init__()
        neck_sizes = config.neck_hidden_sizes
        factors = config.reassemble_factors
        fusion_size = config.fusion_hidden_size
        backbone_hidden = config.hidden_size

        # One Linear readout projection per feature level: (2*hidden → hidden) + GELU
        self.readout_projects = nn.ModuleList(
            [_ReadoutProject(backbone_hidden * 2, backbone_hidden) for _ in neck_sizes]
        )
        # Reassemble: project channels + spatial resize
        self.reassemble_layers = nn.ModuleList(
            [_ReassembleLayer(backbone_hidden, ch, f) for ch, f in zip(neck_sizes, factors)]
        )
        # Per-level conv before fusion
        self.convs = nn.ModuleList(
            [Conv2dNoBias(ch, fusion_size, kernel_size=3, padding=1) for ch in neck_sizes]
        )
        # Feature fusion layers (coarse-to-fine)
        self.fusion_layers = nn.ModuleList(
            [_FeatureFusionLayer(fusion_size) for _ in neck_sizes]
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: list[ir.Value],
        patch_height: ir.Value,
        patch_width: ir.Value,
    ) -> tuple[list[ir.Value], ir.Value]:
        """Reassemble backbone features and fuse them.

        Returns:
            fused_list: list of 4 fused feature maps (coarse→fine order).
            bottleneck: last pre-fusion conv feature (coarsest; fed to metric head).
        """
        reassembled: list[ir.Value] = []
        for i, hs in enumerate(hidden_states):
            # hs: (B, S+1, hidden) — includes CLS token

            # Extract CLS token: (B, 1, hidden)
            cls = op.Slice(
                hs,
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[1]),
                op.Constant(value_ints=[1]),  # axis=1
            )
            # Extract patch tokens: (B, S, hidden)
            patches = op.Slice(
                hs,
                op.Constant(value_ints=[1]),
                op.Constant(value_ints=[2**31 - 1]),
                op.Constant(value_ints=[1]),  # axis=1
            )

            # Expand CLS to (B, S, hidden) for concat
            # Shape of patches: (B, S, hidden)
            cls_expanded = op.Expand(cls, op.Shape(patches))  # (B, S, hidden)

            # Concat along last dim: (B, S, 2*hidden)
            combined = op.Concat(patches, cls_expanded, axis=2)

            # Readout projection + GELU: (B, S, 2*hidden) → (B, S, hidden)
            combined = self.readout_projects[i](op, combined)

            # Reshape to spatial: (B, S, hidden) → (B, hidden, pH, pW)
            batch = op.Shape(combined, start=0, end=1)
            channels = op.Shape(combined, start=2, end=3)
            hs_spatial = op.Transpose(combined, perm=[0, 2, 1])  # (B, hidden, S)
            hs_spatial = op.Reshape(
                hs_spatial,
                op.Concat(batch, channels, patch_height, patch_width, axis=0),
            )

            # Reassemble (project channels + spatial resize): (B, neck_ch, H', W')
            hs_spatial = self.reassemble_layers[i](op, hs_spatial)
            # Channel projection to fusion_hidden: (B, fusion_size, H', W')
            hs_spatial = self.convs[i](op, hs_spatial)
            reassembled.append(hs_spatial)

        # Fusion: coarse-to-fine (reverse order of reassembled list)
        reassembled.reverse()  # [coarsest, ..., finest]
        fused: ir.Value | None = None
        fused_list: list[ir.Value] = []
        for feature, layer in zip(reassembled, self.fusion_layers):
            if fused is None:
                fused = layer(op, feature)
            else:
                fused = layer(op, fused, feature)
            fused_list.append(fused)
        # fused_list order: [coarsest_fused, ..., finest_fused]

        # bottleneck = coarsest pre-fusion conv feature (last before reversal)
        # After reversal, reassembled[0] is the coarsest, which was formerly
        # the last element of the original list (index=-1 before reversal).
        # We exposed this as the "bottleneck" fed to the metric head.
        # But NOTE: we reversed in-place, so reassembled[0] = coarsest.
        # The HF code returns features[-1] (coarsest conv before fusion) via
        # `features = [self.convs[i](feature) for ...]` and `return output, features[-1]`.
        # After reversal our reassembled[0] == coarsest, which = original features[-1].
        bottleneck = reassembled[0]

        return fused_list, bottleneck


class _ReadoutProject(nn.Module):
    """Project concatenated (patch, cls) tokens: (B, S, 2H) → (B, S, H) + GELU."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = Linear(in_features, out_features)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, S, 2H) → Linear uses MatMul which supports 3D inputs
        return op.Gelu(self.proj(op, x))


# ---------------------------------------------------------------------------
# Relative depth estimation head
# ---------------------------------------------------------------------------


class _ZoeDepthRelativeHead(nn.Module):
    """Relative depth estimation head (DPT-style).

    Takes the finest fused feature map and produces both a relative depth
    map and intermediate feature activations used by the metric head.

    Returns:
        predicted_depth: (B, H, W) relative depth map.
        features: (B, num_relative_features, H, W) intermediate features.
    """

    def __init__(self, config: ZoeDepthConfig):
        super().__init__()
        fusion_size = config.fusion_hidden_size
        num_relative_features = config.num_relative_features
        self._head_in_index = config.head_in_index

        self.conv1 = Conv2d(fusion_size, fusion_size // 2, kernel_size=3, padding=1)
        self.conv2 = Conv2d(fusion_size // 2, num_relative_features, kernel_size=3, padding=1)
        self.conv3 = Conv2d(num_relative_features, 1, kernel_size=1, padding=0)

    def forward(
        self,
        op: builder.OpBuilder,
        fused_list: list[ir.Value],
    ) -> tuple[ir.Value, ir.Value]:
        # Use the feature at head_in_index (default -1 = finest)
        x = fused_list[self._head_in_index]  # (B, fusion_size, H, W)

        x = self.conv1(op, x)  # (B, fusion_size//2, H, W)
        # Upsample 2x
        x = op.Resize(
            x,
            None,  # roi
            op.Constant(value_floats=[1.0, 1.0, 2.0, 2.0]),
            mode="linear",
        )
        x = self.conv2(op, x)  # (B, num_relative_features, H, W)
        x = op.Relu(x)
        features = x  # saved for metric head: (B, num_relative_features, H, W)

        x = self.conv3(op, x)  # (B, 1, H, W)
        x = op.Relu(x)
        # Squeeze channel dim: (B, 1, H, W) → (B, H, W)
        predicted_depth = op.Squeeze(x, op.Constant(value_ints=[1]))

        return predicted_depth, features


# ---------------------------------------------------------------------------
# Metric depth head components
# ---------------------------------------------------------------------------


class _ZoeDepthProjector(nn.Module):
    """Two-layer 1x1 conv MLP with ReLU: in_features -> mlp_dim -> out_features."""

    def __init__(self, in_features: int, out_features: int, mlp_dim: int = 128):
        super().__init__()
        self.conv1 = Conv2d(in_features, mlp_dim, kernel_size=1, padding=0)
        self.conv2 = Conv2d(mlp_dim, out_features, kernel_size=1, padding=0)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.conv1(op, x)
        x = op.Relu(x)
        x = self.conv2(op, x)
        return x


class _ZoeDepthSeedBinRegressor(nn.Module):
    """Seed bin center regressor (softplus variant).

    Given the bottleneck feature, produces initial per-pixel bin centers
    using two 1x1 convolutions and a Softplus activation.  The seed centers
    are unbounded (metric scale determined later by attractor layers).

    Returns (bin_values, bin_centers) - for softplus both are the same tensor.
    """

    def __init__(
        self,
        config: ZoeDepthConfig,
        n_bins: int,
        min_depth: float,
        max_depth: float,
    ):
        super().__init__()
        self.conv1 = Conv2d(config.bottleneck_features, 256, kernel_size=1, padding=0)
        self.conv2 = Conv2d(256, n_bins, kernel_size=1, padding=0)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> tuple[ir.Value, ir.Value]:
        # x: (B, bottleneck_features, H, W)
        x = op.Relu(self.conv1(op, x))  # (B, 256, H, W)
        x = self.conv2(op, x)  # (B, n_bins, H, W)
        bin_centers = op.Softplus(x)  # (B, n_bins, H, W)  — unbounded
        return bin_centers, bin_centers


class _ZoeDepthAttractorLayer(nn.Module):
    """Attractor-based bin center update layer (unnormed / softplus variant).

    Predicts ``n_attractors`` attractor points per pixel and updates the
    incoming bin centers using the inverse-attractor rule::

        delta_c_j = sum_i (a_i - c_j) / (1 + alpha * |a_i - c_j|^gamma)

    Bin centers are updated as c'_j = c_j + delta_c_j and returned unbounded.
    """

    def __init__(
        self,
        config: ZoeDepthConfig,
        n_bins: int,
        n_attractors: int,
        min_depth: float,
        max_depth: float,
    ):
        super().__init__()
        self.n_attractors = n_attractors
        self.n_bins = n_bins
        bin_embed_dim = config.bin_embedding_dim

        self.conv1 = Conv2d(bin_embed_dim, bin_embed_dim, kernel_size=1, padding=0)
        self.conv2 = Conv2d(bin_embed_dim, n_attractors, kernel_size=1, padding=0)

        # inv_attractor constants (HF uses hardcoded defaults 300 / 2 in the
        # jit-scripted function rather than the stored self.alpha/gamma).
        self._alpha = 300.0
        self._gamma = 2.0

    def forward(
        self,
        op: builder.OpBuilder,
        x: ir.Value,
        prev_bin: ir.Value,
        prev_bin_embedding: ir.Value | None = None,
    ) -> tuple[ir.Value, ir.Value]:
        # Interpolate previous bin embedding to current spatial size if given
        if prev_bin_embedding is not None:
            target_h = op.Shape(x, start=2, end=3)
            target_w = op.Shape(x, start=3, end=4)
            prev_bin_embedding = op.Resize(
                prev_bin_embedding,
                None,
                None,  # scales
                op.Concat(
                    op.Shape(prev_bin_embedding, start=0, end=2),
                    target_h,
                    target_w,
                    axis=0,
                ),
                mode="linear",
                coordinate_transformation_mode="align_corners",
            )
            x = op.Add(x, prev_bin_embedding)

        # Predict attractor points
        x = op.Relu(self.conv1(op, x))  # (B, bin_embed_dim, H, W)
        attractors = op.Softplus(self.conv2(op, x))  # (B, n_attractors, H, W)

        # Resize prev_bin to current spatial size
        target_h = op.Shape(attractors, start=2, end=3)
        target_w = op.Shape(attractors, start=3, end=4)
        bin_centers = op.Resize(
            prev_bin,
            None,
            None,
            op.Concat(
                op.Shape(prev_bin, start=0, end=2),
                target_h,
                target_w,
                axis=0,
            ),
            mode="linear",
            coordinate_transformation_mode="align_corners",
        )
        # bin_centers: (B, n_bins, H, W)

        # Memory-efficient inv_attractor loop:
        #   delta_c += (a_i - c) / (1 + alpha * |a_i - c|^gamma)
        alpha = op.Constant(value_float=self._alpha)
        gamma = op.Constant(value_float=self._gamma)
        one = op.Constant(value_float=1.0)

        delta_c = op.Mul(bin_centers, op.Constant(value_float=0.0))  # zeros_like
        for i in range(self.n_attractors):
            # attractor_i: (B, 1, H, W)
            attractor_i = op.Slice(
                attractors,
                op.Constant(value_ints=[i]),
                op.Constant(value_ints=[i + 1]),
                op.Constant(value_ints=[1]),  # axis=1
            )
            # d = a_i - c: (B, n_bins, H, W)
            d = op.Sub(attractor_i, bin_centers)
            # inv_attractor(d) = d / (1 + alpha * |d|^gamma)
            abs_d = op.Abs(d)
            gamma_power = op.Pow(abs_d, gamma)
            denominator = op.Add(one, op.Mul(alpha, gamma_power))
            delta_c = op.Add(delta_c, op.Div(d, denominator))

        bin_new_centers = op.Add(bin_centers, delta_c)  # (B, n_bins, H, W)
        return bin_new_centers, bin_new_centers


class _ZoeDepthConditionalLogBinomialSoftmax(nn.Module):
    """Per-pixel MLP producing a log-binomial depth probability distribution.

    Given main features and a conditioning bin embedding, computes per-bin
    probabilities using the log-binomial model:
        y_k = log C(n-1, k) + k * log(p) + (n-1-k) * log(1-p)
        out = softmax(y / temperature, dim=1)   (B, n_bins, H, W)

    The probability ``p`` and temperature are predicted by the MLP.
    """

    def __init__(
        self,
        config: ZoeDepthConfig,
        in_features: int,
        condition_dim: int,
        n_classes: int,
    ):
        super().__init__()
        bottleneck = (in_features + condition_dim) // 2
        self.conv1 = Conv2d(in_features + condition_dim, bottleneck, kernel_size=1, padding=0)
        # 4 outputs: 2 for probability normalisation, 2 for temperature normalisation
        self.conv2 = Conv2d(bottleneck, 4, kernel_size=1, padding=0)
        self.n_classes = n_classes
        self.p_eps = 1e-4
        self.min_temp = config.min_temp
        self.max_temp = config.max_temp

        # Precompute log-binom table: log C(n-1, k) for k=0..n-1 using Stirling.
        # Stored as a constant ONNX initialiser of shape (1, n_classes, 1, 1).
        n = n_classes - 1
        k = np.arange(0, n_classes, dtype=np.float32)
        eps_s = 1e-7
        log_binom_coeff = (
            (n + eps_s) * np.log(n + eps_s)
            - (k + eps_s) * np.log(k + eps_s)
            - (n - k + eps_s) * np.log(n - k + eps_s)
        )
        log_binom_coeff = log_binom_coeff.reshape(1, n_classes, 1, 1)
        self._log_binom_coeff = nn.Parameter(
            [1, n_classes, 1, 1],
            data=ir.tensor(log_binom_coeff),
        )
        # k_idx: (1, n_classes, 1, 1)
        k_idx = k.reshape(1, n_classes, 1, 1)
        self._k_idx = nn.Parameter(
            [1, n_classes, 1, 1],
            data=ir.tensor(k_idx),
        )
        k_minus_1 = np.array([[[[n_classes - 1]]]], dtype=np.float32)
        self._k_minus_1 = nn.Parameter(
            [1, 1, 1, 1],
            data=ir.tensor(k_minus_1),
        )

    def forward(
        self,
        op: builder.OpBuilder,
        main_feature: ir.Value,
        condition_feature: ir.Value,
    ) -> ir.Value:
        # Concat and process through MLP
        x = op.Concat(main_feature, condition_feature, axis=1)
        x = op.Gelu(self.conv1(op, x))  # (B, bottleneck, H, W)
        x = op.Softplus(self.conv2(op, x))  # (B, 4, H, W)

        # Extract probability and temperature estimates
        p_eps = op.Constant(value_float=self.p_eps)
        p1 = op.Add(
            op.Slice(
                x,
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[1]),
                op.Constant(value_ints=[1]),
            ),
            p_eps,
        )
        p2 = op.Add(
            op.Slice(
                x,
                op.Constant(value_ints=[1]),
                op.Constant(value_ints=[2]),
                op.Constant(value_ints=[1]),
            ),
            p_eps,
        )
        t1 = op.Add(
            op.Slice(
                x,
                op.Constant(value_ints=[2]),
                op.Constant(value_ints=[3]),
                op.Constant(value_ints=[1]),
            ),
            p_eps,
        )
        t2 = op.Add(
            op.Slice(
                x,
                op.Constant(value_ints=[3]),
                op.Constant(value_ints=[4]),
                op.Constant(value_ints=[1]),
            ),
            p_eps,
        )

        # prob: (B, 1, H, W) in (0, 1)
        prob = op.Div(p1, op.Add(p1, p2))
        # temperature: scaled to [min_temp, max_temp] — (B, 1, H, W)
        temp_norm = op.Div(t1, op.Add(t1, t2))
        temp = op.Add(
            op.Mul(
                op.Constant(value_float=self.max_temp - self.min_temp),
                temp_norm,
            ),
            op.Constant(value_float=self.min_temp),
        )

        # Clamp probability for numerical stability
        log_eps = op.Constant(value_float=1e-4)
        log_one = op.Constant(value_float=1.0)
        prob_c = op.Clip(prob, log_eps, log_one)
        one_minus_prob_c = op.Clip(op.Sub(log_one, prob), log_eps, log_one)

        # Log-binomial probabilities: (B, n_classes, H, W)
        # y_k = log C(n-1, k) + k * log(p) + (n-1-k) * log(1-p)
        log_p = op.Log(prob_c)  # (B, 1, H, W)
        log_1p = op.Log(one_minus_prob_c)  # (B, 1, H, W)
        y = op.Add(
            self._log_binom_coeff,
            op.Add(
                op.Mul(self._k_idx, log_p),
                op.Mul(op.Sub(self._k_minus_1, self._k_idx), log_1p),
            ),
        )  # (B, n_classes, H, W)

        # Softmax over bin dimension with temperature scaling
        return op.Softmax(op.Div(y, temp), axis=1)  # (B, n_classes, H, W)


class _ZoeDepthMetricHead(nn.Module):
    """Attractor-based metric depth head.

    Converts relative features and fused feature maps into a calibrated
    metric depth map using seed bin regression and progressive attractor
    refinement.
    """

    def __init__(self, config: ZoeDepthConfig):
        super().__init__()
        bin_cfg = config.bin_configurations[0]
        n_bins = bin_cfg["n_bins"]
        min_depth = bin_cfg["min_depth"]
        max_depth = bin_cfg["max_depth"]
        bin_embed_dim = config.bin_embedding_dim
        n_attractors_list = config.num_attractors
        fusion_hidden = config.fusion_hidden_size
        num_relative_features = config.num_relative_features

        # Bottleneck 1x1 conv (applied to the coarsest neck feature)
        self.conv2 = Conv2d(
            config.bottleneck_features,
            config.bottleneck_features,
            kernel_size=1,
            padding=0,
        )
        # Seed bin regressor initialises per-pixel bin centers
        self.seed_bin_regressor = _ZoeDepthSeedBinRegressor(
            config,
            n_bins=n_bins,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        # Projector from bottleneck to bin embedding
        self.seed_projector = _ZoeDepthProjector(
            in_features=config.bottleneck_features,
            out_features=bin_embed_dim,
            mlp_dim=bin_embed_dim,
        )
        # One projector per attractor stage (map fused features → bin embeddings)
        self.projectors = nn.ModuleList(
            [
                _ZoeDepthProjector(
                    in_features=fusion_hidden,
                    out_features=bin_embed_dim,
                    mlp_dim=bin_embed_dim,
                )
                for _ in range(4)
            ]
        )
        # One attractor layer per stage
        self.attractors = nn.ModuleList(
            [
                _ZoeDepthAttractorLayer(
                    config,
                    n_bins=n_bins,
                    n_attractors=n_attractors_list[i],
                    min_depth=min_depth,
                    max_depth=max_depth,
                )
                for i in range(4)
            ]
        )
        # Final conditional log-binomial softmax mixing relative+metric features
        last_in = num_relative_features + 1  # +1 for concatenated relative depth channel
        self.conditional_log_binomial = _ZoeDepthConditionalLogBinomialSoftmax(
            config,
            in_features=last_in,
            condition_dim=bin_embed_dim,
            n_classes=n_bins,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        outconv_activation: ir.Value,
        bottleneck: ir.Value,
        feature_blocks: list[ir.Value],
        relative_depth: ir.Value,
    ) -> ir.Value:
        # Bottleneck projection
        x = self.conv2(op, bottleneck)  # (B, bottleneck_features, H, W)

        # Seed bin centers (per-pixel, unbounded)
        _, prev_bin = self.seed_bin_regressor(op, x)  # (B, n_bins, H, W)

        # Seed projector: bottleneck → bin embedding
        prev_bin_embedding = self.seed_projector(op, x)  # (B, bin_embed_dim, H, W)

        bin_embedding: ir.Value = prev_bin_embedding  # will be updated in loop
        bin_centers: ir.Value = prev_bin

        # Progressive attractor refinement over the 4 fused feature levels
        for projector, attractor, feature in zip(
            self.projectors, self.attractors, feature_blocks
        ):
            bin_embedding = projector(op, feature)  # (B, bin_embed_dim, H_i, W_i)
            bin, bin_centers = attractor(op, bin_embedding, prev_bin, prev_bin_embedding)
            prev_bin = bin
            prev_bin_embedding = bin_embedding

        # Assemble final depth probability distribution
        # Resize relative_depth to match outconv_activation spatial dims
        last = outconv_activation  # (B, num_relative_features, H_out, W_out)
        target_h = op.Shape(last, start=2, end=3)
        target_w = op.Shape(last, start=3, end=4)

        # relative_depth: (B, H, W) → (B, 1, H_out, W_out)
        rel_depth_4d = op.Unsqueeze(relative_depth, op.Constant(value_ints=[1]))
        rel_depth_resized = op.Resize(
            rel_depth_4d,
            None,
            None,
            op.Concat(op.Shape(rel_depth_4d, start=0, end=2), target_h, target_w, axis=0),
            mode="linear",
            coordinate_transformation_mode="align_corners",
        )
        # Concat: (B, num_relative_features+1, H_out, W_out)
        last_combined = op.Concat(last, rel_depth_resized, axis=1)

        # Resize bin_embedding to match last_combined
        bin_embed_resized = op.Resize(
            bin_embedding,
            None,
            None,
            op.Concat(op.Shape(bin_embedding, start=0, end=2), target_h, target_w, axis=0),
            mode="linear",
            coordinate_transformation_mode="align_corners",
        )

        # Log-binomial mixing: (B, n_bins, H_out, W_out)
        x_probs = self.conditional_log_binomial(op, last_combined, bin_embed_resized)

        # Resize bin_centers to match x_probs spatial dims
        x_h = op.Shape(x_probs, start=2, end=3)
        x_w = op.Shape(x_probs, start=3, end=4)
        bin_centers_resized = op.Resize(
            bin_centers,
            None,
            None,
            op.Concat(op.Shape(bin_centers, start=0, end=2), x_h, x_w, axis=0),
            mode="linear",
            coordinate_transformation_mode="align_corners",
        )

        # Weighted sum over bins: (B, n_bins, H, W) → (B, 1, H, W)
        out = op.ReduceSum(
            op.Mul(x_probs, bin_centers_resized),
            op.Constant(value_ints=[1]),
            keepdims=1,
        )
        # Squeeze channel dim: (B, 1, H, W) → (B, H, W)
        return op.Squeeze(out, op.Constant(value_ints=[1]))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class ZoeDepthForDepthEstimation(nn.Module):
    """ZoeDepth model for metric depth estimation.

    Combines a BEiT ViT-Large backbone with a DPT neck, a relative depth
    head, and an attractor-based metric depth calibration head.  Outputs
    a metric depth map (B, H, W) in the depth range specified by the first
    bin configuration (e.g. 0.001-10 m for NYU or 0.001-80 m for KITTI).
    """

    default_task = "image-classification"
    category = "Depth Estimation"
    config_class: type = ZoeDepthConfig

    def __init__(self, config: ZoeDepthConfig):
        super().__init__()
        self.config = config
        self._patch_size = config.patch_size

        self.backbone = _ZoeDepthViTBackbone(config)
        self.neck = _ZoeDepthNeck(config)
        self.relative_head = _ZoeDepthRelativeHead(config)
        self.metric_head = _ZoeDepthMetricHead(config)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value) -> ir.Value:
        # ── Backbone ──────────────────────────────────────────────────────
        feature_maps = self.backbone(op, pixel_values)
        # feature_maps: list of (B, S+1, hidden) at each out_index

        # Compute patch grid size from input
        patch_height = op.Constant(value_int=self.config.image_size // self._patch_size)
        patch_width = op.Constant(value_int=self.config.image_size // self._patch_size)
        patch_height = op.Reshape(patch_height, op.Constant(value_ints=[1]))
        patch_width = op.Reshape(patch_width, op.Constant(value_ints=[1]))

        # ── Neck ──────────────────────────────────────────────────────────
        # fused_list: [fused_coarsest, ..., fused_finest]
        # bottleneck: coarsest conv output (before fusion)
        fused_list, bottleneck = self.neck(op, feature_maps, patch_height, patch_width)

        # ── Relative head ─────────────────────────────────────────────────
        # Uses fused_list[-1] (finest) by default (head_in_index=-1)
        relative_depth, rel_features = self.relative_head(op, fused_list)

        # ── Metric head ───────────────────────────────────────────────────
        # feature_blocks = fused_list (all 4 fused levels, coarse→fine)
        metric_depth = self.metric_head(
            op,
            outconv_activation=rel_features,
            bottleneck=bottleneck,
            feature_blocks=fused_list,
            relative_depth=relative_depth,
        )
        return metric_depth  # (B, H, W)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            new_name = _rename_zoedepth_weight(name, tensor)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------

_BACKBONE_LAYER_RENAMES = {
    "attention.attention.query": "self_attn.q_proj",
    "attention.attention.key": "self_attn.k_proj",
    "attention.attention.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.out_proj",
    "intermediate.dense": "mlp.up_proj",
    "output.dense": "mlp.down_proj",
    "layernorm_before": "layernorm_before",
    "layernorm_after": "layernorm_after",
}

_BACKBONE_LAYER_PATTERN = re.compile(r"^backbone\.encoder\.layer\.(\d+)\.(.+)$")
_NECK_REASSEMBLE_PATTERN = re.compile(r"^neck\.reassemble_stage\.layers\.(.+)$")
_NECK_READOUT_PATTERN = re.compile(
    r"^neck\.reassemble_stage\.readout_projects\.(\d+)\.0\.(.+)$"
)
_NECK_FUSION_PATTERN = re.compile(r"^neck\.fusion_stage\.layers\.(.+)$")


def _rename_zoedepth_weight(name: str, tensor: torch.Tensor) -> str | None:
    """Map HuggingFace ZoeDepth weight names to our ONNX module naming."""
    # ── Backbone embeddings ───────────────────────────────────────────────
    if name == "backbone.embeddings.cls_token":
        return "backbone.cls_token"
    if name.startswith("backbone.embeddings.patch_embeddings.projection."):
        suffix = name[len("backbone.embeddings.patch_embeddings.projection.") :]
        return f"backbone.patch_embeddings.projection.{suffix}"
    # Skip absolute position embeddings (BEiT uses relative; not in state dict)
    if name == "backbone.embeddings.position_embeddings":
        return None

    # ── Backbone final layernorm ──────────────────────────────────────────
    if name.startswith("backbone.layernorm."):
        return name  # backbone.layernorm.{weight,bias}

    # ── Backbone encoder layers ───────────────────────────────────────────
    m = _BACKBONE_LAYER_PATTERN.match(name)
    if m:
        layer_idx, suffix = m.group(1), m.group(2)

        # Layer scale parameters
        if suffix == "layer_scale_1.lambda_1":
            return f"backbone.encoder.{layer_idx}.layer_scale_1"
        if suffix == "layer_scale_2.lambda_2":
            return f"backbone.encoder.{layer_idx}.layer_scale_2"

        # Attention / MLP renames
        for old, new in _BACKBONE_LAYER_RENAMES.items():
            if suffix.startswith(old):
                remainder = suffix[len(old) :]
                return f"backbone.encoder.{layer_idx}.{new}{remainder}"
        return None

    # ── Neck reassemble ───────────────────────────────────────────────────
    m_reassemble = _NECK_REASSEMBLE_PATTERN.match(name)
    if m_reassemble:
        suffix = m_reassemble.group(1)
        return f"neck.reassemble_layers.{suffix}"

    # Neck readout projects: reassemble_stage.readout_projects.N.0.{weight,bias}
    # → neck.readout_projects.N.proj.{weight,bias}
    m_readout = _NECK_READOUT_PATTERN.match(name)
    if m_readout:
        idx, attr = m_readout.group(1), m_readout.group(2)
        return f"neck.readout_projects.{idx}.proj.{attr}"

    # ── Neck convs ────────────────────────────────────────────────────────
    if name.startswith("neck.convs."):
        return name

    # ── Neck fusion ───────────────────────────────────────────────────────
    m_fusion = _NECK_FUSION_PATTERN.match(name)
    if m_fusion:
        suffix = m_fusion.group(1)
        return f"neck.fusion_layers.{suffix}"

    # ── Relative head ─────────────────────────────────────────────────────
    if name.startswith("relative_head."):
        return name  # names match directly

    # ── Metric head ───────────────────────────────────────────────────────
    if name.startswith("metric_head."):
        # mlp.0. and mlp.2. → conv1 and conv2 inside conditional_log_binomial
        name = name.replace(
            "metric_head.conditional_log_binomial.mlp.0.",
            "metric_head.conditional_log_binomial.conv1.",
        )
        name = name.replace(
            "metric_head.conditional_log_binomial.mlp.2.",
            "metric_head.conditional_log_binomial.conv2.",
        )
        return name

    return None
