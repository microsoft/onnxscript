# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""QwenImage 3D causal VAE for video/image latent encoding.

Architecture: 3D causal autoencoder with temporal downsampling, RMS norm,
single-head attention, and residual blocks. Supports both 2D (image) and
3D (video) inputs.

HF diffusers class: AutoencoderKLQwenImage
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius._diffusers_configs import QwenImageVAEConfig
from mobius.components import Conv2d as _Conv2d
from mobius.components import SiLU as _SiLU

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _CausalConv3d(nn.Module):
    """3D convolution with causal (left) padding on the temporal axis.

    Mirrors QwenImageCausalConv3d from diffusers but without feature caching.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.weight = nn.Parameter(
            (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        )
        self.bias = nn.Parameter((out_channels,))
        self._stride = stride
        # Causal padding for 5D input (B, C, T, H, W)
        # PyTorch F.pad format: (W_l, W_r, H_l, H_r, T_l, T_r)
        # ONNX Pad format: [B_begin, C_begin, T_begin, H_begin, W_begin,
        #                    B_end, C_end, T_end, H_end, W_end]
        t_left = 2 * padding[0]  # Causal: all padding on left
        h_pad = padding[1]
        w_pad = padding[2]
        self._onnx_pads = [
            0,
            0,
            t_left,
            h_pad,
            w_pad,  # begin pads
            0,
            0,
            0,
            h_pad,
            w_pad,  # end pads (T has 0 on right = causal)
        ]
        self._needs_pad = any(p > 0 for p in self._onnx_pads)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        if self._needs_pad:
            x = op.Pad(
                x,
                op.Constant(value_ints=self._onnx_pads),
                op.Constant(value_float=0.0),
            )
        return op.Conv(
            x,
            self.weight,
            self.bias,
            strides=list(self._stride),
            pads=[0, 0, 0, 0, 0, 0],  # Already padded above
            dilations=[1, 1, 1],
            group=1,
        )


class _RMSNorm3d(nn.Module):
    """RMS normalization for 3D feature maps (channel-first).

    Normalizes along channel dim=1, then scales by learnable gamma.
    Shape: (B, C, T, H, W) — normalize over C for each spatial position.
    """

    def __init__(self, dim: int):
        super().__init__()
        # gamma shape: (C, 1, 1, 1) for broadcasting over T, H, W
        self.gamma = nn.Parameter((dim, 1, 1, 1))
        self._scale = dim**0.5

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # F.normalize(x, dim=1) * scale * gamma
        # L2 normalize along channel dimension
        norm = op.ReduceL2(x, [1], keepdims=True)
        eps = op.Constant(value_float=1e-12)
        norm = op.Max(norm, eps)
        x_normalized = op.Div(x, norm)
        scale = op.Constant(value_float=self._scale)
        return op.Mul(op.Mul(x_normalized, scale), self.gamma)


class _ResidualBlock(nn.Module):
    """Residual block with two 3D causal convolutions and RMS norm.

    Mirrors QwenImageResidualBlock from diffusers.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = _RMSNorm3d(in_dim)
        self.conv1 = _CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = _RMSNorm3d(out_dim)
        self.conv2 = _CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = _CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else None
        self._silu = _SiLU()

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        h = x if self.conv_shortcut is None else self.conv_shortcut(op, x)
        x = self.norm1(op, x)
        x = self._silu(op, x)
        x = self.conv1(op, x)
        x = self.norm2(op, x)
        x = self._silu(op, x)
        x = self.conv2(op, x)
        return op.Add(x, h)


class _AttentionBlock(nn.Module):
    """Single-head self-attention for 3D feature maps.

    Operates on each frame independently: reshapes (B,C,T,H,W) to (B*T,C,H,W),
    applies attention over spatial positions, then reshapes back.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = _RMSNorm3d(dim)
        # 1x1 conv projections (operates on (B*T, C, H, W))
        self.to_qkv = _Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = _Conv2d(dim, dim, kernel_size=1)
        self._dim = dim

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        identity = x
        # x: (B, C, T, H, W)
        b_shape = op.Shape(x, start=0, end=1)
        c_shape = op.Shape(x, start=1, end=2)
        t_shape = op.Shape(x, start=2, end=3)
        h_shape = op.Shape(x, start=3, end=4)
        w_shape = op.Shape(x, start=4, end=5)

        # Reshape to (B*T, C, H, W) for 2D operations
        bt = op.Mul(b_shape, t_shape)
        neg_one = op.Constant(value_ints=[-1])
        x_2d = op.Reshape(
            op.Transpose(x, perm=[0, 2, 1, 3, 4]),
            op.Concat(bt, c_shape, h_shape, w_shape, axis=0),
        )

        x_2d = self.norm(op, x_2d)

        # QKV projection
        qkv = self.to_qkv(op, x_2d)
        # Reshape for attention: (B*T, 3C, H, W) → (B*T, 1, H*W, 3C) → split
        hw = op.Mul(h_shape, w_shape)
        three_c = op.Constant(value_ints=[3])
        c3 = op.Mul(c_shape, three_c)
        one = op.Constant(value_ints=[1])
        qkv = op.Reshape(qkv, op.Concat(bt, one, c3, neg_one, axis=0))
        qkv = op.Transpose(qkv, perm=[0, 1, 3, 2])
        # Split into Q, K, V each (B*T, 1, H*W, C)
        q, k, v = op.Split(qkv, num_outputs=3, axis=-1, _outputs=3)

        # Scaled dot-product attention
        scale = self._dim**-0.5
        attn = op.Attention(q, k, v, scale=scale, q_num_heads=1, kv_num_heads=1)

        # Reshape back: (B*T, 1, H*W, C) → (B*T, C, H, W)
        attn = op.Transpose(
            op.Reshape(attn, op.Concat(bt, hw, c_shape, axis=0)),
            perm=[0, 2, 1],
        )
        attn = op.Reshape(attn, op.Concat(bt, c_shape, h_shape, w_shape, axis=0))

        # Output projection
        attn = self.proj(op, attn)

        # Reshape back to 5D: (B*T, C, H, W) → (B, T, C, H, W) → (B, C, T, H, W)
        out = op.Transpose(
            op.Reshape(attn, op.Concat(b_shape, t_shape, c_shape, h_shape, w_shape, axis=0)),
            perm=[0, 2, 1, 3, 4],
        )
        return op.Add(out, identity)


class _MidBlock(nn.Module):
    """Middle block: ResNet -> (Attention -> ResNet) x num_layers."""

    def __init__(self, dim: int, num_layers: int = 1):
        super().__init__()
        self.resnets = nn.ModuleList([_ResidualBlock(dim, dim)])
        self.attentions = nn.ModuleList([])
        for _ in range(num_layers):
            self.attentions.append(_AttentionBlock(dim))
            self.resnets.append(_ResidualBlock(dim, dim))

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        x = self.resnets[0](op, x)
        for i in range(len(self.attentions)):
            x = self.attentions[i](op, x)
            x = self.resnets[i + 1](op, x)
        return x


# ---------------------------------------------------------------------------
# Resampling (up/down)
# ---------------------------------------------------------------------------


class _SequentialConv2d(nn.Module):
    """Conv2d wrapped to match PyTorch Sequential(ZeroPad2d, Conv2d) naming.

    PyTorch names the Conv2d at index 1 in the Sequential, so weights are
    stored as ``resample.1.weight`` and ``resample.1.bias``. We replicate
    this naming by using setattr with key "1".
    """

    def __init__(self, conv: _Conv2d, asymmetric_pad: bool = False):
        super().__init__()
        # Use setattr with "1" to match PyTorch Sequential index naming
        setattr(self, "1", conv)
        self._asymmetric_pad = asymmetric_pad

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        if self._asymmetric_pad:
            # Asymmetric pad (0,1,0,1) on H,W — equivalent to ZeroPad2d
            x = op.Pad(
                x,
                op.Constant(value_ints=[0, 0, 0, 0, 0, 0, 1, 1]),
                op.Constant(value_float=0.0),
            )
        conv = getattr(self, "1")
        return conv(op, x)


class _Resample(nn.Module):
    """2D or 3D resampling for encoder (downsample) or decoder (upsample).

    For ONNX export, temporal resampling uses CausalConv3d (downsample3d)
    or CausalConv3d + pixel shuffle (upsample3d).
    """

    def __init__(self, dim: int, mode: str):
        super().__init__()
        self._mode = mode
        if mode == "downsample2d":
            self.resample = _SequentialConv2d(
                _Conv2d(dim, dim, kernel_size=3, stride=2, padding=0),
                asymmetric_pad=True,
            )
        elif mode == "downsample3d":
            self.resample = _SequentialConv2d(
                _Conv2d(dim, dim, kernel_size=3, stride=2, padding=0),
                asymmetric_pad=True,
            )
            self.time_conv = _CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        elif mode == "upsample2d":
            self.resample = _SequentialConv2d(
                _Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = _SequentialConv2d(
                _Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            )
            self.time_conv = _CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: (B, C, T, H, W)
        b_shape = op.Shape(x, start=0, end=1)
        c_shape = op.Shape(x, start=1, end=2)
        t_shape = op.Shape(x, start=2, end=3)
        h_shape = op.Shape(x, start=3, end=4)
        w_shape = op.Shape(x, start=4, end=5)

        if self._mode == "upsample3d":
            # Temporal upsample via CausalConv3d → pixel shuffle along T
            x = self.time_conv(op, x)
            # time_conv output: (B, 2C, T, H, W) → reshape to (B, 2, C, T, H, W)
            two = op.Constant(value_ints=[2])
            x = op.Reshape(
                x, op.Concat(b_shape, two, c_shape, t_shape, h_shape, w_shape, axis=0)
            )
            # Interleave: stack dim=0 and dim=1 along temporal → (B, C, T*2, H, W)
            x0 = op.Gather(x, op.Constant(value_ints=[0]), axis=1)
            x1 = op.Gather(x, op.Constant(value_ints=[1]), axis=1)
            t2 = op.Mul(t_shape, two)
            # Interleave by stacking at dim 3, then reshaping
            # (B, C, T, H, W) each → stack → (B, C, T, 2, H, W) → (B, C, T*2, H, W)
            stacked = op.Concat(
                op.Unsqueeze(x0, [3]),
                op.Unsqueeze(x1, [3]),
                axis=3,
            )
            x = op.Reshape(stacked, op.Concat(b_shape, c_shape, t2, h_shape, w_shape, axis=0))

        if self._mode == "downsample3d":
            # Temporal downsample
            x = self.time_conv(op, x)
            # Update T after temporal conv
            t_shape = op.Shape(x, start=2, end=3)

        # Reshape to (B*T, C, H, W) for spatial operation
        bt = op.Mul(b_shape, t_shape)
        x_2d = op.Reshape(
            op.Transpose(x, perm=[0, 2, 1, 3, 4]),
            op.Concat(bt, c_shape, h_shape, w_shape, axis=0),
        )

        if self._mode in ("upsample2d", "upsample3d"):
            # Nearest-neighbor 2x upsample
            x_2d = op.Resize(
                x_2d,
                None,
                op.Constant(value_floats=[1.0, 1.0, 2.0, 2.0]),
                mode="nearest",
            )

        x_2d = self.resample(op, x_2d)

        # Get new spatial dims after resampling
        new_c = op.Shape(x_2d, start=1, end=2)
        new_h = op.Shape(x_2d, start=2, end=3)
        new_w = op.Shape(x_2d, start=3, end=4)

        # Reshape back to 5D
        out = op.Transpose(
            op.Reshape(x_2d, op.Concat(b_shape, t_shape, new_c, new_h, new_w, axis=0)),
            perm=[0, 2, 1, 3, 4],
        )
        return out


# ---------------------------------------------------------------------------
# Encoder and Decoder
# ---------------------------------------------------------------------------


class _Encoder3d(nn.Module):
    """3D encoder: conv_in → down_blocks → mid_block → norm → conv_out.

    Produces mean + logvar for the latent distribution.
    """

    def __init__(self, config: QwenImageVAEConfig):
        super().__init__()
        dim = config.base_dim
        z_dim = config.z_dim
        dim_mult = config.dim_mult
        num_res_blocks = config.num_res_blocks
        attn_scales = config.attn_scales
        temperal_downsample = config.temperal_downsample

        dims = [dim * u for u in [1, *list(dim_mult)]]
        scale = 1.0

        self.conv_in = _CausalConv3d(3, dims[0], 3, padding=1)

        self.down_blocks = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            for _ in range(num_res_blocks):
                self.down_blocks.append(_ResidualBlock(in_dim, out_dim))
                if scale in attn_scales:
                    self.down_blocks.append(_AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(_Resample(out_dim, mode=mode))
                scale /= 2.0

        self.mid_block = _MidBlock(dims[-1], num_layers=1)
        self.norm_out = _RMSNorm3d(dims[-1])
        self.conv_out = _CausalConv3d(dims[-1], z_dim * 2, 3, padding=1)
        self._silu = _SiLU()

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        x = self.conv_in(op, x)
        x = self.down_blocks(op, x)
        x = self.mid_block(op, x)
        x = self.norm_out(op, x)
        x = self._silu(op, x)
        x = self.conv_out(op, x)
        return x


class _UpBlock(nn.Module):
    """Decoder upsampling block: ResNets + optional upsample."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None = None,
    ):
        super().__init__()
        self.resnets = nn.Sequential()
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(_ResidualBlock(current_dim, out_dim))
            current_dim = out_dim
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([_Resample(out_dim, mode=upsample_mode)])

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        x = self.resnets(op, x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](op, x)
        return x


class _Decoder3d(nn.Module):
    """3D decoder: conv_in → mid_block → up_blocks → norm → conv_out.

    Reconstructs images/video from latent representation.
    """

    def __init__(self, config: QwenImageVAEConfig):
        super().__init__()
        dim = config.base_dim
        z_dim = config.z_dim
        dim_mult = config.dim_mult
        num_res_blocks = config.num_res_blocks
        temperal_upsample = list(reversed(config.temperal_downsample))

        # Decoder dimensions are reversed from encoder
        dims = [dim * u for u in [dim_mult[-1], *list(reversed(dim_mult))]]

        self.conv_in = _CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = _MidBlock(dims[0], num_layers=1)

        self.up_blocks = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            if i > 0:
                in_dim = in_dim // 2
            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
            self.up_blocks.append(_UpBlock(in_dim, out_dim, num_res_blocks, upsample_mode))

        self.norm_out = _RMSNorm3d(dims[-1])
        self.conv_out = _CausalConv3d(dims[-1], 3, 3, padding=1)
        self._silu = _SiLU()

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        x = self.conv_in(op, x)
        x = self.mid_block(op, x)
        x = self.up_blocks(op, x)
        x = self.norm_out(op, x)
        x = self._silu(op, x)
        x = self.conv_out(op, x)
        return op.Clip(x, op.Constant(value_float=-1.0), op.Constant(value_float=1.0))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class AutoencoderKLQwenImageModel(nn.Module):
    """3D causal VAE for QwenImage video/image generation.

    Provides encoder and decoder for 3D latent diffusion.
    HF diffusers class: AutoencoderKLQwenImage
    """

    default_task: str = "qwen-image-vae"
    category: str = "autoencoder"

    def __init__(self, config: QwenImageVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = _Encoder3d(config)
        self.decoder = _Decoder3d(config)
        self.quant_conv = _CausalConv3d(config.z_dim * 2, config.z_dim * 2, 1)
        self.post_quant_conv = _CausalConv3d(config.z_dim, config.z_dim, 1)
