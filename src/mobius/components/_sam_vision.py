# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""SAM ViT-B vision encoder for DeepSeek-OCR-2.

Implements the Segment Anything Model (SAM) image encoder with:
- Patch embedding (Conv2d 16x16)
- Absolute positional embedding (interpolated for non-standard sizes)
- Transformer blocks with window attention (size=14) or global attention
- Decomposed relative position bias (separate H and W params)
- Neck: Conv2d(768→256) → LayerNorm2d → Conv2d(256→256) → LayerNorm2d
- Downsample: Conv2d(256→512, s=2) → Conv2d(512→896, s=2)

Input: (B, 3, 1024, 1024)
Output: (B, 896, 16, 16)

Reference: SAM paper (Kirillov et al. 2023), DeepSeek-OCR-2 deepencoderv2.py
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import Linear


class _SAMPatchEmbed(nn.Module):
    """Conv2d patch embedding for SAM.

    (B, 3, H, W) → Conv2d(3→768, k=16, s=16) → (B, 768, H/16, W/16)
    → transpose to (B, H/16, W/16, 768).

    HF weight names: patch_embed.proj.weight, patch_embed.proj.bias
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        kernel_size: int = 16,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            (embed_dim, in_channels, kernel_size, kernel_size),
            name="proj.weight",
        )
        self.bias = nn.Parameter((embed_dim,), name="proj.bias")
        self._kernel_size = kernel_size

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # (B, 3, H, W) → (B, embed_dim, H/16, W/16)
        x = op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel_size, self._kernel_size],
            strides=[self._kernel_size, self._kernel_size],
        )
        # BCHW → BHWC for transformer
        return op.Transpose(x, perm=[0, 2, 3, 1])


class _SAMLayerNorm2d(nn.Module):
    """LayerNorm applied to channel dimension of BCHW tensors.

    Normalizes across channels at each spatial position.
    Equivalent to: (x - mean) / sqrt(var + eps) * weight + bias
    where mean/var computed over C dimension.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter([num_channels])
        self.bias = nn.Parameter([num_channels])
        self._eps = eps
        self._num_channels = num_channels

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: (B, C, H, W) — normalize over C dimension
        # Reshape to (B, C, H*W), normalize, reshape back
        # Or use InstanceNormalization which normalizes per-channel
        # Actually, LayerNorm2d normalizes across C for each spatial position.
        # Equivalent to: transpose to BHWC, LayerNorm(C), transpose back
        x = op.Transpose(x, perm=[0, 2, 3, 1])  # BHWC
        x = op.LayerNormalization(x, self.weight, self.bias, epsilon=self._eps, axis=-1)
        return op.Transpose(x, perm=[0, 3, 1, 2])  # BCHW


class _SAMAttention(nn.Module):
    """Multi-head attention with decomposed relative position bias.

    Operates on BHWC layout. Supports optional relative position embeddings
    that are decomposed into separate H and W components.

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether Q/K/V projections have bias.
        use_rel_pos: Whether to use decomposed relative position bias.
        input_size: (H, W) spatial resolution for computing rel pos params.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = dim // num_heads

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=True)

        self._use_rel_pos = use_rel_pos
        if use_rel_pos and input_size is not None:
            # Decomposed relative position bias: separate H and W params
            # rel_pos_h: (2*H-1, head_dim), rel_pos_w: (2*W-1, head_dim)
            self.rel_pos_h = nn.Parameter([2 * input_size[0] - 1, self._head_dim])
            self.rel_pos_w = nn.Parameter([2 * input_size[1] - 1, self._head_dim])
            # Precompute relative position indices
            self._input_h = input_size[0]
            self._input_w = input_size[1]
            self._rel_h_indices = self._compute_rel_indices(input_size[0], input_size[0])
            self._rel_w_indices = self._compute_rel_indices(input_size[1], input_size[1])

    @staticmethod
    def _compute_rel_indices(q_size: int, k_size: int) -> np.ndarray:
        """Precompute relative position indices for gathering."""
        q_coords = np.arange(q_size)[:, None]
        k_coords = np.arange(k_size)[None, :]
        relative_coords = q_coords - k_coords + (k_size - 1)
        return relative_coords.astype(np.int64)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: (B, H, W, C)
        # Flatten spatial dims for attention
        B = op.Shape(x, start=0, end=1)  # noqa: N806
        H_val = self._input_h  # noqa: N806
        W_val = self._input_w  # noqa: N806
        N = H_val * W_val  # total spatial tokens  # noqa: N806

        # (B, H, W, C) → (B, N, C)
        x_flat = op.Reshape(x, op.Concat(B, [-1, self._num_heads * self._head_dim], axis=0))

        # Q, K, V projection: (B, N, 3*C) -> 3 x (B, num_heads, N, head_dim)
        qkv = self.qkv(op, x_flat)
        # (B, N, 3*C) → (B, N, 3, num_heads, head_dim)
        qkv = op.Reshape(
            qkv,
            op.Concat(B, [N, 3, self._num_heads, self._head_dim], axis=0),
        )
        # → (3, B, num_heads, N, head_dim)
        qkv = op.Transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = op.Split(qkv, [1, 1, 1], axis=0, _outputs=3)
        q = op.Squeeze(q, [0])  # (B, num_heads, N, head_dim)
        k = op.Squeeze(k, [0])
        v = op.Squeeze(v, [0])

        if self._use_rel_pos:
            # Compute decomposed relative position bias
            # Rh[i,j] = rel_pos_h[q_h - k_h + (H-1)], indexed by (q_h, k_h)
            # → gather: (H, H, head_dim)
            rel_h_idx = op.Constant(value=ir.tensor(self._rel_h_indices))
            Rh = op.Gather(self.rel_pos_h, rel_h_idx)  # noqa: N806  # (H, H, head_dim)
            rel_w_idx = op.Constant(value=ir.tensor(self._rel_w_indices))
            Rw = op.Gather(self.rel_pos_w, rel_w_idx)  # noqa: N806  # (W, W, head_dim)

            # Compute rel_h bias: einsum "bhwc,hkc->bhwk"
            # q reshaped to (BH, H, W, head_dim)
            q_4d = op.Reshape(
                q,
                op.Concat([-1, H_val, W_val, self._head_dim], axis=0),
            )
            # Rh: (H, K, D) → transpose to (H, D, K) for matmul
            Rh_t = op.Transpose(Rh, perm=[0, 2, 1])  # noqa: N806  # (H, D, K)
            # Unsqueeze to (1, H, D, K) for batch broadcast
            Rh_t = op.Unsqueeze(Rh_t, [0])  # noqa: N806
            # q_4d: (BH, H, W, D) @ (1, H, D, K) → (BH, H, W, K)
            # MatMul broadcasts: batch dims (BH, H) vs (1, H) → OK
            rel_h = op.MatMul(q_4d, Rh_t)  # (BH, H, W, K=H)

            # Compute rel_w bias: einsum "bhwc,wkc->bhwk"
            # Need q indexed by w: q_4d transposed to (BH, W, H, D)
            q_for_w = op.Transpose(q_4d, perm=[0, 2, 1, 3])  # (BH, W, H, D)
            # Rw: (W, K, D) → transpose to (W, D, K)
            Rw_t = op.Transpose(Rw, perm=[0, 2, 1])  # noqa: N806  # (W, D, K)
            Rw_t = op.Unsqueeze(Rw_t, [0])  # noqa: N806  # (1, W, D, K)
            # (BH, W, H, D) @ (1, W, D, K) → (BH, W, H, K=W)
            rel_w = op.MatMul(q_for_w, Rw_t)  # (BH, W, H, W)
            # Transpose back: (BH, H, W, W)
            rel_w = op.Transpose(rel_w, perm=[0, 2, 1, 3])

            # Combine: rel_h[:,:,:,:,None] + rel_w[:,:,:,None,:]
            # → (BH, H, W, H, W) → (BH, H*W, H*W)
            rel_h_5d = op.Unsqueeze(rel_h, [-1])  # (BH, H, W, H, 1)
            rel_w_5d = op.Unsqueeze(rel_w, [3])  # (BH, H, W, 1, W)
            attn_bias = op.Add(rel_h_5d, rel_w_5d)  # (BH, H, W, H, W)
            attn_bias = op.Reshape(attn_bias, [-1, N, N])
            # → (B, num_heads, N, N)
            attn_bias = op.Reshape(
                attn_bias,
                op.Concat(B, [self._num_heads, N, N], axis=0),
            )

            # Scaled dot-product attention with bias
            scale = float(self._head_dim**-0.5)
            attn_weights = op.MatMul(q, op.Transpose(k, perm=[0, 1, 3, 2]))
            attn_weights = op.Mul(attn_weights, scale)
            attn_weights = op.Add(attn_weights, attn_bias)
            attn_weights = op.Softmax(attn_weights, axis=-1)
            attn_output = op.MatMul(attn_weights, v)
        else:
            # Standard scaled dot-product attention (no bias)
            scale = float(self._head_dim**-0.5)
            attn_weights = op.MatMul(q, op.Transpose(k, perm=[0, 1, 3, 2]))
            attn_weights = op.Mul(attn_weights, scale)
            attn_weights = op.Softmax(attn_weights, axis=-1)
            attn_output = op.MatMul(attn_weights, v)

        # (B, num_heads, N, head_dim) → (B, N, C) → (B, H, W, C)
        attn_output = op.Transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = op.Reshape(
            attn_output,
            op.Concat(B, [N, self._num_heads * self._head_dim], axis=0),
        )
        attn_output = self.proj(op, attn_output)
        return op.Reshape(
            attn_output,
            op.Concat(B, [H_val, W_val, self._num_heads * self._head_dim], axis=0),
        )


class _SAMMLPBlock(nn.Module):
    """MLP block for SAM: Linear → GELU → Linear."""

    def __init__(self, embedding_dim: int, mlp_dim: int):
        super().__init__()
        self.lin1 = Linear(embedding_dim, mlp_dim, bias=True)
        self.lin2 = Linear(mlp_dim, embedding_dim, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return self.lin2(op, op.Gelu(self.lin1(op, x)))


class _SAMBlock(nn.Module):
    """SAM transformer block with optional window attention.

    Structure: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual

    When window_size > 0, input is partitioned into non-overlapping windows
    before attention and reassembled afterward.

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim = dim * mlp_ratio.
        window_size: Window size for local attention (0 = global).
        input_size: (H, W) spatial dims after patch embedding.
        use_rel_pos: Whether to add decomposed relative position bias.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        window_size: int = 0,
        input_size: tuple[int, int] = (64, 64),
        use_rel_pos: bool = True,
    ):
        super().__init__()
        self.norm1 = _SAMLayerNorm(dim)
        # Attention input_size depends on windowing
        attn_input_size = (window_size, window_size) if window_size > 0 else input_size
        self.attn = _SAMAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            use_rel_pos=use_rel_pos,
            input_size=attn_input_size,
        )
        self.norm2 = _SAMLayerNorm(dim)
        self.mlp = _SAMMLPBlock(dim, int(dim * mlp_ratio))

        self._window_size = window_size
        self._input_size = input_size
        # Precompute padding for window partition
        if window_size > 0:
            H, W = input_size  # noqa: N806
            self._pad_h = (window_size - H % window_size) % window_size
            self._pad_w = (window_size - W % window_size) % window_size
            self._Hp = H + self._pad_h
            self._Wp = W + self._pad_w

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: (B, H, W, C)
        shortcut = x
        x = self.norm1(op, x)

        if self._window_size > 0:
            x = self._window_partition_attn(op, x)
        else:
            x = self.attn(op, x)

        x = op.Add(shortcut, x)

        # MLP: flatten to (B, H*W, C), apply MLP, reshape back
        residual = x
        x = self.norm2(op, x)
        B = op.Shape(x, start=0, end=1)  # noqa: N806
        H, W = self._input_size  # noqa: N806
        x_flat = op.Reshape(x, op.Concat(B, [H * W, -1], axis=0))
        x_flat = self.mlp(op, x_flat)
        x = op.Reshape(x_flat, op.Concat(B, [H, W, -1], axis=0))
        return op.Add(residual, x)

    def _window_partition_attn(self, op, x):
        """Apply windowed attention: partition → attention → unpartition."""
        ws = self._window_size
        H, W = self._input_size  # noqa: N806
        Hp, Wp = self._Hp, self._Wp  # noqa: N806
        B = op.Shape(x, start=0, end=1)  # noqa: N806
        C = self.attn._num_heads * self.attn._head_dim  # noqa: N806

        # Pad if needed: (B, H, W, C) → (B, Hp, Wp, C)
        if self._pad_h > 0 or self._pad_w > 0:
            # Pad spatial dims (H and W) with zeros
            x = op.Pad(
                x,
                [0, 0, 0, 0, 0, self._pad_h, 0, self._pad_w],
                0.0,
            )

        # Window partition:
        # (B, Hp, Wp, C) → (B, Hp/ws, ws, Wp/ws, ws, C)
        nH = Hp // ws  # noqa: N806
        nW = Wp // ws  # noqa: N806
        x = op.Reshape(x, op.Concat(B, [nH, ws, nW, ws, C], axis=0))
        # → (B, nH, nW, ws, ws, C)
        x = op.Transpose(x, perm=[0, 1, 3, 2, 4, 5])
        # → (B*nH*nW, ws, ws, C)
        x = op.Reshape(x, [-1, ws, ws, C])

        # Apply attention to each window
        x = self.attn(op, x)

        # Window unpartition:
        # (B*nH*nW, ws, ws, C) → (B, nH, nW, ws, ws, C)
        x = op.Reshape(x, op.Concat(B, [nH, nW, ws, ws, C], axis=0))
        # → (B, nH, ws, nW, ws, C)
        x = op.Transpose(x, perm=[0, 1, 3, 2, 4, 5])
        # → (B, Hp, Wp, C)
        x = op.Reshape(x, op.Concat(B, [Hp, Wp, C], axis=0))

        # Remove padding
        if self._pad_h > 0 or self._pad_w > 0:
            # Slice to (B, H, W, C)
            x = op.Slice(x, [0, 0], [H, W], [1, 2])

        return x


class _SAMLayerNorm(nn.Module):
    """LayerNorm with bias (standard, not RMSNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self.bias = nn.Parameter([hidden_size])
        self._eps = eps

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.LayerNormalization(x, self.weight, self.bias, epsilon=self._eps, axis=-1)


class _SAMConv2dNoBias(nn.Module):
    """Conv2d without bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.weight = nn.Parameter((out_channels, in_channels, kernel_size, kernel_size))
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        p = self._padding
        return op.Conv(
            x,
            self.weight,
            kernel_shape=[self._kernel_size, self._kernel_size],
            strides=[self._stride, self._stride],
            pads=[p, p, p, p],
        )


class SAMVisionEncoder(nn.Module):
    """SAM ViT-B image encoder for DeepSeek-OCR-2.

    Architecture:
    1. Patch embed: Conv2d(3→768, k=16, s=16) → (B, 64, 64, 768)
    2. Add absolute position embedding
    3. 12 transformer blocks (window_size=14, global at [2,5,8,11])
    4. Neck: Conv2d(768→256, k=1) → LN2d → Conv2d(256→256, k=3, p=1) → LN2d
    5. Downsample: Conv2d(256→512, k=3, s=2, p=1) → Conv2d(512→896, k=3, s=2, p=1)

    Input: (B, 3, 1024, 1024)
    Output: (B, 896, 16, 16)
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        window_size: int = 14,
        global_attn_indexes: tuple[int, ...] = (2, 5, 8, 11),
        downsample_channels: tuple[int, ...] = (512, 896),
    ):
        super().__init__()
        self._img_size = img_size
        self._has_downsample = len(downsample_channels) >= 2
        spatial_size = img_size // patch_size  # 64

        self.patch_embed = _SAMPatchEmbed(
            in_channels=3,
            embed_dim=embed_dim,
            kernel_size=patch_size,
        )

        # Absolute positional embedding: (1, 64, 64, 768)
        self.pos_embed = nn.Parameter([1, spatial_size, spatial_size, embed_dim])

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                _SAMBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    window_size=(window_size if i not in global_attn_indexes else 0),
                    input_size=(spatial_size, spatial_size),
                    use_rel_pos=True,
                )
                for i in range(depth)
            ]
        )

        # Neck: reduce channels from embed_dim to out_chans
        # neck.0: Conv2d(768→256, k=1, no bias)
        # neck.1: LayerNorm2d(256)
        # neck.2: Conv2d(256→256, k=3, p=1, no bias)
        # neck.3: LayerNorm2d(256)
        self.neck = nn.ModuleList(
            [
                _SAMConv2dNoBias(embed_dim, out_chans, kernel_size=1),
                _SAMLayerNorm2d(out_chans),
                _SAMConv2dNoBias(out_chans, out_chans, kernel_size=3, padding=1),
                _SAMLayerNorm2d(out_chans),
            ]
        )

        # Downsample convolutions (optional, for OCR-2 custom pipeline)
        # net_2: Conv2d(256→512, k=3, s=2, p=1, no bias)
        # net_3: Conv2d(512→896, k=3, s=2, p=1, no bias)
        if self._has_downsample:
            self.net_2 = _SAMConv2dNoBias(
                out_chans,
                downsample_channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.net_3 = _SAMConv2dNoBias(
                downsample_channels[0],
                downsample_channels[1],
                kernel_size=3,
                stride=2,
                padding=1,
            )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # pixel_values: (B, 3, 1024, 1024)
        # → (B, 64, 64, 768)
        x = self.patch_embed(op, pixel_values)

        # Add position embedding (same size, no interpolation needed)
        x = op.Add(x, self.pos_embed)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(op, x)

        # Neck: BHWC → BCHW → neck → BCHW
        x = op.Transpose(x, perm=[0, 3, 1, 2])  # (B, 768, 64, 64)
        for layer in self.neck:
            x = layer(op, x)
        # x: (B, 256, 64, 64)

        # Downsample (if configured)
        if self._has_downsample:
            x = self.net_2(op, x)  # (B, 512, 32, 32)
            x = self.net_3(op, x)  # (B, 896, 16, 16)

        return x
