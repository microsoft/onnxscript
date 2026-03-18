# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen2.5-VL vision encoder components.

Provides modules for the Qwen2.5-VL vision backbone:

- ``Qwen25VLPatchEmbed``: Conv3d patch tokenisation (14x14x2, no bias).
- ``Qwen25VLVisionRotaryEmbedding``: 2D rotary embeddings from grid positions.
- ``Qwen25VLVisionAttention``: Packed MHA with cu_seqlens boundaries.
- ``Qwen25VLVisionMLP``: Gate-up-down MLP with SiLU activation.
- ``Qwen25VLVisionBlock``: Pre-norm transformer block (RMSNorm → Attn → MLP).
- ``Qwen25VLPatchMerger``: Spatial merge via RMSNorm → reshape → MLP.
- ``Qwen25VLVisionModel``: Full encoder with windowed + full attention.

Windowed attention is handled by switching cu_seqlens boundaries per block:
blocks at ``fullatt_block_indexes`` use full-image cu_seqlens, others use
windowed cu_seqlens.  Both sets of boundaries are provided as model inputs,
precomputed at runtime by the host.
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import Linear
from mobius.components._rms_norm import RMSNorm
from mobius.components._scan_utils import (
    compact_scan_output,
    create_body_graph,
    rename_subgraph_values,
)


class Qwen25VLPatchEmbed(nn.Module):
    """Conv3d patch embedding (no bias).

    Reshapes flat input ``(total_patches, C * T_p * P * P)`` into 5-D,
    applies Conv3d with kernel = stride = ``(T_p, P, P)``, and flattens
    back to ``(total_patches, hidden_size)``.
    """

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1280,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        # Conv3d weight: [out_channels, in_channels, kD, kH, kW]
        # HF: nn.Conv3d(..., bias=False)
        self.weight = nn.Parameter(
            [hidden_size, in_channels, temporal_patch_size, patch_size, patch_size],
            name="proj.weight",
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        x = op.Reshape(
            hidden_states,
            [-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size],
        )
        x = op.Conv(
            x,
            self.weight,
            kernel_shape=[self.temporal_patch_size, self.patch_size, self.patch_size],
            strides=[self.temporal_patch_size, self.patch_size, self.patch_size],
        )
        return op.Reshape(x, [-1, self.hidden_size])


class Qwen25VLVisionRotaryEmbedding(nn.Module):
    """2D rotary position embeddings for the vision encoder.

    Precomputes a lookup table of frequencies and returns (cos, sin) given
    2D position indices ``(total_patches, 2)`` with ``[h_pos, w_pos]``.

    The HF implementation computes ``inv_freq`` then does ``outer(seq, inv_freq)``
    and gathers by position.  We precompute the full table as initializers.
    """

    def __init__(self, dim: int, theta: float = 10000.0, max_grid_size: int = 512):
        super().__init__()
        self._dim = dim
        half_dim = dim // 2
        inv_freq = 1.0 / (theta ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
        # Precompute freq table: (max_grid_size, half_dim)
        seq = np.arange(max_grid_size, dtype=np.float32)
        freqs = np.outer(seq, inv_freq)  # (max_grid_size, half_dim)
        self.freq_table = nn.Parameter(
            [max_grid_size, half_dim],
            data=ir.tensor(freqs),
        )

    def forward(self, op: builder.OpBuilder, rotary_pos_ids: ir.Value):
        """Compute cos/sin position embeddings.

        Args:
            rotary_pos_ids: ``(total_patches, 2)`` INT64 with [h_pos, w_pos].

        Returns:
            Tuple of (cos, sin) each ``(total_patches, dim)``.
        """
        # Gather freq for h and w positions separately
        h_pos = op.Gather(rotary_pos_ids, [0], axis=1)  # (N, 1)
        w_pos = op.Gather(rotary_pos_ids, [1], axis=1)  # (N, 1)
        h_pos = op.Squeeze(h_pos, [1])  # (N,)
        w_pos = op.Squeeze(w_pos, [1])  # (N,)

        h_freqs = op.Gather(self.freq_table, h_pos, axis=0)  # (N, half_dim)
        w_freqs = op.Gather(self.freq_table, w_pos, axis=0)  # (N, half_dim)

        # Concat h and w frequencies, then duplicate for cos/sin
        freqs = op.Concat(h_freqs, w_freqs, axis=-1)  # (N, dim)
        emb = op.Concat(freqs, freqs, axis=-1)  # (N, 2*dim)
        cos = op.Cos(emb)
        sin = op.Sin(emb)
        return cos, sin


class Qwen25VLVisionAttention(nn.Module):
    """Packed bidirectional MHA for the vision encoder.

    Uses cu_seqlens to define which tokens can attend to each other
    (block-diagonal attention within images/windows).
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
    ):
        """Packed attention with block-diagonal mask from cu_seqlens.

        Args:
            hidden_states: (total_patches, hidden_size)
            cu_seqlens: (num_sub_seqs + 1,) cumulative sequence lengths
            cos, sin: (total_patches, 2 * head_dim) position embeddings
        """
        seq_len_val = op.Shape(hidden_states, start=0, end=1)

        # QKV projection: (N, 3 * hidden) → split into Q, K, V
        qkv = self.qkv(op, hidden_states)
        qkv = op.Reshape(
            qkv,
            op.Concat(seq_len_val, [3, self.num_heads, self.head_dim], axis=0),
        )
        qkv = op.Transpose(qkv, perm=[1, 0, 2, 3])  # (3, N, num_heads, head_dim)
        q = op.Gather(qkv, [0], axis=0)
        k = op.Gather(qkv, [1], axis=0)
        v = op.Gather(qkv, [2], axis=0)
        q = op.Squeeze(q, [0])  # (N, num_heads, head_dim)
        k = op.Squeeze(k, [0])
        v = op.Squeeze(v, [0])

        # Apply rotary embeddings
        q = self._apply_rotary(op, q, cos, sin)
        k = self._apply_rotary(op, k, cos, sin)

        # Reshape for attention: add batch dim
        # (N, num_heads, head_dim) → (1, num_heads, N, head_dim)
        q = op.Transpose(q, perm=[1, 0, 2])  # (num_heads, N, head_dim)
        k = op.Transpose(k, perm=[1, 0, 2])
        v = op.Transpose(v, perm=[1, 0, 2])
        q = op.Unsqueeze(q, [0])  # (1, num_heads, N, head_dim)
        k = op.Unsqueeze(k, [0])
        v = op.Unsqueeze(v, [0])

        # Build block-diagonal attention bias from cu_seqlens
        attn_bias = self._build_block_diagonal_bias(op, cu_seqlens, seq_len_val)
        attn_bias = op.Unsqueeze(attn_bias, [0, 1])  # (1, 1, N, N)

        # Scaled dot-product attention
        attn_out = op.Attention(
            q,
            k,
            v,
            attn_bias,
            scale=float(1.0 / math.sqrt(self.head_dim)),
            _outputs=["attn_out"],
        )

        # Reshape back: (1, num_heads, N, head_dim) → (N, hidden)
        attn_out = op.Squeeze(attn_out, [0])
        attn_out = op.Transpose(attn_out, perm=[1, 0, 2])  # (N, num_heads, head_dim)
        attn_out = op.Reshape(
            attn_out,
            op.Concat(seq_len_val, [-1], axis=0),
        )
        return self.proj(op, attn_out)

    def _apply_rotary(self, op, x, cos, sin):
        """Apply rotary embeddings to (N, num_heads, head_dim)."""
        # Split into two halves
        half = self.head_dim // 2
        x1 = op.Slice(x, [0], [half], [2])
        x2 = op.Slice(x, [half], [self.head_dim], [2])

        # cos/sin: (N, 2*head_dim) → need (N, head_dim) for each half
        cos_half = op.Slice(cos, [0], [self.head_dim], [1])  # (N, head_dim)
        sin_half = op.Slice(sin, [0], [self.head_dim], [1])  # (N, head_dim)

        # Expand cos/sin for broadcasting: (N, 1, head_dim)
        cos_half = op.Unsqueeze(cos_half, [1])
        sin_half = op.Unsqueeze(sin_half, [1])

        # Split cos/sin into two halves
        cos1 = op.Slice(cos_half, [0], [half], [2])
        cos2 = op.Slice(cos_half, [half], [self.head_dim], [2])
        sin1 = op.Slice(sin_half, [0], [half], [2])
        sin2 = op.Slice(sin_half, [half], [self.head_dim], [2])

        # RoPE: (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
        rot_x1 = op.Sub(op.Mul(x1, cos1), op.Mul(x2, sin1))
        rot_x2 = op.Add(op.Mul(x1, sin2), op.Mul(x2, cos2))

        return op.Concat(rot_x1, rot_x2, axis=-1)

    def _build_block_diagonal_bias(self, op, cu_seqlens, seq_len):
        """Build block-diagonal attention bias from cu_seqlens.

        Creates a matrix where positions in the same sub-sequence have 0
        and positions in different sub-sequences have -inf.
        """
        # Create range indices
        indices = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(seq_len, [0]),
            op.Constant(value_int=1),
        )

        # Assign each position to its sub-sequence using cu_seqlens
        # For each position i, find which sub-sequence it belongs to
        # by checking cu_seqlens[j] <= i < cu_seqlens[j+1]
        # Use searchsorted-like approach: CumSum of (i >= cu_seqlens)
        indices_2d = op.Unsqueeze(indices, [1])  # (N, 1)
        cu_2d = op.Unsqueeze(cu_seqlens, [0])  # (1, S+1)
        cu_2d = op.Cast(cu_2d, to=7)  # INT64

        # ge_mask[i, j] = (i >= cu_seqlens[j])
        ge_mask = op.GreaterOrEqual(indices_2d, cu_2d)
        ge_mask_int = op.Cast(ge_mask, to=7)
        # seg_ids[i] = sum of (i >= cu_seqlens[j]) - 1 = segment index for position i
        seg_ids = op.Sub(
            op.ReduceSum(ge_mask_int, [1], keepdims=False),
            op.Constant(value_int=1),
        )

        # Build mask: same_segment[i, j] = (seg_ids[i] == seg_ids[j])
        seg_row = op.Unsqueeze(seg_ids, [1])  # (N, 1)
        seg_col = op.Unsqueeze(seg_ids, [0])  # (1, N)
        same_segment = op.Equal(seg_row, seg_col)

        # Convert to attention bias: 0 where same segment, -inf where different
        neg_inf = op.Constant(value_float=-1e9)
        zero = op.Constant(value_float=0.0)
        return op.Where(same_segment, zero, neg_inf)


class Qwen25VLVisionMLP(nn.Module):
    """Gate-up-down MLP with SiLU activation (bias=True).

    Matches HF Qwen2_5_VLMLP: gate_proj * act(up_proj) → down_proj.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        gate = self.gate_proj(op, hidden_states)
        gate = op.Mul(gate, op.Sigmoid(gate))  # SiLU
        up = self.up_proj(op, hidden_states)
        return self.down_proj(op, op.Mul(gate, up))


class Qwen25VLVisionBlock(nn.Module):
    """Pre-norm vision transformer block.

    norm1 → attn → residual → norm2 → mlp → residual
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Qwen25VLVisionAttention(hidden_size, num_heads)
        self.mlp = Qwen25VLVisionMLP(hidden_size, intermediate_size)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
    ):
        residual = hidden_states
        hidden_states = self.attn(
            op,
            self.norm1(op, hidden_states),
            cu_seqlens=cu_seqlens,
            cos=cos,
            sin=sin,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(op, self.norm2(op, hidden_states))
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class Qwen25VLPatchMerger(nn.Module):
    """Spatial merge via RMSNorm → reshape → MLP.

    Reduces token count by ``spatial_merge_size²`` via reshaping adjacent
    patches and projecting to output dimension.

    HF: ``ln_q`` (RMSNorm) → view(-1, hidden*merge²) → ``mlp`` (Linear → GELU → Linear)
    """

    def __init__(
        self,
        out_hidden_size: int,
        hidden_size: int,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self._merge_size = spatial_merge_size
        merged_dim = hidden_size * spatial_merge_size * spatial_merge_size
        self.ln_q = RMSNorm(hidden_size, eps=1e-6)
        # mlp is Sequential(Linear, GELU, Linear) in HF
        # We match the naming: mlp.0, mlp.2 (index 1 is GELU, no params)
        self.mlp_0 = Linear(merged_dim, merged_dim, bias=True)
        self.mlp_2 = Linear(merged_dim, out_hidden_size, bias=True)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # hidden_states: (N, hidden_size) where N = total_patches after blocks
        # Apply RMSNorm per-token
        hidden_states = self.ln_q(op, hidden_states)
        # Reshape: group merge²=4 consecutive tokens together
        # (N, hidden) → (N/merge², merge² * hidden)
        merge_sq = self._merge_size * self._merge_size
        seq_len = op.Shape(hidden_states, start=0, end=1)
        new_seq = op.Div(seq_len, op.Constant(value_ints=[merge_sq]))
        hidden_dim = op.Shape(hidden_states, start=1, end=2)
        new_hidden = op.Mul(hidden_dim, op.Constant(value_ints=[merge_sq]))
        new_shape = op.Concat(new_seq, new_hidden, axis=0)
        hidden_states = op.Reshape(hidden_states, new_shape)

        # MLP: Linear → GELU → Linear
        hidden_states = self.mlp_0(op, hidden_states)
        hidden_states = op.Gelu(hidden_states)
        hidden_states = self.mlp_2(op, hidden_states)
        return hidden_states

    def preprocess_weights(self, state_dict):
        """Rename Sequential indices to match our flat naming."""
        renamed = {}
        for key, value in state_dict.items():
            new_key = key
            if ".mlp.0." in key:
                new_key = key.replace(".mlp.0.", ".mlp_0.")
            elif ".mlp.2." in key:
                new_key = key.replace(".mlp.2.", ".mlp_2.")
            renamed[new_key] = value
        return renamed


def _rotary_pos_ids_one_image(op, T, H, W, ms):  # noqa: N803
    """Compute 2D rotary position IDs for a single image.

    Creates ``[h_pos, w_pos]`` per patch with spatial-merge permutation:
    ``(H, W) → (H//ms, ms, W//ms, ms) → transpose(0,2,1,3) → flatten``.

    Works with any OpBuilder (main graph or Scan body graph).

    Args:
        op: OpBuilder instance.
        T: Scalar INT64 — temporal frames.
        H: Scalar INT64 — height in patches.
        W: Scalar INT64 — width in patches.
        ms: Python int — spatial merge size.

    Returns:
        ``(T*H*W, 2)`` INT64 position IDs.
    """
    H_m = op.Div(H, op.Constant(value_int=ms))  # noqa: N806
    W_m = op.Div(W, op.Constant(value_int=ms))  # noqa: N806

    # h positions: arange(H) expanded to (H, W) via outer add with zeros
    h_range = op.Range(op.Constant(value_int=0), H, op.Constant(value_int=1))
    w_range = op.Range(op.Constant(value_int=0), W, op.Constant(value_int=1))
    h_2d = op.Unsqueeze(h_range, [1])  # (H, 1)
    w_2d = op.Unsqueeze(w_range, [0])  # (1, W)
    h_grid = op.Add(h_2d, op.Mul(w_2d, op.Constant(value_int=0)))  # (H, W)
    w_grid = op.Add(op.Mul(h_2d, op.Constant(value_int=0)), w_2d)  # (H, W)

    # Spatial-merge permutation: (H, W) → (H_m, ms, W_m, ms)
    # → transpose(0, 2, 1, 3) → flatten to (H*W,)
    shape_4d = op.Concat(
        op.Reshape(H_m, [1]),
        op.Constant(value_ints=[ms]),
        op.Reshape(W_m, [1]),
        op.Constant(value_ints=[ms]),
        axis=0,
    )
    h_grid = op.Reshape(
        op.Transpose(op.Reshape(h_grid, shape_4d), perm=[0, 2, 1, 3]),
        [-1],
    )
    w_grid = op.Reshape(
        op.Transpose(op.Reshape(w_grid, shape_4d), perm=[0, 2, 1, 3]),
        [-1],
    )

    # Stack to (H*W, 2) and tile T times
    pos_ids = op.Concat(
        op.Unsqueeze(h_grid, [1]),
        op.Unsqueeze(w_grid, [1]),
        axis=1,
    )
    tile_t = op.Concat(op.Reshape(T, [1]), op.Constant(value_ints=[1]), axis=0)
    return op.Tile(pos_ids, tile_t)  # (T*H*W, 2)


def _window_index_one_image(op, T, H, W, ms, ws, smu):  # noqa: N803
    """Compute window reordering index and cu_window_seqlens for one image.

    Works with any OpBuilder (main graph or Scan body graph).

    Args:
        op: OpBuilder.
        T, H, W: Scalar INT64 values.
        ms: spatial_merge_size (Python int).
        ws: vit_merger_window_size (Python int).
        smu: spatial_merge_unit = ms² (Python int).

    Returns:
        ``(window_index, cu_window, total_merged, num_windows)`` where:
        - window_index: ``(total_merged,)`` INT64 permutation.
        - cu_window: ``(num_windows,)`` INT64 cumulative per-window
          patch counts (WITHOUT leading 0).
        - total_merged: scalar INT64.
        - num_windows: scalar INT64.
    """
    llm_h = op.Div(H, op.Constant(value_int=ms))
    llm_w = op.Div(W, op.Constant(value_int=ms))
    total_merged = op.Mul(T, op.Mul(llm_h, llm_w))

    # Index array: Range(0, total_merged) → (T, llm_h, llm_w)
    indices = op.Range(
        op.Constant(value_int=0),
        total_merged,
        op.Constant(value_int=1),
    )
    shape_3d = op.Concat(
        op.Reshape(T, [1]),
        op.Reshape(llm_h, [1]),
        op.Reshape(llm_w, [1]),
        axis=0,
    )
    indices = op.Reshape(indices, shape_3d)

    # Pad to make llm_h/llm_w divisible by ws
    ws_c = op.Constant(value_int=ws)
    pad_h = op.Sub(ws_c, op.Mod(llm_h, ws_c))
    pad_w = op.Sub(ws_c, op.Mod(llm_w, ws_c))
    pads = op.Concat(
        op.Constant(value_ints=[0, 0, 0, 0]),
        op.Reshape(pad_h, [1]),
        op.Reshape(pad_w, [1]),
        axis=0,
    )
    padded = op.Pad(indices, pads, op.Constant(value_int=-100))

    # Reshape into windows: (T, num_wh, ws, num_ww, ws)
    num_wh = op.Div(op.Add(llm_h, pad_h), ws_c)
    num_ww = op.Div(op.Add(llm_w, pad_w), ws_c)
    shape_5d = op.Concat(
        op.Reshape(T, [1]),
        op.Reshape(num_wh, [1]),
        op.Constant(value_ints=[ws]),
        op.Reshape(num_ww, [1]),
        op.Constant(value_ints=[ws]),
        axis=0,
    )
    padded = op.Transpose(op.Reshape(padded, shape_5d), perm=[0, 1, 3, 2, 4])

    # Flatten to (num_windows, ws²)
    num_windows = op.Mul(T, op.Mul(num_wh, num_ww))
    shape_2d = op.Concat(
        op.Reshape(num_windows, [1]),
        op.Constant(value_ints=[ws * ws]),
        axis=0,
    )
    padded = op.Reshape(padded, shape_2d)

    # Valid mask and per-window token counts
    valid = op.Not(op.Equal(padded, op.Constant(value_int=-100)))
    seqlens = op.ReduceSum(op.Cast(valid, to=7), [1], keepdims=False)

    # Extract valid indices
    window_index = op.Compress(op.Reshape(padded, [-1]), op.Reshape(valid, [-1]))

    # cu_window: cumsum of per-window actual patch counts (no leading 0)
    seqlens_actual = op.Mul(seqlens, op.Constant(value_int=smu))
    cu_window = op.CumSum(seqlens_actual, op.Constant(value_int=0))

    return window_index, cu_window, total_merged, num_windows


class Qwen25VLVisionModel(nn.Module):
    """Qwen2.5-VL vision encoder.

    Processes packed image/video patches through Conv3d embedding,
    rotary-embedded transformer blocks with windowed/full attention,
    and spatial merge.

    Accepts ``pixel_values`` and ``image_grid_thw`` (matching ORT GenAI's
    ``QwenImageProcessor`` output) and computes all derived values
    internally: cu_seqlens, cu_window_seqlens, window_index, rotary
    position IDs, and the window reordering/reverse-reordering of
    hidden states.

    Args:
        depth: Number of transformer blocks.
        hidden_size: Hidden dimension of the vision encoder.
        intermediate_size: MLP intermediate dimension.
        num_heads: Number of attention heads.
        patch_size: Spatial patch size (default: 14).
        temporal_patch_size: Temporal patch size (default: 2).
        in_channels: Input image channels (default: 3).
        out_hidden_size: Output dimension after merger (default: hidden_size).
        spatial_merge_size: Merge factor for patch merger (default: 2).
        fullatt_block_indexes: Block indices that use full-image attention.
        window_size: Window size in pixels for windowed attention (default: 112).
    """

    def __init__(
        self,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        out_hidden_size: int | None = None,
        spatial_merge_size: int = 2,
        fullatt_block_indexes: list[int] | None = None,
        window_size: int = 112,
    ):
        super().__init__()
        self._fullatt_block_indexes = set(fullatt_block_indexes or [])
        self._spatial_merge_size = spatial_merge_size
        self._patch_size = patch_size
        self._hidden_size = hidden_size
        smu = spatial_merge_size * spatial_merge_size
        self._spatial_merge_unit = smu
        self._vit_merger_window_size = window_size // spatial_merge_size // patch_size

        self.patch_embed = Qwen25VLPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
        )

        head_dim = hidden_size // num_heads
        # HF passes head_dim // 2 as the rotary dim because position
        # embeddings are 2-D (height, width).  Each half covers one
        # spatial dimension, so the per-dimension frequency count is
        # half_dim = (head_dim // 2) // 2 = head_dim // 4.
        self.rotary_pos_emb = Qwen25VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen25VLVisionBlock(hidden_size, intermediate_size, num_heads)
                for _ in range(depth)
            ]
        )

        self.merger = Qwen25VLPatchMerger(
            out_hidden_size=out_hidden_size or hidden_size,
            hidden_size=hidden_size,
            spatial_merge_size=spatial_merge_size,
        )

    def _compute_rotary_pos_ids(self, op, image_grid_thw):
        """Compute 2D rotary position IDs for all images via ONNX Scan.

        Iterates over ``image_grid_thw`` rows, computing per-image spatial-
        merge-permuted position IDs and concatenating.  Matches the HF
        ``Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb()`` loop.

        Args:
            op: Main graph OpBuilder.
            image_grid_thw: ``(num_images, 3)`` INT64.

        Returns:
            ``(total_patches, 2)`` INT64 with ``[h_pos, w_pos]`` per patch.
        """
        ms = self._spatial_merge_size

        # Per-image patch counts for padding/compaction
        T_col = op.Squeeze(op.Slice(image_grid_thw, [0], [1], [1], [1]), [1])  # noqa: N806
        H_col = op.Squeeze(op.Slice(image_grid_thw, [1], [2], [1], [1]), [1])  # noqa: N806
        W_col = op.Squeeze(op.Slice(image_grid_thw, [2], [3], [1], [1]), [1])  # noqa: N806
        patches_per_image = op.Mul(T_col, op.Mul(H_col, W_col))
        max_patches = op.ReduceMax(patches_per_image, keepdims=False)

        # --- Scan body: compute pos_ids for one image, pad to max_patches ---
        body_thw = ir.Value(
            name="body_thw",
            shape=ir.Shape([3]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        body_graph, body_builder = create_body_graph([], [body_thw])
        body_op = body_builder.op

        bT = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=0)))  # noqa: N806
        bH = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=1)))  # noqa: N806
        bW = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=2)))  # noqa: N806

        # Compute pos_ids using the same spatial-merge logic
        pos_ids = _rotary_pos_ids_one_image(body_op, bT, bH, bW, ms)

        # Pad to (max_patches, 2) — implicit input from main graph
        num_p = body_op.Mul(bT, body_op.Mul(bH, bW))
        pad_len = body_op.Reshape(body_op.Sub(max_patches, num_p), [1])
        pads = body_op.Concat(
            body_op.Constant(value_ints=[0, 0]),
            pad_len,
            body_op.Constant(value_ints=[0]),
            axis=0,
        )  # [0, 0, pad_len, 0] for 2D tensor
        padded = body_op.Pad(pos_ids, pads, body_op.Constant(value_int=-1))
        padded.name = "padded_pos_ids"
        body_graph.outputs.append(padded)

        rename_subgraph_values(body_graph, "rotary_body_")

        # Scan over all images (no carry state, 1 scan input)
        scan_result = op.Scan(
            image_grid_thw,
            body=body_graph,
            num_scan_inputs=1,
            _outputs=1,
        )  # (num_images, max_patches, 2)

        return compact_scan_output(op, scan_result, patches_per_image)

    def _compute_window_index(self, op, image_grid_thw):
        """Compute window reordering index and cu_window_seqlens for all images.

        Uses ONNX Scan to iterate over ``image_grid_thw``, computing
        per-image window indices with accumulating offsets.  Matches the
        HF ``get_window_index()`` loop.

        Args:
            op: Main graph OpBuilder.
            image_grid_thw: ``(num_images, 3)`` INT64.

        Returns:
            ``(window_index, cu_window_seqlens)`` where:

            - window_index: ``(total_merged,)`` INT64 global permutation.
            - cu_window_seqlens: ``(num_segments + 1,)`` INT64.
        """
        ms = self._spatial_merge_size
        smu = self._spatial_merge_unit
        ws = self._vit_merger_window_size

        # Compute per-image merged token counts and window counts
        T_col = op.Squeeze(op.Slice(image_grid_thw, [0], [1], [1], [1]), [1])  # noqa: N806
        H_col = op.Squeeze(op.Slice(image_grid_thw, [1], [2], [1], [1]), [1])  # noqa: N806
        W_col = op.Squeeze(op.Slice(image_grid_thw, [2], [3], [1], [1]), [1])  # noqa: N806
        llm_h_col = op.Div(H_col, op.Constant(value_int=ms))
        llm_w_col = op.Div(W_col, op.Constant(value_int=ms))
        merged_per_image = op.Mul(T_col, op.Mul(llm_h_col, llm_w_col))
        max_merged = op.ReduceMax(merged_per_image, keepdims=False)

        # Max windows per image: T * (llm_h/ws + 1) * (llm_w/ws + 1)
        nwh = op.Add(op.Div(llm_h_col, op.Constant(value_int=ws)), op.Constant(value_int=1))
        nww = op.Add(op.Div(llm_w_col, op.Constant(value_int=ws)), op.Constant(value_int=1))
        windows_per_image = op.Mul(T_col, op.Mul(nwh, nww))
        max_windows = op.ReduceMax(windows_per_image, keepdims=False)

        # --- Scan body ---
        # Carry states: merged_offset, cu_offset (both INT64 scalars)
        body_m_off = ir.Value(
            name="merged_offset",
            shape=ir.Shape([]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        body_cu_off = ir.Value(
            name="cu_offset",
            shape=ir.Shape([]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        body_thw = ir.Value(
            name="body_thw",
            shape=ir.Shape([3]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        body_graph, body_builder = create_body_graph(
            [body_m_off, body_cu_off],
            [body_thw],
            name="window_body",
        )
        body_op = body_builder.op

        bT = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=0)))  # noqa: N806
        bH = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=1)))  # noqa: N806
        bW = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=2)))  # noqa: N806

        # Per-image window index and cu_window (no offset yet)
        win_idx, cu_win, total_m, num_w = _window_index_one_image(
            body_op,
            bT,
            bH,
            bW,
            ms,
            ws,
            smu,
        )

        # Add offsets
        win_idx = body_op.Add(win_idx, body_m_off)
        cu_win = body_op.Add(cu_win, body_cu_off)

        # Updated carry states
        new_m_off = body_op.Add(body_m_off, total_m)
        last_cu = body_op.Gather(
            cu_win,
            body_op.Sub(num_w, body_op.Constant(value_int=1)),
        )
        new_cu_off = body_op.Squeeze(last_cu)

        # Pad outputs to max sizes
        # window_index → (max_merged,)
        idx_pad = body_op.Reshape(
            body_op.Sub(max_merged, total_m),
            [1],
        )
        idx_pads = body_op.Concat(
            body_op.Constant(value_ints=[0]),
            idx_pad,
            axis=0,
        )
        padded_idx = body_op.Pad(
            win_idx,
            idx_pads,
            body_op.Constant(value_int=-1),
        )

        # cu_window → (max_windows,)
        cu_pad = body_op.Reshape(
            body_op.Sub(max_windows, num_w),
            [1],
        )
        cu_pads = body_op.Concat(
            body_op.Constant(value_ints=[0]),
            cu_pad,
            axis=0,
        )
        padded_cu = body_op.Pad(
            cu_win,
            cu_pads,
            body_op.Constant(value_int=-1),
        )

        # Body outputs: 2 carry states + 2 scan outputs
        new_m_off.name = "new_merged_offset"
        new_cu_off.name = "new_cu_offset"
        padded_idx.name = "padded_window_index"
        padded_cu.name = "padded_cu_window"
        body_graph.outputs.extend(
            [
                new_m_off,
                new_cu_off,
                padded_idx,
                padded_cu,
            ]
        )

        rename_subgraph_values(body_graph, "win_body_")

        # Run Scan
        init_m_off = op.Constant(value_int=0)
        init_cu_off = op.Constant(value_int=0)
        _final_m, _final_cu, scan_idx, scan_cu = op.Scan(
            init_m_off,
            init_cu_off,
            image_grid_thw,
            body=body_graph,
            num_scan_inputs=1,
            _outputs=4,
        )
        # scan_idx: (N, max_merged), scan_cu: (N, max_windows)

        # Compact both outputs
        window_index = compact_scan_output(
            op,
            scan_idx,
            merged_per_image,
        )
        cu_window_entries = compact_scan_output(
            op,
            scan_cu,
            windows_per_image,
        )

        # Prepend 0 to cu_window_seqlens
        cu_window_seqlens = op.Pad(
            cu_window_entries,
            op.Constant(value_ints=[1, 0]),
            op.Constant(value_int=0),
        )

        return window_index, cu_window_seqlens

    def _compute_cu_seqlens(self, op, image_grid_thw):
        """Compute full-attention cu_seqlens for all images.

        Produces per-frame boundaries: ``[0, hw₀, 2*hw₀, ..., T₀*hw₀,
        T₀*hw₀ + hw₁, ...]``.  Uses ONNX Scan to handle per-image
        ``repeat_interleave(hw, T)`` followed by CumSum.

        Args:
            op: Main graph OpBuilder.
            image_grid_thw: ``(num_images, 3)`` INT64.

        Returns:
            ``(total_frames + 1,)`` INT64 cumulative seq lengths.
        """
        T_col = op.Squeeze(op.Slice(image_grid_thw, [0], [1], [1], [1]), [1])  # noqa: N806
        max_T = op.ReduceMax(T_col, keepdims=False)  # noqa: N806

        # Scan body: for each image, output T copies of hw, padded to max_T
        body_thw = ir.Value(
            name="body_thw",
            shape=ir.Shape([3]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        body_graph, body_builder = create_body_graph([], [body_thw])
        body_op = body_builder.op

        bT = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=0)))  # noqa: N806
        bH = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=1)))  # noqa: N806
        bW = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=2)))  # noqa: N806

        # Expand hw to T copies: [hw, hw, ..., hw] (T times)
        hw = body_op.Mul(bH, bW)
        ones = body_op.Expand(
            body_op.Constant(value_int=1),
            body_op.Reshape(bT, [1]),
        )
        hw_repeated = body_op.Mul(ones, hw)  # (T,)

        # Pad to max_T
        pad_len = body_op.Reshape(body_op.Sub(max_T, bT), [1])
        pads = body_op.Concat(
            body_op.Constant(value_ints=[0]),
            pad_len,
            axis=0,
        )
        padded = body_op.Pad(
            hw_repeated,
            pads,
            body_op.Constant(value_int=0),
        )
        padded.name = "padded_hw"
        body_graph.outputs.append(padded)

        rename_subgraph_values(body_graph, "cu_body_")

        # Scan → (N, max_T)
        scan_hw = op.Scan(
            image_grid_thw,
            body=body_graph,
            num_scan_inputs=1,
            _outputs=1,
        )

        # Compact to remove padding, then CumSum + prepend 0
        hw_flat = compact_scan_output(op, scan_hw, T_col)  # (total_frames,)
        cu = op.CumSum(hw_flat, op.Constant(value_int=0))
        return op.Pad(cu, op.Constant(value_ints=[1, 0]), op.Constant(value_int=0))

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ):
        """Forward pass of the vision encoder.

        Supports multiple images: ``pixel_values`` contains all images'
        patches concatenated, ``image_grid_thw`` has one ``[T, H, W]``
        row per image.  All per-image derived values (rotary position IDs,
        window indices, cu_seqlens) are computed via ONNX Scan.

        Args:
            pixel_values: Flattened patches ``(total_patches, C * T_p * P * P)``.
            image_grid_thw: ``(num_images, 3)`` INT64 with ``[T, H, W]``
                per image.

        Returns:
            image_features: ``(num_merged_patches, out_hidden_size)``.
        """
        smu = self._spatial_merge_unit

        # 1. Patch embedding (operates on all patches at once)
        hidden_states = self.patch_embed(op, pixel_values)

        # 2. Compute per-image derived values via Scan
        rotary_pos_ids = self._compute_rotary_pos_ids(op, image_grid_thw)
        window_index, cu_window_seqlens = self._compute_window_index(
            op,
            image_grid_thw,
        )
        cu_seqlens = self._compute_cu_seqlens(op, image_grid_thw)

        # 4. Reorder hidden_states by window_index at merge-unit level
        seq_len = op.Shape(hidden_states, start=0, end=1)
        num_merged = op.Div(seq_len, op.Constant(value_ints=[smu]))
        hidden_dim = op.Shape(hidden_states, start=1, end=2)
        shape_3d = op.Concat(num_merged, op.Constant(value_ints=[smu]), hidden_dim, axis=0)
        hidden_states = op.Reshape(hidden_states, shape_3d)
        hidden_states = op.Gather(hidden_states, window_index, axis=0)
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(op.Constant(value_ints=[-1]), hidden_dim, axis=0),
        )

        # 5. Reorder rotary_pos_ids by window_index at merge-unit level
        pos_shape_3d = op.Concat(num_merged, op.Constant(value_ints=[smu, 2]), axis=0)
        rotary_pos_ids = op.Reshape(rotary_pos_ids, pos_shape_3d)
        rotary_pos_ids = op.Gather(rotary_pos_ids, window_index, axis=0)
        rotary_pos_ids = op.Reshape(rotary_pos_ids, [-1, 2])

        # 6. Compute cos/sin from reordered position IDs
        cos, sin = self.rotary_pos_emb(op, rotary_pos_ids)

        # 7. Run transformer blocks
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx in self._fullatt_block_indexes:
                block_cu_seqlens = cu_seqlens
            else:
                block_cu_seqlens = cu_window_seqlens

            hidden_states = block(
                op,
                hidden_states,
                cu_seqlens=block_cu_seqlens,
                cos=cos,
                sin=sin,
            )

        # 8. Spatial merge
        merged = self.merger(op, hidden_states)

        # 9. Reverse reorder: argsort(window_index) via TopK
        k = op.Shape(window_index, start=0, end=1)
        _sorted_vals, reverse_index = op.TopK(
            op.Cast(window_index, to=1),  # TopK needs float input
            k,
            largest=0,
            sorted=1,
            _outputs=["_sorted", "reverse_idx"],
        )
        merged = op.Gather(merged, reverse_index, axis=0)

        return merged

    def preprocess_weights(self, state_dict):
        """Map HF weight names to ONNX parameter names."""
        renamed = {}
        for key, value in state_dict.items():
            new_key = key
            # Rename merger.mlp Sequential indices
            if "merger.mlp.0." in key:
                new_key = key.replace("merger.mlp.0.", "merger.mlp_0.")
            elif "merger.mlp.2." in key:
                new_key = key.replace("merger.mlp.2.", "merger.mlp_2.")
            renamed[new_key] = value
        return renamed
