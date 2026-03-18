# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL vision encoder components.

Provides modules for the Qwen3-VL vision backbone:

- ``Qwen3VLPatchEmbed``: Conv3d patch tokenisation (temporal + spatial).
- ``Qwen3VLVisionRotaryEmbedding``: 2D rotary embeddings from grid positions.
- ``Qwen3VLVisionAttention``: Packed bidirectional MHA with ``cu_seqlens``.
- ``Qwen3VLVisionMLP``: Single-gate MLP (Linear → activation → Linear).
- ``Qwen3VLVisionBlock``: Pre-norm transformer block.
- ``Qwen3VLPatchMerger``: Spatial merge to reduce token count.
- ``Qwen3VLVisionModel``: Full encoder stack with DeepStack outputs.

Packed attention loops over sub-sequences indicated by ``cu_seqlens``.
This uses standard ONNX ops; the ``rewrite_rules`` submodule provides
optional rules to replace the loop with a custom packed-attention op.
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import LayerNorm, Linear
from mobius.components._scan_utils import (
    compact_scan_output,
    create_body_graph,
    rename_subgraph_values,
)


class Qwen3VLPatchEmbed(nn.Module):
    """Conv3d patch embedding for video / image tokens.

    Reshapes flat input ``(total_patches, C * T_p * P * P)`` into 5-D,
    applies Conv3d with kernel = stride = ``(T_p, P, P)``, and flattens
    back to ``(total_patches, hidden_size)``.
    """

    def __init__(
        self,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        hidden_size: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        # Conv3d weight: [out_channels, in_channels, kD, kH, kW]
        self.weight = nn.Parameter(
            [hidden_size, in_channels, temporal_patch_size, patch_size, patch_size],
            name="proj.weight",
        )
        self.bias = nn.Parameter([hidden_size], name="proj.bias")

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # hidden_states: (total_patches, C * T_p * P * P)
        # Reshape to (total_patches, C, T_p, P, P) for Conv3d
        x = op.Reshape(
            hidden_states,
            [-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size],
        )
        x = op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self.temporal_patch_size, self.patch_size, self.patch_size],
            strides=[self.temporal_patch_size, self.patch_size, self.patch_size],
        )
        # x: (total_patches, hidden_size, 1, 1, 1) → flatten to (total_patches, hidden_size)
        return op.Reshape(x, [-1, self.hidden_size])


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """2D rotary position embeddings for the vision encoder.

    Precomputes a frequency table from which cos/sin values are looked up
    using 2D grid position IDs.

    In HuggingFace this is ``Qwen3VLVisionModel.rot_pos_emb()``.
    """

    def __init__(self, head_dim: int, max_grid_size: int = 4096):
        super().__init__()
        dim = head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))

        pos = np.arange(0, max_grid_size, dtype=np.float32)
        angles = np.outer(pos, inv_freq)
        cos_table = np.cos(angles).astype(np.float32)
        sin_table = np.sin(angles).astype(np.float32)

        self.cos_table = nn.Parameter(
            list(cos_table.shape),
            name="cos_table",
            data=ir.tensor(cos_table),
        )
        self.sin_table = nn.Parameter(
            list(sin_table.shape),
            name="sin_table",
            data=ir.tensor(sin_table),
        )

    def forward(self, op: builder.OpBuilder, position_ids: ir.Value):
        """Look up cos/sin for 2D position IDs.

        Args:
            op: ONNX op builder.
            position_ids: ``(total_tokens, 2)`` with [h_pos, w_pos] per token.

        Returns:
            Tuple of ``(cos, sin)`` each ``(total_tokens, head_dim // 2)``.
        """
        # position_ids: (total_tokens, 2) — h_indices and w_indices
        h_pos = op.Gather(position_ids, [0], axis=1)  # (total_tokens, 1)
        w_pos = op.Gather(position_ids, [1], axis=1)

        h_pos = op.Squeeze(h_pos, [1])  # (total_tokens,)
        w_pos = op.Squeeze(w_pos, [1])

        cos_h = op.Gather(self.cos_table, h_pos)  # (total_tokens, dim//2)
        sin_h = op.Gather(self.sin_table, h_pos)
        cos_w = op.Gather(self.cos_table, w_pos)
        sin_w = op.Gather(self.sin_table, w_pos)

        cos = op.Concat(cos_h, cos_w, axis=-1)  # (total_tokens, dim)
        sin = op.Concat(sin_h, sin_w, axis=-1)
        return cos, sin


class Qwen3VLVisionAttention(nn.Module):
    """Packed bidirectional multi-head attention for the vision encoder.

    Iterates over sub-sequences delimited by ``cu_seqlens`` and applies
    standard ONNX Attention (opset 23) to each independently.  This avoids
    cross-image attention while processing all patches in a single flat
    sequence.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        # Fused QKV projection (matches HF weight name ``attn.qkv``)
        self.qkv = Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        position_embeddings: tuple,
    ):
        # hidden_states: (total_seq, hidden_size)  — flat packed sequence
        # cu_seqlens: (num_sub_seqs + 1,) — cumulative sequence lengths
        # position_embeddings: (cos, sin) each (total_seq, rotary_dim)

        cos, sin = position_embeddings

        qkv = self.qkv(op, hidden_states)

        # Split into Q, K, V: each (total_seq, hidden_size)
        q, k, v = op.Split(qkv, axis=-1, num_outputs=3, _outputs=3)

        # Reshape to (total_seq, num_heads, head_dim) for RoPE
        q = op.Reshape(q, [0, self.num_heads, self.head_dim])
        k = op.Reshape(k, [0, self.num_heads, self.head_dim])

        # Apply rotary embedding (vision uses full rotation, no partial)
        cos = op.Unsqueeze(cos, [1])  # (total_seq, 1, rotary_dim)
        sin = op.Unsqueeze(sin, [1])

        half = self.head_dim // 2
        q1, q2 = op.Split(q, [half, half], axis=-1, _outputs=2)
        k1, k2 = op.Split(k, [half, half], axis=-1, _outputs=2)

        q_rot = op.Concat(
            op.Sub(op.Mul(cos, q1), op.Mul(sin, q2)),
            op.Add(op.Mul(sin, q1), op.Mul(cos, q2)),
            axis=-1,
        )
        k_rot = op.Concat(
            op.Sub(op.Mul(cos, k1), op.Mul(sin, k2)),
            op.Add(op.Mul(sin, k1), op.Mul(cos, k2)),
            axis=-1,
        )

        # Flatten back to (total_seq, hidden_size)
        q_rot = op.Reshape(q_rot, [0, -1])
        k_rot = op.Reshape(k_rot, [0, -1])

        # Build block-diagonal attention bias from cu_seqlens
        # Each token only attends to tokens in the same sub-sequence
        total_seq = op.Shape(hidden_states, start=0, end=1)
        total_seq_scalar = op.Squeeze(total_seq)
        positions = op.Range(
            op.Constant(value_int=0),
            total_seq_scalar,
            op.Constant(value_int=1),
        )
        positions_2d = op.Unsqueeze(positions, [1])  # (total_seq, 1)
        cu_seqlens_2d = op.Unsqueeze(cu_seqlens, [0])  # (1, num_sub_seqs+1)
        ge = op.GreaterOrEqual(positions_2d, cu_seqlens_2d)  # (total_seq, num_sub_seqs+1)
        ge_int = op.Cast(ge, to=7)  # INT64
        segment_ids = op.Sub(
            op.ReduceSum(ge_int, [1], keepdims=False),
            op.Constant(value_int=1),
        )  # (total_seq,) — segment ID per token

        seg_row = op.Unsqueeze(segment_ids, [1])  # (total_seq, 1)
        seg_col = op.Unsqueeze(segment_ids, [0])  # (1, total_seq)
        same_segment = op.Equal(seg_row, seg_col)  # (total_seq, total_seq)
        attn_bias = op.Where(
            same_segment,
            op.Constant(value_float=0.0),
            op.Constant(value_float=-10000.0),
        )
        # Reshape for Attention: (1, 1, total_seq, total_seq)
        attn_bias = op.Unsqueeze(attn_bias, [0, 1])

        # Add batch dim: (1, total_seq, hidden_size)
        q_out = op.Unsqueeze(q_rot, [0])
        k_out = op.Unsqueeze(k_rot, [0])
        v_out = op.Unsqueeze(v, [0])

        attn_output = op.Attention(
            q_out,
            k_out,
            v_out,
            attn_bias,
            kv_num_heads=self.num_heads,
            q_num_heads=self.num_heads,
            scale=self.scale,
            _outputs=1,
        )

        # Remove batch dim: (total_seq, hidden_size)
        attn_output = op.Squeeze(attn_output, [0])

        return self.proj(op, attn_output)


class Qwen3VLVisionMLP(nn.Module):
    """Single-gate MLP for the vision encoder."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.linear_fc1 = Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        hidden_states = self.linear_fc1(op, hidden_states)
        hidden_states = op.Gelu(hidden_states, approximate="tanh")
        return self.linear_fc2(op, hidden_states)


class Qwen3VLVisionBlock(nn.Module):
    """Pre-norm vision transformer block with packed attention.

    Structure: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(hidden_size, num_heads)
        self.norm2 = LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Qwen3VLVisionMLP(hidden_size, intermediate_size)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        position_embeddings: tuple,
    ):
        residual = hidden_states
        hidden_states = self.norm1(op, hidden_states)
        hidden_states = self.attn(op, hidden_states, cu_seqlens, position_embeddings)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.norm2(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class Qwen3VLPatchMerger(nn.Module):
    """Spatial merge — reduces spatial resolution by merging adjacent patches.

    Reshapes tokens by the merge factor, normalises, then projects.
    ``use_postshuffle_norm=False`` (final merger): LayerNorm before reshape.
    ``use_postshuffle_norm=True`` (deepstack mergers): LayerNorm after reshape.
    """

    def __init__(
        self,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        self.merged_size = hidden_size * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        norm_dim = self.merged_size if use_postshuffle_norm else hidden_size
        self.norm = LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = Linear(self.merged_size, self.merged_size, bias=True)
        self.linear_fc2 = Linear(self.merged_size, out_hidden_size, bias=True)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        if self.use_postshuffle_norm:
            # Reshape to merged dim, then normalise
            x = op.Reshape(hidden_states, [-1, self.merged_size])
            x = self.norm(op, x)
        else:
            # Normalise first, then reshape to merged dim
            x = self.norm(op, hidden_states)
            x = op.Reshape(x, [-1, self.merged_size])

        x = self.linear_fc1(op, x)
        x = op.Gelu(x, approximate="none")
        return self.linear_fc2(op, x)


def _qwen3_rotary_pos_ids_one_image(op, T, H, W, ms):  # noqa: N803
    """Compute 2D rotary position IDs for one image (Qwen3-VL style).

    Uses block_rows * ms + intra indexing for spatial-merge groups.
    Works with any OpBuilder (main graph or Scan body graph).

    Args:
        op: OpBuilder instance.
        T, H, W: Scalar INT64 values.
        ms: Python int — spatial merge size.

    Returns:
        ``(T*H*W, 2)`` INT64 position IDs.
    """
    H_m = op.Div(H, op.Constant(value_int=ms))  # noqa: N806
    W_m = op.Div(W, op.Constant(value_int=ms))  # noqa: N806

    # Block row/col indices and intra-merge indices
    block_rows = op.Range(
        op.Constant(value_int=0),
        H_m,
        op.Constant(value_int=1),
    )
    block_cols = op.Range(
        op.Constant(value_int=0),
        W_m,
        op.Constant(value_int=1),
    )
    intra = op.Range(
        op.Constant(value_int=0),
        op.Constant(value_int=ms),
        op.Constant(value_int=1),
    )

    # row_idx = block_rows[:,None,None,None] * ms + intra[None,None,:,None]
    br = op.Mul(op.Unsqueeze(block_rows, [1, 2, 3]), op.Constant(value_int=ms))
    ir_row = op.Unsqueeze(intra, [0, 1, 3])
    row_idx = op.Add(br, ir_row)

    bc = op.Mul(op.Unsqueeze(block_cols, [0, 2, 3]), op.Constant(value_int=ms))
    ir_col = op.Unsqueeze(intra, [0, 1, 2])
    col_idx = op.Add(bc, ir_col)

    # Expand to (H_m, W_m, ms, ms) and flatten
    row_shape = op.Concat(
        op.Reshape(H_m, [1]),
        op.Reshape(W_m, [1]),
        op.Constant(value_ints=[ms, ms]),
        axis=0,
    )
    row_flat = op.Reshape(op.Expand(row_idx, row_shape), [-1])
    col_flat = op.Reshape(op.Expand(col_idx, row_shape), [-1])

    # Stack to (H*W, 2) and tile T times
    pos_ids = op.Concat(
        op.Unsqueeze(row_flat, [1]),
        op.Unsqueeze(col_flat, [1]),
        axis=1,
    )
    tile_t = op.Concat(op.Reshape(T, [1]), op.Constant(value_ints=[1]), axis=0)
    return op.Tile(pos_ids, tile_t)  # (T*H*W, 2)


class Qwen3VLVisionModel(nn.Module):
    """Full Qwen3-VL vision encoder with DeepStack outputs.

    Processes packed image/video patches through Conv3d embedding,
    bilinear-interpolated position embeddings, transformer blocks with
    packed attention, and spatial merge.

    Args:
        depth: Number of vision transformer blocks.
        hidden_size: Hidden dimension of the vision encoder.
        intermediate_size: MLP intermediate dimension.
        num_heads: Number of attention heads.
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal patch size.
        in_channels: Number of input channels.
        out_hidden_size: Output projection dimension (after merge).
        spatial_merge_size: Factor for spatial merge.
        num_position_embeddings: Size of the learned 2D position grid.
        deepstack_visual_indexes: Layer indices for DeepStack features.
    """

    def __init__(
        self,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        out_hidden_size: int | None = None,
        spatial_merge_size: int = 2,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: list[int] | None = None,
    ):
        super().__init__()
        if out_hidden_size is None:
            out_hidden_size = hidden_size
        if deepstack_visual_indexes is None:
            deepstack_visual_indexes = []

        self.depth = depth
        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.num_grid_per_side = math.isqrt(num_position_embeddings)

        head_dim = hidden_size // num_heads

        self.patch_embed = Qwen3VLPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
        )

        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(
            head_dim=head_dim,
        )

        # Learned position embedding grid (num_grid_per_side^2, hidden_size)
        self.pos_embed = nn.Parameter(
            [num_position_embeddings, hidden_size],
            name="pos_embed.weight",
        )

        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(hidden_size, intermediate_size, num_heads)
                for _ in range(depth)
            ]
        )

        self.merger = Qwen3VLPatchMerger(
            hidden_size=hidden_size,
            out_hidden_size=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
            use_postshuffle_norm=False,
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLPatchMerger(
                    hidden_size=hidden_size,
                    out_hidden_size=out_hidden_size,
                    spatial_merge_size=spatial_merge_size,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(deepstack_visual_indexes))
            ]
        )

    def _interpolate_pos_embed(self, op, grid_thw):
        """Bilinear interpolation of learned position embeddings for all images.

        Iterates over ``grid_thw`` via ONNX Scan, computing per-image
        bilinear interpolation from the learned position grid and
        concatenating results.

        Matches HuggingFace ``Qwen3VLVisionModel.fast_pos_embed_interpolate``.

        Args:
            op: OpBuilder instance.
            grid_thw: ``(num_images, 3)`` INT64 with ``[T, H, W]`` per image.

        Returns:
            Position embeddings ``(total_patches, hidden_size)``.
        """
        n = self.num_grid_per_side
        ms = self.spatial_merge_size
        hidden_size = self.hidden_size
        n_minus_1 = float(n - 1)

        # Per-image patch counts
        T_col = op.Squeeze(op.Slice(grid_thw, [0], [1], [1], [1]), [1])  # noqa: N806
        H_col = op.Squeeze(op.Slice(grid_thw, [1], [2], [1], [1]), [1])  # noqa: N806
        W_col = op.Squeeze(op.Slice(grid_thw, [2], [3], [1], [1]), [1])  # noqa: N806
        patches_per_image = op.Mul(T_col, op.Mul(H_col, W_col))
        max_patches = op.ReduceMax(patches_per_image, keepdims=False)

        # --- Scan body: interpolate pos embeddings for one image ---
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

        # linspace(0, n-1, H) and linspace(0, n-1, W)
        H_f = body_op.Cast(bH, to=1)  # noqa: N806
        W_f = body_op.Cast(bW, to=1)  # noqa: N806
        h_range = body_op.Cast(
            body_op.Range(
                body_op.Constant(value_int=0),
                bH,
                body_op.Constant(value_int=1),
            ),
            to=1,
        )
        w_range = body_op.Cast(
            body_op.Range(
                body_op.Constant(value_int=0),
                bW,
                body_op.Constant(value_int=1),
            ),
            to=1,
        )

        h_idxs = body_op.Div(
            body_op.Mul(h_range, body_op.Constant(value_float=n_minus_1)),
            body_op.Sub(H_f, body_op.Constant(value_float=1.0)),
        )
        w_idxs = body_op.Div(
            body_op.Mul(w_range, body_op.Constant(value_float=n_minus_1)),
            body_op.Sub(W_f, body_op.Constant(value_float=1.0)),
        )

        # Floor/ceil indices
        h_floor = body_op.Cast(body_op.Floor(h_idxs), to=7)
        w_floor = body_op.Cast(body_op.Floor(w_idxs), to=7)
        clip_max = body_op.Constant(value_int=n - 1)
        h_ceil = body_op.Min(
            body_op.Add(h_floor, body_op.Constant(value_int=1)),
            clip_max,
        )
        w_ceil = body_op.Min(
            body_op.Add(w_floor, body_op.Constant(value_int=1)),
            clip_max,
        )

        # Bilinear weights
        dh = body_op.Sub(h_idxs, body_op.Cast(h_floor, to=1))
        dw = body_op.Sub(w_idxs, body_op.Cast(w_floor, to=1))

        n_const = body_op.Constant(value_int=n)
        base_h_floor = body_op.Mul(h_floor, n_const)
        base_h_ceil = body_op.Mul(h_ceil, n_const)

        bh_f2 = body_op.Unsqueeze(base_h_floor, [1])
        bh_c2 = body_op.Unsqueeze(base_h_ceil, [1])
        wf2 = body_op.Unsqueeze(w_floor, [0])
        wc2 = body_op.Unsqueeze(w_ceil, [0])

        idx_00 = body_op.Reshape(body_op.Add(bh_f2, wf2), [-1])
        idx_01 = body_op.Reshape(body_op.Add(bh_f2, wc2), [-1])
        idx_10 = body_op.Reshape(body_op.Add(bh_c2, wf2), [-1])
        idx_11 = body_op.Reshape(body_op.Add(bh_c2, wc2), [-1])

        one_minus_dh = body_op.Sub(body_op.Constant(value_float=1.0), dh)
        one_minus_dw = body_op.Sub(body_op.Constant(value_float=1.0), dw)
        dh2 = body_op.Unsqueeze(dh, [1])
        omdh2 = body_op.Unsqueeze(one_minus_dh, [1])
        dw2 = body_op.Unsqueeze(dw, [0])
        omdw2 = body_op.Unsqueeze(one_minus_dw, [0])

        w_00 = body_op.Reshape(body_op.Mul(omdh2, omdw2), [-1, 1])
        w_01 = body_op.Reshape(body_op.Mul(omdh2, dw2), [-1, 1])
        w_10 = body_op.Reshape(body_op.Mul(dh2, omdw2), [-1, 1])
        w_11 = body_op.Reshape(body_op.Mul(dh2, dw2), [-1, 1])

        # Gather from learned pos_embed (implicit input from parent graph)
        e_00 = body_op.Mul(body_op.Gather(self.pos_embed, idx_00), w_00)
        e_01 = body_op.Mul(body_op.Gather(self.pos_embed, idx_01), w_01)
        e_10 = body_op.Mul(body_op.Gather(self.pos_embed, idx_10), w_10)
        e_11 = body_op.Mul(body_op.Gather(self.pos_embed, idx_11), w_11)
        pos_embeds = body_op.Add(
            body_op.Add(e_00, e_01),
            body_op.Add(e_10, e_11),
        )

        # Tile T times: (H*W, D) → (T*H*W, D)
        T_tile = body_op.Concat(  # noqa: N806
            body_op.Reshape(bT, [1]),
            body_op.Constant(value_ints=[1]),
            axis=0,
        )
        pos_embeds = body_op.Tile(pos_embeds, T_tile)

        # Spatial merge permutation:
        # (T, H//ms, ms, W//ms, ms, D) → (T, H//ms, W//ms, ms, ms, D)
        H_m = body_op.Div(bH, body_op.Constant(value_int=ms))  # noqa: N806
        W_m = body_op.Div(bW, body_op.Constant(value_int=ms))  # noqa: N806
        shape_6d = body_op.Concat(
            body_op.Reshape(bT, [1]),
            body_op.Reshape(H_m, [1]),
            body_op.Constant(value_ints=[ms]),
            body_op.Reshape(W_m, [1]),
            body_op.Constant(value_ints=[ms]),
            body_op.Constant(value_ints=[hidden_size]),
            axis=0,
        )
        pos_embeds = body_op.Reshape(pos_embeds, shape_6d)
        pos_embeds = body_op.Transpose(pos_embeds, perm=[0, 1, 3, 2, 4, 5])
        pos_embeds = body_op.Reshape(pos_embeds, [-1, hidden_size])

        # Pad to (max_patches, hidden_size) — implicit input from main graph
        num_p = body_op.Mul(bT, body_op.Mul(bH, bW))
        pad_len = body_op.Reshape(body_op.Sub(max_patches, num_p), [1])
        pads = body_op.Concat(
            body_op.Constant(value_ints=[0, 0]),
            pad_len,
            body_op.Constant(value_ints=[0]),
            axis=0,
        )
        padded = body_op.Pad(pos_embeds, pads, body_op.Constant(value_float=0.0))
        padded.name = "padded_pos_embed"
        body_graph.outputs.append(padded)

        rename_subgraph_values(body_graph, "posemb_body_")

        scan_result = op.Scan(
            grid_thw,
            body=body_graph,
            num_scan_inputs=1,
            _outputs=1,
        )  # (num_images, max_patches, hidden_size)

        return compact_scan_output(op, scan_result, patches_per_image)

    def _compute_rotary_pos_ids(self, op, grid_thw):
        """Compute 2D rotary position IDs for all images via ONNX Scan.

        Matches HF ``Qwen3VLVisionModel.rot_pos_emb()`` position indexing.
        Iterates over ``grid_thw`` rows, computing per-image spatial-merge-
        permuted position IDs and concatenating.

        Returns ``(total_patches, 2)`` INT64 with ``[h_pos, w_pos]`` per patch.
        """
        ms = self.spatial_merge_size

        # Per-image patch counts for padding/compaction
        T_col = op.Squeeze(op.Slice(grid_thw, [0], [1], [1], [1]), [1])  # noqa: N806
        H_col = op.Squeeze(op.Slice(grid_thw, [1], [2], [1], [1]), [1])  # noqa: N806
        W_col = op.Squeeze(op.Slice(grid_thw, [2], [3], [1], [1]), [1])  # noqa: N806
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

        pos_ids = _qwen3_rotary_pos_ids_one_image(body_op, bT, bH, bW, ms)

        # Pad to (max_patches, 2) — implicit input from main graph
        num_p = body_op.Mul(bT, body_op.Mul(bH, bW))
        pad_len = body_op.Reshape(body_op.Sub(max_patches, num_p), [1])
        pads = body_op.Concat(
            body_op.Constant(value_ints=[0, 0]),
            pad_len,
            body_op.Constant(value_ints=[0]),
            axis=0,
        )
        padded = body_op.Pad(pos_ids, pads, body_op.Constant(value_int=-1))
        padded.name = "padded_pos_ids"
        body_graph.outputs.append(padded)

        rename_subgraph_values(body_graph, "q3_rotary_body_")

        scan_result = op.Scan(
            grid_thw,
            body=body_graph,
            num_scan_inputs=1,
            _outputs=1,
        )
        return compact_scan_output(op, scan_result, patches_per_image)

    def _compute_cu_seqlens(self, op, grid_thw):
        """Compute full-attention cu_seqlens for all images.

        Produces per-frame boundaries across all images using ONNX Scan
        to handle per-image ``repeat_interleave(hw, T)`` + CumSum.

        Returns ``(total_frames + 1,)`` INT64.
        """
        T_col = op.Squeeze(op.Slice(grid_thw, [0], [1], [1], [1]), [1])  # noqa: N806
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

        hw = body_op.Mul(bH, bW)
        ones = body_op.Expand(
            body_op.Constant(value_int=1),
            body_op.Reshape(bT, [1]),
        )
        hw_repeated = body_op.Mul(ones, hw)

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

        rename_subgraph_values(body_graph, "q3_cu_body_")

        scan_hw = op.Scan(
            grid_thw,
            body=body_graph,
            num_scan_inputs=1,
            _outputs=1,
        )
        hw_flat = compact_scan_output(op, scan_hw, T_col)
        cu = op.CumSum(hw_flat, op.Constant(value_int=0))
        return op.Pad(cu, op.Constant(value_ints=[1, 0]), op.Constant(value_int=0))

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        grid_thw: ir.Value,
    ):
        """Run the vision encoder.

        Args:
            hidden_states: Flat patches ``(total_patches, C * T_p * P * P)``.
            grid_thw: ``(num_images, 3)`` INT64 with ``[T, H, W]`` per image.

        Returns:
            Tuple of ``(merged_hidden_states, *deepstack_features)`` where
            ``merged_hidden_states`` has shape
            ``(total_merged_patches, out_hidden_size)`` and each deepstack
            feature tensor has the same shape.
        """
        # Patch embedding
        hidden_states = self.patch_embed(op, hidden_states)

        # Bilinear-interpolated position embeddings from learned grid
        pos_embeds = self._interpolate_pos_embed(op, grid_thw)
        hidden_states = op.Add(hidden_states, pos_embeds)

        # Compute rotary position IDs and embeddings from grid_thw
        rotary_pos_ids = self._compute_rotary_pos_ids(op, grid_thw)
        position_embeddings = self.rotary_pos_emb(op, rotary_pos_ids)

        # Compute cu_seqlens from grid_thw
        cu_seqlens = self._compute_cu_seqlens(op, grid_thw)

        # Transformer blocks
        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(op, hidden_states, cu_seqlens, position_embeddings)

            if layer_idx in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(layer_idx)
                ds_feature = self.deepstack_merger_list[ds_idx](op, hidden_states)
                deepstack_features.append(ds_feature)

        # Final spatial merge
        merged = self.merger(op, hidden_states)

        return merged, *deepstack_features
