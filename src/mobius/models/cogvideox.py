# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""CogVideoX 3D transformer denoiser for video generation.

CogVideoX uses a dual-stream expert transformer architecture where
text and video tokens are processed in parallel streams with joint
attention, similar to SD3/Flux but operating on 3D (temporal + spatial)
video latents.

Architecture:
- 3D patch embedding: [B, T, C, H, W] → [B, T*H'*W', hidden]
- Text embedding: [B, seq, text_dim] → [B, seq, hidden]
- Dual-stream joint attention (text + video concat → attend → split)
- CogVideoXLayerNormZero: shared norm + 6-param dual-stream modulation
- 3D sincos positional embeddings (temporal + spatial)

Replicates HuggingFace diffusers' ``CogVideoXTransformer3DModel``.
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._diffusers_configs import CogVideoXConfig
from mobius.components import LayerNorm as _LayerNorm
from mobius.components import Linear as _Linear
from mobius.components._activations import SiLU as _SiLU
from mobius.components._diffusion import (
    TimestepEmbedding as _TimestepEmbedding,
)

# ---------------------------------------------------------------------------
# 3D sincos positional embedding helpers
# ---------------------------------------------------------------------------


def _get_1d_sincos(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    """1D sinusoidal positional embedding.

    Args:
        embed_dim: Output embedding dimension.
        positions: 1D array of positions.

    Returns:
        Array of shape ``[len(positions), embed_dim]``.
    """
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    out = np.outer(positions.astype(np.float64), omega)  # [N, D/2]
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)  # [N, D]


def _get_2d_sincos(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """2D sinusoidal positional embedding from a spatial grid.

    Args:
        embed_dim: Output embedding dimension (split equally H/W).
        grid: Array of shape ``[2, 1, H, W]``.

    Returns:
        Array of shape ``[H*W, embed_dim]``.
    """
    emb_h = _get_1d_sincos(embed_dim // 2, grid[0].flatten())
    emb_w = _get_1d_sincos(embed_dim // 2, grid[1].flatten())
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_3d_sincos_pos_embed(
    embed_dim: int,
    spatial_size: tuple[int, int],
    temporal_size: int,
    spatial_scale: float = 1.0,
    temporal_scale: float = 1.0,
) -> np.ndarray:
    """3D sincos positional embedding (temporal + spatial).

    Splits embedding dimension: 1/4 temporal, 3/4 spatial.

    Args:
        embed_dim: Total embedding dimension (must be divisible by 4).
        spatial_size: ``(W, H)`` grid dimensions.
        temporal_size: Number of temporal positions.
        spatial_scale: Scale factor for spatial positions.
        temporal_scale: Scale factor for temporal positions.

    Returns:
        Array of shape ``[temporal_size, H*W, embed_dim]``.
    """
    dim_spatial = 3 * embed_dim // 4
    dim_temporal = embed_dim // 4

    # Spatial: 2D sincos over (H, W) grid
    grid_h = np.arange(spatial_size[1], dtype=np.float32) / spatial_scale
    grid_w = np.arange(spatial_size[0], dtype=np.float32) / spatial_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, spatial_size[1], spatial_size[0])
    pos_spatial = _get_2d_sincos(dim_spatial, grid)  # [H*W, 3D/4]

    # Temporal: 1D sincos
    grid_t = np.arange(temporal_size, dtype=np.float32) / temporal_scale
    pos_temporal = _get_1d_sincos(dim_temporal, grid_t)  # [T, D/4]

    # Broadcast: spatial [1, H*W, 3D/4] x T, temporal [T, 1, D/4] x H*W
    pos_spatial = np.repeat(pos_spatial[np.newaxis, :, :], temporal_size, axis=0)
    pos_temporal = np.repeat(
        pos_temporal[:, np.newaxis, :],
        spatial_size[0] * spatial_size[1],
        axis=1,
    )

    # [T, H*W, D] — temporal dims first in the channel dimension
    return np.concatenate([pos_temporal, pos_spatial], axis=-1)


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------


class _CogVideoXLayerNormZero(nn.Module):
    """Dual-stream adaptive layer norm for CogVideoX.

    Shared LayerNorm applied to both video and text streams, with
    6 modulation parameters from the timestep embedding:
    (shift, scale, gate) for video + (enc_shift, enc_scale, enc_gate)
    for text.

    Replicates HF diffusers' ``CogVideoXLayerNormZero``.
    """

    def __init__(self, conditioning_dim: int, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = _LayerNorm(embedding_dim, eps=eps)
        self.linear = _Linear(conditioning_dim, 6 * embedding_dim, bias=True)
        self._silu = _SiLU()

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        temb: ir.Value,
    ):
        emb = self._silu(op, temb)
        emb = self.linear(op, emb)

        # 6 modulation params: video (shift, scale, gate), text (same)
        shift, scale, gate, enc_shift, enc_scale, enc_gate = op.Split(
            emb, num_outputs=6, axis=-1, _outputs=6
        )

        one = op.Constant(value_float=1.0)

        # Modulate video stream
        normed_h = self.norm(op, hidden_states)
        normed_h = op.Mul(normed_h, op.Add(one, op.Unsqueeze(scale, [1])))
        normed_h = op.Add(normed_h, op.Unsqueeze(shift, [1]))

        # Modulate text stream (same shared norm)
        normed_e = self.norm(op, encoder_hidden_states)
        normed_e = op.Mul(normed_e, op.Add(one, op.Unsqueeze(enc_scale, [1])))
        normed_e = op.Add(normed_e, op.Unsqueeze(enc_shift, [1]))

        return (
            normed_h,
            normed_e,
            op.Unsqueeze(gate, [1]),
            op.Unsqueeze(enc_gate, [1]),
        )


class _CogVideoXOutputNorm(nn.Module):
    """Adaptive layer norm for CogVideoX output.

    Like ``AdaLayerNormOutput`` but with shift-first ordering
    (CogVideoX uses ``chunk_dim=1`` in HF).

    Replicates HF diffusers' ``AdaLayerNorm`` with ``chunk_dim=1``.
    """

    def __init__(self, conditioning_dim: int, output_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = _LayerNorm(output_dim, eps=eps)
        self.linear = _Linear(conditioning_dim, output_dim * 2, bias=True)
        self._silu = _SiLU()

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value, temb: ir.Value):
        emb = self._silu(op, temb)
        emb = self.linear(op, emb)
        # CogVideoX shift-first order
        shift, scale = op.Split(emb, num_outputs=2, axis=-1, _outputs=2)
        one = op.Constant(value_float=1.0)
        hidden_states = self.norm(op, hidden_states)
        hidden_states = op.Mul(hidden_states, op.Add(one, op.Unsqueeze(scale, [1])))
        hidden_states = op.Add(hidden_states, op.Unsqueeze(shift, [1]))
        return hidden_states


class _CogVideoXAttention(nn.Module):
    """Joint self-attention with QK-norm for CogVideoX.

    Concatenates text + video tokens, applies shared Q/K/V projections,
    QK normalization (per-head LayerNorm), runs attention, then splits
    the output back into text and video streams.

    Replicates HF diffusers' ``Attention`` with ``CogVideoXAttnProcessor2_0``.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.to_q = _Linear(hidden_size, hidden_size, bias=True)
        self.to_k = _Linear(hidden_size, hidden_size, bias=True)
        self.to_v = _Linear(hidden_size, hidden_size, bias=True)
        # to_out.0 matches HF naming via nn.Sequential
        self.to_out = nn.Sequential(_Linear(hidden_size, hidden_size))
        # Per-head QK normalization
        self.norm_q = _LayerNorm(head_dim, eps=1e-6)
        self.norm_k = _LayerNorm(head_dim, eps=1e-6)
        self._num_heads = num_heads
        self._head_dim = head_dim

    def forward(
        self, op: builder.OpBuilder, hidden_states: ir.Value, encoder_hidden_states: ir.Value
    ):
        text_seq_len = op.Shape(encoder_hidden_states, start=1, end=2)

        # Concatenate text + video for shared projection
        combined = op.Concat(encoder_hidden_states, hidden_states, axis=1)

        q = self.to_q(op, combined)
        k = self.to_k(op, combined)
        v = self.to_v(op, combined)

        # QK-norm: reshape to [B, seq, H, D], normalize D, reshape back
        batch = op.Shape(q, start=0, end=1)
        seq = op.Shape(q, start=1, end=2)
        head_shape = op.Concat(
            batch,
            seq,
            op.Constant(value_ints=[self._num_heads, self._head_dim]),
            axis=0,
        )
        flat_shape = op.Concat(
            batch,
            seq,
            op.Constant(value_ints=[self._num_heads * self._head_dim]),
            axis=0,
        )

        # [B, seq, hidden] → [B, seq, H, D] → LayerNorm on D → [B, seq, hidden]
        q = self.norm_q(op, op.Reshape(q, head_shape))
        q = op.Reshape(q, flat_shape)
        k = self.norm_k(op, op.Reshape(k, head_shape))
        k = op.Reshape(k, flat_shape)

        attn_out = op.Attention(
            q,
            k,
            v,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            is_causal=0,
            scale=float(self._head_dim**-0.5),
        )

        attn_out = self.to_out(op, attn_out)

        # Split back: text (first text_seq_len) + video (rest)
        video_seq_len = op.Sub(seq, text_seq_len)
        enc_out, h_out = op.Split(
            attn_out,
            op.Concat(text_seq_len, video_seq_len, axis=0),
            axis=1,
            _outputs=2,
        )

        return h_out, enc_out


class _CogVideoXFFN(nn.Module):
    """Feed-forward network for CogVideoX.

    GELU (tanh approximation) activation + linear output.

    HF weight names ``ff.net.0.proj.*`` and ``ff.net.2.*`` are mapped
    via ``preprocess_weights`` to ``ff.gelu_proj.*`` and
    ``ff.linear_out.*``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int | None = None):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size * 4
        self.gelu_proj = _Linear(hidden_size, intermediate_size)
        self.linear_out = _Linear(intermediate_size, hidden_size)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        hidden_states = self.gelu_proj(op, hidden_states)
        hidden_states = op.Gelu(hidden_states)
        hidden_states = self.linear_out(op, hidden_states)
        return hidden_states


class _CogVideoXBlock(nn.Module):
    """CogVideoX transformer block with dual-stream joint attention.

    1. CogVideoXLayerNormZero: modulate both streams from timestep
    2. Joint attention: concat(text, video) → Q/K/V → attend → split
    3. Gated residual for both streams
    4. CogVideoXLayerNormZero: modulate for FFN
    5. FFN on concatenated (text + video) → split
    6. Gated residual for both streams

    Replicates HF diffusers' ``CogVideoXBlock``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        time_embed_dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = _CogVideoXLayerNormZero(time_embed_dim, dim, eps)
        self.attn1 = _CogVideoXAttention(dim, num_heads, head_dim)
        self.norm2 = _CogVideoXLayerNormZero(time_embed_dim, dim, eps)
        self.ff = _CogVideoXFFN(dim)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        temb: ir.Value,
    ):
        # --- Self-attention ---
        norm_h, norm_e, gate_msa, enc_gate_msa = self.norm1(
            op, hidden_states, encoder_hidden_states, temb
        )
        attn_h, attn_e = self.attn1(op, norm_h, norm_e)
        hidden_states = op.Add(hidden_states, op.Mul(gate_msa, attn_h))
        encoder_hidden_states = op.Add(encoder_hidden_states, op.Mul(enc_gate_msa, attn_e))

        # --- Feed-forward ---
        norm_h, norm_e, gate_ff, enc_gate_ff = self.norm2(
            op, hidden_states, encoder_hidden_states, temb
        )
        # FFN runs on concatenated text + video
        combined = op.Concat(norm_e, norm_h, axis=1)
        ff_out = self.ff(op, combined)

        # Split back and apply gated residual
        text_len = op.Shape(norm_e, start=1, end=2)
        video_len = op.Shape(norm_h, start=1, end=2)
        ff_e, ff_h = op.Split(
            ff_out,
            op.Concat(text_len, video_len, axis=0),
            axis=1,
            _outputs=2,
        )
        hidden_states = op.Add(hidden_states, op.Mul(gate_ff, ff_h))
        encoder_hidden_states = op.Add(encoder_hidden_states, op.Mul(enc_gate_ff, ff_e))

        return hidden_states, encoder_hidden_states


class _CogVideoXPatchEmbed(nn.Module):
    """3D patch embedding for CogVideoX.

    Patchifies video: [B, T, C, H, W] → [B, T*(H/p)*(W/p), hidden]
    Embeds text: [B, seq, text_dim] → [B, seq, hidden]
    Concatenates and adds 3D sincos positional embedding.

    Replicates HF diffusers' ``CogVideoXPatchEmbed``.
    """

    def __init__(self, config: CogVideoXConfig, embed_dim: int):
        super().__init__()
        p = config.patch_size
        in_ch = config.in_channels

        # Video patch projection: [B, seq, C*p*p] → [B, seq, embed_dim]
        self.proj = _Linear(in_ch * p * p, embed_dim, bias=True)
        # Text projection: [B, seq, text_dim] → [B, seq, embed_dim]
        self.text_proj = _Linear(config.text_embed_dim, embed_dim, bias=True)

        self._patch_size = p
        self._in_channels = in_ch

        # Pre-compute 3D sincos positional embedding as nn.Parameter
        post_patch_h = config.sample_height // p
        post_patch_w = config.sample_width // p
        num_time_patches = (config.sample_frames - 1) // config.temporal_compression_ratio + 1

        pos_embed = _get_3d_sincos_pos_embed(
            embed_dim,
            (post_patch_w, post_patch_h),
            num_time_patches,
            spatial_scale=config.spatial_interpolation_scale,
            temporal_scale=config.temporal_interpolation_scale,
        )  # [T, H'*W', D]
        pos_embed = pos_embed.reshape(-1, embed_dim)  # [T*H'*W', D]

        # Prepend zeros for text tokens (no positional embedding)
        text_zeros = np.zeros((config.max_text_seq_length, embed_dim), dtype=np.float32)
        # [max_text_seq + T*H'*W', D]
        full_pos = np.concatenate([text_zeros, pos_embed], axis=0)
        total_seq = full_pos.shape[0]
        # [1, total_seq, D] for broadcasting with batch dim
        full_pos = full_pos.reshape(1, total_seq, embed_dim)

        self.pos_embedding = nn.Parameter(
            [1, total_seq, embed_dim],
            name="patch_embed.pos_embedding.pos_embedding",
            data=ir.tensor(full_pos),
        )

    def forward(
        self,
        op: builder.OpBuilder,
        text_embeds: ir.Value,
        video: ir.Value,
    ):
        batch = op.Shape(video, start=0, end=1)
        num_frames = op.Shape(video, start=1, end=2)
        height = op.Shape(video, start=3, end=4)
        width = op.Shape(video, start=4, end=5)
        p = self._patch_size
        c = self._in_channels

        h_patches = op.Div(height, op.Constant(value_ints=[p]))
        w_patches = op.Div(width, op.Constant(value_ints=[p]))

        # Patchify: [B, T, C, H, W] → [B, T, C, H/p, p, W/p, p]
        video = op.Reshape(
            video,
            op.Concat(
                batch,
                num_frames,
                op.Constant(value_ints=[c]),
                h_patches,
                op.Constant(value_ints=[p]),
                w_patches,
                op.Constant(value_ints=[p]),
                axis=0,
            ),
        )
        # → [B, T, H/p, W/p, C, p, p]
        video = op.Transpose(video, perm=[0, 1, 3, 5, 2, 4, 6])
        # → [B, T*H'*W', C*p*p]
        video_seq = op.Mul(op.Mul(num_frames, h_patches), w_patches)
        video = op.Reshape(
            video,
            op.Concat(
                batch,
                video_seq,
                op.Constant(value_ints=[c * p * p]),
                axis=0,
            ),
        )
        # Project video patches
        video = self.proj(op, video)  # [B, T*H'*W', hidden]

        # Project text
        text = self.text_proj(op, text_embeds)  # [B, seq, hidden]

        # Concatenate text + video and add positional embedding
        embeds = op.Concat(text, video, axis=1)  # [B, total_seq, hidden]
        embeds = op.Add(embeds, self.pos_embedding)

        return embeds


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class CogVideoXTransformer3DModel(nn.Module):
    """CogVideoX 3D transformer denoiser for video generation.

    Dual-stream expert transformer: text and video tokens processed
    in parallel with joint attention across all tokens.

    Input: sample [B, T, C, H, W], timestep [B], text [B, seq, text_dim]
    Output: noise_pred [B, T, C_out, H, W]

    Replicates HF diffusers' ``CogVideoXTransformer3DModel``.
    """

    default_task: str = "video-denoising"
    category: str = "Diffusion"

    def __init__(self, config: CogVideoXConfig):
        super().__init__()
        self.config = config
        inner_dim = config.num_attention_heads * config.attention_head_dim

        # 1. Patch embedding
        self.patch_embed = _CogVideoXPatchEmbed(config, inner_dim)

        # 2. Timestep embedding: sinusoidal → MLP
        self.time_embedding = _TimestepEmbedding(inner_dim, config.time_embed_dim)

        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            self.transformer_blocks.append(
                _CogVideoXBlock(
                    dim=inner_dim,
                    num_heads=config.num_attention_heads,
                    head_dim=config.attention_head_dim,
                    time_embed_dim=config.time_embed_dim,
                    eps=config.norm_eps,
                )
            )

        # 4. Output
        self.norm_final = _LayerNorm(inner_dim, eps=config.norm_eps)
        self.norm_out = _CogVideoXOutputNorm(
            config.time_embed_dim, inner_dim, eps=config.norm_eps
        )

        if config.patch_size_t is None:
            output_dim = config.patch_size * config.patch_size * config.out_channels
        else:
            output_dim = (
                config.patch_size
                * config.patch_size
                * config.patch_size_t
                * config.out_channels
            )
        self.proj_out = _Linear(inner_dim, output_dim)

        self._time_proj_dim = inner_dim

    def forward(
        self,
        op: builder.OpBuilder,
        sample: ir.Value,
        timestep: ir.Value,
        encoder_hidden_states: ir.Value,
    ):
        # sample: [B, T, C, H, W]
        batch = op.Shape(sample, start=0, end=1)
        num_frames = op.Shape(sample, start=1, end=2)
        height = op.Shape(sample, start=3, end=4)
        width = op.Shape(sample, start=4, end=5)

        # 1. Timestep embedding
        t_emb = self._get_timestep_embedding(op, timestep)
        emb = self.time_embedding(op, t_emb)

        # 2. Patch embedding (patchify video + embed text + add pos embed)
        # Returns [B, text_seq + video_seq, hidden]
        hidden_states = self.patch_embed(op, encoder_hidden_states, sample)

        # 3. Split text and video tokens
        text_seq_len = op.Shape(encoder_hidden_states, start=1, end=2)
        encoder_hidden_states = op.Slice(
            hidden_states,
            op.Constant(value_ints=[0]),
            text_seq_len,
            op.Constant(value_ints=[1]),
        )
        hidden_states = op.Slice(
            hidden_states,
            text_seq_len,
            op.Constant(value_ints=[2147483647]),
            op.Constant(value_ints=[1]),
        )

        # 4. Transformer blocks
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                op, hidden_states, encoder_hidden_states, emb
            )

        # 5. Output normalization and projection
        hidden_states = self.norm_final(op, hidden_states)
        hidden_states = self.norm_out(op, hidden_states, emb)
        hidden_states = self.proj_out(op, hidden_states)

        # 6. Unpatchify: [B, T*H'*W', p*p*C] → [B, T, C, H, W]
        hidden_states = self._unpatchify(op, hidden_states, batch, num_frames, height, width)

        return hidden_states

    def _unpatchify(self, op: builder.OpBuilder, x, batch, num_frames, height, width):
        """Reshape patch tokens back to video: [B, T, C, H, W]."""
        p = self.config.patch_size
        c = self.config.out_channels
        h_patches = op.Div(height, op.Constant(value_ints=[p]))
        w_patches = op.Div(width, op.Constant(value_ints=[p]))

        # [B, T*H'*W', p*p*C] → [B, T, H', W', C, p, p]
        x = op.Reshape(
            x,
            op.Concat(
                batch,
                num_frames,
                h_patches,
                w_patches,
                op.Constant(value_ints=[c, p, p]),
                axis=0,
            ),
        )
        # → [B, T, C, H', p, W', p]
        x = op.Transpose(x, perm=[0, 1, 4, 2, 5, 3, 6])
        # → [B, T, C, H, W]
        x = op.Reshape(
            x,
            op.Concat(
                batch,
                num_frames,
                op.Constant(value_ints=[c]),
                op.Mul(h_patches, op.Constant(value_ints=[p])),
                op.Mul(w_patches, op.Constant(value_ints=[p])),
                axis=0,
            ),
        )
        return x

    def _get_timestep_embedding(self, op: builder.OpBuilder, timestep):
        """Sinusoidal timestep embedding (flip_sin_to_cos=True)."""
        half_dim = self._time_proj_dim // 2
        exponent = -math.log(10000.0) / half_dim
        freqs = np.exp(np.arange(half_dim) * exponent).astype(np.float32)
        freq_const = op.Constant(value_floats=freqs.tolist())
        t = op.Cast(timestep, to=1)  # Cast to FLOAT
        t = op.Unsqueeze(t, [1])
        args = op.Mul(t, op.Unsqueeze(freq_const, [0]))
        # flip_sin_to_cos=True: cos first, then sin
        return op.Concat(op.Cos(args), op.Sin(args), axis=-1)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF diffusers weight names to ONNX parameter names.

        Renames:
        - ``ff.net.0.proj.*`` → ``ff.gelu_proj.*``
        - ``ff.net.2.*`` → ``ff.linear_out.*``
        """
        new_state_dict: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            name = name.replace(".ff.net.0.proj.", ".ff.gelu_proj.")
            name = name.replace(".ff.net.2.", ".ff.linear_out.")
            new_state_dict[name] = tensor
        return new_state_dict
