# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import INT64_MAX


def _get_default_inv_freq(config: ArchitectureConfig) -> np.ndarray:
    dim = int(config.head_dim * config.partial_rotary_factor)
    return 1.0 / (config.rope_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))


def _get_cos_sin_cache(
    max_position_embeddings: int,
    inv_freq: np.ndarray,
    attention_scaling: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.arange(0, max_position_embeddings, dtype=np.float32)
    angles = np.outer(pos, inv_freq).astype(np.float32)
    return (
        (np.cos(angles) * attention_scaling).astype(np.float32),
        (np.sin(angles) * attention_scaling).astype(np.float32),
    )


def get_rotary_pos_emb(op: builder.OpBuilder, position_ids, cos_cache, sin_cache):
    """Retrieve cos/sin positional embeddings based on position IDs.

    Uses Gather to look up the embeddings for the given position IDs.

    Args:
        op: The OpBuilder.
        position_ids: Position IDs of shape (batch_size, seq_length).
        cos_cache: Cosine cache parameter of shape (max_pos, rotary_dim).
        sin_cache: Sine cache parameter of shape (max_pos, rotary_dim).

    Returns:
        Tuple of (cos_emb, sin_emb), each of shape (batch_size, seq_length, rotary_dim).
    """
    cos_emb = op.Gather(cos_cache, position_ids)
    sin_emb = op.Gather(sin_cache, position_ids)
    return cos_emb, sin_emb


def apply_rotary_pos_emb(
    op: builder.OpBuilder,
    x,
    position_embeddings: tuple,
    num_heads: int,
    rotary_embedding_dim: int = 0,
    interleaved: bool = False,
):
    """Apply Rotary Positional Embedding (RoPE) to the input.

    Uses the ONNX opset 23 ``RotaryEmbedding`` op with pre-gathered
    cos/sin embeddings (3D tensors without position_ids).

    Args:
        op: The OpBuilder.
        x: Input tensor of shape ``(batch_size, seq_length, num_heads * head_dim)``.
        position_embeddings: Tuple of ``(cos, sin)`` embeddings, each
            ``(batch_size, seq_length, rotary_dim)``.
        num_heads: Number of attention heads.
        rotary_embedding_dim: Dimension for partial RoPE (0 = full embedding).
        interleaved: If True, use interleaved RoPE layout where real/imag
            pairs are adjacent (d0_re, d0_im, d1_re, d1_im, ...) instead
            of the default half-split layout (d0_re, d1_re, ..., d0_im, d1_im, ...).

    Returns:
        Tensor with RoPE applied, same shape as input.
    """
    cos, sin = position_embeddings
    return op.RotaryEmbedding(
        x,
        cos,
        sin,
        num_heads=num_heads,
        rotary_embedding_dim=rotary_embedding_dim,
        interleaved=1 if interleaved else 0,
    )


class BaseRope(nn.Module):
    """Base class for rotary position embeddings."""

    def __init__(self, cos_cache_data: np.ndarray, sin_cache_data: np.ndarray):
        super().__init__()
        self.cos_cache = nn.Parameter(
            list(cos_cache_data.shape),
            name="cos_cache",
            data=ir.tensor(cos_cache_data),
        )
        self.sin_cache = nn.Parameter(
            list(sin_cache_data.shape),
            name="sin_cache",
            data=ir.tensor(sin_cache_data),
        )

    def forward(self, op: builder.OpBuilder, position_ids: ir.Value):
        return get_rotary_pos_emb(op, position_ids, self.cos_cache, self.sin_cache)


class DefaultRope(BaseRope):
    def __init__(self, config: ArchitectureConfig):
        inv_freq = _get_default_inv_freq(config)
        cos_cache, sin_cache = _get_cos_sin_cache(config.max_position_embeddings, inv_freq)
        super().__init__(cos_cache, sin_cache)


class LinearRope(BaseRope):
    def __init__(self, config: ArchitectureConfig):
        inv_freq = _get_default_inv_freq(config)
        inv_freq = inv_freq / config.rope_scaling["factor"]
        cos_cache, sin_cache = _get_cos_sin_cache(config.max_position_embeddings, inv_freq)
        super().__init__(cos_cache, sin_cache)


class DynamicNTKRope(BaseRope):
    """NTK-aware dynamic RoPE scaling.

    Scales the base theta rather than dividing frequencies directly:
        new_theta = theta * factor^(dim / (dim - 2))
    This spreads frequencies more evenly across the extended context,
    preserving the model's positional inductive bias better than linear
    scaling for long-context extrapolation.
    """

    def __init__(self, config: ArchitectureConfig):
        dim = int(config.head_dim * config.partial_rotary_factor)
        factor = config.rope_scaling["factor"]
        # NTK-aware base scaling
        new_theta = config.rope_theta * (factor ** (dim / (dim - 2)))
        inv_freq = 1.0 / (new_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        cos_cache, sin_cache = _get_cos_sin_cache(config.max_position_embeddings, inv_freq)
        super().__init__(cos_cache, sin_cache)


class Llama3Rope(BaseRope):
    def __init__(self, config: ArchitectureConfig):
        inv_freq = _get_default_inv_freq(config)

        factor = config.rope_scaling["factor"]
        low_freq_factor = config.rope_scaling["low_freq_factor"]
        high_freq_factor = config.rope_scaling["high_freq_factor"]
        old_context_len = config.original_max_position_embeddings

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = np.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
        inv_freq_llama = np.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        cos_cache, sin_cache = _get_cos_sin_cache(
            config.max_position_embeddings, inv_freq_llama
        )
        super().__init__(cos_cache, sin_cache)


class LongRope(BaseRope):
    def __init__(self, config: ArchitectureConfig):
        inv_freq = _get_default_inv_freq(config)

        long_factor = np.array(config.rope_scaling["long_factor"], dtype=np.float32)
        short_factor = np.array(config.rope_scaling["short_factor"], dtype=np.float32)

        original_max_pos = (
            config.original_max_position_embeddings or config.max_position_embeddings
        )
        factor = config.max_position_embeddings / original_max_pos
        if factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max_pos))

        self.original_max_position_embeddings = original_max_pos
        self.has_long_cache = original_max_pos != config.max_position_embeddings

        short_cos, short_sin = _get_cos_sin_cache(
            original_max_pos, inv_freq / short_factor, attention_factor
        )
        if not self.has_long_cache:
            super().__init__(short_cos, short_sin)
            return

        long_cos, long_sin = _get_cos_sin_cache(
            config.max_position_embeddings,
            inv_freq / long_factor,
            attention_factor,
        )
        cos_cache = np.concatenate([short_cos, long_cos], axis=0)
        sin_cache = np.concatenate([short_sin, long_sin], axis=0)
        super().__init__(cos_cache, sin_cache)

    def forward(self, op: builder.OpBuilder, position_ids: ir.Value):
        if self.has_long_cache:
            max_pos = op.ReduceMax(position_ids, keepdims=False)
            use_long = op.Cast(
                op.Greater(max_pos, self.original_max_position_embeddings - 1),
                to=ir.DataType.INT64,
            )
            offset = op.Mul(use_long, self.original_max_position_embeddings)
            position_ids = op.Add(position_ids, offset)
        return get_rotary_pos_emb(op, position_ids, self.cos_cache, self.sin_cache)


class YarnRope(BaseRope):
    """YaRN (Yet another RoPE extensioN) rotary embeddings.

    Used by DeepSeek-V2/V3. Blends interpolated and extrapolated
    frequencies with a linear ramp, and applies mscale attention factor.

    Reference: https://huggingface.co/papers/2309.00071
    """

    def __init__(self, config: ArchitectureConfig):
        rope_scaling = config.rope_scaling or {}
        factor = rope_scaling.get("factor", 1.0)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        mscale = rope_scaling.get("mscale", 1.0)
        mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.0)
        original_max_pos = (
            rope_scaling.get("original_max_position_embeddings")
            or config.max_position_embeddings
        )

        dim = int(config.head_dim * config.partial_rotary_factor)
        base = config.rope_theta

        # Compute attention scaling from mscale
        def get_mscale(scale, ms=1.0):
            if scale <= 1:
                return 1.0
            return 0.1 * ms * math.log(scale) + 1.0

        if mscale and mscale_all_dim:
            attention_factor = float(
                get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            )
        else:
            attention_factor = get_mscale(factor)

        # Compute correction range for frequency interpolation
        def find_correction_dim(num_rotations):
            return (
                dim
                * math.log(original_max_pos / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(find_correction_dim(beta_fast)), 0)
        high = min(math.ceil(find_correction_dim(beta_slow)), dim - 1)

        # Linear ramp between interpolation and extrapolation
        # ramp=0 → extrapolation (use base freq), ramp=1 → interpolation
        if low == high:
            high += 0.001
        t = np.arange(dim // 2, dtype=np.float32)
        ramp = np.clip((t - low) / (high - low), 0.0, 1.0)

        inv_freq_base = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        inv_freq_interpolation = inv_freq_base / factor
        # extrapolation_factor = 1 - ramp:
        # - ramp=0 (below low) → extrapolation_factor=1 → use base freq
        # - ramp=1 (above high) → extrapolation_factor=0 → use interp freq
        extrapolation_factor = 1.0 - ramp
        inv_freq = (
            inv_freq_interpolation * (1 - extrapolation_factor)
            + inv_freq_base * extrapolation_factor
        )

        cos_cache, sin_cache = _get_cos_sin_cache(
            config.max_position_embeddings, inv_freq, attention_factor
        )
        super().__init__(cos_cache, sin_cache)


class _MRopeBase(BaseRope):
    """Base class for multi-dimensional RoPE variants.

    Handles 3D position_ids ``(3, batch, seq_len)`` for temporal, height,
    and width dimensions.  Subclasses compute different ``h_mask`` / ``w_mask``
    arrays to control how T, H, W frequencies are assigned to channels.

    For 2D position_ids ``(batch, seq_len)``, all three dimensions use the
    same positions (equivalent to standard 1D RoPE), enabling text-only use.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        h_mask: np.ndarray,
        w_mask: np.ndarray,
    ):
        inv_freq = _get_default_inv_freq(config)
        cos_cache, sin_cache = _get_cos_sin_cache(config.max_position_embeddings, inv_freq)
        super().__init__(cos_cache, sin_cache)

        rotary_dim = len(inv_freq)
        self.h_mask = nn.Parameter(
            [rotary_dim],
            name="h_mask",
            data=ir.tensor(h_mask),
        )
        self.w_mask = nn.Parameter(
            [rotary_dim],
            name="w_mask",
            data=ir.tensor(w_mask),
        )

    def forward(self, op: builder.OpBuilder, position_ids: ir.Value):
        """Compute MRoPE cos/sin embeddings.

        Args:
            op: ONNX op builder.
            position_ids: Either ``(batch, seq)`` for text-only or
                ``(3, batch, seq)`` for multimodal (T, H, W dimensions).
                For 2D input, the same positions are used for all 3 dims.

        Returns:
            Tuple of ``(cos, sin)`` each with shape ``(batch, seq, rotary_dim)``.
        """
        # For 2D text-only position_ids (batch, seq), expand to (3, batch, seq)
        # by stacking the same positions for T, H, W dimensions.
        # For 3D position_ids, this is a no-op reshape.
        pos_shape = op.Shape(position_ids)
        # Reshape to (3, batch, seq) — works for both 2D and 3D:
        # 2D (batch, seq): Unsqueeze → (1, batch, seq) → Expand → (3, batch, seq)
        # 3D (3, batch, seq): passes through directly
        batch_seq = op.Slice(pos_shape, [-2], [INT64_MAX], [0])
        target_shape = op.Concat(op.Constant(value_ints=[3]), batch_seq, axis=0)
        position_ids = op.Expand(op.Unsqueeze(position_ids, [0]), target_shape)
        # Ensure exactly (3, batch, seq) — remove extra leading dims from 3D input
        position_ids = op.Reshape(position_ids, target_shape)

        # Gather cos/sin for T dimension (dimension 0)
        pos_t = op.Gather(position_ids, [0], axis=0)
        pos_t = op.Squeeze(pos_t, [0])  # (batch, seq)
        cos_t = op.Gather(self.cos_cache, pos_t)
        sin_t = op.Gather(self.sin_cache, pos_t)

        # Gather cos/sin for H dimension (dimension 1)
        pos_h = op.Gather(position_ids, [1], axis=0)
        pos_h = op.Squeeze(pos_h, [0])
        cos_h = op.Gather(self.cos_cache, pos_h)
        sin_h = op.Gather(self.sin_cache, pos_h)

        # Gather cos/sin for W dimension (dimension 2)
        pos_w = op.Gather(position_ids, [2], axis=0)
        pos_w = op.Squeeze(pos_w, [0])
        cos_w = op.Gather(self.cos_cache, pos_w)
        sin_w = op.Gather(self.sin_cache, pos_w)

        # Mix T, H, W channels according to subclass masks
        cos = op.Where(self.h_mask, cos_h, cos_t)
        cos = op.Where(self.w_mask, cos_w, cos)
        sin = op.Where(self.h_mask, sin_h, sin_t)
        sin = op.Where(self.w_mask, sin_w, sin)

        return cos, sin


class ChunkedMRope(_MRopeBase):
    """Chunked Multimodal RoPE for Qwen2.5-VL / Qwen3-VL.

    Frequencies are assigned in contiguous chunks::

        freq layout: [T0, T1, ..., Ts0, Hs0+1, ..., Hs0+s1, Ws0+s1+1, ..., Wend]

    where ``s0, s1, s2 = mrope_section``.
    """

    def __init__(self, config: ArchitectureConfig):
        mrope_section = config.mrope_section
        assert mrope_section is not None

        dim = int(config.head_dim * config.partial_rotary_factor)
        rotary_dim = dim // 2

        # Contiguous section masks: [T...T | H...H | W...W]
        h_mask = np.zeros(rotary_dim, dtype=np.bool_)
        w_mask = np.zeros(rotary_dim, dtype=np.bool_)
        s0 = mrope_section[0]
        s1 = mrope_section[1]
        h_mask[s0 : s0 + s1] = True
        w_mask[s0 + s1 :] = True

        super().__init__(config, h_mask, w_mask)


class InterleavedMRope(_MRopeBase):
    """Interleaved Multimodal RoPE for Qwen3.5.

    Frequencies are interleaved in groups of 3 (T, H, W) with any remaining
    channels assigned to T::

        freq layout: [T, H, W, T, H, W, ..., T, T]

    Uses the ``mrope_section`` channel assignment to determine how many
    channels each dimension occupies.
    """

    def __init__(self, config: ArchitectureConfig):
        mrope_section = config.mrope_section
        assert mrope_section is not None

        dim = int(config.head_dim * config.partial_rotary_factor)
        rotary_dim = dim // 2

        # Interleaved channel masks:
        # H channels: positions [1, 4, 7, ...] up to mrope_section[1]*3
        # W channels: positions [2, 5, 8, ...] up to mrope_section[2]*3
        h_mask = np.zeros(rotary_dim, dtype=np.bool_)
        w_mask = np.zeros(rotary_dim, dtype=np.bool_)
        h_length = mrope_section[1] * 3
        w_length = mrope_section[2] * 3
        for i in range(1, h_length, 3):
            if i < rotary_dim:
                h_mask[i] = True
        for i in range(2, w_length, 3):
            if i < rotary_dim:
                w_mask[i] = True

        super().__init__(config, h_mask, w_mask)


def initialize_rope(config: ArchitectureConfig) -> nn.Module:
    """Factory function to create the appropriate RoPE variant."""
    if config.mrope_section is not None:
        if config.mrope_interleaved:
            return InterleavedMRope(config)
        return ChunkedMRope(config)
    if config.rope_type == "default":
        return DefaultRope(config)
    if config.rope_type == "linear":
        return LinearRope(config)
    if config.rope_type == "dynamic":
        return DynamicNTKRope(config)
    if config.rope_type == "llama3":
        return Llama3Rope(config)
    if config.rope_type == "longrope":
        return LongRope(config)
    if config.rope_type == "yarn":
        return YarnRope(config)

    raise ValueError(f"Unsupported rope type: {config.rope_type}")
