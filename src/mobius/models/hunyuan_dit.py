# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""HunyuanDiT (Hunyuan Diffusion Transformer) 2D model.

Transformer-based denoiser with U-Net-style skip connections. Used by
Tencent's HunyuanDiT text-to-image model.

Key differences from standard DiT (dit.py):
- AdaLN-Shift (shift only, no scale/gate) for self-attention norm
- QK normalization (LayerNorm on Q and K before attention)
- Skip connections: second half of blocks receives residuals from first half
- GEGLU activation in feed-forward network
- Separate LayerNorm for cross-attention and FFN (norm2, norm3)

Architecture:
1. Patch embedding (2D → sequence of patches)
2. Sinusoidal timestep embedding → MLP
3. Transformer blocks with skip connections and AdaLN-Shift
4. AdaLN-Continuous output norm → Linear → unpatchify

HF diffusers class: HunyuanDiT2DModel
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius.components import LayerNorm as _LayerNorm
from mobius.components import Linear as _Linear
from mobius.components import SiLU as _SiLU
from mobius.components._diffusion import (
    PatchEmbed as _PatchEmbed,
)
from mobius.components._diffusion import (
    TimestepEmbedding as _TimestepEmbedding,
)

if TYPE_CHECKING:
    import onnx_ir as ir


@dataclasses.dataclass
class HunyuanDiTConfig:
    """Configuration for HunyuanDiT transformer denoiser.

    Note: hidden_size should equal num_attention_heads * attention_head_dim.
    In HF diffusers, these are separate params; here we use hidden_size
    directly.
    """

    in_channels: int = 4
    patch_size: int = 2
    hidden_size: int = 1408
    num_layers: int = 40
    num_attention_heads: int = 16
    cross_attention_dim: int = 1024
    mlp_ratio: float = 4.0
    learn_sigma: bool = True
    sample_size: int = 128
    norm_eps: float = 1e-6
    qk_norm: bool = True

    @classmethod
    def from_diffusers(cls, config: dict) -> HunyuanDiTConfig:
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        num_heads = config.get("num_attention_heads", 16)
        head_dim = config.get("attention_head_dim", 88)
        hidden = config.get("hidden_size", num_heads * head_dim)
        return cls(
            in_channels=config.get("in_channels", 4),
            patch_size=config.get("patch_size", 2),
            hidden_size=hidden,
            num_layers=config.get("num_layers", 40),
            num_attention_heads=num_heads,
            cross_attention_dim=config.get("cross_attention_dim", 1024),
            mlp_ratio=config.get("mlp_ratio", 4.0),
            learn_sigma=config.get("learn_sigma", True),
            sample_size=config.get("sample_size", 128),
            norm_eps=config.get("norm_eps", 1e-6),
            qk_norm=config.get("qk_norm", True),
        )


# ---------------------------------------------------------------------------
# Components (shared: _LayerNorm, _PatchEmbed, _TimestepEmbedding imported
# from components; remaining model-specific below)
# ---------------------------------------------------------------------------


class _LayerNormNoAffine(nn.Module):
    """Layer Normalization without learnable parameters."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self._dim = dim
        self._eps = eps

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # Use ReduceMean + variance computation for norm without params
        mean = op.ReduceMean(x, [-1], keepdims=True)
        diff = op.Sub(x, mean)
        var = op.ReduceMean(op.Mul(diff, diff), [-1], keepdims=True)
        eps = op.Constant(value_float=self._eps)
        return op.Div(diff, op.Sqrt(op.Add(var, eps)))


class _AdaLayerNormShift(nn.Module):
    """Adaptive Layer Norm with shift-only modulation.

    Unlike AdaLN-Zero (DiT), this only applies a shift from the timestep
    embedding — no scale or gate. HF name: ``AdaLayerNormShift``.

    HF weights: norm.weight, norm.bias, linear.weight, linear.bias
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = _LayerNorm(hidden_size, eps=eps)
        self.linear = _Linear(hidden_size, hidden_size)
        self._silu = _SiLU()

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value, temb: ir.Value):
        # shift = Linear(SiLU(temb))
        shift = self._silu(op, temb)
        shift = self.linear(op, shift)
        # LayerNorm(x) + shift
        normed = self.norm(op, hidden_states)
        return op.Add(normed, op.Unsqueeze(shift, [1]))


class _HunyuanSelfAttention(nn.Module):
    """Multi-head self-attention with optional QK normalization.

    HF weights: to_q, to_k, to_v, to_out.0, norm_q, norm_k
    """

    def __init__(self, hidden_size: int, num_heads: int, qk_norm: bool = True):
        super().__init__()
        self.to_q = _Linear(hidden_size, hidden_size)
        self.to_k = _Linear(hidden_size, hidden_size)
        self.to_v = _Linear(hidden_size, hidden_size)
        self.to_out = nn.Sequential(_Linear(hidden_size, hidden_size))
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads
        self._qk_norm = qk_norm
        if qk_norm:
            self.norm_q = _LayerNorm(self._head_dim, eps=1e-6)
            self.norm_k = _LayerNorm(self._head_dim, eps=1e-6)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        q = self.to_q(op, hidden_states)
        k = self.to_k(op, hidden_states)
        v = self.to_v(op, hidden_states)

        if self._qk_norm:
            # Reshape to [B, seq, num_heads, head_dim] for per-head norm
            batch_seq = op.Shape(q, start=0, end=2)
            head_shape = op.Constant(value_ints=[self._num_heads, self._head_dim])
            reshape_to = op.Concat(batch_seq, head_shape, axis=0)
            q = op.Reshape(q, reshape_to)
            k = op.Reshape(k, reshape_to)
            q = self.norm_q(op, q)
            k = self.norm_k(op, k)
            # Reshape back to [B, seq, hidden]
            orig_shape = op.Concat(
                batch_seq, op.Constant(value_ints=[self._num_heads * self._head_dim]), axis=0
            )
            q = op.Reshape(q, orig_shape)
            k = op.Reshape(k, orig_shape)

        attn_out = op.Attention(
            q,
            k,
            v,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            is_causal=0,
            scale=float(self._head_dim**-0.5),
        )
        return self.to_out(op, attn_out)


class _HunyuanCrossAttention(nn.Module):
    """Multi-head cross-attention with QK normalization.

    Q from latent, KV from encoder output.
    HF weights: to_q, to_k, to_v, to_out.0, norm_q, norm_k
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_heads: int,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.to_q = _Linear(hidden_size, hidden_size)
        self.to_k = _Linear(cross_attention_dim, hidden_size)
        self.to_v = _Linear(cross_attention_dim, hidden_size)
        self.to_out = nn.Sequential(_Linear(hidden_size, hidden_size))
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads
        self._qk_norm = qk_norm
        if qk_norm:
            self.norm_q = _LayerNorm(self._head_dim, eps=1e-6)
            self.norm_k = _LayerNorm(self._head_dim, eps=1e-6)

    def forward(
        self, op: builder.OpBuilder, hidden_states: ir.Value, encoder_hidden_states: ir.Value
    ):
        q = self.to_q(op, hidden_states)
        k = self.to_k(op, encoder_hidden_states)
        v = self.to_v(op, encoder_hidden_states)

        if self._qk_norm:
            # Per-head QK norm
            q_batch_seq = op.Shape(q, start=0, end=2)
            k_batch_seq = op.Shape(k, start=0, end=2)
            head_shape = op.Constant(value_ints=[self._num_heads, self._head_dim])

            q = op.Reshape(q, op.Concat(q_batch_seq, head_shape, axis=0))
            k = op.Reshape(k, op.Concat(k_batch_seq, head_shape, axis=0))
            q = self.norm_q(op, q)
            k = self.norm_k(op, k)
            hidden_dim = op.Constant(value_ints=[self._num_heads * self._head_dim])
            q = op.Reshape(q, op.Concat(q_batch_seq, hidden_dim, axis=0))
            k = op.Reshape(k, op.Concat(k_batch_seq, hidden_dim, axis=0))

        attn_out = op.Attention(
            q,
            k,
            v,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            is_causal=0,
            scale=float(self._head_dim**-0.5),
        )
        return self.to_out(op, attn_out)


class _GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation.

    Projects input to 2x intermediate, splits, and gates.
    HF weights: proj.weight, proj.bias
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.proj = _Linear(hidden_size, intermediate_size * 2)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        projected = self.proj(op, hidden_states)
        # Split into gate and value: [B, seq, 2*inter] → 2x [B, seq, inter]
        hidden, gate = op.Split(projected, num_outputs=2, axis=-1, _outputs=2)
        return op.Mul(hidden, op.Gelu(gate))


class _HunyuanFFN(nn.Module):
    """Feed-forward network with GEGLU activation.

    HF structure: ff.net = [GEGLU(index=0), Dropout(index=1), Linear(index=2)]
    We map ff.net.0 → geglu, ff.net.2 → linear_out in preprocess_weights.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.geglu = _GEGLU(hidden_size, intermediate_size)
        self.linear_out = _Linear(intermediate_size, hidden_size)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        hidden_states = self.geglu(op, hidden_states)
        return self.linear_out(op, hidden_states)


class _HunyuanDiTBlock(nn.Module):
    """HunyuanDiT transformer block.

    Structure:
    1. AdaLN-Shift → self-attention (with QK-norm)
    2. LayerNorm → cross-attention (with QK-norm)
    3. LayerNorm → GEGLU FFN
    Optional: skip connection (concat + norm + linear) for U-Net style.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        cross_attention_dim: int,
        intermediate_size: int,
        eps: float = 1e-6,
        qk_norm: bool = True,
        skip: bool = False,
    ):
        super().__init__()
        # 1. Self-Attention with AdaLN-Shift
        self.norm1 = _AdaLayerNormShift(hidden_size, eps=eps)
        self.attn1 = _HunyuanSelfAttention(hidden_size, num_heads, qk_norm)

        # 2. Cross-Attention
        self.norm2 = _LayerNorm(hidden_size, eps=eps)
        self.attn2 = _HunyuanCrossAttention(
            hidden_size, cross_attention_dim, num_heads, qk_norm
        )

        # 3. Feed-Forward
        self.norm3 = _LayerNorm(hidden_size, eps=eps)
        self.ff = _HunyuanFFN(hidden_size, intermediate_size)

        # 4. Optional skip connection
        self._has_skip = skip
        if skip:
            self.skip_norm = _LayerNorm(hidden_size * 2, eps=eps)
            self.skip_linear = _Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        temb: ir.Value,
        skip_connection: ir.Value | None = None,
    ):
        # Skip connection: concat → norm → linear
        if self._has_skip and skip_connection is not None:
            cat = op.Concat(hidden_states, skip_connection, axis=-1)
            cat = self.skip_norm(op, cat)
            hidden_states = self.skip_linear(op, cat)

        # 1. Self-Attention with AdaLN-Shift
        normed = self.norm1(op, hidden_states, temb)
        attn_out = self.attn1(op, normed)
        hidden_states = op.Add(hidden_states, attn_out)

        # 2. Cross-Attention
        normed = self.norm2(op, hidden_states)
        cross_out = self.attn2(op, normed, encoder_hidden_states)
        hidden_states = op.Add(hidden_states, cross_out)

        # 3. Feed-Forward
        normed = self.norm3(op, hidden_states)
        ff_out = self.ff(op, normed)
        hidden_states = op.Add(hidden_states, ff_out)

        return hidden_states


# ---------------------------------------------------------------------------
# Output modules (PatchEmbed and TimestepEmbedding imported from components)
# ---------------------------------------------------------------------------


class _AdaLayerNormContinuous(nn.Module):
    """Adaptive LayerNorm with continuous conditioning (scale + shift).

    Used for the final output normalization. No learnable norm parameters
    (elementwise_affine=False in HF).

    HF weights: linear.weight, linear.bias (norm has no params)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = _LayerNormNoAffine(hidden_size, eps=eps)
        self.linear = _Linear(hidden_size, hidden_size * 2)
        self._silu = _SiLU()

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value, temb: ir.Value):
        emb = self._silu(op, temb)
        emb = self.linear(op, emb)
        # Split into scale and shift
        scale, shift = op.Split(emb, num_outputs=2, axis=-1, _outputs=2)
        # Norm → (1 + scale) * normed + shift
        normed = self.norm(op, hidden_states)
        one = op.Constant(value_float=1.0)
        normed = op.Mul(normed, op.Add(one, op.Unsqueeze(scale, [1])))
        return op.Add(normed, op.Unsqueeze(shift, [1]))


# ---------------------------------------------------------------------------
# Full HunyuanDiT model
# ---------------------------------------------------------------------------


class HunyuanDiT2DModel(nn.Module):
    """HunyuanDiT transformer denoiser for latent diffusion.

    Patch-based transformer with AdaLN-Shift conditioning, QK normalization,
    GEGLU feed-forward, and U-Net-style skip connections between first and
    second halves of the transformer blocks.

    The forward method accepts the standard denoising interface:
    (sample, timestep, encoder_hidden_states). For HunyuanDiT pipelines,
    encoder_hidden_states should contain pre-processed text conditioning
    (CLIP + projected T5 embeddings concatenated along sequence dim).

    Simplifications vs HF diffusers HunyuanDiT2DModel:
    - ``text_embedder`` (PixArtAlphaTextProjection) and
      ``text_embedding_padding`` are omitted — T5 projection should be
      done externally before passing encoder_hidden_states.
    - ``time_extra_emb`` is simplified to sinusoidal + MLP; the full HF
      version fuses pooled text, image_meta_size, and style embeddings.
    - ``image_rotary_emb`` (2D RoPE) is not applied — positional info
      comes from the patch embedding only.
    These modules' weights are handled by ``preprocess_weights()`` renames
    but the extra conditioning paths are not wired in the forward graph.

    HF diffusers class: HunyuanDiT2DModel
    """

    default_task: str = "denoising"
    category: str = "Diffusion"

    def __init__(self, config: HunyuanDiTConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        out_channels = config.in_channels * 2 if config.learn_sigma else config.in_channels
        self._out_channels = out_channels
        intermediate_size = int(hidden_size * config.mlp_ratio)

        # Patch embedding
        self.pos_embed = _PatchEmbed(config.in_channels, hidden_size, config.patch_size)

        # Timestep embedding (simplified from HF's combined embedding)
        self._time_proj_dim = 256
        self.time_embed = _TimestepEmbedding(self._time_proj_dim, hidden_size)

        # Transformer blocks with skip connections in second half
        num_layers = config.num_layers
        self.blocks = nn.ModuleList()
        for layer_idx in range(num_layers):
            has_skip = layer_idx > num_layers // 2
            self.blocks.append(
                _HunyuanDiTBlock(
                    hidden_size=hidden_size,
                    num_heads=config.num_attention_heads,
                    cross_attention_dim=config.cross_attention_dim,
                    intermediate_size=intermediate_size,
                    eps=config.norm_eps,
                    qk_norm=config.qk_norm,
                    skip=has_skip,
                )
            )

        # Output: AdaLN-Continuous → Linear → unpatchify
        self.norm_out = _AdaLayerNormContinuous(hidden_size, eps=config.norm_eps)
        self.proj_out = _Linear(
            hidden_size,
            config.patch_size * config.patch_size * out_channels,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        sample: ir.Value,
        timestep: ir.Value,
        encoder_hidden_states: ir.Value,
    ):
        """Forward pass.

        Args:
            op: ONNX op builder.
            sample: Noisy latent [batch, in_channels, height, width]
            timestep: Diffusion timestep [batch]
            encoder_hidden_states: Text conditioning
                [batch, seq_len, cross_attention_dim]

        Returns:
            noise_pred: [batch, out_channels, height, width]
        """
        num_layers = self.config.num_layers

        # Patch embedding: [B, C, H, W] → [B, num_patches, hidden]
        hidden_states = self.pos_embed(op, sample)

        # Timestep embedding: sinusoidal → MLP
        t_emb = self._get_timestep_embedding(op, timestep)
        temb = self.time_embed(op, t_emb)

        # Transformer blocks with U-Net skip connections
        skips: list = []
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx > num_layers // 2:
                # Second half: pop skip from first half
                skip = skips.pop()
                hidden_states = block(
                    op,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    skip_connection=skip,
                )
            else:
                hidden_states = block(
                    op,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                )

            # First half (excluding transition): save for skip connections
            if layer_idx < (num_layers // 2 - 1):
                skips.append(hidden_states)

        # Output normalization + projection
        hidden_states = self.norm_out(op, hidden_states, temb)
        hidden_states = self.proj_out(op, hidden_states)

        # Unpatchify: [B, num_patches, p*p*C] → [B, C, H, W]
        hidden_states = self._unpatchify(op, hidden_states, sample)

        return hidden_states

    def _get_timestep_embedding(self, op: builder.OpBuilder, timestep):
        """Sinusoidal timestep embedding."""
        half_dim = self._time_proj_dim // 2
        exponent = -math.log(10000.0) / half_dim
        freqs = np.exp(np.arange(half_dim) * exponent).astype(np.float32)
        freq_const = op.Constant(value_floats=freqs.tolist())
        t = op.Cast(timestep, to=1)
        t = op.Unsqueeze(t, [1])
        args = op.Mul(t, op.Unsqueeze(freq_const, [0]))
        return op.Concat(op.Cos(args), op.Sin(args), axis=-1)

    def _unpatchify(self, op: builder.OpBuilder, hidden_states, original):
        """Reshape patches back to spatial dimensions."""
        p = self.config.patch_size
        c = self._out_channels
        batch = op.Shape(original, start=0, end=1)
        h = op.Shape(original, start=2, end=3)
        w = op.Shape(original, start=3, end=4)
        h_patches = op.Div(h, op.Constant(value_ints=[p]))
        w_patches = op.Div(w, op.Constant(value_ints=[p]))

        # [B, num_patches, p*p*c] → [B, h_p, w_p, p, p, c]
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                h_patches,
                w_patches,
                op.Constant(value_ints=[p, p, c]),
                axis=0,
            ),
        )
        # [B, h_p, w_p, p, p, c] → [B, c, h_p, p, w_p, p]
        hidden_states = op.Transpose(hidden_states, perm=[0, 5, 1, 3, 2, 4])
        # [B, c, h_p*p, w_p*p] = [B, c, H, W]
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                op.Constant(value_ints=[c]),
                op.Mul(h_patches, op.Constant(value_ints=[p])),
                op.Mul(w_patches, op.Constant(value_ints=[p])),
                axis=0,
            ),
        )
        return hidden_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF diffusers weight names to ONNX parameter names.

        Key renames:
        - blocks.{i}.ff.net.0.proj → blocks.{i}.ff.geglu.proj
        - blocks.{i}.ff.net.2 → blocks.{i}.ff.linear_out
        - time_extra_emb.timestep_embedder → time_embed (simplified)
        """
        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            # GEGLU FFN renames
            new_key = new_key.replace(".ff.net.0.proj.", ".ff.geglu.proj.")
            new_key = new_key.replace(".ff.net.2.", ".ff.linear_out.")
            # Timestep embedding (simplified from HF's combined embedding)
            new_key = new_key.replace("time_extra_emb.timestep_embedder.", "time_embed.")
            new_state_dict[new_key] = value
        return new_state_dict
