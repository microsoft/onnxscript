# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""QwenImage transformer denoiser for text-to-image generation.

Architecture: double-stream joint attention transformer with RoPE,
AdaLayerNorm modulation, and GELU-approximate FFN. Similar to Flux
but with QwenImage-specific block design (no single-stream blocks).

HF diffusers class: QwenImageTransformer2DModel
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._diffusers_configs import QwenImageConfig
from mobius.components import (
    INT64_MAX,
)
from mobius.components import (
    LayerNormNoAffine as _LayerNormNoAffine,
)
from mobius.components import (
    Linear as _Linear,
)
from mobius.components import (
    SiLU as _SiLU,
)

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Model-specific building blocks
# ---------------------------------------------------------------------------


class _TimestepMLP(nn.Module):
    """Projects timestep embedding to hidden dim: Linear -> SiLU -> Linear."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = _Linear(in_channels, time_embed_dim)
        self.linear_2 = _Linear(time_embed_dim, time_embed_dim)

    def forward(self, op: builder.OpBuilder, sample: ir.Value):
        sample = self.linear_1(op, sample)
        sample = op.Mul(sample, op.Sigmoid(sample))  # SiLU
        sample = self.linear_2(op, sample)
        return sample


class _TimestepEmbedding(nn.Module):
    """Wraps _TimestepMLP with ``timestep_embedder`` nesting to match HF naming.

    HF: ``time_text_embed.timestep_embedder.linear_1.weight``
    """

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.timestep_embedder = _TimestepMLP(in_channels, time_embed_dim)

    def forward(self, op: builder.OpBuilder, sample: ir.Value):
        return self.timestep_embedder(op, sample)


class _AdaLayerNormOutput(nn.Module):
    """Final adaptive layer norm with scale+shift from timestep embedding.

    Uses non-affine LayerNorm (matching HF ``AdaLayerNormContinuous(elementwise_affine=False)``).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = _LayerNormNoAffine(hidden_size, eps=eps)
        self.linear = _Linear(hidden_size, hidden_size * 2)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value, timestep_emb: ir.Value):
        emb = op.Mul(timestep_emb, op.Sigmoid(timestep_emb))  # SiLU
        emb = self.linear(op, emb)
        shift, scale = op.Split(emb, num_outputs=2, axis=-1, _outputs=2)
        one = op.Constant(value_float=1.0)
        hidden_states = self.norm(op, hidden_states)
        hidden_states = op.Mul(hidden_states, op.Add(one, op.Unsqueeze(scale, [1])))
        hidden_states = op.Add(hidden_states, op.Unsqueeze(shift, [1]))
        return hidden_states


class _RMSNormQK(nn.Module):
    """RMS normalization for QK with learnable weight (used in QwenImage attention)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter((dim,))
        self._eps = eps

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.RMSNormalization(x, self.weight, axis=-1, epsilon=self._eps, stash_type=1)


class _RMSNorm(nn.Module):
    """RMS normalization with learnable weight (matches HF ``txt_norm.weight``)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter((dim,))
        self._eps = eps

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.RMSNormalization(x, self.weight, axis=-1, epsilon=self._eps, stash_type=1)


class _GELUGate(nn.Module):
    """Matches HF GEGLU wrapper: Linear stored as ``.proj`` sub-attribute."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = _Linear(in_features, out_features)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.Gelu(self.proj(op, x), approximate="tanh")


class _NoOpModule(nn.Module):
    """Placeholder for HF nn.Dropout (no parameters, identity at inference)."""

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return x


class _GeluApproxFFN(nn.Module):
    """Feed-forward network with GELU approximate activation.

    Matches diffusers ``FeedForward(activation_fn="gelu-approximate")``
    naming: ``net.0.proj.weight``, ``net.2.weight``.

    HF structure: ``nn.Sequential(GEGLU(Linear), Dropout, Linear)``.
    """

    def __init__(self, dim: int, dim_out: int | None = None):
        super().__init__()
        dim_out = dim_out or dim
        inner_dim = dim * 4
        self.net = nn.Sequential(
            _GELUGate(dim, inner_dim),
            _NoOpModule(),
            _Linear(inner_dim, dim_out),
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # Sequential chains: GELUGate → NoOp → Linear
        return self.net(op, x)


class _QwenImageJointAttention(nn.Module):
    """Joint attention for QwenImage double-stream blocks.

    Computes Q/K/V projections for both image and text streams,
    concatenates for joint attention, then splits back.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        # Image stream projections
        self.to_q = _Linear(dim, inner_dim)
        self.to_k = _Linear(dim, inner_dim)
        self.to_v = _Linear(dim, inner_dim)
        self.to_out = nn.Sequential(_Linear(inner_dim, dim))

        # Text stream projections (added_kv)
        self.add_q_proj = _Linear(dim, inner_dim)
        self.add_k_proj = _Linear(dim, inner_dim)
        self.add_v_proj = _Linear(dim, inner_dim)
        self.to_add_out = _Linear(inner_dim, dim)

        # QK normalization
        self.norm_q = _RMSNormQK(head_dim, eps=eps)
        self.norm_k = _RMSNormQK(head_dim, eps=eps)
        self.norm_added_q = _RMSNormQK(head_dim, eps=eps)
        self.norm_added_k = _RMSNormQK(head_dim, eps=eps)

    def forward(self, op: builder.OpBuilder, img_hidden: ir.Value, txt_hidden: ir.Value):
        batch = op.Shape(img_hidden, start=0, end=1)

        # Image stream QKV
        q = self.to_q(op, img_hidden)
        k = self.to_k(op, img_hidden)
        v = self.to_v(op, img_hidden)

        # Text stream QKV
        add_q = self.add_q_proj(op, txt_hidden)
        add_k = self.add_k_proj(op, txt_hidden)
        add_v = self.add_v_proj(op, txt_hidden)

        # Reshape to [batch, seq, heads, head_dim]
        head_shape = op.Concat(
            batch,
            op.Constant(value_ints=[-1]),
            op.Constant(value_ints=[self.num_heads, self.head_dim]),
            axis=0,
        )
        q = op.Reshape(q, head_shape)
        k = op.Reshape(k, head_shape)
        v = op.Reshape(v, head_shape)
        add_q = op.Reshape(add_q, head_shape)
        add_k = op.Reshape(add_k, head_shape)
        add_v = op.Reshape(add_v, head_shape)

        # QK normalization
        q = self.norm_q(op, q)
        k = self.norm_k(op, k)
        add_q = self.norm_added_q(op, add_q)
        add_k = self.norm_added_k(op, add_k)

        # Concatenate image + text for joint attention
        # [batch, img_seq + txt_seq, heads, head_dim]
        q_cat = op.Concat(q, add_q, axis=1)
        k_cat = op.Concat(k, add_k, axis=1)
        v_cat = op.Concat(v, add_v, axis=1)

        # Transpose to [batch, heads, seq, head_dim]
        q_cat = op.Transpose(q_cat, perm=[0, 2, 1, 3])
        k_cat = op.Transpose(k_cat, perm=[0, 2, 1, 3])
        v_cat = op.Transpose(v_cat, perm=[0, 2, 1, 3])

        # Scaled dot-product attention
        scale = float(self.head_dim**-0.5)
        attn_out = op.Attention(
            q_cat,
            k_cat,
            v_cat,
            scale=scale,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
        )

        # [batch, heads, total_seq, head_dim] -> [batch, total_seq, heads, head_dim]
        attn_out = op.Transpose(attn_out, perm=[0, 2, 1, 3])

        # Split back into image and text parts
        img_seq_len = op.Shape(img_hidden, start=1, end=2)
        img_attn = op.Slice(
            attn_out,
            op.Constant(value_ints=[0]),
            img_seq_len,
            op.Constant(value_ints=[1]),
        )
        txt_attn = op.Slice(
            attn_out,
            img_seq_len,
            op.Constant(value_ints=[INT64_MAX]),
            op.Constant(value_ints=[1]),
        )

        # Reshape and project back
        out_shape = op.Concat(
            batch,
            op.Constant(value_ints=[-1]),
            op.Constant(value_ints=[self.num_heads * self.head_dim]),
            axis=0,
        )
        img_out = self.to_out(op, op.Reshape(img_attn, out_shape))
        txt_out = self.to_add_out(op, op.Reshape(txt_attn, out_shape))

        return img_out, txt_out


class _QwenImageTransformerBlock(nn.Module):
    """QwenImage double-stream transformer block.

    Architecture (per block):
    - Image: AdaLN → joint attention → residual + gate → AdaLN → FFN → residual + gate
    - Text:  AdaLN → joint attention → residual + gate → AdaLN → FFN → residual + gate
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        # Image stream modulation: nn.Sequential(SiLU, Linear(dim, 6*dim))
        # Matches HF nn.Sequential(nn.SiLU(), nn.Linear(...)) → img_mod.1.weight
        self.img_mod = nn.Sequential(_SiLU(), _Linear(dim, 6 * dim))
        self.img_norm1 = _LayerNormNoAffine(dim, eps=eps)
        self.attn = _QwenImageJointAttention(dim, num_heads, head_dim, eps=eps)
        self.img_norm2 = _LayerNormNoAffine(dim, eps=eps)
        self.img_mlp = _GeluApproxFFN(dim)

        # Text stream modulation
        self.txt_mod = nn.Sequential(_SiLU(), _Linear(dim, 6 * dim))
        self.txt_norm1 = _LayerNormNoAffine(dim, eps=eps)
        self.txt_norm2 = _LayerNormNoAffine(dim, eps=eps)
        self.txt_mlp = _GeluApproxFFN(dim)

    def forward(
        self, op: builder.OpBuilder, img_hidden: ir.Value, txt_hidden: ir.Value, temb: ir.Value
    ):
        one = op.Constant(value_float=1.0)

        # Modulation parameters (SiLU → Linear → chunk)
        img_mod = self.img_mod(op, temb)
        img_mod1, img_mod2 = op.Split(img_mod, num_outputs=2, axis=-1, _outputs=2)
        img_shift1, img_scale1, img_gate1 = op.Split(
            img_mod1, num_outputs=3, axis=-1, _outputs=3
        )
        img_shift2, img_scale2, img_gate2 = op.Split(
            img_mod2, num_outputs=3, axis=-1, _outputs=3
        )

        txt_mod = self.txt_mod(op, temb)
        txt_mod1, txt_mod2 = op.Split(txt_mod, num_outputs=2, axis=-1, _outputs=2)
        txt_shift1, txt_scale1, txt_gate1 = op.Split(
            txt_mod1, num_outputs=3, axis=-1, _outputs=3
        )
        txt_shift2, txt_scale2, txt_gate2 = op.Split(
            txt_mod2, num_outputs=3, axis=-1, _outputs=3
        )

        # Image stream: norm1 + modulate
        img_normed = self.img_norm1(op, img_hidden)
        img_modulated = op.Add(
            op.Mul(img_normed, op.Add(one, op.Unsqueeze(img_scale1, [1]))),
            op.Unsqueeze(img_shift1, [1]),
        )

        # Text stream: norm1 + modulate
        txt_normed = self.txt_norm1(op, txt_hidden)
        txt_modulated = op.Add(
            op.Mul(txt_normed, op.Add(one, op.Unsqueeze(txt_scale1, [1]))),
            op.Unsqueeze(txt_shift1, [1]),
        )

        # Joint attention
        img_attn, txt_attn = self.attn(op, img_modulated, txt_modulated)

        # Residual + gate
        img_hidden = op.Add(img_hidden, op.Mul(op.Unsqueeze(img_gate1, [1]), img_attn))
        txt_hidden = op.Add(txt_hidden, op.Mul(op.Unsqueeze(txt_gate1, [1]), txt_attn))

        # Image stream: norm2 + modulate + FFN
        img_normed2 = self.img_norm2(op, img_hidden)
        img_modulated2 = op.Add(
            op.Mul(img_normed2, op.Add(one, op.Unsqueeze(img_scale2, [1]))),
            op.Unsqueeze(img_shift2, [1]),
        )
        img_ff = self.img_mlp(op, img_modulated2)
        img_hidden = op.Add(img_hidden, op.Mul(op.Unsqueeze(img_gate2, [1]), img_ff))

        # Text stream: norm2 + modulate + FFN
        txt_normed2 = self.txt_norm2(op, txt_hidden)
        txt_modulated2 = op.Add(
            op.Mul(txt_normed2, op.Add(one, op.Unsqueeze(txt_scale2, [1]))),
            op.Unsqueeze(txt_shift2, [1]),
        )
        txt_ff = self.txt_mlp(op, txt_modulated2)
        txt_hidden = op.Add(txt_hidden, op.Mul(op.Unsqueeze(txt_gate2, [1]), txt_ff))

        return img_hidden, txt_hidden


class QwenImageTransformer2DModel(nn.Module):
    """QwenImage transformer denoiser for text-to-image generation.

    Architecture:
    1. Patch embedding: Linear(in_channels → hidden_dim)
    2. Text embedding: RMSNorm → Linear(joint_attention_dim → hidden_dim)
    3. Timestep embedding: sinusoidal → MLP
    4. N x QwenImageTransformerBlock (double-stream joint attention)
    5. AdaLN output + unpatchify

    Forward: sample(latent) + timestep + encoder_hidden_states → noise_pred
    """

    default_task: str = "denoising"
    category: str = "Diffusion"

    def __init__(self, config: QwenImageConfig):
        super().__init__()
        self.config = config
        hidden_size = config.num_attention_heads * config.attention_head_dim

        self.img_in = _Linear(config.in_channels, hidden_size)
        self.txt_norm = _RMSNorm(config.joint_attention_dim, eps=config.norm_eps)
        self.txt_in = _Linear(config.joint_attention_dim, hidden_size)
        self.time_text_embed = _TimestepEmbedding(hidden_size, hidden_size)

        self.transformer_blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            self.transformer_blocks.append(
                _QwenImageTransformerBlock(
                    hidden_size,
                    config.num_attention_heads,
                    config.attention_head_dim,
                    eps=config.norm_eps,
                )
            )

        self.norm_out = _AdaLayerNormOutput(hidden_size, eps=config.norm_eps)
        self.proj_out = _Linear(
            hidden_size,
            config.patch_size * config.patch_size * config.out_channels,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        sample: ir.Value,
        timestep: ir.Value,
        encoder_hidden_states: ir.Value,
    ):
        config = self.config
        hidden_size = config.num_attention_heads * config.attention_head_dim

        # Patch embed: [batch, channels, H, W] → [batch, num_patches, hidden_size]
        hidden_states = self.img_in(op, sample)

        # Text: RMSNorm → Linear
        txt = self.txt_norm(op, encoder_hidden_states)
        txt = self.txt_in(op, txt)

        # Timestep embedding
        t_emb = self._get_timestep_embedding(op, timestep, hidden_size)
        temb = self.time_text_embed(op, t_emb)

        # Double-stream transformer blocks
        for block in self.transformer_blocks:
            hidden_states, txt = block(op, hidden_states, txt, temb)

        # Output: AdaLN + projection
        hidden_states = self.norm_out(op, hidden_states, temb)
        output = self.proj_out(op, hidden_states)

        return output

    def _get_timestep_embedding(self, op: builder.OpBuilder, timestep, dim):
        """Compute sinusoidal timestep embedding."""
        half_dim = 256 // 2
        exponent = -math.log(10000.0) / half_dim
        freqs = np.exp(np.arange(half_dim) * exponent).astype(np.float32)
        freq_const = op.Constant(value_floats=freqs.tolist())
        t = op.Cast(timestep, to=1)  # to float
        t = op.Mul(t, op.Constant(value_float=1000.0))
        t = op.Unsqueeze(t, [1])
        args = op.Mul(t, op.Unsqueeze(freq_const, [0]))
        return op.Concat(op.Sin(args), op.Cos(args), axis=-1)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """No renaming needed — parameter names match diffusers directly."""
        return state_dict
