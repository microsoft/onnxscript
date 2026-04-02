# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""CLAP: Contrastive Language-Audio Pretraining (laion/clap-htsat-fused).

Architecture:
  - Text encoder: RoBERTa-style BERT (12 layers, hidden=768)
  - Audio encoder: HTSAT (Hierarchical Token-Semantic Audio Transformer)
    — a 4-stage Swin Transformer on mel spectrograms
  - Projection: linear1 → ReLU → linear2 (768 → 768 → 512)

Two separate models are exported:
  ClapTextModel  (model_type: clap_text_model)  → text_embeds (batch, 512)
  ClapAudioModel (model_type: clap_audio_model) → audio_embeds (batch, 512)

The non-fusion path is implemented (single mel-spectrogram crop, no long-audio
fusion mechanism).  Users should pass a mel spectrogram padded/truncated to
exactly ``spec_size * freq_ratio = 1024`` time frames and ``num_mel_bins = 64``
frequency bins.

Reference: https://huggingface.co/laion/clap-htsat-fused
HuggingFace classes: ClapModel, ClapTextModel, ClapAudioModel
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._activations import ACT2FN
from mobius.components._common import LayerNorm, Linear
from mobius.components._conv import Conv2d
from mobius.models.bert import _BertEmbeddings, _BertEncoder

# ---------------------------------------------------------------------------
# Shared: projection layer
# ---------------------------------------------------------------------------


class _ClapProjectionLayer(nn.Module):
    """Two-layer MLP projection: linear1 → ReLU → linear2.

    Matches HF ``ClapProjectionLayer`` weight naming: ``linear1``, ``linear2``.
    ``activation`` (ReLU) has no parameters so it is not a sub-module.
    """

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.linear1 = Linear(in_features, hidden, bias=True)
        self.linear2 = Linear(hidden, out_features, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.linear1(op, x)
        x = op.Relu(x)
        return self.linear2(op, x)


# ---------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------


class _ClapTextPooler(nn.Module):
    """CLS-token pooler: Linear + Tanh.

    Matches HF ``BertPooler`` with attribute ``dense``.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = Linear(hidden_size, hidden_size, bias=True)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # Take [CLS] token (first token) and project with tanh
        cls = op.Slice(hidden_states, [0], [1], [1])  # (batch, 1, hidden)
        cls = op.Squeeze(cls, [1])  # (batch, hidden)
        return op.Tanh(self.dense(op, cls))


class _ClapTextEncoder(nn.Module):
    """BERT-style text encoder with CLS pooler for CLAP.

    Contains ``embeddings``, ``encoder``, ``pooler`` sub-modules matching HF
    ``ClapTextModel`` weight naming exactly.  ``type_vocab_size=1`` because the
    CLAP text model (RoBERTa) uses a single token type embedding (all zeros).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embeddings = _BertEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=1,
            layer_norm_eps=config.rms_norm_eps,
            pad_token_id=config.pad_token_id or 1,
        )
        self.encoder = _BertEncoder(config)
        self.pooler = _ClapTextPooler(config.hidden_size)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
    ) -> ir.Value:
        # CLAP processor does not produce token_type_ids — use all-zero placeholders
        batch = op.Shape(input_ids, start=0, end=1)
        seq_len = op.Shape(input_ids, start=1, end=2)
        token_type_ids = op.ConstantOfShape(
            op.Concat(batch, seq_len, axis=0),
        )  # (batch, seq_len) zeros, int64
        token_type_ids = op.Cast(token_type_ids, to=7)  # int64

        hidden_states = self.embeddings(op, input_ids, token_type_ids)
        hidden_states = self.encoder(op, hidden_states, attention_mask)
        # pooler: CLS → (batch, hidden)
        return self.pooler(op, hidden_states)


class ClapTextModel(nn.Module):
    """CLAP text model: RoBERTa-style text encoder + projection.

    Inputs:
        input_ids:      ``(batch, seq_len)`` token IDs
        attention_mask: ``(batch, seq_len)`` 1 for real tokens, 0 for padding

    Output:
        text_embeds: ``(batch, 512)`` normalised text embeddings

    HuggingFace class: ``ClapTextModel``
    """

    default_task: str = "feature-extraction"
    category: str = "Encoder"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.text_model = _ClapTextEncoder(config)
        self.projection = _ClapProjectionLayer(
            config.hidden_size,
            config.hidden_size,
            config.projection_dim,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        token_type_ids: ir.Value,  # unused; kept for FeatureExtractionTask compat
    ) -> ir.Value:
        pooled = self.text_model(op, input_ids, attention_mask)
        return self.projection(op, pooled)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map weights from combined ClapModel checkpoint to ClapTextModel.

        - Keeps ``text_model.*`` keys unchanged (exact attribute path match)
        - Maps ``text_projection.*`` → ``projection.*``
        - Drops all other keys (audio_model, audio_projection, logit_scale)
        """
        result: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            if name.startswith("text_model."):
                result[name] = tensor
            elif name.startswith("text_projection."):
                result["projection." + name[len("text_projection.") :]] = tensor
        return result


# ---------------------------------------------------------------------------
# Audio model — HTSAT Swin Transformer
# ---------------------------------------------------------------------------


# ---- Mel-frequency batch normalisation (eval mode) ----


class _ClapAudioBatchNorm(nn.Module):
    """BatchNorm2d in eval mode for mel-spectrogram frequency normalisation.

    Operates on the frequency (last) dimension of shape
    ``(batch, time, freq)``.  All statistics are stored as parameters so they
    become ONNX initialisers.

    Parameters (loaded from HF ``batch_norm.*`` weights):
        weight, bias, running_mean, running_var — each ``(num_mel_bins,)``
    """

    EPS = 1e-5

    def __init__(self, num_mel_bins: int):
        super().__init__()
        self.weight = nn.Parameter((num_mel_bins,))
        self.bias = nn.Parameter((num_mel_bins,))
        self.running_mean = nn.Parameter((num_mel_bins,))
        self.running_var = nn.Parameter((num_mel_bins,))

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # x: (batch, time, freq) — normalise along freq dimension
        # eval-mode BN: (x - mean) / sqrt(var + eps) * weight + bias
        # All parameters broadcast over batch and time dims automatically
        x_centred = op.Sub(x, self.running_mean)
        denom = op.Sqrt(op.Add(self.running_var, op.Constant(value_float=self.EPS)))
        return op.Add(op.Mul(op.Div(x_centred, denom), self.weight), self.bias)


# ---- Patch embedding ----


class _ClapAudioPatchEmbed(nn.Module):
    """4x4 Conv2d patch embedding with LayerNorm (non-fusion path).

    Converts a ``(batch, 1, img_size, img_size)`` "audio image" into a
    sequence of patch embeddings: ``(batch, H*W, hidden_size)``.

    Parameters:
        proj: Conv2d(1, patch_embeds_hidden_size, 4, 4)
        norm: LayerNorm(patch_embeds_hidden_size)
    """

    def __init__(self, img_size: int, patch_size: int, hidden_size: int):
        super().__init__()
        self._h = img_size // patch_size  # spatial height after patching
        self._w = img_size // patch_size  # spatial width after patching
        self.proj = Conv2d(1, hidden_size, patch_size, patch_size)
        self.norm = LayerNorm(hidden_size, eps=1e-5)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # x: (batch, 1, img_size, img_size)
        x = self.proj(op, x)  # (batch, C, H, W) where H=W=img_size/patch_size
        # Flatten spatial dims and transpose: (batch, H*W, C)
        batch = op.Shape(x, start=0, end=1)
        x = op.Reshape(x, op.Concat(batch, [self._h * self._w, -1], axis=0))
        return self.norm(op, x)


# ---- Swin attention ----


def _precompute_relative_position_index(window_size: int) -> np.ndarray:
    """Compute the (window^2, window^2) relative position index table."""
    wh = ww = window_size
    coords_h = np.arange(wh)
    coords_w = np.arange(ww)
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"), axis=0)  # (2, wH, wW)
    coords_flat = coords.reshape(2, -1)  # (2, wH*wW)
    rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, wH*wW, wH*wW)
    rel = rel.transpose(1, 2, 0)  # (wH*wW, wH*wW, 2)
    rel[:, :, 0] += wh - 1  # shift to non-negative
    rel[:, :, 1] += ww - 1
    rel[:, :, 0] *= 2 * ww - 1
    return rel.sum(axis=-1).astype(np.int64)  # (wH*wW, wH*wW)


def _precompute_shift_attn_mask(
    h: int, w: int, window_size: int, shift_size: int
) -> np.ndarray | None:
    """Compute the shifted-window attention mask (nWindows, wH*wW, wH*wW).

    Returns ``None`` if ``shift_size == 0``.
    """
    if shift_size == 0:
        return None
    img_mask = np.zeros((1, h, w, 1), dtype=np.float32)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = cnt
            cnt += 1
    # Partition img_mask into windows
    nh, nw = h // window_size, w // window_size
    mw = img_mask.reshape(1, nh, window_size, nw, window_size, 1)
    mw = mw.transpose(0, 1, 3, 2, 4, 5).reshape(nh * nw, window_size * window_size)
    attn_mask = mw[:, np.newaxis, :] - mw[:, :, np.newaxis]  # (nWindows, wH*wW, wH*wW)
    attn_mask = np.where(attn_mask != 0, -100.0, 0.0).astype(np.float32)
    return attn_mask  # (nWindows, wH*wW, wH*wW)


class _ClapAudioSelfAttention(nn.Module):
    """Swin windowed self-attention with relative position bias.

    Implements the ``ClapAudioSelfAttention`` HF class with matching
    attribute names so that ``preprocess_weights`` needs no renaming.

    Parameters:
        query, key, value: Linear(dim, dim, bias=True)
        relative_position_bias_table: (2*wH-1)*(2*wW-1), num_heads)
        relative_position_index:      (wH*wW, wH*wW) precomputed index buffer
    """

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = dim // num_heads
        self._n_tokens = window_size * window_size
        self._scale = (dim // num_heads) ** -0.5
        self.query = Linear(dim, dim, bias=True)
        self.key = Linear(dim, dim, bias=True)
        self.value = Linear(dim, dim, bias=True)
        # Learnable relative position bias table
        n_rel = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter((n_rel, num_heads))
        # Precomputed index (buffer in HF; treated as parameter for ONNX)
        rpi = _precompute_relative_position_index(window_size)
        self.relative_position_index = nn.Parameter(
            rpi.shape,
            data=ir.tensor(rpi),
        )

    def forward(
        self,
        op: builder.OpBuilder,
        x: ir.Value,
        attn_mask: np.ndarray | None,
    ) -> ir.Value:
        # x: (batch_windows, window_tokens, dim)
        nh = self._num_heads
        hd = self._head_dim
        n = self._n_tokens

        batch_wins = op.Shape(x, start=0, end=1)
        q = op.Reshape(self.query(op, x), op.Concat(batch_wins, [n, nh, hd], axis=0))
        k = op.Reshape(self.key(op, x), op.Concat(batch_wins, [n, nh, hd], axis=0))
        v = op.Reshape(self.value(op, x), op.Concat(batch_wins, [n, nh, hd], axis=0))

        # Transpose to (batch_wins, nh, n, hd)
        q = op.Transpose(q, perm=[0, 2, 1, 3])
        k = op.Transpose(k, perm=[0, 2, 1, 3])
        v = op.Transpose(v, perm=[0, 2, 1, 3])

        # Scaled dot-product attention
        attn = op.Mul(
            op.MatMul(q, op.Transpose(k, perm=[0, 1, 3, 2])),
            op.Constant(value_float=self._scale),
        )  # (batch_wins, nh, n, n)

        # Relative position bias: (1, nh, n, n)
        flat_idx = op.Reshape(self.relative_position_index, [-1])  # (n*n,)
        bias = op.Gather(self.relative_position_bias_table, flat_idx, axis=0)  # (n*n, nh)
        bias = op.Reshape(bias, [n, n, nh])
        bias = op.Transpose(bias, perm=[2, 0, 1])  # (nh, n, n)
        bias = op.Unsqueeze(bias, [0])  # (1, nh, n, n)
        attn = op.Add(attn, bias)

        # Add shifted-window attention mask (precomputed constant)
        if attn_mask is not None:
            # attn_mask: (num_windows, n, n) — add head dim → (num_windows, 1, n, n)
            mask_const = op.Constant(value=ir.tensor(attn_mask[:, np.newaxis]))
            # Reshape attn to (batch, num_windows, nh, n, n) and add mask
            n_wins = attn_mask.shape[0]
            b = op.Div(batch_wins, op.Constant(value_int=n_wins))
            attn = op.Reshape(attn, op.Concat(b, [n_wins, nh, n, n], axis=0))
            attn = op.Add(attn, mask_const)
            attn = op.Reshape(attn, op.Concat(batch_wins, [nh, n, n], axis=0))

        attn = op.Softmax(attn, axis=-1)

        # Weighted sum: (batch_wins, nh, n, hd)
        out = op.MatMul(attn, v)
        # (batch_wins, n, nh, hd) → (batch_wins, n, dim)
        out = op.Transpose(out, perm=[0, 2, 1, 3])
        return op.Reshape(out, op.Concat(batch_wins, [n, nh * hd], axis=0))


class _ClapAudioAttentionOutput(nn.Module):
    """Attention output projection (no LayerNorm — block-level pre/post norms)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dense = Linear(dim, dim, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return self.dense(op, x)


class _ClapAudioAttention(nn.Module):
    """Combined attention: ``self`` + ``output`` sub-modules (HF naming)."""

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self_attn = _ClapAudioSelfAttention(dim, num_heads, window_size)
        # Attribute named 'self' matches HF weight key prefix 'attention.self.*'
        self.self = self_attn
        self.output = _ClapAudioAttentionOutput(dim)

    def forward(
        self,
        op: builder.OpBuilder,
        x: ir.Value,
        attn_mask: np.ndarray | None,
    ) -> ir.Value:
        attn_out = self.self(op, x, attn_mask)
        return self.output(op, attn_out)


class _ClapAudioIntermediate(nn.Module):
    """FFN up-projection with GELU activation (HF naming: ``dense``)."""

    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.dense = Linear(dim, intermediate_size, bias=True)
        self._gelu = ACT2FN["gelu"]

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return self._gelu(op, self.dense(op, x))


class _ClapAudioOutput(nn.Module):
    """FFN down-projection (HF naming: ``dense``)."""

    def __init__(self, intermediate_size: int, dim: int):
        super().__init__()
        self.dense = Linear(intermediate_size, dim, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return self.dense(op, x)


# ---- Swin block ----


class _ClapSwinBlock(nn.Module):
    """One Swin Transformer block with optional cyclic shift.

    Attribute names match HF ``ClapAudioSwinTransformerBlock``:
        layernorm_before, attention, layernorm_after, intermediate, output

    The ``drop_path`` is Identity at ``drop_path_rate=0.0`` and is omitted.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        h: int,
        w: int,
        window_size: int,
        shift_size: int,
        intermediate_multiplier: int = 4,
    ):
        super().__init__()
        self._H = h
        self._W = w
        self._window_size = window_size
        self._shift_size = shift_size
        self._n_tokens = window_size * window_size

        intermediate_size = dim * intermediate_multiplier
        self.layernorm_before = LayerNorm(dim, eps=1e-5)
        self.attention = _ClapAudioAttention(dim, num_heads, window_size)
        self.layernorm_after = LayerNorm(dim, eps=1e-5)
        self.intermediate = _ClapAudioIntermediate(dim, intermediate_size)
        self.output = _ClapAudioOutput(intermediate_size, dim)

        # Precompute shift attention mask (None if shift_size==0)
        self._attn_mask = _precompute_shift_attn_mask(h, w, window_size, shift_size)

    def _cyclic_shift(self, op: builder.OpBuilder, x: ir.Value, neg: bool) -> ir.Value:
        """Cyclic shift of (batch, h, w, c) along h and w dimensions."""
        s = self._shift_size
        h, w = self._H, self._W
        if neg:
            # Shift by -s: x[s:, :] + x[:s, :]
            a = op.Slice(x, [s], [h], [1])
            b = op.Slice(x, [0], [s], [1])
            x = op.Concat(a, b, axis=1)
            a = op.Slice(x, [s], [w], [2])
            b = op.Slice(x, [0], [s], [2])
            return op.Concat(a, b, axis=2)
        else:
            # Reverse shift by +s: x[h-s:, :] + x[:h-s, :]
            a = op.Slice(x, [h - s], [h], [1])
            b = op.Slice(x, [0], [h - s], [1])
            x = op.Concat(a, b, axis=1)
            a = op.Slice(x, [w - s], [w], [2])
            b = op.Slice(x, [0], [w - s], [2])
            return op.Concat(a, b, axis=2)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # hidden_states: (batch, H*W, C)
        h, w = self._H, self._W
        ws = self._window_size
        nh, nw = h // ws, w // ws
        n_tokens = self._n_tokens

        batch = op.Shape(hidden_states, start=0, end=1)
        shortcut = hidden_states

        x = self.layernorm_before(op, hidden_states)

        # Reshape sequence to 2D spatial: (batch, h, w, C)
        x = op.Reshape(x, op.Concat(batch, [h, w, -1], axis=0))

        # Optional cyclic shift
        if self._shift_size > 0:
            x = self._cyclic_shift(op, x, neg=True)

        # Window partition: (batch, h, w, C) → (batch*nh*nw, wh*ww, C)
        # Reshape to (batch, nh, wh, nw, ww, C) → transpose → reshape
        x = op.Reshape(x, op.Concat(batch, [nh, ws, nw, ws, -1], axis=0))
        x = op.Transpose(x, perm=[0, 1, 3, 2, 4, 5])  # (batch, nh, nw, wh, ww, C)
        x = op.Reshape(
            x,
            op.Concat(
                op.Mul(batch, op.Constant(value_int=nh * nw)),
                [n_tokens, -1],
                axis=0,
            ),
        )  # (batch*nh*nw, wh*ww, C)

        # Self-attention
        x = self.attention(op, x, self._attn_mask)

        # Window unpartition: → (batch, h, w, C)
        x = op.Reshape(x, op.Concat(batch, [nh, nw, ws, ws, -1], axis=0))
        x = op.Transpose(x, perm=[0, 1, 3, 2, 4, 5])  # (batch, nh, wh, nw, ww, C)
        x = op.Reshape(x, op.Concat(batch, [h, w, -1], axis=0))

        # Reverse cyclic shift
        if self._shift_size > 0:
            x = self._cyclic_shift(op, x, neg=False)

        # Flatten back to sequence: (batch, h*w, C)
        x = op.Reshape(x, op.Concat(batch, [h * w, -1], axis=0))

        # First residual (drop_path is Identity)
        hidden_states = op.Add(shortcut, x)

        # FFN
        layer_out = self.layernorm_after(op, hidden_states)
        layer_out = self.intermediate(op, layer_out)
        layer_out = self.output(op, layer_out)

        # Second residual
        return op.Add(hidden_states, layer_out)


# ---- Patch merging (downsample) ----


class _ClapPatchMerging(nn.Module):
    """Patch merging: combine 2x2 patches → double channels, halve spatial.

    Matches HF ``ClapAudioPatchMerging`` weight naming: ``norm``, ``reduction``.
    """

    def __init__(self, h: int, w: int, in_dim: int):
        super().__init__()
        self._H = h
        self._W = w
        self._in_dim = in_dim
        self.norm = LayerNorm(4 * in_dim, eps=1e-5)
        self.reduction = Linear(4 * in_dim, 2 * in_dim, bias=False)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # x: (batch, H*W, C) — merge 2x2 patches
        h, w, c = self._H, self._W, self._in_dim
        batch = op.Shape(x, start=0, end=1)

        x = op.Reshape(x, op.Concat(batch, [h, w, c], axis=0))

        # Gather the 4 patch groups using stride-2 Slice on axes 1 (H) and 2 (W):
        # op.Slice(data, starts, ends, axes, steps) — ONNX Slice 5-input form
        # x0: even rows, even cols → (batch, H/2, W/2, C)
        x0 = op.Slice(op.Slice(x, [0], [h], [1], [2]), [0], [w], [2], [2])
        x1 = op.Slice(op.Slice(x, [1], [h], [1], [2]), [0], [w], [2], [2])
        x2 = op.Slice(op.Slice(x, [0], [h], [1], [2]), [1], [w], [2], [2])
        x3 = op.Slice(op.Slice(x, [1], [h], [1], [2]), [1], [w], [2], [2])

        # Concatenate along channel dim: (batch, H/2, W/2, 4C)
        x = op.Concat(x0, x1, x2, x3, axis=-1)

        # Flatten spatial: (batch, H/2*W/2, 4C)
        x = op.Reshape(x, op.Concat(batch, [(h // 2) * (w // 2), -1], axis=0))

        x = self.norm(op, x)
        return self.reduction(op, x)


# ---- HTSAT encoder stage ----


class _ClapAudioEncoderStage(nn.Module):
    """One HTSAT stage: a sequence of Swin blocks + optional patch merging.

    Attribute names match HF ``ClapAudioSwinStage``: ``blocks``, ``downsample``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        h: int,
        w: int,
        depth: int,
        window_size: int,
        downsample: bool,
    ):
        super().__init__()
        # For stage where H == window_size, both blocks use shift_size=0
        effective_shifts = [
            0 if (min(h, w) <= window_size) else (0 if i % 2 == 0 else window_size // 2)
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(
            [
                _ClapSwinBlock(dim, num_heads, h, w, window_size, shift)
                for shift in effective_shifts
            ]
        )
        if downsample:
            self.downsample = _ClapPatchMerging(h, w, dim)
        else:
            # Placeholder so the attribute always exists (no weights loaded)
            self.downsample = None  # type: ignore[assignment]

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        for block in self.blocks:
            hidden_states = block(op, hidden_states)
        if self.downsample is not None:
            hidden_states = self.downsample(op, hidden_states)
        return hidden_states


# ---- Full HTSAT encoder ----


class _ClapAudioEncoder(nn.Module):
    """HTSAT (Hierarchical Token-Semantic Audio Transformer) encoder.

    Non-fusion path: accepts a single mel-spectrogram crop.

    Forward:
        input_features: ``(batch, time, num_mel_bins)``
            where ``time = spec_size x freq_ratio = 1024`` and
            ``num_mel_bins = 64``.

    Output:
        audio_features: ``(batch, hidden_size)`` — mean-pooled patch embeddings.

    Attribute names match HF ``ClapAudioEncoder`` exactly:
        ``batch_norm``, ``patch_embed``, ``layers``, ``norm``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        ac = config.audio
        assert ac is not None, "ClapAudioEncoder requires an AudioConfig"

        self._spec_size = ac.spec_size
        self._num_mel_bins = ac.num_mel_bins
        self._freq_ratio = 4  # hard-coded in HTSAT
        self._patch_size = ac.patch_size
        img_size = ac.spec_size  # 256 — the "image" is spec_size x spec_size

        # After patch_embed: h = w = img_size / patch_size = 64
        h0 = img_size // self._patch_size
        w0 = img_size // self._patch_size

        depths = ac.depths  # [2, 2, 6, 2]
        heads = ac.num_attention_heads  # [4, 8, 16, 32]
        window_size = ac.window_size  # 8
        patch_embeds_hidden_size = ac.patch_embeds_hidden_size  # 96

        self.batch_norm = _ClapAudioBatchNorm(ac.num_mel_bins)
        self.patch_embed = _ClapAudioPatchEmbed(
            img_size, self._patch_size, patch_embeds_hidden_size
        )

        stages = []
        h, w = h0, w0
        for i, (depth, n_heads) in enumerate(zip(depths, heads)):
            dim = patch_embeds_hidden_size * (2**i)
            downsample = i < len(depths) - 1
            stages.append(
                _ClapAudioEncoderStage(dim, n_heads, h, w, depth, window_size, downsample)
            )
            if downsample:
                h, w = h // 2, w // 2

        self.layers = nn.ModuleList(stages)
        # Final hidden dim after all stages
        final_dim = patch_embeds_hidden_size * (2 ** (len(depths) - 1))
        self.norm = LayerNorm(final_dim, eps=1e-5)

    def forward(self, op: builder.OpBuilder, input_features: ir.Value) -> ir.Value:
        """Forward pass: mel spectrogram → pooled audio features.

        Args:
            input_features: ``(batch, time=1024, freq=64)``

        Returns:
            ``(batch, hidden_size=768)``
        """
        batch = op.Shape(input_features, start=0, end=1)
        freq_ratio = self._freq_ratio
        spec_size = self._spec_size  # 256
        # time = spec_size * freq_ratio = 1024 (assumed by caller)

        # --- Mel batch normalisation (operates on freq axis) ---
        x = self.batch_norm(op, input_features)  # (batch, 1024, 64)

        # --- reshape_mel2img: (batch, 1024, 64) → (batch, 1, 256, 256) ---
        # Step 1: unsqueeze channels: (batch, 1, 1024, 64)
        x = op.Unsqueeze(x, [1])
        # Step 2: reshape(batch, freq_ratio, 1024/freq_ratio, 64) = (batch, 4, 256, 64)
        x = op.Reshape(
            x, op.Concat(batch, [freq_ratio, spec_size, self._num_mel_bins], axis=0)
        )
        # Step 3: permute(0, 1, 3, 2) → (batch, 4, 64, 256)
        x = op.Transpose(x, perm=[0, 1, 3, 2])
        # Step 4: reshape(batch, 1, 64*freq_ratio, 256) = (batch, 1, 256, 256)
        x = op.Reshape(
            x, op.Concat(batch, [1, self._num_mel_bins * freq_ratio, spec_size], axis=0)
        )

        # --- Patch embedding (non-fusion): (batch, 1, 256, 256) → (batch, 4096, 96) ---
        x = self.patch_embed(op, x)

        # --- 4 Swin stages ---
        for stage in self.layers:
            x = stage(op, x)

        # --- Final layer norm ---
        x = self.norm(op, x)  # (batch, n_tokens, hidden)

        # --- Mean pooling over spatial tokens → (batch, hidden) ---
        return op.ReduceMean(x, [1], keepdims=0)


class ClapAudioModel(nn.Module):
    """CLAP audio model: HTSAT audio encoder + projection.

    Inputs:
        input_features: ``(batch, 1024, 64)``
            — mel spectrogram with 1024 time frames x 64 frequency bins.
            The CLAP feature extractor (48 kHz, hop_length=480) produces
            ~3200 frames for 30s audio; callers should pad/truncate to 1024.

    Output:
        audio_embeds: ``(batch, 512)`` normalised audio embeddings

    HuggingFace class: ``ClapAudioModel``
    """

    default_task: str = "clap-audio-feature-extraction"
    category: str = "Audio"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.audio_encoder = _ClapAudioEncoder(config)
        projection_dim = config.projection_dim
        hidden_size = config.hidden_size
        self.projection = _ClapProjectionLayer(hidden_size, hidden_size, projection_dim)

    def forward(
        self,
        op: builder.OpBuilder,
        input_features: ir.Value,
    ) -> ir.Value:
        audio_feats = self.audio_encoder(op, input_features)
        return self.projection(op, audio_feats)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map weights from combined ClapModel checkpoint to ClapAudioModel.

        - Strips ``audio_model.`` prefix from audio encoder weights
        - Maps ``audio_projection.*`` → ``projection.*``
        - Drops fusion-model weights (``patch_embed.fusion_model.*``,
          ``patch_embed.mel_conv2d.*``) — not used in the non-fusion path
        - Drops all non-audio keys (text_model, logit_scale, etc.)
        """
        fusion_prefixes = (
            "audio_encoder.patch_embed.fusion_model.",
            "audio_encoder.patch_embed.mel_conv2d.",
        )
        result: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            if name.startswith("audio_model."):
                stripped = name[len("audio_model.") :]
                # Skip fusion-only weights
                if any(stripped.startswith(p) for p in fusion_prefixes):
                    continue
                result[stripped] = tensor
            elif name.startswith("audio_projection."):
                result["projection." + name[len("audio_projection.") :]] = tensor
        return result
