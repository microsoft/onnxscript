# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Facebook Speech2Text (S2T) encoder-decoder ASR model.

Speech2Text is a seq2seq model for automatic speech recognition.  The encoder
uses a Conv1d subsampler (2× stride-2 convolutions with GLU) to reduce the
frame rate before feeding mel spectrogram features into a standard transformer
encoder.  The decoder is an autoregressive transformer with cross-attention
over the encoder output.

Architecture:
    input_features (B, T, feat_dim)
        → Conv1dSubsampler (stride-4 total, GLU activation)
        → sinusoidal positional embedding
        → TransformerEncoder (post-LN)
        → encoder_hidden_states (B, T//4, d_model)
        → TransformerDecoder (post-LN, with cross-attention)
        → logits (B, dec_seq, vocab_size)

HuggingFace reference: ``Speech2TextForConditionalGeneration``
(model_type="speech_to_text").

Weight naming (after stripping ``model.`` prefix) matches HuggingFace exactly
with a single exception: sinusoidal PE weights are non-persistent buffers in
HF (not in state_dict) and are therefore computed in ``preprocess_weights``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import Speech2TextConfig
from mobius.components._common import LayerNorm, Linear
from mobius.components._encoder_decoder_attention import EncoderDecoderAttention

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Conv1d wrapper
# ---------------------------------------------------------------------------


class _Conv1d(nn.Module):
    """Single 1-D convolution layer with weight and bias parameters.

    Stores parameters in the same layout as ``nn.Conv1d``:
        weight: (out_channels, in_channels, kernel_size)
        bias:   (out_channels,)

    Parameter names align with HuggingFace ``Conv1d.weight/bias`` exactly,
    enabling direct weight loading from the HF checkpoint.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self._stride = stride
        self._padding = padding
        self._kernel_size = kernel_size
        self.weight = nn.Parameter([out_channels, in_channels, kernel_size])
        self.bias = nn.Parameter([out_channels])

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        """Apply 1-D convolution.

        Args:
            x: (batch, in_channels, seq_len) — channels-first layout.

        Returns:
            (batch, out_channels, seq_len') where seq_len' = (T + 2*pad - k) / stride + 1.
        """
        return op.Conv(
            x,
            self.weight,
            self.bias,
            dilations=[1],
            kernel_shape=[self._kernel_size],
            pads=[self._padding, self._padding],
            strides=[self._stride],
        )


# ---------------------------------------------------------------------------
# Conv1d subsampler with GLU
# ---------------------------------------------------------------------------


class _Conv1dSubsampler(nn.Module):
    """Downsampling conv stack that reduces frame rate and projects to d_model.

    Each layer applies Conv1d(stride=2) followed by Gated Linear Unit (GLU),
    which halves the channel count.  After ``num_conv_layers`` layers the
    total temporal downsampling is 2^num_conv_layers (4× for 2 layers).

    The final layer produces ``hidden_size * 2`` channels so that after GLU
    the output has exactly ``hidden_size = d_model`` channels.

    Parameters match HuggingFace ``Speech2TextConv1dSubsampler.conv_layers``.
    """

    def __init__(self, config: Speech2TextConfig) -> None:
        super().__init__()
        layers = []
        in_ch = config.input_feat_per_channel * config.input_channels
        for i, k in enumerate(config.conv_kernel_sizes):
            is_last = i == len(config.conv_kernel_sizes) - 1
            out_ch = config.hidden_size * 2 if is_last else config.conv_channels
            layers.append(_Conv1d(in_ch, out_ch, kernel_size=k, stride=2, padding=k // 2))
            in_ch = (config.conv_channels if not is_last else config.hidden_size * 2) // 2
        # nn.ModuleList gives paths: conv_layers.{i}.weight / conv_layers.{i}.bias
        self.conv_layers = nn.ModuleList(layers)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        """Subsample audio features.

        Args:
            x: (batch, seq_len, input_feat) — time-major float input.

        Returns:
            (batch, seq_len // 2^num_layers, hidden_size)
        """
        # Transpose to channels-first for ONNX Conv: (B, feat, T)
        hidden = op.Transpose(x, perm=[0, 2, 1])  # (B, feat, T)

        for conv in self.conv_layers:
            hidden = conv(op, hidden)  # (B, out_ch, T')
            # GLU: split channels in half and gate: hidden[:, :C//2] * sigmoid(hidden[:, C//2:])
            hidden = op.Split(hidden, axis=1, num_outputs=2, _outputs=2)
            first, second = hidden
            hidden = op.Mul(first, op.Sigmoid(second))  # (B, out_ch//2, T')

        # Transpose back to time-major: (B, T', hidden_size)
        hidden = op.Transpose(hidden, perm=[0, 2, 1])
        return hidden


# ---------------------------------------------------------------------------
# Sinusoidal positional embedding
# ---------------------------------------------------------------------------


class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional embedding table.

    Stores the precomputed PE table as a ``weights`` parameter whose values
    are injected during ``Speech2TextForConditionalGeneration.preprocess_weights``
    (since HF stores them as non-persistent buffers not included in state_dict).

    The offset of 2 matches HuggingFace's ``Speech2TextSinusoidalPositionalEmbedding``.

    Parameter name ``weights`` aligns with the HF buffer attribute, though the
    values must be synthesized at export time since HF does not persist them.
    """

    OFFSET: int = 2

    def __init__(self, max_positions: int, embed_dim: int) -> None:
        super().__init__()
        # +OFFSET to match HF's table size (positions are 2-indexed)
        self.weights = nn.Parameter([max_positions + self.OFFSET, embed_dim])

    def forward(self, op: builder.OpBuilder, positions: ir.Value) -> ir.Value:
        """Look up sinusoidal embeddings by position index.

        Args:
            positions: (batch, seq_len) INT64 — position indices (include offset).

        Returns:
            (batch, seq_len, embed_dim)
        """
        return op.Gather(self.weights, positions, axis=0)


# ---------------------------------------------------------------------------
# Encoder and Decoder blocks
# ---------------------------------------------------------------------------


class _S2TEncoderBlock(nn.Module):
    """Speech2Text encoder layer: post-norm self-attention + post-norm FFN.

    Structure (post-LayerNorm like BART):
        residual + self_attn(x) → self_attn_layer_norm
        residual + FFN(x) → final_layer_norm
    """

    def __init__(self, config: Speech2TextConfig) -> None:
        super().__init__()
        self.self_attn = EncoderDecoderAttention(config)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc1 = Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._hidden_act = config.hidden_act

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # Self-attention (non-causal, encoder) with post-LN
        residual = hidden_states
        hidden_states, _ = self.self_attn(op, hidden_states)
        hidden_states = self.self_attn_layer_norm(op, op.Add(residual, hidden_states))

        # FFN with post-LN
        residual = hidden_states
        hidden_states = op.Relu(self.fc1(op, hidden_states))  # ReLU activation for S2T
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = self.final_layer_norm(op, op.Add(residual, hidden_states))

        return hidden_states


class _S2TDecoderBlock(nn.Module):
    """Speech2Text decoder layer: causal self-attn + cross-attn + FFN (post-LN).

    Structure:
        residual + self_attn(x) → self_attn_layer_norm
        residual + encoder_attn(x, enc_hidden) → encoder_attn_layer_norm
        residual + FFN(x) → final_layer_norm
    """

    def __init__(self, config: Speech2TextConfig) -> None:
        super().__init__()
        self.self_attn = EncoderDecoderAttention(config, is_causal=True)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.encoder_attn = EncoderDecoderAttention(config)
        self.encoder_attn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc1 = Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        past_key_value: tuple | None = None,
        cross_past_key_value: tuple | None = None,
    ) -> tuple[ir.Value, tuple, tuple]:
        # Causal self-attention
        residual = hidden_states
        hidden_states, self_kv = self.self_attn(
            op, hidden_states, past_key_value=past_key_value
        )
        hidden_states = self.self_attn_layer_norm(op, op.Add(residual, hidden_states))

        # Cross-attention over encoder output
        residual = hidden_states
        hidden_states, cross_kv = self.encoder_attn(
            op,
            hidden_states,
            key_value_states=encoder_hidden_states,
            past_key_value=cross_past_key_value,
        )
        hidden_states = self.encoder_attn_layer_norm(op, op.Add(residual, hidden_states))

        # FFN
        residual = hidden_states
        hidden_states = op.Relu(self.fc1(op, hidden_states))
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = self.final_layer_norm(op, op.Add(residual, hidden_states))

        return hidden_states, self_kv, cross_kv


# ---------------------------------------------------------------------------
# Encoder and Decoder modules
# ---------------------------------------------------------------------------


class _S2TEncoder(nn.Module):
    """Speech2Text encoder: conv subsampler → sinusoidal PE → transformer layers.

    Input: mel spectrogram features (B, T, feat_dim)
    Output: encoder hidden states (B, T//4, d_model)
    """

    def __init__(self, config: Speech2TextConfig) -> None:
        super().__init__()
        self._embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        # Conv subsampler: paths as conv.conv_layers.{i}.weight/bias
        self.conv = _Conv1dSubsampler(config)
        # Sinusoidal PE table: weights injected during preprocess_weights
        self.embed_positions = _SinusoidalPE(config.max_source_positions, config.hidden_size)
        self.layers = nn.ModuleList(
            [_S2TEncoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        input_features: ir.Value,
    ) -> ir.Value:
        """Encode mel features to hidden states.

        Args:
            input_features: (batch, seq_len, feat_dim) FLOAT.

        Returns:
            (batch, seq_len // 4, hidden_size)
        """
        # Conv subsampling: (B, T, feat) → (B, T//4, d_model)
        hidden_states = self.conv(op, input_features)

        # Apply embedding scale
        if self._embed_scale != 1.0:
            hidden_states = op.Mul(
                hidden_states,
                op.CastLike(op.Constant(value_float=self._embed_scale), hidden_states),
            )

        # Sinusoidal positional embedding: positions start at offset=2
        seq_len = op.Shape(hidden_states, start=1, end=2)  # (1,)
        batch = op.Shape(hidden_states, start=0, end=1)    # (1,)
        offset = op.Constant(value_ints=[_SinusoidalPE.OFFSET])
        positions = op.Range(
            offset,
            op.Add(seq_len, offset),
            op.Constant(value_ints=[1]),
        )  # (seq_len,)
        positions = op.Cast(positions, to=7)   # INT64
        positions = op.Unsqueeze(positions, [0])  # (1, seq_len)
        # Broadcast to (batch, seq_len) via Expand
        shape = op.Concat(batch, seq_len, axis=0)
        positions = op.Expand(positions, shape)  # (B, seq_len)
        pos_embeds = self.embed_positions(op, positions)  # (B, seq_len, d_model)

        hidden_states = op.Add(hidden_states, op.CastLike(pos_embeds, hidden_states))

        for layer in self.layers:
            hidden_states = layer(op, hidden_states)

        hidden_states = self.layer_norm(op, hidden_states)
        return hidden_states


class _S2TDecoder(nn.Module):
    """Speech2Text decoder: embedding + sinusoidal PE → transformer layers → logits.

    Input: decoder_input_ids (B, dec_seq), encoder_hidden_states (B, enc_seq, d_model)
    Output: logits (B, dec_seq, vocab_size)
    """

    def __init__(self, config: Speech2TextConfig) -> None:
        super().__init__()
        num_dec_layers = config.num_decoder_layers or config.num_hidden_layers
        self._embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        from mobius.components._common import Embedding  # avoid circular import

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        # Decoder sinusoidal PE: max_target_positions + 2 rows
        self.embed_positions = _SinusoidalPE(config.max_target_positions, config.hidden_size)
        self.layers = nn.ModuleList(
            [_S2TDecoderBlock(config) for _ in range(num_dec_layers)]
        )
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        encoder_hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
        past_key_values: list | None = None,
        cross_past_key_values: list | None = None,
    ) -> tuple[ir.Value, list, list]:
        """Decode with cross-attention over encoder output.

        Args:
            input_ids:              (batch, dec_seq_len) INT64.
            encoder_hidden_states:  (batch, enc_seq_len, hidden_size).
            attention_mask:         Ignored (sinusoidal positions handle causality).
            past_key_values:        Per-layer (key, value) for self-attention cache.
            cross_past_key_values:  Per-layer (key, value) for cross-attention cache.

        Returns:
            logits:             (batch, dec_seq_len, vocab_size)
            present_self_kvs:   Updated self-attention KV cache.
            present_cross_kvs:  Updated cross-attention KV cache.
        """
        # Token + position embeddings
        inputs_embeds = self.embed_tokens(op, input_ids)  # (B, T, H)
        if self._embed_scale != 1.0:
            inputs_embeds = op.Mul(
                inputs_embeds,
                op.CastLike(op.Constant(value_float=self._embed_scale), inputs_embeds),
            )

        # Compute positions: [past_len + 2, ..., past_len + 2 + seq_len)
        seq_len = op.Shape(input_ids, start=1, end=2)
        batch = op.Shape(input_ids, start=0, end=1)
        if past_key_values is not None:
            past_len = op.Shape(past_key_values[0][0], start=2, end=3)
        else:
            past_len = op.Constant(value_ints=[0])
        offset = op.Constant(value_ints=[_SinusoidalPE.OFFSET])
        start = op.Add(past_len, offset)
        positions = op.Range(start, op.Add(start, seq_len), op.Constant(value_ints=[1]))
        positions = op.Cast(positions, to=7)
        positions = op.Unsqueeze(positions, [0])  # (1, seq_len)
        shape = op.Concat(batch, seq_len, axis=0)
        positions = op.Expand(positions, shape)   # (B, seq_len)
        pos_embeds = self.embed_positions(op, positions)  # (B, seq_len, H)

        hidden_states = op.Add(inputs_embeds, op.CastLike(pos_embeds, inputs_embeds))

        past_kvs = past_key_values or [None] * len(self.layers)
        cross_past_kvs = cross_past_key_values or [None] * len(self.layers)
        present_self_kvs = []
        present_cross_kvs = []

        for layer, past_kv, cross_kv in zip(self.layers, past_kvs, cross_past_kvs):
            hidden_states, self_kv, cross_kv_out = layer(
                op,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=past_kv,
                cross_past_key_value=cross_kv,
            )
            present_self_kvs.append(self_kv)
            present_cross_kvs.append(cross_kv_out)

        hidden_states = self.layer_norm(op, hidden_states)
        logits = self.lm_head(op, hidden_states)  # (B, T, V)
        return logits, present_self_kvs, present_cross_kvs


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class Speech2TextForConditionalGeneration(nn.Module):
    """Facebook Speech2Text encoder-decoder model for ASR.

    Exposes ``encoder`` and ``decoder`` sub-modules that are consumed by
    :class:`Speech2TextSeq2SeqTask` to produce two separate ONNX graphs.

    Weight names match HuggingFace after stripping the ``model.`` prefix.
    The only non-trivial rename is the sinusoidal positional embedding weights
    which must be synthesised in :meth:`preprocess_weights` because HuggingFace
    stores them as non-persistent buffers (excluded from state_dict).
    """

    default_task = "speech2text-seq2seq"
    category = "speech"

    def __init__(self, config: Speech2TextConfig) -> None:
        super().__init__()
        self._config = config
        self.encoder = _S2TEncoder(config)
        self.decoder = _S2TDecoder(config)

    def preprocess_weights(self, state_dict: dict) -> dict:
        """Map HuggingFace weights to our module paths and inject sinusoidal PE.

        Key transformations:
        1. Strip ``model.`` prefix.
        2. Compute and inject ``encoder.embed_positions.weights`` and
           ``decoder.embed_positions.weights`` (non-persistent HF buffers).
        3. Tie lm_head from embed_tokens if not separately provided.
        """
        result: dict = {}
        for key, value in state_dict.items():
            k = key
            if k.startswith("model."):
                k = k[len("model."):]
            # lm_head is a top-level key in HF (not under model.)
            # Our lm_head lives at decoder.lm_head
            if k == "lm_head.weight":
                k = "decoder.lm_head.weight"
            result[k] = value

        # Compute sinusoidal PE and inject (HF persistent=False → not in state_dict)
        config = self._config
        if "encoder.embed_positions.weights" not in result:
            result["encoder.embed_positions.weights"] = _compute_sinusoidal_pe(
                config.max_source_positions, config.hidden_size
            )
        if "decoder.embed_positions.weights" not in result:
            result["decoder.embed_positions.weights"] = _compute_sinusoidal_pe(
                config.max_target_positions, config.hidden_size
            )

        # Tie lm_head weight to decoder embed_tokens if not already present
        if "decoder.lm_head.weight" not in result:
            embed = result.get("decoder.embed_tokens.weight")
            if embed is not None:
                result["decoder.lm_head.weight"] = embed

        return result


# ---------------------------------------------------------------------------
# Sinusoidal PE computation
# ---------------------------------------------------------------------------


def _compute_sinusoidal_pe(
    max_positions: int,
    embedding_dim: int,
    padding_idx: int = 1,
    offset: int = _SinusoidalPE.OFFSET,
) -> np.ndarray:
    """Compute sinusoidal positional embedding table.

    Matches ``Speech2TextSinusoidalPositionalEmbedding.get_embedding`` in HF.
    The returned table has shape ``(max_positions + offset, embedding_dim)``
    and the row at ``padding_idx`` is zeroed out.
    """
    num_rows = max_positions + offset
    half_dim = embedding_dim // 2
    denom = math.log(10000) / (half_dim - 1)
    freqs = np.exp(np.arange(half_dim) * -denom)              # (half_dim,)
    pos = np.arange(num_rows).reshape(-1, 1)                  # (N, 1)
    angles = pos * freqs                                       # (N, half_dim)
    table = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)  # (N, dim)
    if padding_idx is not None and 0 <= padding_idx < num_rows:
        table[padding_idx] = 0.0
    return table.astype(np.float32)
