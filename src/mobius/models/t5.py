# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""T5 encoder-decoder model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._activations import ACT2FN
from mobius.components._common import Embedding, Linear
from mobius.components._encoder_decoder_attention import (
    EncoderDecoderAttention,
)
from mobius.components._rms_norm import RMSNorm

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# T5 Relative Position Bias
# ---------------------------------------------------------------------------


def _relative_position_bucket(
    op: builder.OpBuilder,
    relative_position,
    *,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
):
    """Map relative positions to bucket indices (T5-style log-linear).

    Implements the same bucketing as HuggingFace
    ``T5Attention._relative_position_bucket``.

    Args:
        op: ONNX op builder.
        relative_position: INT64 tensor [query_length, key_length].
        bidirectional: True for encoder, False for decoder.
        num_buckets: Number of relative position buckets (e.g. 32).
        max_distance: Maximum distance for bucketing (e.g. 128).

    Returns:
        INT64 tensor [query_length, key_length] of bucket indices
        in ``[0, num_buckets)``.
    """
    zero = op.Constant(value_int=0)

    if bidirectional:
        # Half buckets for positive, half for negative relative positions
        half_buckets = num_buckets // 2
        is_positive = op.Greater(relative_position, zero)
        is_positive_int = op.Cast(is_positive, to=7)  # INT64
        relative_buckets = op.Mul(is_positive_int, op.Constant(value_int=half_buckets))
        abs_position = op.Abs(relative_position)
        effective_buckets = half_buckets
    else:
        # Unidirectional: only negative relative positions (past tokens)
        neg_position = op.Neg(op.Min(relative_position, zero))
        abs_position = neg_position
        relative_buckets = op.Expand(zero, op.Shape(relative_position))
        effective_buckets = num_buckets

    max_exact = effective_buckets // 2

    # Small positions: use direct index
    is_small = op.Less(abs_position, op.Constant(value_int=max_exact))

    # Large positions: log-linear bucketing
    abs_float = op.Cast(abs_position, to=1)  # FLOAT32
    # Clamp to avoid log(0); doesn't affect result because Where
    # selects the is_small path for abs_position < max_exact
    abs_clamped = op.Max(abs_float, op.Constant(value_float=1.0))
    log_ratio = op.Log(op.Div(abs_clamped, op.Constant(value_float=float(max_exact))))
    log_scale = math.log(max_distance / max_exact)
    bucket_float = op.Add(
        op.Constant(value_float=float(max_exact)),
        op.Mul(
            log_ratio,
            op.Constant(value_float=float(effective_buckets - max_exact) / log_scale),
        ),
    )
    large_bucket = op.Cast(bucket_float, to=7)  # INT64
    large_bucket = op.Min(large_bucket, op.Constant(value_int=effective_buckets - 1))

    # Select small or large bucket
    final_offset = op.Where(is_small, abs_position, large_bucket)
    return op.Add(relative_buckets, final_offset)


def _compute_position_bias(
    op: builder.OpBuilder,
    embedding: Embedding,
    query_length,
    key_length,
    *,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
    num_heads: int,
    query_offset=None,
):
    """Compute T5-style relative position bias from learned embeddings.

    Uses log-linear bucketing of relative positions (HuggingFace
    ``T5Attention.compute_bias``). Bidirectional for encoder,
    unidirectional for decoder self-attention.

    Args:
        op: ONNX op builder.
        embedding: Learned relative attention bias embedding
            with shape ``[num_buckets, num_heads]``.
        query_length: Scalar INT64 — number of query positions.
        key_length: Scalar INT64 — number of key positions.
        bidirectional: True for encoder, False for decoder.
        num_buckets: Number of relative position buckets.
        max_distance: Maximum distance for bucketing.
        num_heads: Number of attention heads.
        query_offset: Scalar INT64 — offset for query positions
            (e.g. past_sequence_length for decode steps). If None,
            query positions start at 0.

    Returns:
        Position bias tensor of shape
        ``[1, num_heads, query_length, key_length]`` (FLOAT32).
    """
    # Query positions: [query_offset, query_offset + query_length)
    if query_offset is None:
        query_offset = op.Constant(value_int=0)
    query_end = op.Add(query_offset, query_length)
    # context_position: [query_length]
    context_position = op.Range(query_offset, query_end, op.Constant(value_int=1))
    # memory_position: [key_length]
    memory_position = op.Range(op.Constant(value_int=0), key_length, op.Constant(value_int=1))

    # Relative position: memory - context → [query_length, key_length]
    context_2d = op.Unsqueeze(context_position, [1])
    memory_2d = op.Unsqueeze(memory_position, [0])
    relative_position = op.Sub(memory_2d, context_2d)

    # Map relative positions to bucket indices
    bucket_indices = _relative_position_bucket(
        op,
        relative_position,
        bidirectional=bidirectional,
        num_buckets=num_buckets,
        max_distance=max_distance,
    )

    # Gather from learned embedding → [query_len, key_len, num_heads]
    values = embedding(op, bucket_indices)
    # Transpose to [num_heads, query_length, key_length]
    values = op.Transpose(values, perm=[2, 0, 1])
    # Unsqueeze to [1, num_heads, query_length, key_length]
    values = op.Unsqueeze(values, [0])
    return values


# ---------------------------------------------------------------------------
# T5 Components
# ---------------------------------------------------------------------------


class T5EncoderBlock(nn.Module):
    """T5 encoder block: pre-norm self-attention + pre-norm FFN."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        # T5 does not scale attention scores by 1/sqrt(d_k)
        self.self_attn = EncoderDecoderAttention(config, bias=False, scale=1.0)
        self.self_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = _T5FFN(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None = None,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_norm(op, hidden_states)
        hidden_states, _ = self.self_attn(op, hidden_states, attention_bias=attention_bias)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = self.ffn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states


class T5DecoderBlock(nn.Module):
    """T5 decoder block: self-attn + cross-attn + FFN, all pre-norm."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        # T5 does not scale attention scores by 1/sqrt(d_k)
        self.self_attn = EncoderDecoderAttention(
            config,
            is_causal=True,
            bias=False,
            scale=1.0,
        )
        self.self_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn = EncoderDecoderAttention(
            config,
            bias=False,
            scale=1.0,
        )
        self.cross_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = _T5FFN(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        attention_bias: ir.Value | None = None,
        past_key_value: tuple | None = None,
        cross_past_key_value: ir.Value | None = None,
    ):
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn_norm(op, hidden_states)
        hidden_states, self_kv = self.self_attn(
            op, hidden_states, attention_bias=attention_bias, past_key_value=past_key_value
        )
        hidden_states = op.Add(residual, hidden_states)

        # Cross-attention
        residual = hidden_states
        hidden_states = self.cross_attn_norm(op, hidden_states)
        hidden_states, cross_kv = self.cross_attn(
            op,
            hidden_states,
            key_value_states=encoder_hidden_states,
            past_key_value=cross_past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # FFN
        residual = hidden_states
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = self.ffn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, self_kv, cross_kv


class _T5FFN(nn.Module):
    """T5 feed-forward network.

    Standard T5 uses ``wi → act → wo``.
    Gated variants (mT5, FLAN-T5, UL2) use ``(wi_0(x) * act(wi_1(x))) → wo``
    where wi_0 is the gate and wi_1 is the up-projection.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._is_gated = config.is_gated_act
        if self._is_gated:
            # Gated FFN: gate (wi_0) and up-projection (wi_1)
            self.wi_0 = Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.wi_1 = Linear(config.hidden_size, config.intermediate_size, bias=False)
        else:
            self.wi = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.wo = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self._act_fn = ACT2FN[config.hidden_act]

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        if self._is_gated:
            gate = self._act_fn(op, self.wi_0(op, hidden_states))
            hidden_states = op.Mul(gate, self.wi_1(op, hidden_states))
        else:
            hidden_states = self.wi(op, hidden_states)
            hidden_states = self._act_fn(op, hidden_states)
        return self.wo(op, hidden_states)


# ---------------------------------------------------------------------------
# T5 Encoder and Decoder top-level models
# ---------------------------------------------------------------------------


class T5Encoder(nn.Module):
    """T5 encoder stack."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        # Relative position bias: learned embedding shared across all blocks.
        # Only block 0 has this weight in HF, but we lift it to the
        # encoder level so the onnxscript parameter name resolves correctly.
        self.relative_attention_bias = Embedding(
            config.relative_attention_num_buckets, config.num_attention_heads
        )
        self._num_buckets = config.relative_attention_num_buckets
        self._max_distance = config.relative_attention_max_distance
        self._num_heads = config.num_attention_heads
        self.block = nn.ModuleList(
            [T5EncoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        # Compute T5 relative position bias (bidirectional for encoder).
        # Shape: [1, num_heads, seq_len, seq_len]
        seq_len = op.Shape(input_ids, start=1, end=2)
        position_bias = _compute_position_bias(
            op,
            self.relative_attention_bias,
            seq_len,
            seq_len,
            bidirectional=True,
            num_buckets=self._num_buckets,
            max_distance=self._max_distance,
            num_heads=self._num_heads,
        )
        for block in self.block:
            hidden_states = block(op, hidden_states, attention_bias=position_bias)
        hidden_states = self.final_layer_norm(op, hidden_states)
        return hidden_states


class T5Decoder(nn.Module):
    """T5 decoder stack with cross-attention and KV cache."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        # Relative position bias: learned embedding for decoder self-attention.
        self.relative_attention_bias = Embedding(
            config.relative_attention_num_buckets, config.num_attention_heads
        )
        self._num_buckets = config.relative_attention_num_buckets
        self._max_distance = config.relative_attention_max_distance
        self._num_heads = config.num_attention_heads
        self._hidden_size = config.hidden_size
        # HF T5 recently introduced scale_decoder_outputs (decoupled from
        # tie_word_embeddings). Original T5 sets it True; FLAN-T5/UL2 set
        # it False. MT5 doesn't have this field and never scales.
        self._scale_decoder_outputs = bool(config.scale_decoder_outputs)
        num_decoder_layers = getattr(config, "num_decoder_layers", config.num_hidden_layers)
        self.block = nn.ModuleList([T5DecoderBlock(config) for _ in range(num_decoder_layers)])
        self.final_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        encoder_hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
        past_key_values: list | None = None,
        cross_past_key_values: ir.Value | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)

        # Compute T5 relative position bias for decoder self-attention.
        # Unidirectional (bidirectional=False) since decoder is causal.
        # query_offset = past_sequence_length for decode steps.
        query_length = op.Shape(input_ids, start=1, end=2)
        if past_key_values is not None:
            # past_key_values[0][0]: [batch, heads, past_seq_len, head_dim]
            past_len = op.Shape(past_key_values[0][0], start=2, end=3)
            key_length = op.Add(past_len, query_length)
        else:
            past_len = None
            key_length = query_length
        position_bias = _compute_position_bias(
            op,
            self.relative_attention_bias,
            query_length,
            key_length,
            bidirectional=False,
            num_buckets=self._num_buckets,
            max_distance=self._max_distance,
            num_heads=self._num_heads,
            query_offset=past_len,
        )

        past_kvs = past_key_values or [None] * len(self.block)
        cross_past_kvs = cross_past_key_values or [None] * len(self.block)
        present_self_kvs = []
        present_cross_kvs = []

        for block, past_kv, cross_kv in zip(self.block, past_kvs, cross_past_kvs):
            hidden_states, self_kv, cross_kv_out = block(
                op,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_bias=position_bias,
                past_key_value=past_kv,
                cross_past_key_value=cross_kv,
            )
            present_self_kvs.append(self_kv)
            present_cross_kvs.append(cross_kv_out)

        hidden_states = self.final_layer_norm(op, hidden_states)
        # T5 scales hidden states by 1/sqrt(d_model) before projecting to
        # vocab. Controlled by scale_decoder_outputs (newer HF) or
        # tie_word_embeddings (legacy). Original T5 and mT5 use True;
        # FLAN-T5/UL2 use False (separate lm_head weights).
        if self._scale_decoder_outputs:
            hidden_states = op.Mul(
                hidden_states,
                op.Constant(value_float=float(self._hidden_size**-0.5)),
            )
        logits = self.lm_head(op, hidden_states)
        return logits, present_self_kvs, present_cross_kvs


# ---------------------------------------------------------------------------
# T5 Model (wraps encoder + decoder)
# ---------------------------------------------------------------------------


class T5ForConditionalGeneration(nn.Module):
    """T5 encoder-decoder model for conditional generation (seq2seq).

    This model produces a ModelPackage with separate encoder and decoder
    components for efficient inference.
    """

    default_task = "seq2seq"
    category = "encoder-decoder"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.encoder = T5Encoder(config)
        self.decoder = T5Decoder(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_t5_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        # Shared embeddings: encoder and decoder use the same embedding
        if "encoder.embed_tokens.weight" not in new_state_dict:
            shared = new_state_dict.get("shared.weight")
            if shared is not None:
                new_state_dict["encoder.embed_tokens.weight"] = shared
                new_state_dict["decoder.embed_tokens.weight"] = shared
        # Tied lm_head
        if "decoder.lm_head.weight" not in new_state_dict:
            embed = new_state_dict.get("encoder.embed_tokens.weight")
            if embed is not None:
                new_state_dict["decoder.lm_head.weight"] = embed
        return new_state_dict


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------

_T5_COMMON_RENAMES = {
    # Self-attention (same for encoder and decoder)
    "layer.0.SelfAttention.q.": "self_attn.q_proj.",
    "layer.0.SelfAttention.k.": "self_attn.k_proj.",
    "layer.0.SelfAttention.v.": "self_attn.v_proj.",
    "layer.0.SelfAttention.o.": "self_attn.out_proj.",
    "layer.0.layer_norm.": "self_attn_norm.",
}

# Encoder: layer.0 = self-attn, layer.1 = FFN
_T5_ENCODER_RENAMES = {
    "layer.1.DenseReluDense.wi.": "ffn.wi.",
    # Gated FFN variants (mT5, FLAN-T5, UL2)
    "layer.1.DenseReluDense.wi_0.": "ffn.wi_0.",
    "layer.1.DenseReluDense.wi_1.": "ffn.wi_1.",
    "layer.1.DenseReluDense.wo.": "ffn.wo.",
    "layer.1.layer_norm.": "ffn_norm.",
}

# Decoder: layer.0 = self-attn, layer.1 = cross-attn, layer.2 = FFN
_T5_DECODER_RENAMES = {
    "layer.1.EncDecAttention.q.": "cross_attn.q_proj.",
    "layer.1.EncDecAttention.k.": "cross_attn.k_proj.",
    "layer.1.EncDecAttention.v.": "cross_attn.v_proj.",
    "layer.1.EncDecAttention.o.": "cross_attn.out_proj.",
    "layer.1.layer_norm.": "cross_attn_norm.",
    "layer.2.DenseReluDense.wi.": "ffn.wi.",
    # Gated FFN variants (mT5, FLAN-T5, UL2)
    "layer.2.DenseReluDense.wi_0.": "ffn.wi_0.",
    "layer.2.DenseReluDense.wi_1.": "ffn.wi_1.",
    "layer.2.DenseReluDense.wo.": "ffn.wo.",
    "layer.2.layer_norm.": "ffn_norm.",
}


def _rename_t5_weight(name: str) -> str | None:
    """Rename a HF T5 weight to our naming convention.

    Encoder sublayers: layer.0=self-attn, layer.1=FFN.
    Decoder sublayers: layer.0=self-attn, layer.1=cross-attn, layer.2=FFN.
    """
    # Keep shared embedding as-is for now (handled by preprocess_weights)
    if name == "shared.weight":
        return "shared.weight"
    if name == "lm_head.weight":
        return "decoder.lm_head.weight"

    # encoder.block.{i}.layer.X.{...} or decoder.block.{i}.layer.X.{...}
    for prefix in ("encoder.", "decoder."):
        if not name.startswith(prefix):
            continue

        rest = name[len(prefix) :]

        # Final layer norm
        if rest.startswith("final_layer_norm."):
            return name  # Already correct naming

        # Block weights
        if rest.startswith("block."):
            parts = rest.split(".", 2)  # block, idx, remainder
            if len(parts) < 3:
                return None
            block_idx = parts[1]
            remainder = parts[2]

            # Relative position bias is lifted from block.0 to the
            # encoder/decoder level (only block 0 has it in HF).
            rel_bias_key = "layer.0.SelfAttention.relative_attention_bias."
            if remainder.startswith(rel_bias_key):
                suffix = remainder[len(rel_bias_key) :]
                return f"{prefix}relative_attention_bias.{suffix}"

            # Pick context-specific rename table
            extra = _T5_ENCODER_RENAMES if prefix == "encoder." else _T5_DECODER_RENAMES

            # Try common renames first, then context-specific
            for table in (_T5_COMMON_RENAMES, extra):
                for old, new in table.items():
                    if remainder.startswith(old):
                        suffix = remainder[len(old) :]
                        return f"{prefix}block.{block_idx}.{new}{suffix}"

    return None
