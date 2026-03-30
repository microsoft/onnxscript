# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS causal language model with Mixture-of-Experts and attention sinks.

GPT-OSS (openai/gpt-oss-20b) features:
- Alternating sliding/full attention layers (config.layer_types)
- Attention sinks: learnable per-head scalar appended to the softmax denominator,
  allowing each head to "discard" tokens into a virtual null position
- GQA with attention projection biases (config.attention_bias=True)
- YaRN RoPE
- MoE FFN: top-k routing with softmax scores and additive router bias
- Custom gated activation: (up.clamp(-L,L) + 1) * silu_alpha(gate.clamp(max=L))
  where silu_alpha(x) = x * sigmoid(alpha * x), alpha=1.702, L=7.0

HuggingFace reference: ``GptOssForCausalLM``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class _GptOssGate(nn.Module):
    """Top-k router with additive bias for GPT-OSS.

    HF ``GptOssTopKRouter``: router_logits = hidden @ weight.T + bias,
    then top-k selection followed by softmax over the selected scores.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter([num_experts, hidden_size])
        self.bias = nn.Parameter([num_experts])

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # router_logits: [B, S, N_exp] = [B, S, H] @ [H, N_exp] + [N_exp]
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)
        router_logits = op.Add(router_logits, self.bias)
        k = op.Constant(value_ints=[self.top_k])
        routing_weights, selected_experts = op.TopK(router_logits, k, axis=-1, _outputs=2)
        # Softmax over top-k logits only
        routing_weights = op.Softmax(routing_weights, axis=-1)
        return routing_weights, selected_experts


class _GptOssExpertMLP(nn.Module):
    """Expert MLP with GPT-OSS custom gated activation and projection biases.

    Implements:
        gate = gate_proj(x)                   # [B, S, inter]
        up   = up_proj(x)                     # [B, S, inter]
        gate_clamped = min(gate, L)           # clamp from above
        up_clamped   = clip(up, -L, L)       # symmetric clamp
        glu  = gate_clamped * sigmoid(alpha * gate_clamped)   # SiLU with alpha
        out  = (up_clamped + 1) * glu         # gated output
        return down_proj(out)

    HF ``GptOssExperts._apply_gate`` uses alpha=1.702 and L=7.0 (``swiglu_limit``).
    Note: in HF gate/up are interleaved in a packed weight; here they are split
    in ``preprocess_weights`` into separate ``gate_proj``/``up_proj`` parameters.
    """

    _ALPHA: float = 1.702
    _LIMIT: float = 7.0

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        gate = self.gate_proj(op, hidden_states)  # [B, S, inter]
        up = self.up_proj(op, hidden_states)  # [B, S, inter]

        # gate.clamp(max=limit): min(gate, 7.0)
        gate_clamped = op.Min(gate, self._LIMIT)
        # up.clamp(-limit, limit)
        up_clamped = op.Clip(up, -self._LIMIT, self._LIMIT)

        # glu = gate * sigmoid(alpha * gate)  — SiLU with custom alpha
        glu = op.Mul(gate_clamped, op.Sigmoid(op.Mul(self._ALPHA, gate_clamped)))

        # gated_output = (up + 1) * glu
        gated = op.Mul(op.Add(up_clamped, 1.0), glu)
        return self.down_proj(op, gated)  # [B, S, hidden]


class _GptOssMoELayer(nn.Module):
    """MoE layer for GPT-OSS with biased router and custom expert MLPs."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.gate = _GptOssGate(config.hidden_size, self.num_experts, self.top_k)
        self.experts = nn.ModuleList(
            [
                _GptOssExpertMLP(config.hidden_size, config.intermediate_size)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        routing_weights, selected_experts = self.gate(op, hidden_states)

        # Loop over experts: mask-and-accumulate dispatch
        result = None
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(op, hidden_states)
            expert_id = op.Constant(value_int=expert_idx)
            match = op.Equal(selected_experts, expert_id)
            match_float = op.Cast(match, to=1)  # FLOAT
            weighted = op.Mul(routing_weights, match_float)
            weight = op.ReduceSum(weighted, [-1], keepdims=True)
            contribution = op.Mul(expert_output, weight)
            if result is None:
                result = contribution
            else:
                result = op.Add(result, contribution)

        return result


class _GptOssAttention(nn.Module):
    """GQA attention with learned per-head sinks for GPT-OSS.

    HF ``eager_attention_forward`` appends one extra logit per token per head
    (the learnable ``sinks`` value) to the attention scores before softmax.
    This lets each head "discard" a token's weight into a virtual null position:

        combined = cat([attn_weights, sinks_expanded], dim=-1)  # [B, H, S, S_kv+1]
        combined = combined - max(combined)                      # numerical stability
        probs    = softmax(combined, dim=-1)[..., :-1]           # drop sink, [B, H, S, S_kv]
        out      = probs @ V

    Implements this manually (cannot use fused op.Attention with sinks).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.scale = config.head_dim**-0.5
        self._rotary_embedding_dim = (
            0
            if math.isclose(config.partial_rotary_factor, 1.0)
            else int(self.head_dim * config.partial_rotary_factor)
        )
        self._rope_interleave = config.rope_interleave

        # QKV projections with bias (attention_bias=True for GPT-OSS)
        self.q_proj = Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.k_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.v_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.o_proj = Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attn_o_bias,
        )

        # Learnable sink logit: one scalar per attention head [num_heads]
        self.sinks = nn.Parameter([config.num_attention_heads])

    def _expand_kv_for_gqa(
        self,
        op: builder.OpBuilder,
        kv: ir.Value,
        batch_1d: ir.Value,
        kv_len_1d: ir.Value,
    ) -> ir.Value:
        """Expand KV from [B, kv_heads, S, d] to [B, q_heads, S, d] for GQA.

        Uses unsqueeze+expand+reshape to replicate each KV head ``num_kv_groups``
        times consecutively: [kv0]*g, [kv1]*g, ..., which is what ``repeat_kv`` does.
        """
        # [B, kv_heads, S, d] → [B, kv_heads, 1, S, d]
        kv_5d = op.Unsqueeze(kv, [2])
        # Expand to [B, kv_heads, num_kv_groups, S, d]
        expand_shape = op.Concat(
            batch_1d,
            [self.num_key_value_heads, self.num_kv_groups],
            kv_len_1d,
            [self.head_dim],
            axis=0,
        )
        kv_exp = op.Expand(kv_5d, expand_shape)
        # Flatten to [B, q_heads, S, d]
        flat_shape = op.Concat(
            batch_1d,
            [self.num_attention_heads],
            kv_len_1d,
            [self.head_dim],
            axis=0,
        )
        return op.Reshape(kv_exp, flat_shape)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
    ):
        # hidden_states: [B, S, H]
        batch_1d = op.Shape(hidden_states, start=0, end=1)  # 1D tensor containing B
        seq_1d = op.Shape(hidden_states, start=1, end=2)  # 1D tensor containing S

        # QKV projections: [B, S, heads * d]
        query = self.q_proj(op, hidden_states)
        key = self.k_proj(op, hidden_states)
        value = self.v_proj(op, hidden_states)

        # Apply RoPE on 3D packed format [B, S, heads * d]
        if position_embeddings is not None:
            query = apply_rotary_pos_emb(
                op,
                x=query,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=self._rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )
            key = apply_rotary_pos_emb(
                op,
                x=key,
                position_embeddings=position_embeddings,
                num_heads=self.num_key_value_heads,
                rotary_embedding_dim=self._rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )

        # Reshape to 4D and transpose: [B, S, heads, d] → [B, heads, S, d]
        query = op.Transpose(
            op.Reshape(query, [0, 0, self.num_attention_heads, self.head_dim]),
            perm=[0, 2, 1, 3],
        )  # [B, q_heads, S, d]
        key = op.Transpose(
            op.Reshape(key, [0, 0, self.num_key_value_heads, self.head_dim]),
            perm=[0, 2, 1, 3],
        )  # [B, kv_heads, S, d]
        value = op.Transpose(
            op.Reshape(value, [0, 0, self.num_key_value_heads, self.head_dim]),
            perm=[0, 2, 1, 3],
        )  # [B, kv_heads, S, d]

        # KV cache: prepend past tokens
        if past_key_value is not None:
            key = op.Concat(past_key_value[0], key, axis=2)  # [B, kv_heads, past+S, d]
            value = op.Concat(past_key_value[1], value, axis=2)  # [B, kv_heads, past+S, d]
        present_key_value = (key, value)

        # Total KV sequence length (after cache concatenation)
        kv_len_1d = op.Shape(key, start=2, end=3)  # [total_S]

        # GQA: expand key/value from kv_heads to q_heads
        if self.num_kv_groups > 1:
            key_exp = self._expand_kv_for_gqa(op, key, batch_1d, kv_len_1d)
            value_exp = self._expand_kv_for_gqa(op, value, batch_1d, kv_len_1d)
        else:
            key_exp = key
            value_exp = value

        # Attention scores: [B, q_heads, S_q, S_kv]
        # query @ key.T: [B, q_heads, S_q, d] @ [B, q_heads, d, S_kv]
        key_t = op.Transpose(key_exp, perm=[0, 1, 3, 2])  # [B, q_heads, d, S_kv]
        attn_scores = op.MatMul(query, key_t)  # [B, q_heads, S_q, S_kv]
        attn_scores = op.Mul(attn_scores, self.scale)

        # Add causal+sliding_window+padding mask (float additive bias)
        if attention_bias is not None:
            # attention_bias: [B, 1, S_q, S_kv] — broadcasts over q_heads
            attn_scores = op.Add(attn_scores, attention_bias)

        # Append sinks column: [q_heads] → [B, q_heads, S_q, 1]
        sinks_4d = op.Reshape(
            self.sinks,
            [1, self.num_attention_heads, 1, 1],
        )
        expand_shape = op.Concat(
            batch_1d,
            [self.num_attention_heads],
            seq_1d,
            [1],
            axis=0,
        )
        sinks_expanded = op.Expand(sinks_4d, expand_shape)  # [B, q_heads, S_q, 1]
        # combined: [B, q_heads, S_q, S_kv+1]
        combined = op.Concat(attn_scores, sinks_expanded, axis=-1)

        # Numerical stability: subtract per-row max before softmax
        row_max = op.ReduceMax(combined, [-1], keepdims=True)  # [B, q_heads, S_q, 1]
        combined = op.Sub(combined, row_max)

        # Softmax over the extended sequence (S_kv + 1) dimension
        probs = op.Softmax(combined, axis=-1)  # [B, q_heads, S_q, S_kv+1]

        # Drop sink column: slice axis=3 from 0 to -1 (all but last)
        scores = op.Slice(probs, [0], [-1], [3])  # [B, q_heads, S_q, S_kv]

        # Weighted sum with value: [B, q_heads, S_q, d]
        attn_out = op.MatMul(scores, value_exp)

        # Transpose and flatten heads: [B, q_heads, S_q, d] → [B, S_q, q_heads*d]
        attn_out = op.Transpose(attn_out, perm=[0, 2, 1, 3])  # [B, S_q, q_heads, d]
        attn_out = op.Reshape(attn_out, [0, 0, -1])  # [B, S_q, q_heads*d]

        # Output projection
        attn_out = self.o_proj(op, attn_out)
        return attn_out, present_key_value


class _GptOssDecoderLayer(nn.Module):
    """GPT-OSS decoder layer: pre-norm attention (with sinks) + pre-norm MoE FFN."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = _GptOssAttention(config)
        self.mlp = _GptOssMoELayer(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        # Pre-norm attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_out, present_kv = self.self_attn(
            op, hidden_states, attention_bias, position_embeddings, past_key_value
        )
        hidden_states = op.Add(residual, attn_out)

        # Pre-norm MoE FFN block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class _GptOssTextModel(nn.Module):
    """GPT-OSS text backbone with alternating sliding/full attention layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self._layer_types = config.layer_types  # list of 'sliding_attention'/'full_attention'
        self._sliding_window = config.sliding_window
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [_GptOssDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)

        # Create attention biases (full and sliding window) for dynamic dispatch
        full_attn_bias = (
            create_attention_bias(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                dtype=self._dtype,
            )
            if attention_mask is not None
            else None
        )
        sliding_attn_bias = None
        if self._sliding_window is not None and attention_mask is not None:
            sliding_attn_bias = create_attention_bias(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                sliding_window=self._sliding_window,
                dtype=self._dtype,
            )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for i, (layer, past_kv) in enumerate(zip(self.layers, past_kvs)):
            # Select bias: sliding window for 'sliding_attention' layers, full for others
            layer_type = self._layer_types[i] if self._layer_types else "full_attention"
            if layer_type == "sliding_attention" and sliding_attn_bias is not None:
                attn_bias = sliding_attn_bias
            else:
                attn_bias = full_attn_bias

            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attn_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class GPTOSSCausalLMModel(CausalLMModel):
    """GPT-OSS causal language model with MoE FFN and attention sinks.

    Architecture highlights:
    - Alternating sliding/full attention layers (``config.layer_types``)
    - Attention sinks: learned per-head scalar extends softmax denominator
    - GQA with YaRN RoPE and attention projection biases
    - MoE FFN: top-k routing with softmax scores and additive router bias
    - Custom activation: (up+1) * silu_alpha(gate), alpha=1.702

    HuggingFace model_type: ``gpt_oss``.
    """

    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = _GptOssTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF GPT-OSS weight names to our ONNX parameter names.

        Key transformations:
        - ``mlp.router.{weight,bias}`` → ``mlp.gate.{weight,bias}``
        - ``mlp.experts.gate_up_proj [N, hidden, 2*inter]``: de-interleave and
          transpose to per-expert ``gate_proj.weight`` and ``up_proj.weight [inter, hidden]``
          (HF stores transposed packed weights; gate/up interleaved at every other column)
        - ``mlp.experts.gate_up_proj_bias [N, 2*inter]``: de-interleave to per-expert
          ``gate_proj.bias`` and ``up_proj.bias [inter]``
        - ``mlp.experts.down_proj [N, inter, hidden]``: transpose per-expert to
          ``down_proj.weight [hidden, inter]``
        - ``mlp.experts.down_proj_bias [N, hidden]``: split to per-expert ``down_proj.bias``
        """
        result: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            # Router weight/bias rename
            if "mlp.router.weight" in name:
                result[name.replace("mlp.router.weight", "mlp.gate.weight")] = tensor
            elif "mlp.router.bias" in name:
                result[name.replace("mlp.router.bias", "mlp.gate.bias")] = tensor

            # Expert fused gate+up bias: [N, 2*inter] — interleaved gate/up
            elif "mlp.experts.gate_up_proj_bias" in name:
                prefix = name.replace(".mlp.experts.gate_up_proj_bias", "")
                n_exp = tensor.shape[0]
                for i in range(n_exp):
                    result[f"{prefix}.mlp.experts.{i}.gate_proj.bias"] = tensor[
                        i, ::2
                    ].contiguous()
                    result[f"{prefix}.mlp.experts.{i}.up_proj.bias"] = tensor[
                        i, 1::2
                    ].contiguous()

            # Expert fused gate+up weight: [N, hidden, 2*inter] — transposed + interleaved
            elif "mlp.experts.gate_up_proj" in name:
                prefix = name.replace(".mlp.experts.gate_up_proj", "")
                n_exp = tensor.shape[0]
                for i in range(n_exp):
                    w = tensor[i]  # [hidden, 2*inter]
                    # Gate at even columns, up at odd columns; transpose for nn.Linear format
                    result[f"{prefix}.mlp.experts.{i}.gate_proj.weight"] = w[
                        :, ::2
                    ].T.contiguous()  # [inter, hidden]
                    result[f"{prefix}.mlp.experts.{i}.up_proj.weight"] = w[
                        :, 1::2
                    ].T.contiguous()  # [inter, hidden]

            # Expert down projection bias: [N, hidden] → per-expert
            elif "mlp.experts.down_proj_bias" in name:
                prefix = name.replace(".mlp.experts.down_proj_bias", "")
                n_exp = tensor.shape[0]
                for i in range(n_exp):
                    result[f"{prefix}.mlp.experts.{i}.down_proj.bias"] = tensor[i].contiguous()

            # Expert down projection weight: [N, inter, hidden] — transposed
            elif "mlp.experts.down_proj" in name:
                prefix = name.replace(".mlp.experts.down_proj", "")
                n_exp = tensor.shape[0]
                for i in range(n_exp):
                    w = tensor[i]  # [inter, hidden]
                    result[f"{prefix}.mlp.experts.{i}.down_proj.weight"] = (
                        w.T.contiguous()  # [hidden, inter]
                    )

            else:
                result[name] = tensor

        return super().preprocess_weights(result)
