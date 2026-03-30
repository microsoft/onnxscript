# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""JetMoE causal language model with Mixture-of-Attention (MoA).

JetMoE uses expert-routed Q and O projections with shared KV, called
Mixture-of-Attention (MoA). Each of the top_k expert slots produces a
separate Q by conditionally accumulating per-expert input_linear projections,
then runs standard GQA attention, and applies the O projection from the same
selected expert scaled by the routing gate.

The MLP block is a standard MoE FFN (TopKGate + per-expert SiLU MLPs)
with an additive output bias.

HuggingFace reference: ``JetMoeForCausalLM``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    Embedding,
    Linear,
    MoELayer,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._attention import _apply_attention
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class _JetMoARouter(nn.Module):
    """Top-k routing gate for Mixture-of-Attention.

    Computes routing logits via a weight matrix, selects top-k experts,
    and normalizes weights with softmax.

    Attribute ``weight`` maps to HF ``self_attention.experts.router.layer.weight``
    after preprocess_weights renames it to ``self_attn.router.weight``.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter([num_experts, hidden_size])

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # routing logits: [bsz, seq, n_experts]
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        logits = op.MatMul(hidden_states, weight_t)
        k = op.Constant(value_ints=[self.top_k])
        top_k_logits, selected_experts = op.TopK(logits, k, axis=-1, _outputs=2)
        # routing_weights [bsz, seq, top_k], selected_experts [bsz, seq, top_k] (int64)
        routing_weights = op.Softmax(top_k_logits, axis=-1)
        return routing_weights, selected_experts


class _JetMoeAttention(nn.Module):
    """Mixture-of-Attention (MoA) module for JetMoE.

    Selects top_k experts for Q and O projections; K and V are shared
    (no routing). Each expert slot produces a separate Q using its
    selected expert's input_linear weight, runs GQA attention, then
    accumulates the O projection of the selected expert weighted by
    the routing gate.

    Args:
        config: Architecture configuration. ``num_attention_heads`` must
            equal ``num_experts_per_tok * num_key_value_heads``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        # num_q_heads = top_k * num_kv_heads (all Q heads across all slots)
        self.num_q_heads = config.num_attention_heads
        # q_size is the Q/K/V projection size per-expert = num_kv_heads * head_dim
        self.q_size = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        # Per-expert Q projections: weight [q_size, hidden]
        # Named q_proj_experts.{i} to match preprocess_weights output keys.
        self.q_proj_experts = nn.ModuleList(
            [
                Linear(self.hidden_size, self.q_size, bias=False)
                for _ in range(self.num_experts)
            ]
        )

        # Per-expert O projections: weight [hidden, q_size]
        # Named o_proj_experts.{i} to match preprocess_weights output keys.
        self.o_proj_experts = nn.ModuleList(
            [
                Linear(self.q_size, self.hidden_size, bias=False)
                for _ in range(self.num_experts)
            ]
        )

        # Expert router: weight [n_experts, hidden]
        self.router = _JetMoARouter(self.hidden_size, self.num_experts, self.top_k)

        # Shared KV projection: weight [2*q_size, hidden]
        self.kv_proj = Linear(self.hidden_size, 2 * self.q_size, bias=False)

        # Additive output bias: shape [hidden]
        self.bias = nn.Parameter([self.hidden_size])

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        """MoA forward pass.

        Returns:
            tuple[ir.Value, tuple]: (output [bsz, seq, hidden], (present_key, present_value))
        """
        # --- Step 1: Route tokens to top_k expert slots ---
        # routing_weights: [bsz, seq, top_k]  (softmax-normalized)
        # selected_experts: [bsz, seq, top_k]  (int64 expert indices)
        routing_weights, selected_experts = self.router(op, hidden_states)

        # --- Step 2: Precompute all expert Q projections ---
        # q_e: [bsz, seq, q_size]  for each expert e
        q_per_expert = [
            self.q_proj_experts[e](op, hidden_states) for e in range(self.num_experts)
        ]

        # --- Step 3: Gather Q for each top_k slot via conditional accumulation ---
        q_slots = []
        for k in range(self.top_k):
            k_scalar = op.Constant(value_int=k)
            # sel_k: [bsz, seq]  — which expert index was selected for slot k
            sel_k = op.Gather(selected_experts, k_scalar, axis=2)
            # Accumulate weighted expert Qs: q_k[b,s] = q_{sel_k[b,s]}[b,s]
            q_k = None
            for e in range(self.num_experts):
                e_id = op.Constant(value_int=e)
                # match: [bsz, seq] bool  — tokens where slot k chose expert e
                match = op.Equal(sel_k, e_id)
                # mask_3d: [bsz, seq, 1] float  — for broadcasting over q_size
                mask_3d = op.Unsqueeze(op.Cast(match, to=1), op.Constant(value_ints=[-1]))
                weighted_q = op.Mul(q_per_expert[e], mask_3d)  # [bsz, seq, q_size]
                q_k = weighted_q if q_k is None else op.Add(q_k, weighted_q)
            # q_k: [bsz, seq, q_size] — Q for slot k
            # Unsqueeze along slot dim for later Concat → [bsz, seq, 1, q_size]
            q_slots.append(op.Unsqueeze(q_k, op.Constant(value_ints=[2])))

        # --- Step 4: Stack Q slots, rearrange heads for ONNX GQA, and reshape ---
        # Concat → [bsz, seq, top_k, q_size]  where q_size = num_kv_heads * head_dim
        query_4d = op.Concat(*q_slots, axis=2)
        # Expand to 5D: [bsz, seq, top_k, num_kv_heads, head_dim]
        query_5d = op.Reshape(query_4d, [0, 0, self.top_k, self.num_kv_heads, self.head_dim])
        # Transpose from [slot, kv_head] to [kv_head, slot] order.
        # ONNX GQA groups Q heads in contiguous blocks of size top_k and maps each
        # block to one KV head.  Without this transpose the grouping would be
        # [q_s0_h0, q_s0_h1, q_s1_h0, q_s1_h1] → groups {h0,h1} and {h0,h1} (wrong),
        # while HF JetMoE pairs Q[i] with K[i % num_kv_heads] via repeat().
        # After transpose the order becomes [q_s0_h0, q_s1_h0, q_s0_h1, q_s1_h1]:
        #   group 0 → K head 0 and group 1 → K head 1 (correct GQA pairing).
        # [bsz, seq, top_k, num_kv_heads, head_dim] → [bsz, seq, num_kv_heads, top_k, head_dim]
        query_5d = op.Transpose(query_5d, perm=[0, 1, 3, 2, 4])
        # Flatten to [bsz, seq, num_q_heads * head_dim]
        query_states = op.Reshape(query_5d, [0, 0, -1])

        # --- Step 5: Shared KV projection ---
        # kv: [bsz, seq, 2*q_size]
        kv = self.kv_proj(op, hidden_states)
        # Split into K and V: each [bsz, seq, q_size] = [bsz, seq, num_kv_heads * head_dim]
        key_states, value_states = op.Split(kv, num_outputs=2, axis=-1, _outputs=2)

        # --- Step 6: Apply RoPE to Q (num_q_heads) and K (num_kv_heads) ---
        query_states = apply_rotary_pos_emb(
            op,
            query_states,
            position_embeddings,
            num_heads=self.num_q_heads,
            rotary_embedding_dim=self.head_dim,  # full head_dim used for RoPE
        )
        key_states = apply_rotary_pos_emb(
            op,
            key_states,
            position_embeddings,
            num_heads=self.num_kv_heads,
            rotary_embedding_dim=self.head_dim,
        )

        # --- Step 7: Standard GQA attention ---
        # attn_output: [bsz, seq, num_q_heads * head_dim]
        # present_key/value: [bsz, num_kv_heads, total_seq, head_dim]
        attn_output, present_key, present_value = _apply_attention(
            op,
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            num_attention_heads=self.num_q_heads,
            num_key_value_heads=self.num_kv_heads,
            scale=self.scale,
        )

        # --- Step 8: Reverse-transpose attn output back to [slot, kv_head] order ---
        # [bsz, seq, num_q_heads * head_dim] → [bsz, seq, num_kv_heads, top_k, head_dim]
        attn_out_5d = op.Reshape(
            attn_output, [0, 0, self.num_kv_heads, self.top_k, self.head_dim]
        )
        # Transpose back from [kv_head, slot] to [slot, kv_head]:
        # [bsz, seq, num_kv_heads, top_k, head_dim] → [bsz, seq, top_k, num_kv_heads, head_dim]
        attn_out_5d = op.Transpose(attn_out_5d, perm=[0, 1, 3, 2, 4])
        # Reshape to [bsz, seq, top_k, q_size] for per-slot O projection
        attn_output_4d = op.Reshape(attn_out_5d, [0, 0, self.top_k, self.q_size])

        # --- Step 9: Apply expert O projections with routing gate weighting ---
        # For each slot k and expert e, accumulate: gate_k * (sel_k == e) * o_e(attn_k)
        output = None
        for k in range(self.top_k):
            k_scalar = op.Constant(value_int=k)
            # sel_k: [bsz, seq]  (expert index selected for slot k)
            sel_k = op.Gather(selected_experts, k_scalar, axis=2)
            # gate_k: [bsz, seq, 1]  (routing weight for slot k)
            gate_k = op.Unsqueeze(
                op.Gather(routing_weights, k_scalar, axis=2),
                op.Constant(value_ints=[-1]),
            )
            # attn_k: [bsz, seq, q_size]  (attention output for slot k)
            attn_k = op.Gather(attn_output_4d, k_scalar, axis=2)

            for e in range(self.num_experts):
                e_id = op.Constant(value_int=e)
                # match_3d: [bsz, seq, 1]  (1 where slot k chose expert e)
                match_3d = op.Unsqueeze(
                    op.Cast(op.Equal(sel_k, e_id), to=1),
                    op.Constant(value_ints=[-1]),
                )
                # o_e: [bsz, seq, hidden]
                o_e = self.o_proj_experts[e](op, attn_k)
                contribution = op.Mul(op.Mul(gate_k, match_3d), o_e)
                output = contribution if output is None else op.Add(output, contribution)

        # Add shared output bias: [hidden] broadcasts over [bsz, seq, hidden]
        output = op.Add(output, self.bias)
        return output, (present_key, present_value)


class _JetMoeMLP(nn.Module):
    """JetMoE MoE FFN with additive bias.

    Wraps a standard ``MoELayer`` (TopKGate + per-expert SiLU MLPs)
    and adds a learnable bias to the output.

    Attribute ``moe`` maps to ``mlp.moe.gate.weight`` / ``mlp.moe.experts.*``
    and ``bias`` maps to ``mlp.bias`` after preprocess_weights.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.moe = MoELayer(config)
        # Additive output bias: [hidden]
        self.bias = nn.Parameter([config.hidden_size])

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        return op.Add(self.moe(op, hidden_states), self.bias)


class JetMoeDecoderLayer(nn.Module):
    """JetMoE decoder layer: pre-norm with MoA attention and MoE FFN.

    Structure (pre-norm):
        residual = x
        x = input_layernorm(x)
        x, kv = self_attn(x, ...)  # MoA attention
        x = residual + x
        residual = x
        x = post_attention_layernorm(x)
        x = mlp(x)                 # MoE FFN
        x = residual + x

    Args:
        config: Architecture configuration.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = _JetMoeAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = _JetMoeMLP(config)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass.

        Args:
            hidden_states: [bsz, seq, hidden]
            attention_bias: Optional padding attention bias.
            position_embeddings: (cos, sin) tuple from RoPE.
            past_key_value: Optional (past_key, past_value) KV cache.

        Returns:
            tuple: (hidden_states [bsz, seq, hidden], (present_key, present_value))
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_output, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class _JetMoeTextModel(nn.Module):
    """JetMoE text model: embedding + JetMoE decoder layers + final RMSNorm."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [JetMoeDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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

        if attention_mask is not None:
            attention_bias = create_attention_bias(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                dtype=self._dtype,
            )
        else:
            attention_bias = None

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class JetMoeCausalLMModel(CausalLMModel):
    """JetMoE causal language model with Mixture-of-Attention.

    Differs from generic MoE models by routing Q and O projections
    through experts (MoA), while K and V projections are shared.

    HuggingFace reference: ``JetMoeForCausalLM``.
    """

    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = _JetMoeTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF JetMoE weight names to ONNX module attribute names.

        Key transforms:
        - ``self_attention.*`` → ``self_attn.*``
        - ``self_attn.experts.input_linear.weight`` [n, q, h] → per-expert q_proj
        - ``self_attn.experts.output_linear.weight`` [n, h, q] → per-expert o_proj
        - ``self_attn.experts.router.layer.weight`` → ``self_attn.router.weight``
        - ``self_attn.experts.bias`` → ``self_attn.bias``
        - ``mlp.input_linear.weight`` [n, 2*inter, h] → per-expert gate+up proj
        - ``mlp.output_linear.weight`` [n, h, inter] → per-expert down proj
        - ``mlp.router.layer.weight`` → ``mlp.moe.gate.weight``
        """
        assert self.config.num_local_experts is not None
        assert self.config.num_experts_per_tok is not None
        n_experts = self.config.num_local_experts

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Rename self_attention → self_attn
            key = key.replace("self_attention.", "self_attn.")

            # Split fused Q input_linear [n_experts, q_size, hidden] → per-expert q_proj
            if key.endswith("self_attn.experts.input_linear.weight"):
                prefix = key[: -len("self_attn.experts.input_linear.weight")]
                for i in range(n_experts):
                    renamed[f"{prefix}self_attn.q_proj_experts.{i}.weight"] = value[i]
                continue

            # Split fused O output_linear [n_experts, hidden, q_size] → per-expert o_proj
            if key.endswith("self_attn.experts.output_linear.weight"):
                prefix = key[: -len("self_attn.experts.output_linear.weight")]
                for i in range(n_experts):
                    renamed[f"{prefix}self_attn.o_proj_experts.{i}.weight"] = value[i]
                continue

            # router.layer.weight → router.weight
            key = key.replace("self_attn.experts.router.layer.", "self_attn.router.")

            # experts.bias → bias (attention output bias)
            key = key.replace("self_attn.experts.bias", "self_attn.bias")

            # Split FFN input_linear [n_experts, 2*intermediate, hidden]
            # → per-expert gate_proj [intermediate, hidden] and up_proj [intermediate, hidden]
            if key.endswith("mlp.input_linear.weight"):
                prefix = key[: -len("mlp.input_linear.weight")]
                mid = value.shape[1] // 2
                for i in range(n_experts):
                    renamed[f"{prefix}mlp.moe.experts.{i}.gate_proj.weight"] = value[i, :mid]
                    renamed[f"{prefix}mlp.moe.experts.{i}.up_proj.weight"] = value[i, mid:]
                continue

            # Split FFN output_linear [n_experts, hidden, intermediate] → per-expert down_proj
            if key.endswith("mlp.output_linear.weight"):
                prefix = key[: -len("mlp.output_linear.weight")]
                for i in range(n_experts):
                    renamed[f"{prefix}mlp.moe.experts.{i}.down_proj.weight"] = value[i]
                continue

            # mlp.router.layer.weight → mlp.moe.gate.weight
            key = key.replace("mlp.router.layer.", "mlp.moe.gate.")

            renamed[key] = value

        if self.config.tie_word_embeddings:
            tie_word_embeddings(renamed)
        return renamed
