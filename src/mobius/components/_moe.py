# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Mixture of Experts (MoE) components."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._mlp import MLP

if TYPE_CHECKING:
    import onnx_ir as ir


class TopKGate(nn.Module):
    """Standard top-k expert routing gate.

    Selects top-k experts by logit value and normalizes routing weights
    with softmax over the selected experts.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter([num_experts, hidden_size])

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)
        k = op.Constant(value_ints=[self.top_k])
        routing_weights, selected_experts = op.TopK(router_logits, k, axis=-1, _outputs=2)
        routing_weights = op.Softmax(routing_weights, axis=-1)
        return routing_weights, selected_experts


class SoftmaxTopKGate(nn.Module):
    """Softmax-first top-k expert routing gate (Qwen3-Next style).

    Applies softmax over all expert logits first, then selects top-k.
    Optionally renormalizes the selected weights to sum to 1.
    """

    def __init__(
        self, hidden_size: int, num_experts: int, top_k: int, *, norm_topk_prob: bool = True
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter([num_experts, hidden_size])

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)
        # Softmax over all experts first
        routing_probs = op.Softmax(router_logits, axis=-1)
        k = op.Constant(value_ints=[self.top_k])
        routing_weights, selected_experts = op.TopK(routing_probs, k, axis=-1, _outputs=2)
        if self.norm_topk_prob:
            # Renormalize selected weights to sum to 1
            weight_sum = op.ReduceSum(routing_weights, [-1], keepdims=True)
            routing_weights = op.Div(routing_weights, weight_sum)
        return routing_weights, selected_experts


class SigmoidTopKGate(nn.Module):
    """Sigmoid-first top-k expert routing gate (GLM4-MoE style).

    Applies element-wise sigmoid over all expert logits, selects top-k,
    and optionally renormalizes the selected weights to sum to 1.
    Used by GLM4-MoE where group routing collapses to standard top-k
    when n_group=1 (all experts in one group).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.weight = nn.Parameter([num_experts, hidden_size])

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)
        # Sigmoid instead of softmax: each expert scored independently
        routing_probs = op.Sigmoid(router_logits)
        k = op.Constant(value_ints=[self.top_k])
        routing_weights, selected_experts = op.TopK(routing_probs, k, axis=-1, _outputs=2)
        if self.norm_topk_prob:
            # Renormalize selected weights to sum to 1 (prevents vanishing gradients)
            weight_sum = op.ReduceSum(routing_weights, [-1], keepdims=True)
            eps = op.CastLike(op.Constant(value_float=1e-9), routing_weights)
            routing_weights = op.Div(routing_weights, op.Add(weight_sum, eps))
        if self.routed_scaling_factor != 1.0:  # noqa: RUF069
            scale = op.CastLike(
                op.Constant(value_float=self.routed_scaling_factor), routing_weights
            )
            routing_weights = op.Mul(routing_weights, scale)
        return routing_weights, selected_experts


class SparseMixerGate(nn.Module):
    """Sparsemixer-style routing gate (used by PhiMoE).

    Implements the inference-mode routing from HuggingFace PhiMoE:
    experts are selected sequentially with a threshold mask that filters
    out experts whose logits are relatively far from the maximum. For
    each round, softmax is computed over the non-masked experts, and the
    weight of the selected expert is its softmax probability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        jitter_eps: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_eps = jitter_eps
        self.weight = nn.Parameter([num_experts, hidden_size])

    def _threshold_mask_and_select(self, op, scores, jitter_eps):
        """Apply threshold mask and select the top expert."""
        max_score = op.ReduceMax(scores, [-1], keepdims=True)
        abs_scores = op.Abs(scores)
        factor = op.Max(abs_scores, max_score)
        diff = op.Sub(max_score, scores)
        ratio = op.Div(diff, factor)
        threshold = op.Constant(value_float=2.0 * jitter_eps)
        mask = op.Greater(ratio, threshold)
        neg_inf = op.Constant(value_float=-1e30)
        masked_scores = op.Where(mask, neg_inf, scores)
        weights = op.Softmax(masked_scores, axis=-1)
        k_one = op.Constant(value_ints=[1])
        _top_val, expert_idx = op.TopK(masked_scores, k_one, axis=-1, _outputs=2)
        expert_weight = op.GatherElements(weights, expert_idx, axis=-1)
        return expert_weight, expert_idx

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)

        all_weights = []
        all_experts = []
        current_scores = router_logits

        for _k in range(self.top_k):
            weight_k, expert_k = self._threshold_mask_and_select(
                op, current_scores, self.jitter_eps
            )
            all_weights.append(weight_k)
            all_experts.append(expert_k)
            neg_inf = op.Constant(value_float=-1e30)
            current_scores = op.ScatterElements(
                current_scores,
                expert_k,
                op.Expand(neg_inf, op.Shape(expert_k)),
                axis=-1,
            )

        routing_weights = op.Concat(*all_weights, axis=-1)
        selected_experts = op.Concat(*all_experts, axis=-1)
        return routing_weights, selected_experts


class MoELayer(nn.Module):
    """Mixture of Experts layer.

    Routes each token to top-k experts via a gating mechanism, applies
    each expert MLP, and accumulates weighted results.

    Uses loop-over-experts dispatch: each expert processes all tokens,
    then results are masked and weighted.
    """

    def __init__(self, config: ArchitectureConfig, gate: nn.Module | None = None):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        if gate is not None:
            self.gate = gate
        else:
            self.gate = TopKGate(config.hidden_size, self.num_experts, self.top_k)
        # Use moe_intermediate_size for experts when specified (Qwen2-MoE, Qwen3-MoE).
        expert_config = (
            dataclasses.replace(config, intermediate_size=config.moe_intermediate_size)
            if config.moe_intermediate_size is not None
            else config
        )
        self.experts = nn.ModuleList([MLP(expert_config) for _ in range(self.num_experts)])

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        routing_weights, selected_experts = self.gate(op, hidden_states)

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
