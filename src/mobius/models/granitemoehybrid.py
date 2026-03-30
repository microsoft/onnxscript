# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""GraniteMoeHybrid: Mamba2/SSD + Attention hybrid with MoE FFN on all layers.

Every layer has both a routed MoE block (``block_sparse_moe``) and a dense
shared MLP (``shared_mlp``). The layer type ("mamba2" or "full_attention")
controls whether the attention sub-block is a Mamba2/SSD or standard GQA.
Attention layers use NoPE (no rotary position embeddings).

Forward pass per layer::

    residual = x
    x = input_layernorm(x)
    x = mamba(x)  OR  self_attn(x, position_embeddings=None)
    x = residual + x * residual_multiplier
    residual = x
    x = post_attention_layernorm(x)
    x = block_sparse_moe(x) + shared_mlp(x)
    x = residual + x * residual_multiplier

HuggingFace reference: ``GraniteMoeHybridForCausalLM``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import GraniteMoeHybridConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    Linear,
    Mamba2Block,
    MoELayer,
    RMSNorm,
    TopKGate,
    create_attention_bias,
)

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


class _GraniteMoeHybridMambaDecoderLayer(nn.Module):
    """GraniteMoeHybrid Mamba2 layer.

    input_layernorm → Mamba2Block → residual+scale →
    post_attention_layernorm → block_sparse_moe + shared_mlp → residual+scale.

    Args:
        config: GraniteMoeHybrid architecture config.
    """

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        d_inner = config.hidden_size * config.mamba_expand

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mamba = Mamba2Block(
            d_model=config.hidden_size,
            d_inner=d_inner,
            num_heads=config.mamba_n_heads,
            d_head=config.mamba_d_head,
            d_state=config.mamba_d_state,
            n_groups=config.mamba_n_groups,
            conv_kernel=config.mamba_d_conv,
            conv_bias=config.mamba_conv_bias,
            proj_bias=config.mamba_proj_bias,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Routed MoE: top-k expert selection with softmax weighting
        gate = TopKGate(
            config.hidden_size, config.num_local_experts, config.num_experts_per_tok
        )
        self.block_sparse_moe = MoELayer(config, gate=gate)

        # Dense shared MLP: runs on every token unconditionally
        shared_config = dataclasses.replace(
            config, intermediate_size=config.shared_intermediate_size
        )
        self.shared_mlp = MLP(shared_config)

        self._residual_multiplier = config.residual_multiplier

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (conv_state, ssm_state)).

        attention_bias and position_embeddings are unused by Mamba layers
        but accepted for a uniform interface with attention layers.
        """
        del attention_bias, position_embeddings  # unused by mamba layers

        # Mamba2/SSD path with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        conv_state, ssm_state = past_key_value if past_key_value is not None else (None, None)
        mamba_out, new_conv_state, new_ssm_state = self.mamba(
            op, hidden_states, conv_state, ssm_state
        )
        # residual + output * residual_multiplier
        rm = op.CastLike(op.Constant(value_float=self._residual_multiplier), mamba_out)
        hidden_states = op.Add(residual, op.Mul(mamba_out, rm))

        # MoE + shared-MLP path with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        # Both routed MoE and shared MLP run on every layer; outputs are summed
        hidden_states = op.Add(
            self.block_sparse_moe(op, hidden_states),
            self.shared_mlp(op, hidden_states),
        )
        rm = op.CastLike(op.Constant(value_float=self._residual_multiplier), hidden_states)
        hidden_states = op.Add(residual, op.Mul(hidden_states, rm))

        return hidden_states, (new_conv_state, new_ssm_state)


class _GraniteMoeHybridAttentionDecoderLayer(nn.Module):
    """GraniteMoeHybrid attention layer.

    input_layernorm → GQA (NoPE) → residual+scale →
    post_attention_layernorm → block_sparse_moe + shared_mlp → residual+scale.

    Args:
        config: GraniteMoeHybrid architecture config.
    """

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # GraniteMoeHybrid attention uses NoPE (no position embeddings)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Routed MoE: top-k expert selection with softmax weighting
        gate = TopKGate(
            config.hidden_size, config.num_local_experts, config.num_experts_per_tok
        )
        self.block_sparse_moe = MoELayer(config, gate=gate)

        # Dense shared MLP: runs on every token unconditionally
        shared_config = dataclasses.replace(
            config, intermediate_size=config.shared_intermediate_size
        )
        self.shared_mlp = MLP(shared_config)

        self._residual_multiplier = config.residual_multiplier

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (key, value))."""
        del position_embeddings  # GraniteMoeHybrid uses NoPE: no rotary embeddings

        # GQA attention path (no RoPE)
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_out, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=None,  # NoPE: skip rotary embedding application
            past_key_value=past_key_value,
        )
        rm = op.CastLike(op.Constant(value_float=self._residual_multiplier), attn_out)
        hidden_states = op.Add(residual, op.Mul(attn_out, rm))

        # MoE + shared-MLP path with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = op.Add(
            self.block_sparse_moe(op, hidden_states),
            self.shared_mlp(op, hidden_states),
        )
        rm = op.CastLike(op.Constant(value_float=self._residual_multiplier), hidden_states)
        hidden_states = op.Add(residual, op.Mul(hidden_states, rm))

        return hidden_states, present_kv


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class _GraniteMoeHybridTextModel(nn.Module):
    """GraniteMoeHybrid text backbone: embedding -> N x (Mamba2|Attention) layers -> norm.

    Layer type ("mamba2" or "full_attention") is read from ``config.layer_types``.
    No rotary embeddings are used (NoPE for attention layers).
    """

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        layer_types = config.layer_types or []
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            ltype = layer_types[i] if i < len(layer_types) else "mamba2"
            if ltype == "mamba2":
                self.layers.append(_GraniteMoeHybridMambaDecoderLayer(config))
            else:
                self.layers.append(_GraniteMoeHybridAttentionDecoderLayer(config))

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # No rotary_emb: GraniteMoeHybrid uses NoPE (no positional encodings)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        del position_ids  # unused: NoPE architecture has no positional embeddings

        # (batch, seq, hidden)
        hidden_states = self.embed_tokens(op, input_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=None,  # NoPE: no RoPE
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.final_layernorm(op, hidden_states)
        return hidden_states, present_key_values


class GraniteMoeHybridCausalLMModel(nn.Module):
    """GraniteMoeHybrid hybrid Mamba2+Attention causal language model with MoE FFN.

    Every layer has both a routed MoE block (``block_sparse_moe``) and a dense
    shared MLP (``shared_mlp``). Mamba2 layers use the SSD selective scan;
    attention layers use standard GQA without rotary position embeddings (NoPE).

    Uses ``HybridCausalLMTask`` with mixed ``"mamba2"`` and ``"full_attention"``
    layer types for the KV/SSM cache.

    HuggingFace reference: ``GraniteMoeHybridForCausalLM``.
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid SSM+Attention"
    config_class: type = GraniteMoeHybridConfig

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        self.config = config
        self.model = _GraniteMoeHybridTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace GraniteMoeHybridForCausalLM weights to ONNX parameters.

        Handles:
        1. Weight tying (embed_tokens ↔ lm_head)
        2. Mamba2 SSM params: A_log, D, dt_bias nested under mamba.ssm
        3. MoE gate: block_sparse_moe.router.layer.weight → block_sparse_moe.gate.weight
        4. MoE fused input: block_sparse_moe.input_linear [n_experts, 2*mid, hidden]
           → per-expert block_sparse_moe.experts.{e}.{gate,up}_proj.weight
        5. MoE fused output: block_sparse_moe.output_linear [n_experts, hidden, mid]
           → per-expert block_sparse_moe.experts.{e}.down_proj.weight
        6. Shared MLP fused gate+up: shared_mlp.input_linear [2*shared_mid, hidden]
           → shared_mlp.gate_proj.weight + shared_mlp.up_proj.weight
        7. Shared MLP down proj: shared_mlp.output_linear → shared_mlp.down_proj
        """
        if self.config.tie_word_embeddings:
            tie_word_embeddings(state_dict)

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_granitemoehybrid_weight(key, value, new_state_dict)
            if new_key is not None:
                new_state_dict[new_key] = value

        return new_state_dict


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------

# Mamba2 SSM params stored flat on HF "mamba" that we nest under "mamba.ssm"
_MAMBA2_SSM_PARAMS = (".mamba.A_log", ".mamba.D", ".mamba.dt_bias")


def _rename_granitemoehybrid_weight(
    key: str,
    value: torch.Tensor,
    out: dict[str, torch.Tensor],
) -> str | None:
    """Rename a single HF GraniteMoeHybrid weight key to our ONNX naming.

    Returns the new key, or None if the weight was handled inline
    (fused tensors split into multiple per-expert outputs).
    """
    # SSM params: nest A_log, D, dt_bias under mamba.ssm
    # e.g. "model.layers.0.mamba.A_log" → "model.layers.0.mamba.ssm.A_log"
    for param in _MAMBA2_SSM_PARAMS:
        if key.endswith(param):
            return key.replace(".mamba.", ".mamba.ssm.")

    # MoE gate: router.layer.weight → gate.weight
    # e.g. "…block_sparse_moe.router.layer.weight" → "…block_sparse_moe.gate.weight"
    key = key.replace(".block_sparse_moe.router.layer.", ".block_sparse_moe.gate.")

    # MoE fused input_linear [n_experts, 2*intermediate, hidden]
    # → per-expert {gate,up}_proj.weight
    if ".block_sparse_moe.input_linear" in key:
        _split_moe_fused_gate_up(key, value, out)
        return None  # handled inline

    # MoE fused output_linear [n_experts, hidden, intermediate]
    # → per-expert down_proj.weight
    if ".block_sparse_moe.output_linear" in key:
        _split_moe_fused_down(key, value, out)
        return None  # handled inline

    # Shared MLP fused input_linear [2*shared_intermediate, hidden]
    # → gate_proj.weight + up_proj.weight
    if ".shared_mlp.input_linear" in key:
        _split_shared_mlp_fused_gate_up(key, value, out)
        return None  # handled inline

    # Shared MLP down proj: output_linear → down_proj
    # e.g. "…shared_mlp.output_linear.weight" → "…shared_mlp.down_proj.weight"
    if ".shared_mlp.output_linear" in key:
        return key.replace(".shared_mlp.output_linear", ".shared_mlp.down_proj")

    return key


def _split_moe_fused_gate_up(
    key: str,
    value: torch.Tensor,
    out: dict[str, torch.Tensor],
) -> None:
    """Expand fused MoE input_linear into per-expert gate_proj + up_proj.

    HF stores:
        ``layers.{i}.block_sparse_moe.input_linear.weight``
        with shape ``[n_experts, 2*intermediate, hidden]``

    We need per-expert:
        ``layers.{i}.block_sparse_moe.experts.{e}.gate_proj.weight``
        ``layers.{i}.block_sparse_moe.experts.{e}.up_proj.weight``
    """
    # Derive the base path: e.g. "model.layers.0.block_sparse_moe"
    sep = ".block_sparse_moe.input_linear"
    prefix = key[: key.index(sep)]
    base = f"{prefix}.block_sparse_moe"

    n_experts = value.shape[0]
    intermediate = value.shape[1] // 2
    for e in range(n_experts):
        expert_w = value[e]  # [2*intermediate, hidden]
        out[f"{base}.experts.{e}.gate_proj.weight"] = expert_w[:intermediate]
        out[f"{base}.experts.{e}.up_proj.weight"] = expert_w[intermediate:]


def _split_moe_fused_down(
    key: str,
    value: torch.Tensor,
    out: dict[str, torch.Tensor],
) -> None:
    """Expand fused MoE output_linear into per-expert down_proj.

    HF stores:
        ``layers.{i}.block_sparse_moe.output_linear.weight``
        with shape ``[n_experts, hidden, intermediate]``

    We need per-expert:
        ``layers.{i}.block_sparse_moe.experts.{e}.down_proj.weight``
    """
    sep = ".block_sparse_moe.output_linear"
    prefix = key[: key.index(sep)]
    base = f"{prefix}.block_sparse_moe"

    n_experts = value.shape[0]
    for e in range(n_experts):
        out[f"{base}.experts.{e}.down_proj.weight"] = value[e]


def _split_shared_mlp_fused_gate_up(
    key: str,
    value: torch.Tensor,
    out: dict[str, torch.Tensor],
) -> None:
    """Expand fused shared_mlp input_linear into gate_proj + up_proj.

    HF stores:
        ``layers.{i}.shared_mlp.input_linear.weight``
        with shape ``[2*shared_intermediate, hidden]``

    We need:
        ``layers.{i}.shared_mlp.gate_proj.weight``
        ``layers.{i}.shared_mlp.up_proj.weight``
    """
    sep = ".shared_mlp.input_linear"
    prefix = key[: key.index(sep)]

    intermediate = value.shape[0] // 2
    out[f"{prefix}.shared_mlp.gate_proj.weight"] = value[:intermediate]
    out[f"{prefix}.shared_mlp.up_proj.weight"] = value[intermediate:]
