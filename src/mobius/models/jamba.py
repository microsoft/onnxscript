# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Jamba hybrid SSM+Attention causal language model.

Jamba interleaves Mamba SSM layers with Transformer attention layers.
Some layers use Mixture-of-Experts (MoE) MLPs instead of dense MLPs.

Layer type selection (per HuggingFace JambaConfig):
    - Attention if ``(i - attn_layer_offset) % attn_layer_period == 0``
    - Mamba otherwise
    - MoE MLP if ``(i - expert_layer_offset) % expert_layer_period == 0``
    - Dense MLP otherwise

State per layer:
    Attention layers: KV cache (key + value)
    Mamba layers: conv_state + ssm_state

HuggingFace reference: ``JambaForCausalLM``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import JambaConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    Linear,
    MambaBlock,
    MoELayer,
    RMSNorm,
    TopKGate,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._ssm import JambaSelectiveScan

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


class JambaMambaDecoderLayer(nn.Module):
    """Jamba Mamba layer: RMSNorm → MambaBlock → residual + optional MoE MLP.

    Uses JambaSelectiveScan (with dt/B/C layernorms) inside MambaBlock.

    Args:
        config: Jamba architecture config.
        use_moe: Whether this layer uses MoE MLP (vs dense MLP).
    """

    def __init__(self, config: JambaConfig, *, use_moe: bool = False):
        super().__init__()
        d_inner = config.hidden_size * config.mamba_expand
        dt_rank = config.mamba_dt_rank
        if dt_rank == -1 or dt_rank == 0:
            dt_rank = math.ceil(config.hidden_size / 16)

        # Mamba block with Jamba-specific SSM (layernormed dt/B/C)
        self.mamba = _JambaMambaBlock(
            d_model=config.hidden_size,
            d_inner=d_inner,
            d_state=config.mamba_d_state,
            dt_rank=dt_rank,
            conv_kernel=config.mamba_d_conv,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP: MoE or dense
        if use_moe:
            gate = TopKGate(
                config.hidden_size,
                config.num_local_experts,
                config.num_experts_per_tok,
            )
            self.feed_forward = MoELayer(config, gate=gate)
        else:
            self.feed_forward = MLP(config)
        self.pre_moe_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        but accepted for uniform interface with attention layers.
        """
        del attention_bias, position_embeddings  # unused

        # Mamba path with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        conv_state, ssm_state = past_key_value if past_key_value is not None else (None, None)
        mamba_out, new_conv_state, new_ssm_state = self.mamba(
            op, hidden_states, conv_state, ssm_state
        )
        hidden_states = op.Add(residual, mamba_out)

        # MLP path with pre-norm
        residual = hidden_states
        hidden_states = self.pre_moe_layernorm(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, (new_conv_state, new_ssm_state)


class JambaAttentionDecoderLayer(nn.Module):
    """Jamba attention layer: RMSNorm → Attention → residual + optional MoE.

    Standard transformer decoder layer with optional MoE MLP.

    Args:
        config: Jamba architecture config.
        use_moe: Whether this layer uses MoE MLP (vs dense MLP).
    """

    def __init__(self, config: JambaConfig, *, use_moe: bool = False):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP: MoE or dense
        if use_moe:
            gate = TopKGate(
                config.hidden_size,
                config.num_local_experts,
                config.num_experts_per_tok,
            )
            self.feed_forward = MoELayer(config, gate=gate)
        else:
            self.feed_forward = MLP(config)
        self.pre_moe_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (key, value))."""
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

        # MLP path with pre-norm
        residual = hidden_states
        hidden_states = self.pre_moe_layernorm(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class _JambaTextModel(nn.Module):
    """Jamba text backbone: embedding -> N x (Mamba|Attention) layers -> norm.

    Alternates between Mamba and Attention layers based on
    ``attn_layer_period`` / ``attn_layer_offset``. MoE vs dense MLP
    is controlled by ``expert_layer_period`` / ``expert_layer_offset``.
    """

    def __init__(self, config: JambaConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        layer_types = config.layer_types or []
        self.layers = nn.ModuleList([])
        expert_period = getattr(config, "expert_layer_period", 1)
        expert_offset = getattr(config, "expert_layer_offset", 0)
        for i in range(config.num_hidden_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            use_moe = (
                config.num_local_experts is not None
                and config.num_local_experts > 1
                and (i - expert_offset) % expert_period == 0
            )
            if ltype == "mamba":
                self.layers.append(JambaMambaDecoderLayer(config, use_moe=use_moe))
            else:
                self.layers.append(JambaAttentionDecoderLayer(config, use_moe=use_moe))

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)

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
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.final_layernorm(op, hidden_states)
        return hidden_states, present_key_values


class JambaCausalLMModel(nn.Module):
    """Jamba hybrid SSM+Attention causal language model.

    Uses ``HybridCausalLMTask`` with mixed ``"mamba"`` and
    ``"full_attention"`` layer types for the KV/SSM cache.

    HuggingFace reference: ``JambaForCausalLM``.
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid SSM+Attention"
    config_class: type = JambaConfig

    def __init__(self, config: JambaConfig):
        super().__init__()
        self.config = config
        self.model = _JambaTextModel(config)
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
        """Map HuggingFace JambaForCausalLM weights to ONNX parameters.

        Handles:
        1. Weight tying (embed_tokens ↔ lm_head)
        2. MoE expert weight renames (w1→gate_proj, w2→down_proj, w3→up_proj)
        3. Fused gate_up_proj splitting for per-expert tensors
        4. SSM params nested under mamba.ssm
        5. Attribute name mapping (mamba_mixer → mamba)
        """
        if self.config.tie_word_embeddings:
            tie_word_embeddings(state_dict)

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_jamba_weight(key, value, new_state_dict)
            if new_key is not None:
                new_state_dict[new_key] = value

        return new_state_dict


# ---------------------------------------------------------------------------
# MambaBlock variant with JambaSelectiveScan
# ---------------------------------------------------------------------------


class _JambaMambaBlock(MambaBlock):
    """MambaBlock that uses JambaSelectiveScan (with dt/B/C layernorms)."""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int = 16,
        dt_rank: int | None = None,
        conv_kernel: int = 4,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__(d_model, d_inner, d_state, dt_rank, conv_kernel)
        # Replace the SSM with the Jamba variant (layernormed dt/B/C)
        self.ssm = JambaSelectiveScan(
            d_inner, d_state, self.dt_rank, layer_norm_epsilon=rms_norm_eps
        )


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------

# SSM params that HF stores flat on mamba but we nest under mamba.ssm
_SSM_PARAMS = ("A_log", "D", "x_proj.weight", "dt_proj.weight", "dt_proj.bias")

# Jamba SSM layernorm params that we nest under mamba.ssm
_SSM_LAYERNORM_PARAMS = (
    "dt_layernorm.weight",
    "b_layernorm.weight",
    "c_layernorm.weight",
)


def _rename_jamba_weight(
    key: str,
    value: torch.Tensor,
    out: dict[str, torch.Tensor],
) -> str | None:
    """Rename a single HF Jamba weight key to our naming convention.

    Returns the new key, or None if the weight was handled inline
    (e.g. fused expert weights that produce multiple outputs).
    """
    # Mamba mixer rename: HF "mamba" or "mamba_mixer" → our "mamba"
    # (Some HF versions use "mamba", others "mamba_mixer")
    key = key.replace(".mamba_mixer.", ".mamba.")

    # SSM params: HF stores flat on mamba, we nest under mamba.ssm
    for param in _SSM_PARAMS:
        if f".mamba.{param}" in key:
            key = key.replace(f".mamba.{param}", f".mamba.ssm.{param}")
            return key

    # SSM layernorm params: also nested under mamba.ssm
    for param in _SSM_LAYERNORM_PARAMS:
        if f".mamba.{param}" in key:
            key = key.replace(f".mamba.{param}", f".mamba.ssm.{param}")
            return key

    # HF pre_ff_layernorm → our pre_moe_layernorm
    key = key.replace(".pre_ff_layernorm.", ".pre_moe_layernorm.")

    # MoE expert weight renames
    if ".experts." in key:
        # Fused gate_up_proj: [num_experts, 2*intermediate, hidden]
        # → split into per-expert gate_proj + up_proj
        if ".gate_up_proj" in key:
            _split_fused_expert_gate_up(key, value, out)
            return None  # handled inline
        # Fused down_proj: [num_experts, hidden, intermediate]
        # → split into per-expert down_proj
        if key.endswith(".experts.down_proj") and value.dim() == 3:
            _split_fused_expert_down(key, value, out)
            return None  # handled inline
        # w1 → gate_proj, w2 → down_proj, w3 → up_proj
        key = key.replace(".w1.", ".gate_proj.")
        key = key.replace(".w2.", ".down_proj.")
        key = key.replace(".w3.", ".up_proj.")

    # MoE gate rename: HF "router" → our "gate"
    key = key.replace(".feed_forward.router.", ".feed_forward.gate.")

    return key


def _split_fused_expert_gate_up(
    key: str,
    value: torch.Tensor,
    out: dict[str, torch.Tensor],
) -> None:
    """Split fused expert gate_up_proj into per-expert gate and up tensors.

    HF stores: ``layers.{i}.feed_forward.experts.gate_up_proj``
        with shape ``[num_experts, 2*intermediate_size, hidden_size]``

    We need per-expert:
        ``layers.{i}.feed_forward.experts.{e}.gate_proj.weight``
        ``layers.{i}.feed_forward.experts.{e}.up_proj.weight``
    """
    new_key = key

    # value shape: [num_experts, 2*intermediate, hidden]
    num_experts = value.shape[0]
    intermediate = value.shape[1] // 2

    # Base path: e.g. "layers.0.feed_forward.experts"
    base = new_key.replace(".gate_up_proj", "")

    for e in range(num_experts):
        expert_w = value[e]  # [2*intermediate, hidden]
        gate_w = expert_w[:intermediate]
        up_w = expert_w[intermediate:]
        out[f"{base}.{e}.gate_proj.weight"] = gate_w
        out[f"{base}.{e}.up_proj.weight"] = up_w


def _split_fused_expert_down(
    key: str,
    value: torch.Tensor,
    out: dict[str, torch.Tensor],
) -> None:
    """Split fused expert down_proj into per-expert tensors.

    HF stores: ``layers.{i}.feed_forward.experts.down_proj``
        with shape ``[num_experts, hidden_size, intermediate_size]``

    We need per-expert:
        ``layers.{i}.feed_forward.experts.{e}.down_proj.weight``
    """
    num_experts = value.shape[0]
    base = key.replace(".down_proj", "")
    for e in range(num_experts):
        out[f"{base}.{e}.down_proj.weight"] = value[e]
