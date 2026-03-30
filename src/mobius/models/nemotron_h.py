# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""NemotronH hybrid Mamba2 + Attention + MLP causal language model.

NemotronH interleaves three layer types in a configurable pattern:
- Mamba2/SSD layers for efficient recurrent processing
- Transformer attention layers for global context
- Dense MLP layers for feedforward computation

Each layer is a single-mixer block: RMSNorm → mixer → residual.
Unlike Jamba/Bamba where every layer has a mixer AND MLP, NemotronH
treats MLP as a standalone layer type.

Layer types are specified via ``layers_block_type`` in the HF config:
``M`` = mamba, ``*`` = attention, ``-`` = mlp (dense feedforward).

State per layer:
    Mamba2: conv_state (batch, conv_dim, d_conv-1)
            ssm_state (batch, num_heads, d_head, d_state)
    Attention: standard KV cache (key + value)
    MLP: stateless — no cache

HuggingFace reference: ``NemotronHForCausalLM``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import NemotronHConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    FCMLP,
    Attention,
    Embedding,
    Linear,
    Mamba2Block,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


class NemotronHMambaLayer(nn.Module):
    """NemotronH Mamba2 layer: RMSNorm → Mamba2Block → residual.

    Single-mixer block — no MLP path.

    Args:
        config: NemotronH architecture config.
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        # d_inner = num_heads * head_dim (not hidden_size * expand)
        d_inner = config.mamba_n_heads * config.mamba_d_head

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
            # NemotronH uses grouped RMSNorm: normalize within each
            # group of heads_per_group * head_dim dimensions.
            norm_group_size=d_inner // config.mamba_n_groups,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        # Pre-norm → Mamba2 → residual
        residual = hidden_states
        hidden_states = self.norm(op, hidden_states)

        conv_state, ssm_state = past_key_value if past_key_value is not None else (None, None)
        mamba_out, new_conv_state, new_ssm_state = self.mamba(
            op, hidden_states, conv_state, ssm_state
        )
        hidden_states = op.Add(residual, mamba_out)

        return hidden_states, (new_conv_state, new_ssm_state)


class NemotronHAttentionLayer(nn.Module):
    """NemotronH attention layer: RMSNorm → Attention → residual.

    Single-mixer block — no MLP path.

    Args:
        config: NemotronH architecture config.
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states = self.norm(op, hidden_states)

        attn_output, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, attn_output)

        return hidden_states, present_kv


class NemotronHMLPLayer(nn.Module):
    """NemotronH MLP layer: RMSNorm → FCMLP → residual.

    Single-mixer block — stateless, no cache.

    Args:
        config: NemotronH architecture config.
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act,
            bias=config.mlp_bias,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (None, None)).

        MLP layers are stateless — the None pair keeps the cache
        list aligned with all layers.
        """
        del attention_bias, position_embeddings, past_key_value  # unused

        # Pre-norm → MLP → residual
        residual = hidden_states
        hidden_states = self.norm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, (None, None)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class _NemotronHTextModel(nn.Module):
    """NemotronH text backbone: embedding → N x (Mamba2|Attention|MLP) → norm.

    Layer types are selected based on ``config.layer_types``:
        ``"mamba2"`` → NemotronHMambaLayer
        ``"full_attention"`` → NemotronHAttentionLayer
        ``"mlp"`` → NemotronHMLPLayer
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        layer_types = config.layer_types or []
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            if ltype == "mamba2":
                self.layers.append(NemotronHMambaLayer(config))
            elif ltype == "mlp":
                self.layers.append(NemotronHMLPLayer(config))
            else:
                self.layers.append(NemotronHAttentionLayer(config))

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class NemotronHCausalLMModel(nn.Module):
    """NemotronH hybrid Mamba2+Attention+MLP causal language model.

    Uses ``HybridCausalLMTask`` with mixed ``"mamba2"``,
    ``"full_attention"``, and ``"mlp"`` layer types for the cache.

    HuggingFace reference: ``NemotronHForCausalLM``.
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid SSM+Attention"
    config_class: type = NemotronHConfig

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.config = config
        self.model = _NemotronHTextModel(config)
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
        """Map HuggingFace NemotronHForCausalLM weights to ONNX parameters.

        Handles:
        1. Weight tying (embed_tokens ↔ lm_head)
        2. ``backbone.`` → ``model.`` prefix rename
        3. ``backbone.embeddings.`` → ``model.embed_tokens.``
        4. ``backbone.norm_f.`` → ``model.norm.``
        5. Per-layer ``mixer.`` rename based on layer type:
           - mamba: ``mixer.`` → ``mamba.`` (SSM params nested under ``mamba.ssm.``)
           - attention: ``mixer.`` → ``self_attn.``
           - mlp: ``mixer.`` → ``mlp.``
        """
        layer_types = self.config.layer_types or []

        if self.config.tie_word_embeddings:
            # Tie before renaming so both old-name keys exist
            tie_word_embeddings(
                state_dict,
                embed_key="backbone.embeddings.weight",
                head_key="lm_head.weight",
            )

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_nemotron_h_weight(key, layer_types)
            new_state_dict[new_key] = value

        return new_state_dict


# Layer index regex: backbone.layers.<N>.<rest>
_LAYER_RE = re.compile(r"^backbone\.layers\.(\d+)\.(.+)$")

# Mamba SSM params that need to be nested under mamba.ssm
_MAMBA_SSM_PARAMS = ("A_log", "D", "dt_bias")


def _rename_nemotron_h_weight(key: str, layer_types: list[str]) -> str:
    """Rename a single HF weight key to match ONNX module structure.

    HF NemotronHForCausalLM weight naming:
        backbone.embeddings.weight
        backbone.norm_f.weight
        backbone.layers.N.norm.weight
        backbone.layers.N.mixer.{in_proj, conv1d, out_proj, norm, A_log, D, dt_bias}  (mamba)
        backbone.layers.N.mixer.{q_proj, k_proj, v_proj, o_proj}.weight              (attention)
        backbone.layers.N.mixer.{up_proj, down_proj}.weight                          (mlp)
        lm_head.weight

    ONNX parameter naming:
        model.embed_tokens.weight
        model.norm.weight
        model.layers.N.norm.weight
        model.layers.N.mamba.{in_proj, conv1d, out_proj, norm}   (mamba direct)
        model.layers.N.mamba.ssm.{A_log, D, dt_bias}            (mamba SSM nested)
        model.layers.N.self_attn.{q_proj, k_proj, v_proj, o_proj}.weight
        model.layers.N.mlp.{up_proj, down_proj}.weight
        lm_head.weight
    """
    # Global prefix renames
    if key.startswith("backbone.embeddings."):
        return key.replace("backbone.embeddings.", "model.embed_tokens.", 1)
    if key.startswith("backbone.norm_f."):
        return key.replace("backbone.norm_f.", "model.norm.", 1)

    # Per-layer renames
    m = _LAYER_RE.match(key)
    if m:
        layer_idx = int(m.group(1))
        rest = m.group(2)
        ltype = layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"

        if rest.startswith("mixer."):
            mixer_rest = rest[len("mixer.") :]
            if ltype == "mamba2":
                # Check if this is an SSM param that needs nesting
                param_name = mixer_rest.split(".")[0]
                if param_name in _MAMBA_SSM_PARAMS:
                    return f"model.layers.{layer_idx}.mamba.ssm.{mixer_rest}"
                return f"model.layers.{layer_idx}.mamba.{mixer_rest}"
            elif ltype == "full_attention":
                return f"model.layers.{layer_idx}.self_attn.{mixer_rest}"
            else:  # mlp
                return f"model.layers.{layer_idx}.mlp.{mixer_rest}"

        # norm.weight stays as norm.weight (already matching)
        return f"model.layers.{layer_idx}.{rest}"

    # Catch-all for backbone.X → model.X (shouldn't normally hit)
    if key.startswith("backbone."):
        return key.replace("backbone.", "model.", 1)

    return key
