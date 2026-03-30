# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ModernBERT encoder and decoder models.

ModernBERT is a modernized BERT architecture using pre-norm layers,
rotary position embeddings (RoPE), and GeGLU (gated) FFN.

- ModernBertModel: Bidirectional encoder (feature-extraction task)
- ModernBertDecoderModel: Causal decoder (text-generation task)

Key differences from standard BERT:
- Pre-norm (LayerNorm before attention/FFN) instead of post-norm
- RoPE instead of absolute position embeddings
- GeGLU FFN (gated linear units) instead of standard MLP
- No token type embeddings
- Layer 0 skips the attention norm (Identity)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._activations import ACT2FN
from mobius.components._common import Embedding, LayerNorm, Linear
from mobius.components._rotary_embedding import (
    apply_rotary_pos_emb,
    initialize_rope,
)
from mobius.models.base import CausalLMModel
from mobius.models.base import TextModel as _TextModel

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# ModernBERT components
# ---------------------------------------------------------------------------


class _ModernBertAttention(nn.Module):
    """Bidirectional multi-head attention with RoPE (no KV cache).

    Attribute names match HF ModernBERT naming (Wo instead of o_proj).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = Linear(config.hidden_size, config.hidden_size, bias=config.attn_qkv_bias)
        self.k_proj = Linear(config.hidden_size, config.hidden_size, bias=config.attn_qkv_bias)
        self.v_proj = Linear(config.hidden_size, config.hidden_size, bias=config.attn_qkv_bias)
        self.Wo = Linear(config.hidden_size, config.hidden_size, bias=config.attn_o_bias)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_embeddings: tuple,
    ):
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        # Apply RoPE
        query_states = apply_rotary_pos_emb(
            op,
            x=query_states,
            position_embeddings=position_embeddings,
            num_heads=self.num_heads,
        )
        key_states = apply_rotary_pos_emb(
            op,
            x=key_states,
            position_embeddings=position_embeddings,
            num_heads=self.num_heads,
        )

        # Bidirectional attention (no KV cache)
        attn_output = op.Attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=float(self.head_dim**-0.5),
        )

        return self.Wo(op, attn_output)


class _ModernBertMLP(nn.Module):
    """GeGLU feed-forward network.

    HF uses fused Wi → split(input, gate) → act(input) * gate → Wo.
    We split Wi into gate_proj + up_proj during preprocess_weights.
    Attribute name Wo matches HF ModernBERT naming.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.gate_proj = Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.Wo = Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self._act_fn = ACT2FN[config.hidden_act]

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        gate = self._act_fn(op, self.gate_proj(op, hidden_states))
        up = self.up_proj(op, hidden_states)
        return self.Wo(op, op.Mul(gate, up))


class _ModernBertLayer(nn.Module):
    """Pre-norm encoder layer with RoPE attention and GeGLU FFN.

    Attribute names match HF ModernBERT naming (attn instead of self_attn).
    """

    def __init__(self, config: ArchitectureConfig, layer_id: int):
        super().__init__()
        # Layer 0 has no attn_norm (Identity in HF)
        self._skip_attn_norm = layer_id == 0
        if not self._skip_attn_norm:
            self.attn_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.attn = _ModernBertAttention(config)
        self.mlp_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = _ModernBertMLP(config)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_embeddings: tuple,
    ):
        residual = hidden_states
        if not self._skip_attn_norm:
            hidden_states = self.attn_norm(op, hidden_states)

        hidden_states = self.attn(op, hidden_states, attention_mask, position_embeddings)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.mlp_norm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# ModernBERT Model
# ---------------------------------------------------------------------------


class ModernBertModel(nn.Module):
    """ModernBERT encoder model for feature extraction.

    Pre-norm encoder with RoPE and GeGLU FFN. Uses bidirectional
    attention without KV cache.
    """

    default_task = "feature-extraction"
    category = "encoder"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.embeddings_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [_ModernBertLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        token_type_ids: ir.Value,  # Unused but required by feature-extraction task
    ):
        hidden_states = self.word_embeddings(op, input_ids)
        hidden_states = self.embeddings_norm(op, hidden_states)

        # Compute position_ids and RoPE embeddings
        seq_len = op.Shape(input_ids, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(seq_len),
            op.Constant(value_int=1),
        )
        position_ids = op.Cast(position_ids, to=7)  # INT64
        position_ids = op.Unsqueeze(position_ids, [0])
        position_embeddings = self.rotary_emb(op, position_ids)

        for layer in self.layers:
            hidden_states = layer(op, hidden_states, attention_mask, position_embeddings)

        hidden_states = self.final_norm(op, hidden_states)
        return hidden_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_modernbert_weight(name, tensor, new_state_dict)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


# ---------------------------------------------------------------------------
# Weight renaming
# ---------------------------------------------------------------------------


def _rename_modernbert_weight(
    name: str, tensor: torch.Tensor, out: dict[str, torch.Tensor]
) -> str | None:
    """Rename HF ModernBERT weight and split fused projections.

    After attribute alignment, only prefix stripping, embedding
    remapping, and fused QKV/Wi splits remain.
    """
    # Strip model. prefix
    if name.startswith("model."):
        name = name[len("model.") :]

    # Skip classifier heads
    if name.startswith(("classifier.", "head.")):
        return None

    # Embeddings
    if name.startswith("embeddings.tok_embeddings."):
        return name.replace("embeddings.tok_embeddings.", "word_embeddings.")
    if name.startswith("embeddings.norm."):
        return name.replace("embeddings.norm.", "embeddings_norm.")

    # Final norm
    if name.startswith("final_norm."):
        return name

    # Encoder layers: layers.{i}.{component}
    if name.startswith("layers."):
        parts = name.split(".", 2)
        if len(parts) < 3:
            return None
        layer_idx = parts[1]
        remainder = parts[2]

        # Fused QKV: attn.Wqkv.weight [3*H, H] → split into q/k/v
        if remainder.startswith("attn.Wqkv."):
            param_type = remainder[len("attn.Wqkv.") :]  # weight or bias
            dim = tensor.shape[0] // 3
            q, k, v = tensor.split(dim, dim=0)
            prefix = f"layers.{layer_idx}.attn"
            out[f"{prefix}.q_proj.{param_type}"] = q
            out[f"{prefix}.k_proj.{param_type}"] = k
            out[f"{prefix}.v_proj.{param_type}"] = v
            return None  # Already added to out

        # Output proj: attn.Wo now matches directly
        if remainder.startswith("attn.Wo."):
            return name

        # Fused MLP Wi: mlp.Wi.weight [2*I, H] → split into gate/up
        if remainder.startswith("mlp.Wi."):
            param_type = remainder[len("mlp.Wi.") :]
            dim = tensor.shape[0] // 2
            gate, up = tensor.split(dim, dim=0)
            prefix = f"layers.{layer_idx}.mlp"
            out[f"{prefix}.gate_proj.{param_type}"] = gate
            out[f"{prefix}.up_proj.{param_type}"] = up
            return None

        # MLP output: mlp.Wo now matches directly
        if remainder.startswith("mlp.Wo."):
            return name

        # Norms pass through
        if remainder.startswith(("attn_norm.", "mlp_norm.")):
            return name

        # Skip rotary embeddings (computed from config)
        if remainder.startswith("attn.rotary_emb."):
            return None

        return None

    return None


# ---------------------------------------------------------------------------
# ModernBERT Decoder (CausalLM variant)
# ---------------------------------------------------------------------------


class _ModernBertDecoderTextModel(_TextModel):
    """TextModel subclass that applies LayerNorm after the embedding lookup.

    Used by ModernBertDecoderModel so that ``model.embeddings_norm.*``
    params are correctly named under the ``model.`` namespace in the ONNX
    graph (calling sub-modules through self.model.forward ensures the
    module tree is walked correctly by onnxscript).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.embeddings_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is None:
            # Embedding lookup + LayerNorm (HF: model.embeddings.norm)
            inputs_embeds = self.embeddings_norm(op, self.embed_tokens(op, input_ids))
        return super().forward(
            op,
            input_ids=input_ids,  # Used by padding mask; embed_tokens skipped when inputs_embeds set
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )


class _ModernBertDecoderLMHead(nn.Module):
    """ModernBERT decoder LM head: dense → LayerNorm → decoder projection.

    HF ModernBertForCausalLM uses a 3-component head:
      lm_head.dense    (hidden→hidden, no bias)
      lm_head.norm     (LayerNorm)
      decoder.weight   (hidden→vocab, with bias, tied with embed_tokens)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.decoder = Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # dense → norm → decoder projection with vocab bias
        hidden_states = self.dense(op, hidden_states)
        hidden_states = self.norm(op, hidden_states)
        return self.decoder(op, hidden_states)


class ModernBertDecoderModel(CausalLMModel):
    """ModernBERT decoder (causal LM variant).

    Extends CausalLMModel with:
    - Embedding LayerNorm (via _ModernBertDecoderTextModel)
    - Three-component LM head: dense → LayerNorm → decoder (with bias)

    Replicates HuggingFace ``ModernBertForCausalLM``.

    HF weight naming differences vs standard CausalLMModel:
    - Separate q_proj/k_proj/v_proj (not fused Wqkv)
    - Fused MLP Wi → split into gate_proj + up_proj
    - Custom layer-norm names (attn_norm / mlp_norm / final_norm)
    - Embedding LayerNorm at model.embeddings.norm
    - Three-part LM head (lm_head.dense + lm_head.norm + decoder)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Replace TextModel with the subclass that injects embedding norm
        self.model = _ModernBertDecoderTextModel(config)
        # Replace simple Linear lm_head with the three-component head
        self.lm_head = _ModernBertDecoderLMHead(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            _rename_modernbert_decoder_weight(name, tensor, new_state_dict)

        # Tie embeddings: HF decoder.weight is tied with model.embeddings.tok_embeddings
        if self.config.tie_word_embeddings:
            dec_w = new_state_dict.get("lm_head.decoder.weight")
            if dec_w is not None:
                new_state_dict["model.embed_tokens.weight"] = dec_w
            else:
                emb_w = new_state_dict.get("model.embed_tokens.weight")
                if emb_w is not None:
                    new_state_dict["lm_head.decoder.weight"] = emb_w

        return new_state_dict


def _rename_modernbert_decoder_weight(
    name: str, tensor: torch.Tensor, out: dict[str, torch.Tensor]
) -> None:
    """Rename HF ModernBERT decoder weights to our CausalLMModel naming.

    HF ModernBertForCausalLM weight structure (after stripping 'model.' prefix):
      embeddings.tok_embeddings.*  → model.embed_tokens.*
      embeddings.norm.*            → model.embeddings_norm.*
      layers.N.attn.q_proj.*       → model.layers.N.self_attn.q_proj.*
      layers.N.attn.k_proj.*       → model.layers.N.self_attn.k_proj.*
      layers.N.attn.v_proj.*       → model.layers.N.self_attn.v_proj.*
      layers.N.attn.Wo.*           → model.layers.N.self_attn.o_proj.*
      layers.N.mlp.Wi.*            → split → model.layers.N.mlp.{gate,up}_proj.*
      layers.N.mlp.Wo.*            → model.layers.N.mlp.down_proj.*
      layers.N.attn_norm.*         → model.layers.N.input_layernorm.*
      layers.N.mlp_norm.*          → model.layers.N.post_attention_layernorm.*
      final_norm.*                 → model.norm.*
    Top-level:
      lm_head.dense.*              → lm_head.dense.*  (same)
      lm_head.norm.*               → lm_head.norm.*   (same)
      decoder.*                    → lm_head.decoder.*
    """
    # Strip top-level 'model.' prefix (HF wraps transformer body under 'model.')
    if name.startswith("model."):
        name = name[len("model.") :]

    # Skip rotary embeddings (computed at runtime, not stored as weights)
    if "rotary_emb." in name:
        return

    # Embeddings
    if name.startswith("embeddings.tok_embeddings."):
        out[name.replace("embeddings.tok_embeddings.", "model.embed_tokens.")] = tensor
        return
    if name.startswith("embeddings.norm."):
        # Embedding LayerNorm lives under model.embeddings_norm in ONNX
        out[name.replace("embeddings.norm.", "model.embeddings_norm.")] = tensor
        return

    # Final norm → model.norm
    if name.startswith("final_norm."):
        out[name.replace("final_norm.", "model.norm.")] = tensor
        return

    # LM head dense and norm (same names in HF and ONNX)
    if name.startswith("lm_head."):
        out[name] = tensor
        return

    # Top-level decoder projection (tied with embeddings in HF)
    if name.startswith("decoder."):
        out[f"lm_head.{name}"] = tensor
        return

    # Transformer layers
    if name.startswith("layers."):
        parts = name.split(".", 2)
        if len(parts) < 3:
            return
        layer_idx = parts[1]
        remainder = parts[2]
        prefix = f"model.layers.{layer_idx}"

        # Separate Q/K/V projections (HF decoder uses q_proj/k_proj/v_proj)
        if remainder.startswith("attn.q_proj."):
            out[f"{prefix}.self_attn.q_proj.{remainder[len('attn.q_proj.') :]}"] = tensor
            return
        if remainder.startswith("attn.k_proj."):
            out[f"{prefix}.self_attn.k_proj.{remainder[len('attn.k_proj.') :]}"] = tensor
            return
        if remainder.startswith("attn.v_proj."):
            out[f"{prefix}.self_attn.v_proj.{remainder[len('attn.v_proj.') :]}"] = tensor
            return

        # Output projection: attn.Wo → self_attn.o_proj
        if remainder.startswith("attn.Wo."):
            out[f"{prefix}.self_attn.o_proj.{remainder[len('attn.Wo.') :]}"] = tensor
            return

        # Fused MLP gate+up: Wi → split into gate_proj + up_proj
        if remainder.startswith("mlp.Wi."):
            param_type = remainder[len("mlp.Wi.") :]
            dim = tensor.shape[0] // 2
            gate, up = tensor.split(dim, dim=0)
            out[f"{prefix}.mlp.gate_proj.{param_type}"] = gate
            out[f"{prefix}.mlp.up_proj.{param_type}"] = up
            return

        # MLP output: mlp.Wo → mlp.down_proj
        if remainder.startswith("mlp.Wo."):
            out[f"{prefix}.mlp.down_proj.{remainder[len('mlp.Wo.') :]}"] = tensor
            return

        # Norms: attn_norm → input_layernorm, mlp_norm → post_attention_layernorm
        if remainder.startswith("attn_norm."):
            out[f"{prefix}.input_layernorm.{remainder[len('attn_norm.') :]}"] = tensor
            return
        if remainder.startswith("mlp_norm."):
            out[f"{prefix}.post_attention_layernorm.{remainder[len('mlp_norm.') :]}"] = tensor
            return
