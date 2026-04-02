# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""NomicBERT encoder model for sentence embeddings and retrieval.

NomicBERT is a BERT variant that replaces learned positional embeddings
with rotary position embeddings (RoPE) and uses SwiGLU-style gated MLP
instead of BERT's standard two-layer GELU MLP.

Architecture (per layer, post-norm):
    - Attention: fused QKV → split → RoPE → bidirectional Attention → out_proj
    - MLP: SwiGLU — fc11(x)*sigmoid(fc11(x)) * fc12(x) → fc2
    - LayerNorm after each sub-layer (post-norm, same as BERT)

Weight naming (HF → ONNX):
    - ``embeddings.word_embeddings.weight`` → same
    - ``embeddings.token_type_embeddings.weight`` → same
    - ``emb_ln.*`` → same
    - ``encoder.layers.{i}.attn.Wqkv.weight`` → split into q_proj/k_proj/v_proj
    - ``encoder.layers.{i}.attn.out_proj.weight`` → same
    - ``encoder.layers.{i}.mlp.fc11.weight`` → ``gate_proj`` (SiLU branch)
    - ``encoder.layers.{i}.mlp.fc12.weight`` → ``up_proj`` (linear branch)
    - ``encoder.layers.{i}.mlp.fc2.weight`` → ``down_proj``
    - ``encoder.layers.{i}.norm1.*`` → same
    - ``encoder.layers.{i}.norm2.*`` → same

HF reference: ``nomic-ai/nomic-embed-text-v1.5``
(model_type ``nomic_bert``, requires ``trust_remote_code=True``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius.components._common import Embedding, LayerNorm, Linear, builder
from mobius.components._rotary_embedding import (
    apply_rotary_pos_emb,
    initialize_rope,
)

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class _NomicBertAttention(nn.Module):
    """Bidirectional multi-head attention with RoPE (no KV cache).

    Nomic stores QKV as a fused ``Wqkv`` tensor [3*H, H] which is
    split into separate q_proj/k_proj/v_proj in preprocess_weights.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = Linear(config.hidden_size, config.hidden_size, bias=False)

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

        # Apply RoPE to queries and keys
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

        # Bidirectional attention (no causal mask, no KV cache)
        attn_output = op.Attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=float(self.head_dim**-0.5),
        )

        return self.out_proj(op, attn_output)


# ---------------------------------------------------------------------------
# MLP — SwiGLU gated feed-forward
# ---------------------------------------------------------------------------


class _NomicBertMLP(nn.Module):
    """SwiGLU MLP: SiLU(gate_proj(x)) * up_proj(x) → down_proj.

    HF names: fc11 (gate), fc12 (up), fc2 (down).
    We rename to gate_proj/up_proj/down_proj in preprocess_weights
    for consistency with other mobius models.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.gate_proj = Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        gate = self.gate_proj(op, hidden_states)
        up = self.up_proj(op, hidden_states)
        # SiLU(gate) * up
        activated = op.Mul(op.Mul(gate, op.Sigmoid(gate)), up)
        return self.down_proj(op, activated)


# ---------------------------------------------------------------------------
# Encoder layer — post-norm (default NomicBERT configuration)
# ---------------------------------------------------------------------------


class _NomicBertLayer(nn.Module):
    """NomicBERT encoder layer (post-norm).

    Structure: x + attn(x) → norm1 → x + mlp(x) → norm2
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.attn = _NomicBertAttention(config)
        self.mlp = _NomicBertMLP(config)
        self.norm1 = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_embeddings: tuple,
    ):
        # Post-norm: residual + sublayer → norm
        hidden_states = self.norm1(
            op, op.Add(hidden_states, self.attn(op, hidden_states, attention_mask, position_embeddings))
        )
        hidden_states = self.norm2(
            op, op.Add(hidden_states, self.mlp(op, hidden_states))
        )
        return hidden_states


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class NomicBertModel(nn.Module):
    """NomicBERT encoder for sentence embeddings and retrieval.

    RoPE-based bidirectional encoder with SwiGLU MLP and token type
    embeddings. Produces ``last_hidden_state`` for feature extraction.

    HF reference: ``nomic-ai/nomic-embed-text-v1.5``
    """

    default_task = "feature-extraction"
    category = "encoder"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config

        # Embeddings: word + token_type (no learned position embeddings)
        self.embeddings = _NomicBertEmbeddings(config)
        self.emb_ln = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Encoder layers
        self.encoder = _NomicBertEncoder(config)

        # RoPE (rotary_emb_base=1000 in NomicBERT, not default 10000)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        token_type_ids: ir.Value,
    ):
        # Embeddings: word + token_type → LayerNorm
        hidden_states = self.embeddings(op, input_ids, token_type_ids)
        hidden_states = self.emb_ln(op, hidden_states)

        # Compute position_ids from sequence length → RoPE cos/sin
        seq_len = op.Shape(input_ids, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(seq_len),
            op.Constant(value_int=1),
        )
        position_ids = op.Cast(position_ids, to=7)  # INT64
        position_ids = op.Unsqueeze(position_ids, [0])
        position_embeddings = self.rotary_emb(op, position_ids)

        # Encoder layers
        hidden_states = self.encoder(op, hidden_states, attention_mask, position_embeddings)

        return hidden_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            new_name = _rename_nomic_bert_weight(name, tensor, new_state_dict)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


# ---------------------------------------------------------------------------
# Sub-modules (for weight name alignment)
# ---------------------------------------------------------------------------


class _NomicBertEmbeddings(nn.Module):
    """Word + token type embeddings (no position embeddings — uses RoPE)."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.word_embeddings = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.token_type_embeddings = Embedding(
            config.type_vocab_size or 2, config.hidden_size
        )

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        token_type_ids: ir.Value,
    ):
        word_embeds = self.word_embeddings(op, input_ids)
        token_type_embeds = self.token_type_embeddings(op, token_type_ids)
        return op.Add(word_embeds, token_type_embeds)


class _NomicBertEncoder(nn.Module):
    """Stack of NomicBERT encoder layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [_NomicBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_embeddings: tuple,
    ):
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, attention_mask, position_embeddings)
        return hidden_states


# ---------------------------------------------------------------------------
# Weight renaming
# ---------------------------------------------------------------------------


def _rename_nomic_bert_weight(
    name: str, tensor: ir.TensorType, out: dict[str, ir.TensorType]
) -> str | None:
    """Rename HF NomicBERT weight names to match ONNX module attributes.

    Main transforms:
    - Split fused Wqkv [3*H, H] → q_proj, k_proj, v_proj
    - Rename fc11/fc12/fc2 → gate_proj/up_proj/down_proj
    - Pass through aligned names (embeddings, emb_ln, norm1/norm2, out_proj)
    """
    # Skip rotary embedding buffers (computed dynamically)
    if "rotary_emb" in name:
        return None

    # Embeddings: already aligned
    if name.startswith("embeddings."):
        return name

    # emb_ln: already aligned
    if name.startswith("emb_ln."):
        return name

    # Encoder layers
    if name.startswith("encoder.layers."):
        parts = name.split(".", 3)  # encoder.layers.{i}.{remainder}
        if len(parts) < 4:
            return None
        layer_idx = parts[2]
        remainder = parts[3]
        prefix = f"encoder.layers.{layer_idx}"

        # Fused QKV: attn.Wqkv.weight [3*H, H] → split into q/k/v
        if remainder.startswith("attn.Wqkv."):
            param_type = remainder[len("attn.Wqkv."):]  # "weight"
            dim = tensor.shape[0] // 3
            q, k, v = tensor.split(dim, dim=0)
            out[f"{prefix}.attn.q_proj.{param_type}"] = q
            out[f"{prefix}.attn.k_proj.{param_type}"] = k
            out[f"{prefix}.attn.v_proj.{param_type}"] = v
            return None  # Already added to out

        # Attention output: attn.out_proj — already aligned
        if remainder.startswith("attn.out_proj."):
            return name

        # MLP: rename fc11→gate_proj, fc12→up_proj, fc2→down_proj
        if remainder.startswith("mlp.fc11."):
            return name.replace("mlp.fc11.", "mlp.gate_proj.")
        if remainder.startswith("mlp.fc12."):
            return name.replace("mlp.fc12.", "mlp.up_proj.")
        if remainder.startswith("mlp.fc2."):
            return name.replace("mlp.fc2.", "mlp.down_proj.")

        # Norms: norm1, norm2 — already aligned
        if remainder.startswith(("norm1.", "norm2.")):
            return name

    return None
