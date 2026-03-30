# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""XLM causal language model.

Replicates HuggingFace's ``XLMWithLMHeadModel``. Key differences from GPT-2:
- Embedding LayerNorm (``layer_norm_emb``) applied after token + position embed sum
- No final LayerNorm (``ln_f`` is skipped in forward)
- Flat-indexed per-layer sub-modules: ``attentions.N``, ``layer_norm1.N``,
  ``ffns.N``, ``layer_norm2.N`` (not nested under ``h.N``)
- LM head: ``pred_layer.proj`` (weight + bias, always present)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import Embedding, LayerNorm, Linear, create_padding_mask
from mobius.models.gpt2 import GPT2CausalLMModel, _GPT2DecoderLayer

if TYPE_CHECKING:
    import onnx_ir as ir


class _XLMTextModel(nn.Module):
    """XLM text backbone.

    Extends the GPT-2 backbone by adding an embedding LayerNorm
    (``layer_norm_emb``) applied after the token + positional embedding sum,
    and omitting the final LayerNorm (``ln_f``).

    Attribute names follow HF XLMModel naming (wte, wpe, layer_norm_emb, h).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.wte = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.wpe = Embedding(config.max_position_embeddings, config.hidden_size)
        # Post-embedding LayerNorm (unique to XLM vs GPT-2)
        self.layer_norm_emb = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.h = nn.ModuleList(
            # XLM uses post-norm (LayerNorm after residual, like BERT/OpenAI-GPT)
            [
                _GPT2DecoderLayer(config, post_norm=True)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.wte(op, input_ids)

        position_embeds = self.wpe(op, position_ids)
        hidden_states = op.Add(hidden_states, position_embeds)
        # XLM applies LayerNorm immediately after the embedding sum
        hidden_states = self.layer_norm_emb(op, hidden_states)

        attention_bias = create_padding_mask(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.h)
        for layer, past_kv in zip(self.h, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        # XLM has no final LayerNorm (unlike GPT-2 which has ln_f)
        return hidden_states, present_key_values


class XLMCausalLMModel(GPT2CausalLMModel):
    """XLM causal language model (``XLMWithLMHeadModel``).

    Uses ``_XLMTextModel`` instead of ``_GPT2TextModel`` to add the
    embedding LayerNorm and skip the final norm. The lm_head always
    has a bias (from ``pred_layer.proj``).
    """

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.transformer = _XLMTextModel(config)
        # XLM pred_layer.proj always has a bias
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.transformer(
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
        new: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            result = _rename_xlm_weight(name, tensor)
            if result is not None:
                new.update(result)
        # XLM uses tied word embeddings when tie_word_embeddings=True
        tie_word_embeddings(new, embed_key="transformer.wte.weight")
        return new


# Per-layer rename table for XLM attention and MLP sub-modules.
# HF XLM uses flat-indexed ModuleLists (attentions.N, layer_norm1.N, etc.)
# mapped from state dict keys like ``transformer.attentions.0.q_lin.*``.
_XLM_LAYER_RENAMES: list[tuple[str, str]] = [
    ("q_lin", "attn.q_proj"),
    ("k_lin", "attn.k_proj"),
    ("v_lin", "attn.v_proj"),
    ("out_lin", "attn.o_proj"),
]


def _rename_xlm_weight(name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor] | None:
    """Rename a single HF XLM weight to our GPT-2-compatible naming.

    HF XLM uses flat-indexed sub-modules under ``transformer.*``:

    - ``transformer.embeddings.weight``      → token embedding
    - ``transformer.position_embeddings.weight`` → position embedding
    - ``transformer.layer_norm_emb.*``       → post-embedding LayerNorm
    - ``transformer.attentions.N.*``         → per-layer attention
    - ``transformer.layer_norm1.N.*``        → per-layer pre-attention LayerNorm
    - ``transformer.ffns.N.lin1.*``          → MLP up projection
    - ``transformer.ffns.N.lin2.*``          → MLP down projection
    - ``transformer.layer_norm2.N.*``        → per-layer pre-MLP LayerNorm
    - ``pred_layer.proj.*``                  → lm_head weight + bias
    """
    # LM head: pred_layer.proj → lm_head
    if name.startswith("pred_layer.proj."):
        suffix = name[len("pred_layer.proj.") :]
        return {f"lm_head.{suffix}": tensor}

    if not name.startswith("transformer."):
        return None

    rest = name[len("transformer.") :]

    # Token embedding
    if rest == "embeddings.weight":
        return {"transformer.wte.weight": tensor}

    # Position embedding
    if rest == "position_embeddings.weight":
        return {"transformer.wpe.weight": tensor}

    # Post-embedding LayerNorm
    if rest.startswith("layer_norm_emb."):
        suffix = rest[len("layer_norm_emb.") :]
        return {f"transformer.layer_norm_emb.{suffix}": tensor}

    # Per-layer attention: attentions.{N}.{q_lin|k_lin|v_lin|out_lin}.{weight|bias}
    if rest.startswith("attentions."):
        parts = rest.split(".", 3)
        # attentions . N . sub_module . weight/bias
        if len(parts) < 4:
            return None
        layer_idx = parts[1]
        sub = ".".join(parts[2:])
        for old, new in _XLM_LAYER_RENAMES:
            if sub.startswith(old):
                remainder = sub[len(old) :]
                return {f"transformer.h.{layer_idx}.{new}{remainder}": tensor}
        return None

    # Pre-attention LayerNorm: layer_norm1.{N}.{weight|bias}
    if rest.startswith("layer_norm1."):
        parts = rest.split(".", 2)
        if len(parts) < 3:
            return None
        layer_idx = parts[1]
        param = parts[2]
        return {f"transformer.h.{layer_idx}.ln_1.{param}": tensor}

    # MLP: ffns.{N}.lin1.{weight|bias} / ffns.{N}.lin2.{weight|bias}
    if rest.startswith("ffns."):
        parts = rest.split(".", 3)
        if len(parts) < 4:
            return None
        layer_idx = parts[1]
        sub_and_param = ".".join(parts[2:])
        if sub_and_param.startswith("lin1."):
            param = sub_and_param[len("lin1.") :]
            return {f"transformer.h.{layer_idx}.mlp.up_proj.{param}": tensor}
        if sub_and_param.startswith("lin2."):
            param = sub_and_param[len("lin2.") :]
            return {f"transformer.h.{layer_idx}.mlp.down_proj.{param}": tensor}
        return None

    # Pre-MLP LayerNorm: layer_norm2.{N}.{weight|bias}
    if rest.startswith("layer_norm2."):
        parts = rest.split(".", 2)
        if len(parts) < 3:
            return None
        layer_idx = parts[1]
        param = parts[2]
        return {f"transformer.h.{layer_idx}.ln_2.{param}": tensor}

    return None
