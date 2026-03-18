# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""OPT model with absolute positional embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.models.gpt2 import GPT2CausalLMModel, _GPT2TextModel

if TYPE_CHECKING:
    import onnx_ir as ir
    from onnxscript._internal import builder


class OPTCausalLMModel(GPT2CausalLMModel):
    """OPT causal language model.

    Same architecture as GPT-2 (pre-norm LayerNorm, absolute positional
    embeddings, standard attention with KV cache) but with different HF
    weight naming convention and a position embedding offset of 2.
    """

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = _GPT2TextModel(config)
        from mobius.components import Linear

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
        new_state_dict = {}
        for name, tensor in state_dict.items():
            entries = _rename_opt_weight(name, tensor)
            if entries is not None:
                new_state_dict.update(entries)
        # Tied embeddings
        tie_word_embeddings(new_state_dict)
        return new_state_dict


# Weight rename mapping: HF OPT → our GPT-2-compatible naming
_OPT_LAYER_RENAMES = {
    "self_attn.q_proj": "self_attn.q_proj",
    "self_attn.k_proj": "self_attn.k_proj",
    "self_attn.v_proj": "self_attn.v_proj",
    "self_attn.out_proj": "self_attn.o_proj",
    "self_attn_layer_norm": "input_layernorm",
    "fc1": "mlp.up_proj",
    "fc2": "mlp.down_proj",
    "final_layer_norm": "post_attention_layernorm",
}


def _rename_opt_weight(name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor] | None:
    """Rename and transform a single OPT weight."""
    # Strip model.decoder. prefix
    if name.startswith("model.decoder."):
        name = name[len("model.decoder.") :]
    elif name.startswith("lm_head."):
        return {name: tensor}
    else:
        return None

    # Embeddings
    if name == "embed_tokens.weight":
        return {"model.embed_tokens.weight": tensor}
    if name == "embed_positions.weight":
        # OPT uses offset=2, so positions start at index 2.
        # Strip the first 2 rows to align with 0-based position_ids.
        return {"model.embed_positions.weight": tensor[2:]}
    if name == "final_layer_norm.weight":
        return {"model.norm.weight": tensor}
    if name == "final_layer_norm.bias":
        return {"model.norm.bias": tensor}

    # Layer weights: layers.{i}.{...}
    if not name.startswith("layers."):
        return None

    parts = name.split(".", 2)
    if len(parts) < 3:
        return None
    layer_idx = parts[1]
    rest = parts[2]
    prefix = f"model.layers.{layer_idx}"

    for old, new in _OPT_LAYER_RENAMES.items():
        if rest.startswith(old):
            remainder = rest[len(old) :]
            return {f"{prefix}.{new}{remainder}": tensor}

    return None
