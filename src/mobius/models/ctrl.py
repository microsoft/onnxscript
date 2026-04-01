# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""CTRL causal language model with sinusoidal positional embeddings.

Replicates HuggingFace's ``CTRLLMHeadModel``. Key differences from GPT-2:
- Sinusoidal position embeddings (not learned; computed in preprocess_weights)
- Different attention module naming: ``multi_head_attention.Wq/Wk/Wv/dense``
- MLP uses a sequential FFN with indices: ``ffn.0`` (up) and ``ffn.2`` (down)
- Final LayerNorm is ``transformer.layernorm`` (not ``transformer.ln_f``)
- ``lm_head`` has a bias term (separate from tied weight)
"""

from __future__ import annotations

import torch
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius.components import Linear
from mobius.models.gpt2 import GPT2CausalLMModel


def _sinusoidal_pos_embed(max_pos: int, dim: int) -> torch.Tensor:
    """Compute sinusoidal position embeddings matching HF CTRL's layout.

    HF CTRL uses ``[sines || cosines]`` concatenation (not interleaved):
        pos_encoding = cat([sin(angle_rads[:, 0::2]), cos(angle_rads[:, 1::2])], dim=-1)

    Output shape: (max_pos, dim). The first ``dim//2`` columns are sines,
    the last ``dim//2`` columns are cosines (same frequency grid for both).
    """
    i = torch.arange(dim)
    angle_rates = 1.0 / torch.pow(torch.tensor(10000.0), (2 * (i // 2)).float() / dim)
    pos = torch.arange(max_pos).unsqueeze(1).float()
    angle_rads = pos * angle_rates.unsqueeze(0)  # (max_pos, dim)
    sines = torch.sin(angle_rads[:, 0::2])  # (max_pos, dim//2)
    cosines = torch.cos(angle_rads[:, 1::2])  # (max_pos, dim//2)
    return torch.cat([sines, cosines], dim=-1)  # (max_pos, dim)


class CTRLCausalLMModel(GPT2CausalLMModel):
    """CTRL causal language model.

    Extends ``GPT2CausalLMModel`` with CTRL-specific weight renaming and
    sinusoidal position embedding injection. CTRL uses biases in all
    attention projections and MLP layers, including a separate ``lm_head``
    bias (not tied to the embedding weight).

    Replicates HuggingFace's ``CTRLLMHeadModel``.
    """

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        from mobius.models.gpt2 import _GPT2TextModel

        self.config = config
        self.transformer = _GPT2TextModel(config)
        # CTRL lm_head always has a bias (separate from the tied weight)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=True)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            result = _rename_ctrl_weight(name, tensor)
            if result is not None:
                new.update(result)
            else:
                # Pass through already-ONNX-aligned names unchanged
                new[name] = tensor

        # CTRL scales token embeddings by sqrt(d_model) at runtime in the forward
        # pass (``inputs_embeds *= np.sqrt(self.d_model_size)``).  Pre-apply this
        # scaling so our Embedding lookup produces the same values.
        if "transformer.wte.weight" in new:
            import math

            scale = math.sqrt(self.config.hidden_size)
            new["transformer.wte.weight"] = new["transformer.wte.weight"] * scale

        # CTRL uses sinusoidal PE (not in state dict); pre-compute and inject.
        # The Embedding table is indexed by position_ids (0-based), so we
        # compute max_position_embeddings entries.
        if "transformer.wpe.weight" not in new:
            max_pos = self.config.max_position_embeddings
            dim = self.config.hidden_size
            new["transformer.wpe.weight"] = _sinusoidal_pos_embed(max_pos, dim)

        # Handle tied embeddings: lm_head.weight uses the UNSCALED embedding
        # (HF applies the scale only during the forward pass, not in state dict).
        # If lm_head.weight was not set from state_dict (only happens when NOT
        # tie_word_embeddings), derive it from the unscaled weight.
        if "lm_head.weight" not in new and "transformer.wte.weight" in new:
            # De-scale: lm_head uses the original unscaled token embedding
            import math

            scale = math.sqrt(self.config.hidden_size)
            new["lm_head.weight"] = new["transformer.wte.weight"] / scale

        return new


# Per-layer rename table for CTRL attention and MLP sub-modules
_CTRL_LAYER_RENAMES: list[tuple[str, str]] = [
    ("multi_head_attention.Wq", "attn.q_proj"),
    ("multi_head_attention.Wk", "attn.k_proj"),
    ("multi_head_attention.Wv", "attn.v_proj"),
    ("multi_head_attention.dense", "attn.o_proj"),
    ("layernorm1", "ln_1"),
    ("layernorm2", "ln_2"),
    ("ffn.0", "mlp.up_proj"),
    ("ffn.2", "mlp.down_proj"),
]


def _rename_ctrl_weight(name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor] | None:
    """Rename a single HF CTRL weight to our GPT-2-compatible naming.

    HF CTRL naming:
    - ``transformer.w.weight``           → token embedding (scaled by sqrt(d_model) in preprocess)
    - ``transformer.h.N.multi_head_attention.Wq.*`` → per-layer attention
    - ``transformer.h.N.ffn.0.*``        → MLP up projection
    - ``transformer.h.N.ffn.2.*``        → MLP down projection
    - ``transformer.h.N.layernorm1.*``   → pre-attention LayerNorm
    - ``transformer.h.N.layernorm2.*``   → pre-MLP LayerNorm
    - ``transformer.layernorm.*``        → final LayerNorm
    - ``lm_head.weight``                 → lm_head weight (unscaled, passed through)
    - ``lm_head.bias``                   → lm_head bias (passed through)
    """
    # Pass lm_head.weight and lm_head.bias through unchanged (unscaled).
    # lm_head uses unscaled embeddings even though the forward pass scales them.
    if name.startswith("lm_head."):
        return {name: tensor}

    # Token embedding: transformer.w → transformer.wte
    if name == "transformer.w.weight":
        return {"transformer.wte.weight": tensor}

    # Final LayerNorm: transformer.layernorm → transformer.ln_f
    if name.startswith("transformer.layernorm."):
        suffix = name[len("transformer.layernorm.") :]
        return {f"transformer.ln_f.{suffix}": tensor}

    # Per-layer weights: transformer.h.{N}.{sub_module}.{param}
    if name.startswith("transformer.h."):
        rest = name[len("transformer.h.") :]
        dot = rest.index(".")
        layer_idx = rest[:dot]
        sub = rest[dot + 1 :]

        for old, new in _CTRL_LAYER_RENAMES:
            if sub.startswith(old):
                remainder = sub[len(old) :]
                return {f"transformer.h.{layer_idx}.{new}{remainder}": tensor}

    return None
