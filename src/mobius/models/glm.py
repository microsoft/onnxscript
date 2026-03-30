# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM and GLM4 causal language models.

GLM (ChatGLM-4 open-source) and GLM4 use a fused ``gate_up_proj`` in
the MLP: a single linear ``hidden → 2 * intermediate`` replaces the
separate ``gate_proj`` + ``up_proj`` used by standard LLaMA-style models.

Additionally, GLM4 uses a 4-norm decoder layer:

- ``input_layernorm`` (pre-attention)
- ``post_self_attn_layernorm`` (post-attention, before residual add)
- ``post_attention_layernorm`` (pre-MLP)
- ``post_mlp_layernorm`` (post-MLP, before residual add)

Both models use ``attention_bias=True`` and explicit ``head_dim=128``
with ``partial_rotary_factor=0.5``.

HuggingFace classes: ``GlmForCausalLM``, ``Glm4ForCausalLM``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._attention import StaticCacheState
from mobius.components._decoder import DecoderLayer
from mobius.components._rms_norm import RMSNorm
from mobius.models.base import CausalLMModel, TextModel

if TYPE_CHECKING:
    import onnx_ir as ir


def _split_gate_up_proj(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Split fused ``gate_up_proj`` into separate ``gate_proj`` + ``up_proj``.

    HF GLM/GLM4 store: ``mlp.gate_up_proj.weight`` [2*intermediate, hidden]
    ONNX MLP expects: ``mlp.gate_proj.weight`` [intermediate, hidden]
                      ``mlp.up_proj.weight``   [intermediate, hidden]
    """
    new_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if ".mlp.gate_up_proj." in key:
            # Split along dim 0: first half = gate, second half = up
            mid = value.shape[0] // 2
            gate_key = key.replace(".gate_up_proj.", ".gate_proj.")
            up_key = key.replace(".gate_up_proj.", ".up_proj.")
            new_state_dict[gate_key] = value[:mid]
            new_state_dict[up_key] = value[mid:]
        else:
            new_state_dict[key] = value
    return new_state_dict


class GlmCausalLMModel(CausalLMModel):
    """GLM text model (standard pre-norm + fused gate_up_proj).

    GLM is architecturally identical to LLaMA except for the fused
    ``gate_up_proj`` in the MLP.  ``preprocess_weights`` splits it into
    separate ``gate_proj`` + ``up_proj`` tensors.
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = _split_gate_up_proj(state_dict)
        return super().preprocess_weights(state_dict)


# ---------------------------------------------------------------------------
# GLM4: 4-norm decoder layer (pre+post norm on both attention and MLP)
# ---------------------------------------------------------------------------


class Glm4DecoderLayer(DecoderLayer):
    """GLM4 decoder layer with 4 RMSNorm layers.

    GLM4 uses pre-norm + post-norm on both attention and MLP sub-layers:

    .. code-block:: text

        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)           # pre-attn
        hidden_states = self_attn(hidden_states)
        hidden_states = post_self_attn_layernorm(hidden_states)  # post-attn
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)  # pre-MLP
        hidden_states = mlp(hidden_states)
        hidden_states = post_mlp_layernorm(hidden_states)        # post-MLP
        hidden_states = residual + hidden_states
    """

    def __init__(self, config: ArchitectureConfig):
        # Initialize the base DecoderLayer (pre-norm mode) to get
        # input_layernorm, post_attention_layernorm, self_attn, mlp.
        super().__init__(config)

        # Add the two extra post-sub-layer norms
        self.post_self_attn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | StaticCacheState | None,
    ):
        if isinstance(past_key_value, StaticCacheState):
            static_cache = past_key_value
            past_key_value = None
        else:
            static_cache = None

        # --- Attention with pre+post norm ---
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        attn_output, present_key_value = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            static_cache=static_cache,
        )

        # Post-attention norm before residual addition
        hidden_states = self.post_self_attn_layernorm(op, attn_output)
        hidden_states = op.Add(residual, hidden_states)

        # --- MLP with pre+post norm ---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)

        # Post-MLP norm before residual addition
        hidden_states = self.post_mlp_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_key_value


class Glm4CausalLMModel(CausalLMModel):
    """GLM4 text model (4-norm decoder + fused gate_up_proj).

    Uses ``Glm4DecoderLayer`` with pre+post norms on both sub-layers,
    and splits the fused ``gate_up_proj`` in ``preprocess_weights``.
    """

    def __init__(self, config: ArchitectureConfig):
        # Bypass CausalLMModel.__init__ to use custom decoder layer
        nn.Module.__init__(self)
        self.config = config
        self.model = Glm4TextModel(config)
        from mobius.components import Linear

        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = _split_gate_up_proj(state_dict)
        return super().preprocess_weights(state_dict)


class Glm4TextModel(TextModel):
    """GLM4 text model backbone using 4-norm decoder layers."""

    def __init__(self, config: ArchitectureConfig):
        # Override TextModel init to use Glm4DecoderLayer
        nn.Module.__init__(self)
        from mobius.components import Embedding, initialize_rope

        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [Glm4DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
