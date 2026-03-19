# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Attention,
    Embedding,
    LayerNorm,
    Linear,
    MoELayer,
    RMSNorm,
    SparseMixerGate,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._attention import StaticCacheState
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class MoEDecoderLayer(nn.Module):
    """Decoder layer with MoE replacing the standard MLP.

    Uses RMSNorm by default (Mixtral, GraniteMoE, OLMoE, Qwen2-MoE).
    PhiMoE overrides with LayerNorm via norm_class parameter.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        gate: nn.Module | None = None,
        norm_class: type = RMSNorm,
    ):
        super().__init__()
        self.self_attn = Attention(config)
        self.block_sparse_moe = MoELayer(config, gate=gate)
        self.input_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | StaticCacheState | None,
    ):
        # Dispatch StaticCacheState to the static_cache parameter;
        # custom MoEDecoderLayer subclasses must add this check themselves.
        if isinstance(past_key_value, StaticCacheState):
            static_cache = past_key_value
            past_key_value = None
        else:
            static_cache = None

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
        hidden_states = op.Add(residual, attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.block_sparse_moe(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_key_value


class MoETextModel(nn.Module):
    """Text model using MoE decoder layers instead of standard MLP layers.

    Uses RMSNorm by default (Mixtral, GraniteMoE, OLMoE, Qwen2-MoE).
    PhiMoE overrides with LayerNorm via norm_class parameter.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        gate_factory: type[nn.Module] | None = None,
        norm_class: type = RMSNorm,
    ):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        def _make_gate():
            if gate_factory is None:
                return None
            return gate_factory(
                config.hidden_size, config.num_local_experts, config.num_experts_per_tok
            )

        self.layers = nn.ModuleList(
            [
                MoEDecoderLayer(config, gate=_make_gate(), norm_class=norm_class)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)

        # When attention_mask is None (static cache mode), skip bias
        # creation entirely — the Attention op uses is_causal=1 instead.
        if attention_mask is not None:
            attention_bias = create_attention_bias(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                dtype=self._dtype,
            )
        else:
            attention_bias = None

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


class Phi3MoECausalLMModel(CausalLMModel):
    """Phi-3 Mixture of Experts model.

    Uses MoE decoder layers with sparse expert routing. Each layer replaces
    the standard MLP with a MoELayer containing num_local_experts expert MLPs.
    """

    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = MoETextModel(config, gate_factory=SparseMixerGate, norm_class=LayerNorm)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=True)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = _rename_moe_expert_weights(state_dict)
        return super().preprocess_weights(state_dict)


class GPTOSSCausalLMModel(CausalLMModel):
    """GPTOSS Mixture of Experts model with sink attention and local attention patterns.

    Features sink attention, local attention patterns, and MoE with
    mscale-applied RoPE.
    """

    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = MoETextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = _rename_moe_expert_weights(state_dict)
        return super().preprocess_weights(state_dict)


class MoECausalLMModel(CausalLMModel):
    """Generic Mixture of Experts causal language model.

    Compatible with Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE, GraniteMoE, and
    other standard MoE architectures using top-k sparse routing.
    """

    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = MoETextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = _rename_moe_expert_weights(state_dict)
        return super().preprocess_weights(state_dict)


def _rename_moe_expert_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rename HuggingFace MoE expert weights to match our naming convention.

    Handles two HF weight formats:

    1. **Individual experts** (Mixtral, PhiMoE safetensors):
       ``w1/w2/w3`` → ``gate_proj/down_proj/up_proj``

    2. **Fused 3D experts** (GraniteMoE, newer PhiMoE transformers):
       ``input_linear.weight [N, 2*inter, hidden]`` → per-expert gate_proj + up_proj
       ``output_linear.weight [N, hidden, inter]`` → per-expert down_proj
       ``router.layer.weight`` → ``gate.weight``

       Also handles PhiMoE fused format:
       ``experts.gate_up_proj [N, 2*inter, hidden]`` → per-expert gate_proj + up_proj
       ``experts.down_proj [N, hidden, inter]`` → per-expert down_proj
       ``router.weight`` → ``gate.weight``
    """
    # First pass: individual expert renames (w1/w2/w3)
    rename_map = {".w1.": ".gate_proj.", ".w2.": ".down_proj.", ".w3.": ".up_proj."}
    renamed: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        new_name = name
        for old, new in rename_map.items():
            if old in name:
                new_name = name.replace(old, new)
                break
        renamed[new_name] = tensor

    # Second pass: split fused 3D expert weights and rename router
    fused: dict[str, torch.Tensor] = {}
    for name, tensor in list(renamed.items()):
        # GraniteMoE: router.layer.weight → gate.weight
        if ".router.layer.weight" in name:
            new_name = name.replace(".router.layer.weight", ".gate.weight")
            fused[new_name] = tensor
            del renamed[name]
        # PhiMoE (fused): router.weight → gate.weight (when inside mlp)
        elif ".mlp.router.weight" in name:
            new_name = name.replace(".mlp.router.weight", ".block_sparse_moe.gate.weight")
            fused[new_name] = tensor
            del renamed[name]
        # GraniteMoE: input_linear.weight [N, 2*inter, hidden] → gate_proj + up_proj
        elif ".input_linear.weight" in name and tensor.dim() == 3:
            prefix = name.replace(".input_linear.weight", "")
            num_experts = tensor.shape[0]
            half = tensor.shape[1] // 2
            for i in range(num_experts):
                expert_w = tensor[i]  # [2*inter, hidden]
                gate_w, up_w = expert_w.split(half, dim=0)
                fused[f"{prefix}.experts.{i}.gate_proj.weight"] = gate_w
                fused[f"{prefix}.experts.{i}.up_proj.weight"] = up_w
            del renamed[name]
        # GraniteMoE: output_linear.weight [N, hidden, inter] → down_proj
        elif ".output_linear.weight" in name and tensor.dim() == 3:
            prefix = name.replace(".output_linear.weight", "")
            num_experts = tensor.shape[0]
            for i in range(num_experts):
                fused[f"{prefix}.experts.{i}.down_proj.weight"] = tensor[i]
            del renamed[name]
        # PhiMoE (fused): experts.gate_up_proj [N, 2*inter, hidden]
        elif ".experts.gate_up_proj" in name and tensor.dim() == 3:
            prefix = name.split(".experts.gate_up_proj")[0]
            num_experts = tensor.shape[0]
            half = tensor.shape[1] // 2
            for i in range(num_experts):
                expert_w = tensor[i]
                gate_w, up_w = expert_w.split(half, dim=0)
                fused[f"{prefix}.experts.{i}.gate_proj.weight"] = gate_w
                fused[f"{prefix}.experts.{i}.up_proj.weight"] = up_w
            del renamed[name]
        # PhiMoE (fused): experts.down_proj [N, hidden, inter]
        elif ".experts.down_proj" in name and tensor.dim() == 3 and "experts." in name:
            # Only match fused format (3D tensor at experts.down_proj, not experts.{i}.down_proj)
            parts = name.split(".experts.down_proj")
            if len(parts) == 2 and not parts[0].endswith(tuple("0123456789")):
                prefix = parts[0]
                num_experts = tensor.shape[0]
                for i in range(num_experts):
                    fused[f"{prefix}.experts.{i}.down_proj.weight"] = tensor[i]
                del renamed[name]

    renamed.update(fused)
    return renamed
