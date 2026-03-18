# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._attention import Attention
from mobius.components._mlp import MLP
from mobius.components._rms_norm import RMSNorm

if TYPE_CHECKING:
    import onnx_ir as ir


class DecoderLayer(nn.Module):
    """Configurable transformer decoder layer.

    Consolidates pre-norm and post-norm decoder layer patterns with
    configurable norm classes and optional residual scaling.

    For genuinely different architectures (DeepSeek MLA, Gemma2 4-norm),
    use dedicated subclasses instead.

    Args:
        config: Architecture configuration.
        norm_class: Norm class to use (default: RMSNorm). Pass OffsetRMSNorm
            for Gemma/Qwen3.5 models. The norm class handles its own behavior
            (e.g., +1 offset) internally.
        residual_multiplier: Scale factor applied to attention and MLP outputs
            before residual addition (default: 1.0). Used by Granite models.
        attention_scale: Custom attention scale factor (default: None, meaning
            1/sqrt(head_dim)). Used by Granite models.
        post_norm: If True, apply norms after sub-layer outputs before residual
            addition (OLMo-2 style). Default is False (pre-norm).
        linear_class: Factory callable ``(in_features, out_features, bias=...)``
            for creating projection layers in Attention and MLP. Defaults to
            ``Linear``. Pass a LoRA factory for LoRA-adapted layers.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        *,
        norm_class: type[nn.Module] | None = None,
        residual_multiplier: float = 1.0,
        attention_scale: float | None = None,
        post_norm: bool = False,
        linear_class: type | None = None,
    ):
        super().__init__()
        if norm_class is None:
            norm_class = RMSNorm

        self._post_norm = post_norm
        self._residual_multiplier = residual_multiplier

        self.self_attn = Attention(
            config,
            rms_norm_class=norm_class,
            scale=attention_scale,
            linear_class=linear_class,
        )
        self.mlp = MLP(config, linear_class=linear_class)

        if post_norm:
            self.post_attention_layernorm = norm_class(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm = norm_class(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.input_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = norm_class(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        if self._post_norm:
            return self._forward_post_norm(
                op,
                hidden_states,
                attention_bias,
                position_embeddings,
                past_key_value,
            )
        return self._forward_pre_norm(
            op,
            hidden_states,
            attention_bias,
            position_embeddings,
            past_key_value,
        )

    def _forward_pre_norm(
        self,
        op: builder.OpBuilder,
        hidden_states,
        attention_bias,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        attn_output, present_key_value = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )

        if not math.isclose(self._residual_multiplier, 1.0):
            attn_output = op.Mul(attn_output, self._residual_multiplier)
        hidden_states = op.Add(residual, attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)

        if not math.isclose(self._residual_multiplier, 1.0):
            hidden_states = op.Mul(hidden_states, self._residual_multiplier)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_key_value

    def _forward_post_norm(
        self,
        op: builder.OpBuilder,
        hidden_states,
        attention_bias,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        residual = hidden_states
        attn_output, present_key_value = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = self.post_attention_layernorm(op, attn_output)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = self.post_feedforward_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_key_value


def create_decoder_layer(
    config: ArchitectureConfig,
    *,
    norm_class: type[nn.Module] | None = None,
    post_norm: bool = False,
    linear_class: type | None = None,
) -> DecoderLayer:
    """Config-driven factory for creating decoder layers.

    Reads ``residual_multiplier`` and ``attention_multiplier`` from the config
    automatically. Use explicit ``norm_class`` and ``post_norm`` parameters for
    variations not encoded in the config.

    Args:
        config: Architecture configuration. Fields ``residual_multiplier`` and
            ``attention_multiplier`` are read when present.
        norm_class: Norm class override (default: RMSNorm). Use OffsetRMSNorm
            for Gemma models.
        post_norm: If True, use post-norm residual connections (OLMo-2 style).
        linear_class: Factory callable for projection layers. Pass a LoRA
            factory for LoRA-adapted layers.

    Returns:
        A configured DecoderLayer instance.
    """
    residual_multiplier = getattr(config, "residual_multiplier", 1.0) or 1.0
    attention_scale = getattr(config, "attention_multiplier", None)

    return DecoderLayer(
        config,
        norm_class=norm_class,
        residual_multiplier=residual_multiplier,
        attention_scale=attention_scale,
        post_norm=post_norm,
        linear_class=linear_class,
    )


class PostNormDecoderLayer(DecoderLayer):
    """Post-norm decoder layer (OLMo-2 style). Backward-compatible alias.

    Equivalent to ``DecoderLayer(config, post_norm=True)``.
    Kept for backward compatibility with existing imports.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config, post_norm=True)
