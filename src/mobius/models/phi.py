# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phi model variants: Phi-1/2, Phi-4MM (multimodal), Phi-3 Small.

Replicates HuggingFace's ``PhiForCausalLM``, ``Phi4MMForCausalLM``,
and ``Phi3SmallForCausalLM``. Phi-4MM adds LoRA adapters for vision
and audio modalities. Phi-3 Small uses block-sparse attention with
MuP (maximal update parameterization) scaling.
"""

from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import split_fused_qkv, split_gate_up_proj
from mobius.components import (
    FCMLP,
    ConformerEncoder,
    Embedding,
    InputMixer,
    LayerNorm,
    Linear,
    PatchEmbedding,
    RMSNorm,
    VisionEncoder,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._attention import Attention
from mobius.components._decoder import DecoderLayer
from mobius.components._lora import LoRALinear
from mobius.models.base import CausalLMModel, TextModel
from mobius.models.phi3 import Phi3CausalLMModel


class _PhiDecoderLayer(nn.Module):
    """Phi-1/2 decoder layer with single-norm parallel residual.

    A single LayerNorm is applied to the hidden states, then both attention
    and MLP receive the same normalized output. Their results are summed
    with the residual in a single addition:

        ln_out = input_layernorm(hidden)
        out = hidden + attn(ln_out) + mlp(ln_out)

    This is the same pattern as GPT-J/CodeGen. Unlike Phi-3 which is
    sequential (with a separate ``post_attention_layernorm`` before the MLP),
    Phi-1/2 shares one norm between both branches.

    Attribute names match HF ``PhiDecoderLayer``:
    - ``input_layernorm`` for the shared LayerNorm
    - ``self_attn`` for the attention module
    - ``mlp`` for the FCMLP module
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        # Single shared norm (no post_attention_layernorm in Phi-1/2)
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)  # 'self_attn' matches HF attribute name
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act or "gelu",
            bias=config.mlp_bias,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ) -> tuple[ir.Value, tuple]:
        residual = hidden_states

        # Single norm shared between attention and MLP
        ln_out = self.input_layernorm(op, hidden_states)  # (B, S, H)
        attn_out, present_kv = self.self_attn(
            op, ln_out, attention_bias, position_embeddings, past_key_value
        )
        mlp_out = self.mlp(op, ln_out)  # same ln_out as attention

        # Parallel residual: both branches added in one step
        hidden_states = op.Add(residual, op.Add(attn_out, mlp_out))
        return hidden_states, present_kv


class _PhiTextModel(nn.Module):
    """Phi-1/2 backbone with RoPE and full LayerNorm.

    Attribute names match HF ``PhiModel``:
    - ``embed_tokens`` for the token embedding
    - ``layers`` for the decoder layer list
    - ``final_layernorm`` for the output norm (HF uses this name, not ``norm``)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [_PhiDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # HF Phi names the final norm "final_layernorm" (not "norm" as in Llama)
        self.final_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
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
                op, hidden_states, attention_bias, position_embeddings, past_kv
            )
            present_key_values.append(present_kv)

        hidden_states = self.final_layernorm(op, hidden_states)
        return hidden_states, present_key_values


class PhiCausalLMModel(CausalLMModel):
    """Phi-1/2 causal language model with parallel attention.

    Differences from the Llama-style ``CausalLMModel``:
    - Single-norm parallel residual (like GPT-J, not sequential like Llama)
    - Non-gated FCMLP instead of gated MLP
    - Full LayerNorm throughout (not RMSNorm)
    - LM head has a bias term (``lm_head.bias``)
    - Final norm is ``model.final_layernorm`` (matches HF attribute)

    Replicates HuggingFace's ``PhiForCausalLM``.
    """

    default_task: str = "text-generation"
    category: str = "Text Generation"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = _PhiTextModel(config)
        # Phi LM head has a bias term
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op, input_ids, attention_mask, position_ids, past_key_values
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF Phi weight names to our ONNX attribute names.

        Most paths match directly (model.embed_tokens, model.layers.N.*,
        model.final_layernorm, lm_head, self_attn.q/k/v_proj). Three renames:

        1. Output proj: ``self_attn.dense.*`` → ``self_attn.o_proj.*``
        2. MLP up:   ``mlp.fc1.*`` → ``mlp.up_proj.*``
        3. MLP down: ``mlp.fc2.*`` → ``mlp.down_proj.*``
        """
        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            key = key.replace(".self_attn.dense.", ".self_attn.o_proj.")
            key = key.replace(".mlp.fc1.", ".mlp.up_proj.")
            key = key.replace(".mlp.fc2.", ".mlp.down_proj.")
            new_state_dict[key] = value
        return super().preprocess_weights(new_state_dict)


def _parse_lora_adapters(
    config: ArchitectureConfig,
) -> list[tuple[str, int, float]]:
    """Extract LoRA adapter specs from the config."""
    adapters = []
    for name, sub in (("vision", config.vision), ("speech", config.audio)):
        lora_cfg = getattr(sub, "lora", None) if sub is not None else None
        if lora_cfg is None:
            continue
        rank = lora_cfg["r"]
        alpha = lora_cfg["lora_alpha"]
        scale = alpha / rank
        adapters.append((name, rank, scale))
    return adapters


def _make_lora_linear_factory(
    lora_adapters: list[tuple[str, int, float]],
):
    """Create a LoRALinear factory that captures lora_adapters via closure."""

    def factory(in_features: int, out_features: int, bias: bool = True) -> LoRALinear:
        return LoRALinear(in_features, out_features, bias=bias, lora_adapters=lora_adapters)

    return factory


class _LoRATextModel(TextModel):
    """Text model with LoRA-aware decoder layers."""

    def __init__(
        self, config: ArchitectureConfig, lora_adapters: list[tuple[str, int, float]]
    ):
        nn.Module.__init__(self)
        lora_factory = _make_lora_linear_factory(lora_adapters)
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, linear_class=lora_factory)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)


def _preprocess_phi4mm_weights(
    config: ArchitectureConfig, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Shared weight preprocessing for Phi4MM models (LoRA + fused weight splitting)."""
    intermediate_size = config.intermediate_size

    # Strip "base_layer." from LoRA-wrapped weight names
    # HF stores e.g. "qkv_proj.base_layer.weight" → we need "qkv_proj.weight"
    for key in list(state_dict.keys()):
        if ".base_layer." in key:
            new_key = key.replace(".base_layer.", ".")
            state_dict[new_key] = state_dict.pop(key)

    for key in list(state_dict.keys()):
        # Split qkv_proj base weight/bias
        if ("qkv_proj.weight" in key or "qkv_proj.bias" in key) and "lora" not in key:
            w = state_dict.pop(key)
            base = key.split("qkv_proj.")[0]
            suffix = key.split("qkv_proj.")[1]
            q, k, v = split_fused_qkv(
                w,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
            )
            state_dict[f"{base}q_proj.{suffix}"] = q
            state_dict[f"{base}k_proj.{suffix}"] = k
            state_dict[f"{base}v_proj.{suffix}"] = v

        # Split qkv_proj LoRA B weights (output dim split, same layout as base)
        elif "qkv_proj.lora_B." in key:
            w = state_dict.pop(key)
            base = key.split("qkv_proj.")[0]
            suffix = key.split("qkv_proj.")[1]
            q, k, v = split_fused_qkv(
                w,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
            )
            state_dict[f"{base}q_proj.{suffix}"] = q
            state_dict[f"{base}k_proj.{suffix}"] = k
            state_dict[f"{base}v_proj.{suffix}"] = v

        # Split qkv_proj LoRA A weights (same A for q/k/v)
        elif "qkv_proj.lora_A." in key:
            w = state_dict.pop(key)
            base = key.split("qkv_proj.")[0]
            suffix = key.split("qkv_proj.")[1]
            state_dict[f"{base}q_proj.{suffix}"] = w
            state_dict[f"{base}k_proj.{suffix}"] = w.clone()
            state_dict[f"{base}v_proj.{suffix}"] = w.clone()

        # Split gate_up_proj base weight/bias
        elif (
            "gate_up_proj.weight" in key or "gate_up_proj.bias" in key
        ) and "lora" not in key:
            w = state_dict.pop(key)
            base = key.split("gate_up_proj.")[0]
            suffix = key.split("gate_up_proj.")[1]
            gate, up = split_gate_up_proj(w, intermediate_size)
            state_dict[f"{base}gate_proj.{suffix}"] = gate
            state_dict[f"{base}up_proj.{suffix}"] = up

        # Split gate_up_proj LoRA B weights (output dim split)
        elif "gate_up_proj.lora_B." in key:
            w = state_dict.pop(key)
            base = key.split("gate_up_proj.")[0]
            suffix = key.split("gate_up_proj.")[1]
            gate, up = split_gate_up_proj(w, intermediate_size)
            state_dict[f"{base}gate_proj.{suffix}"] = gate
            state_dict[f"{base}up_proj.{suffix}"] = up

        # Split gate_up_proj LoRA A weights (same A for gate/up)
        elif "gate_up_proj.lora_A." in key:
            w = state_dict.pop(key)
            base = key.split("gate_up_proj.")[0]
            suffix = key.split("gate_up_proj.")[1]
            state_dict[f"{base}gate_proj.{suffix}"] = w
            state_dict[f"{base}up_proj.{suffix}"] = w.clone()

    # Weight tying
    if config.tie_word_embeddings:
        embed_key = "model.embed_tokens.weight"
        lm_head_key = "lm_head.weight"
        if lm_head_key not in state_dict and embed_key in state_dict:
            state_dict[lm_head_key] = state_dict[embed_key]

    return state_dict


class Phi4MMCausalLMModel(Phi3CausalLMModel):
    """Phi4-MM text-only model with LoRA adapters.

    LoRA weights are kept as separate parameters in the ONNX model.
    The forward pass computes the LoRA contribution alongside the base linear.

    Replicates HuggingFace's ``Phi4MMForCausalLM``.
    """

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        lora_adapters = _parse_lora_adapters(config)

        self.model = _LoRATextModel(config, lora_adapters)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _preprocess_phi4mm_weights(self.config, state_dict)


# -----------------------------------------------------------------------
# Phi4-MM Multimodal Model (Vision + Audio + Text with LoRA)
# -----------------------------------------------------------------------


class _Phi4MMSigLIPEncoder(nn.Module):
    """SigLIP vision encoder for Phi4MM (no post_layernorm)."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        image_size = vc.image_size or 336 if vc else 336
        patch_size = vc.patch_size or 14 if vc else 14
        hidden_size = vc.hidden_size or config.hidden_size if vc else config.hidden_size
        intermediate_size = vc.intermediate_size or hidden_size * 4 if vc else hidden_size * 4
        num_heads = vc.num_attention_heads or 4 if vc else 4
        num_layers = vc.num_hidden_layers or 2 if vc else 2
        norm_eps = vc.norm_eps if vc else 1e-6
        self.embeddings = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
        )
        self.encoder = VisionEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            norm_eps=norm_eps,
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        hidden_states = self.embeddings(op, pixel_values)
        return self.encoder(op, hidden_states)


class _GELUModule(nn.Module):
    """GELU activation as an nn.Module (no parameters)."""

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.Gelu(x)


class _Phi4MMProjectionMLP(nn.Module):
    """Sequential MLP: Linear → GELU → Linear.

    Registers children at string indices "0", "1", "2" to match
    HuggingFace Sequential naming convention. Uses nn.Module with
    indexed setattr rather than subclassing ModuleList to avoid
    doubled path segments in ONNX initializer names.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        layers = [
            Linear(in_features, out_features),
            _GELUModule(),
            Linear(out_features, out_features),
        ]
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)
        self._layers = layers

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        for layer in self._layers:
            x = layer(op, x)
        return x


class _Phi4MMImageEmbedding(nn.Module):
    """Phi4MM image embedding: SigLIP + projection + HD transform params."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        vision_hidden_size = vc.hidden_size or config.hidden_size if vc else config.hidden_size
        text_hidden_size = config.hidden_size

        self.img_processor = _Phi4MMSigLIPEncoder(config)
        self.img_projection = _Phi4MMProjectionMLP(vision_hidden_size, text_hidden_size)
        self.glb_GN = nn.Parameter([1, 1, vision_hidden_size])
        self.sub_GN = nn.Parameter([1, 1, 1, vision_hidden_size])

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        vision_features = self.img_processor(op, pixel_values)
        return self.img_projection(op, vision_features)


class _Phi4MMAudioEmbedding(nn.Module):
    """Phi4MM audio embedding: Conformer encoder + speech/vision projections."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        audio = config.audio
        audio_dim = (audio.attention_dim if audio else None) or 256
        text_hidden_size = config.hidden_size

        self.encoder = ConformerEncoder(
            input_size=(audio.input_size if audio else None) or 80,
            attention_dim=audio_dim,
            attention_heads=(audio.attention_heads if audio else None) or 4,
            num_blocks=(audio.num_blocks if audio else None) or 2,
            linear_units=(audio.linear_units if audio else None) or 1024,
            kernel_size=(audio.kernel_size if audio else None) or 3,
            conv_channels=(audio.conv_channels if audio else None) or audio_dim,
            t5_bias_max_distance=((audio.t5_bias_max_distance if audio else None) or 500),
        )

        # Audio projection "speech" and "vision" branches.
        # Named directly to match ONNX initializer paths; the HF
        # "audio_projection." prefix is stripped in preprocess_weights.
        self.speech = _Phi4MMProjectionMLP(audio_dim, text_hidden_size)
        self.vision = _Phi4MMProjectionMLP(audio_dim, text_hidden_size)

    def forward(self, op: builder.OpBuilder, audio_features: ir.Value):
        audio_hidden = self.encoder(op, audio_features)
        return self.speech(op, audio_hidden)


class _Phi4MMImageAudioEmbedding(nn.Module):
    """Combined image + audio embedding (embed_tokens_extend)."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.image_embed = _Phi4MMImageEmbedding(config)
        self.audio_embed = _Phi4MMAudioEmbedding(config)

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value | None = None,
        audio_features: ir.Value | None = None,
    ):
        image_embeddings = None
        if pixel_values is not None:
            image_embeddings = self.image_embed(op, pixel_values)
        audio_embeddings = None
        if audio_features is not None:
            audio_embeddings = self.audio_embed(op, audio_features)
        return image_embeddings, audio_embeddings


class _Phi4MMMultiModalTextModel(nn.Module):
    """Phi4MM inner model: text embeddings + multimodal mixing + LoRA decoder."""

    def __init__(
        self,
        config: ArchitectureConfig,
        lora_adapters: list[tuple[str, int, float]],
    ):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        lora_factory = _make_lora_linear_factory(lora_adapters)
        self.embed_tokens_extend = _Phi4MMImageAudioEmbedding(config)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, linear_class=lora_factory)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self._image_mixer = InputMixer(image_token_id=config.image_token_id or 200010)
        self._audio_mixer = InputMixer(
            image_token_id=(config.audio.token_id if config.audio else None) or 200011
        )

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        pixel_values: ir.Value | None = None,
        audio_features: ir.Value | None = None,
    ):
        has_multimodal = pixel_values is not None or audio_features is not None
        if has_multimodal:
            text_embeddings = self.embed_tokens(op, input_ids)
            image_embeddings, audio_embeddings = self.embed_tokens_extend(
                op, pixel_values=pixel_values, audio_features=audio_features
            )
            hidden_states = text_embeddings
            if image_embeddings is not None:
                hidden_states = self._image_mixer(
                    op, hidden_states, image_embeddings, input_ids
                )
            if audio_embeddings is not None:
                hidden_states = self._audio_mixer(
                    op, hidden_states, audio_embeddings, input_ids
                )
        else:
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


# -----------------------------------------------------------------------
# Phi4-MM Four-Model Split Sub-Modules
# -----------------------------------------------------------------------


class _Phi4MMVisionModel(nn.Module):
    """Phi4MM vision encoder: SigLIP + projection MLP + HD transform params.

    Takes raw pixel values, encodes through SigLIP, and projects to the
    text decoder's hidden dimension. Includes glb_GN and sub_GN
    parameters for HD spatial merge.

    Inputs:
        pixel_values: [batch, 3, image_size, image_size]
        image_sizes: [num_images, 2] — (height, width) per image for HD crop
    Outputs:
        image_features: [num_image_tokens, hidden_size]
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        vision_hidden = (vc.hidden_size if vc else None) or config.hidden_size
        text_hidden = config.hidden_size

        self.img_processor = _Phi4MMSigLIPEncoder(config)
        self.img_projection = _Phi4MMProjectionMLP(vision_hidden, text_hidden)
        self.glb_GN = nn.Parameter([1, 1, vision_hidden])
        self.sub_GN = nn.Parameter([1, 1, 1, vision_hidden])

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
        image_sizes: ir.Value,
    ):
        # image_sizes is plumbed for the I/O contract with ORT GenAI.
        # It will be used by the HD dynamic crop transform in a follow-up.
        vision_features = self.img_processor(op, pixel_values)
        return self.img_projection(op, vision_features)


class _Phi4MMSpeechModel(nn.Module):
    """Phi4MM speech encoder: Conformer + projection MLPs.

    Encodes mel spectrogram audio features through a Conformer encoder
    and projects to the text decoder's hidden dimension. Includes both
    "speech" and "vision" projection branches, selected at runtime by
    ``audio_projection_mode``.

    Inputs:
        audio_embeds: [batch, audio_seq_len, num_mel_bins]
        audio_sizes: [num_audio_clips] — number of frames per clip
        audio_projection_mode: scalar int — 0=speech branch, 1=vision branch
    Outputs:
        audio_features: [num_speech_tokens, hidden_size]
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        audio = config.audio
        audio_dim = (audio.attention_dim if audio else None) or 256
        text_hidden = config.hidden_size

        self.encoder = ConformerEncoder(
            input_size=(audio.input_size if audio else None) or 80,
            attention_dim=audio_dim,
            attention_heads=(audio.attention_heads if audio else None) or 4,
            num_blocks=(audio.num_blocks if audio else None) or 2,
            linear_units=(audio.linear_units if audio else None) or 1024,
            kernel_size=(audio.kernel_size if audio else None) or 3,
            conv_channels=(audio.conv_channels if audio else None) or audio_dim,
            t5_bias_max_distance=(audio.t5_bias_max_distance if audio else None) or 500,
        )

        # Both projection branches for speech-only and combined modes.
        # Named "speech"/"vision" directly; the HF "audio_projection."
        # prefix is stripped in preprocess_weights since onnxscript
        # doesn't propagate intermediate nn.Module container names.
        self.speech = _Phi4MMProjectionMLP(audio_dim, text_hidden)
        self.vision = _Phi4MMProjectionMLP(audio_dim, text_hidden)

    def forward(
        self,
        op: builder.OpBuilder,
        audio_embeds: ir.Value,
        audio_sizes: ir.Value,
        audio_projection_mode: ir.Value,
    ):
        # audio_sizes is plumbed for the I/O contract with ORT GenAI.
        # It will be used for variable-length batching in a follow-up.
        audio_hidden = self.encoder(op, audio_embeds)

        # Both branches run (ONNX graphs are static), then select via mode
        speech_branch = self.speech(op, audio_hidden)
        vision_branch = self.vision(op, audio_hidden)

        # audio_projection_mode: 0=speech, 1=vision
        is_vision_mode = op.Equal(audio_projection_mode, 1)
        return op.Where(is_vision_mode, vision_branch, speech_branch)


class _Phi4MMEmbeddingModel(nn.Module):
    """Phi4MM embedding: token embedding + InputMixer fusion.

    Embeds text tokens and replaces image/audio placeholder positions
    with the corresponding projected features from the vision and
    speech encoders.

    Inputs:
        input_ids: [batch, seq_len]
        image_features: [num_image_tokens, hidden_size]
        audio_features: [num_speech_tokens, hidden_size]
    Outputs:
        inputs_embeds: [batch, seq_len, hidden_size]
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self._image_mixer = InputMixer(image_token_id=config.image_token_id or 200010)
        self._audio_mixer = InputMixer(
            image_token_id=(config.audio.token_id if config.audio else None) or 200011
        )

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
        audio_features: ir.Value,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        # Add batch dim: [num_tokens, hidden] → [1, num_tokens, hidden]
        # InputMixer expects [batch, seq, hidden] for GatherElements
        image_features_3d = op.Unsqueeze(image_features, [0])
        audio_features_3d = op.Unsqueeze(audio_features, [0])
        hidden_states = self._image_mixer(op, hidden_states, image_features_3d, input_ids)
        hidden_states = self._audio_mixer(op, hidden_states, audio_features_3d, input_ids)
        return hidden_states


class _Phi4MMDecoderModel(nn.Module):
    """Phi4MM text decoder with LoRA adapters.

    Takes fused input embeddings (text + vision + audio) and runs
    through the transformer decoder with LoRA-adapted attention and
    MLP layers.

    Inputs:
        inputs_embeds: [batch, seq_len, hidden_size]
        attention_mask: [batch, past_seq_len + seq_len]
        position_ids: [batch, seq_len]
        past_key_values: list of (key, value) tuples
    Outputs:
        logits: [batch, seq_len, vocab_size]
        present_key_values: list of (key, value) tuples
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        lora_adapters = _parse_lora_adapters(config)
        lora_factory = _make_lora_linear_factory(lora_adapters)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, linear_class=lora_factory)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(op, position_ids)
        # Use inputs_embeds for query_length since decoder has no input_ids
        attention_bias = create_attention_bias(
            op,
            input_ids=inputs_embeds,
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
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values


class Phi4MMMultiModalModel(nn.Module):
    """Phi-4 Multimodal model (4-model split).

    Produces four separate ONNX models via ``Phi4MMMultiModalTask``:

    - ``vision``: SigLIP encoder + projection -> image_features
    - ``speech``: Conformer encoder + projection -> audio_features
    - ``embedding``: token embedding + InputMixer fusion -> inputs_embeds
    - ``decoder``: LoRA text decoder + lm_head -> logits + KV cache

    Replicates HuggingFace's ``Phi4MMForCausalLM``.
    """

    default_task: str = "phi4mm-multimodal"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = _Phi4MMVisionModel(config)
        self.speech_encoder = _Phi4MMSpeechModel(config)
        self.embedding = _Phi4MMEmbeddingModel(config)
        self.decoder = _Phi4MMDecoderModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "Phi4MMMultiModalModel uses Phi4MMMultiModalTask "
            "which calls each sub-module (vision_encoder, "
            "speech_encoder, embedding, decoder) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Preprocess and remap weights for the 4-model split.

        First applies LoRA unwrapping and fused weight splitting,
        then remaps HuggingFace prefixes to 4-model structure:

        - ``model.embed_tokens_extend.image_embed.*``
          -> ``vision_encoder.*``
        - ``model.embed_tokens_extend.audio_embed.*``
          -> ``speech_encoder.*``
        - ``model.embed_tokens.*``
          -> ``embedding.embed_tokens.*``
        - ``model.layers.*`` -> ``decoder.layers.*``
        - ``model.norm.*`` -> ``decoder.norm.*``
        - ``lm_head.*`` -> ``decoder.lm_head.*``
        """
        state_dict = _preprocess_phi4mm_weights(self.config, state_dict)

        # Fix vision position embedding: 3D [1,N,H] -> 2D [N,H]
        for key in list(state_dict.keys()):
            if "img_processor.embeddings.position_embedding.weight" in key:
                if state_dict[key].dim() == 3:
                    state_dict[key] = state_dict[key].squeeze(0)

        # Remap prefixes to 4-model structure
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _remap_phi4mm_weight_key(key)
            renamed[new_key] = value

        # Duplicate embed_tokens weight for decoder (tied weights)
        embed_key = "embedding.embed_tokens.weight"
        lm_head_key = "decoder.lm_head.weight"
        if self.config.tie_word_embeddings:
            if embed_key in renamed and lm_head_key not in renamed:
                renamed[lm_head_key] = renamed[embed_key]

        return renamed


def _remap_phi4mm_weight_key(key: str) -> str:
    """Remap a single HuggingFace weight key to 4-model prefix."""
    # Vision encoder: image_embed sub-tree
    img_prefix = "model.embed_tokens_extend.image_embed."
    if key.startswith(img_prefix):
        return "vision_encoder." + key[len(img_prefix) :]

    # Speech encoder: audio_embed sub-tree
    audio_prefix = "model.embed_tokens_extend.audio_embed."
    if key.startswith(audio_prefix):
        suffix = key[len(audio_prefix) :]
        # Strip "audio_projection." since onnxscript resolves the speech
        # and vision projection branches directly on the module (Bug 5).
        suffix = suffix.removeprefix("audio_projection.")
        return "speech_encoder." + suffix

    # Token embeddings -> embedding model
    if key == "model.embed_tokens.weight":
        return "embedding.embed_tokens.weight"

    # Decoder layers
    if key.startswith("model.layers."):
        return "decoder." + key[len("model.") :]

    # Final norm
    if key.startswith("model.norm."):
        return "decoder." + key[len("model.") :]

    # LM head
    if key.startswith("lm_head."):
        return "decoder." + key

    return key


class Phi3SmallCausalLMModel(Phi3CausalLMModel):
    """Phi3-Small model with block-sparse attention.

    Uses block-sparse attention with alternating dense/sparse layers,
    MuP (maximal update parameterization) scaling, and GeGELU activation
    with clamping.

    Replicates HuggingFace's ``Phi3SmallForCausalLM``.
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = super().preprocess_weights(state_dict)
        q_size = self.config.num_attention_heads * self.config.head_dim
        kv_size = self.config.num_key_value_heads * self.config.head_dim
        num_kv_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        group_size = num_kv_groups + 2

        for key in list(state_dict.keys()):
            # Handle combined query_key_value projection
            if "query_key_value.weight" in key:
                weight = state_dict.pop(key)
                wqkv = weight.t().reshape(
                    self.config.hidden_size,
                    self.config.num_key_value_heads,
                    group_size,
                    self.config.head_dim,
                )
                q_weight = (
                    wqkv[:, :, :num_kv_groups, :].reshape(self.config.hidden_size, q_size).t()
                )
                k_weight = wqkv[:, :, [-2], :].reshape(self.config.hidden_size, kv_size).t()
                v_weight = wqkv[:, :, [-1], :].reshape(self.config.hidden_size, kv_size).t()

                prefix = key.replace("query_key_value.weight", "")
                state_dict[f"{prefix}q_proj.weight"] = q_weight
                state_dict[f"{prefix}k_proj.weight"] = k_weight
                state_dict[f"{prefix}v_proj.weight"] = v_weight

            elif "query_key_value.bias" in key:
                bias = state_dict.pop(key)
                bias_grouped = bias.reshape(
                    self.config.num_key_value_heads,
                    group_size,
                    self.config.head_dim,
                )
                q_bias = bias_grouped[:, :num_kv_groups, :].reshape(q_size)
                k_bias = bias_grouped[:, [-2], :].reshape(kv_size)
                v_bias = bias_grouped[:, [-1], :].reshape(kv_size)

                prefix = key.replace("query_key_value.bias", "")
                state_dict[f"{prefix}q_proj.bias"] = q_bias
                state_dict[f"{prefix}k_proj.bias"] = k_bias
                state_dict[f"{prefix}v_proj.bias"] = v_bias

        return state_dict
