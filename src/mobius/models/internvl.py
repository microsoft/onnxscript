# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""InternVL2 multimodal model (vision + text) — 3-model split.

Splits the InternVL2 architecture into three ONNX models for
onnxruntime-genai:

- **decoder**: text decoder (Qwen2/LLaMA) taking ``inputs_embeds``
- **vision**: InternViT encoder + pixel shuffle + MLP projector
- **embedding**: token embedding + image feature fusion

Key differences from LLaVA:
- InternViT vision encoder with fused QKV, CLS token, and layer scale
- Pixel shuffle downsampling (``downsample_ratio=0.5``) before projection
- 4-element MLP projector: LayerNorm → Linear → GELU → Linear

HuggingFace reference: ``InternVLChatModel`` in
``OpenGVLab/InternVL2-*`` (custom code, model_type ``internvl_chat``).

HuggingFace weight names:
- ``vision_model.embeddings.*``
- ``vision_model.encoder.layers.N.*``
- ``mlp1.{0,1,3}.*``
- ``language_model.model.* / language_model.lm_head.*``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import (
    vlm_decoder_weights,
    vlm_embedding_weights,
)
from mobius.components import (
    Conv2d,
    Embedding,
    Linear,
)
from mobius.components._vision import VisionLayerNorm
from mobius.models.base import TextModel

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# InternViT vision encoder components
# ---------------------------------------------------------------------------


class _InternVisionLinear(nn.Module):
    """Linear layer with bias for InternViT."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter([out_features, in_features])
        self.bias = nn.Parameter([out_features])

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        result = op.MatMul(x, weight_t)
        return op.Add(result, self.bias)


class _InternVisionEmbeddings(nn.Module):
    """InternViT patch embedding with CLS token and position embeddings.

    Unlike SigLIP/CLIP, InternViT prepends a learnable CLS token and
    includes it in the position embedding table (num_patches + 1 positions).

    HF reference: ``InternVisionEmbeddings`` in ``modeling_intern_vit.py``.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        num_channels: int = 3,
    ):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        # CLS token — bare parameter (no .weight suffix in HF)
        self.class_embedding = nn.Parameter([1, 1, hidden_size])
        # Conv2d patch embedding — produces .weight and .bias
        self.patch_embedding = Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # Position embedding includes CLS position — bare parameter
        self.position_embedding = nn.Parameter([1, self.num_patches + 1, hidden_size])

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # pixel_values: [batch, channels, height, width]
        patch_embeds = self.patch_embedding(op, pixel_values)
        # patch_embeds: [batch, hidden_size, grid_h, grid_w]

        # Flatten spatial dims: [batch, hidden_size, num_patches]
        batch_size = op.Shape(patch_embeds, start=0, end=1)
        hidden_dim = op.Shape(patch_embeds, start=1, end=2)
        minus_one = op.Constant(value_ints=[-1])
        flat_shape = op.Concat(batch_size, hidden_dim, minus_one, axis=0)
        patch_embeds = op.Reshape(patch_embeds, flat_shape)
        # Transpose to [batch, num_patches, hidden_size]
        patch_embeds = op.Transpose(patch_embeds, perm=[0, 2, 1])

        # Prepend CLS token: [batch, 1, hidden_size]
        cls_token = op.Expand(
            self.class_embedding,
            op.Concat(batch_size, op.Constant(value_ints=[1]), hidden_dim, axis=0),
        )
        # [batch, num_patches + 1, hidden_size]
        embeddings = op.Concat(cls_token, patch_embeds, axis=1)

        # Add position embeddings
        embeddings = op.Add(embeddings, self.position_embedding)
        return embeddings


class _InternVisionAttention(nn.Module):
    """InternViT attention with fused QKV projection.

    Uses a single ``qkv`` linear layer (3x hidden_size) instead of separate
    Q, K, V projections. The fused weight is split in forward, matching
    HF weight names directly (no preprocess_weights rename needed).

    HF reference: ``InternAttention`` in ``modeling_intern_vit.py``.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        # Fused QKV — matches HF's attn.qkv.weight / attn.qkv.bias
        self.qkv = _InternVisionLinear(hidden_size, 3 * hidden_size)
        self.proj = _InternVisionLinear(hidden_size, hidden_size)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # hidden_states: [batch, seq_len, hidden_size]
        qkv = self.qkv(op, hidden_states)  # [batch, seq, 3*hidden]
        # Split into Q, K, V along last dimension
        q, k, v = op.Split(qkv, num_outputs=3, axis=-1, _outputs=3)

        # Bidirectional attention (no causal mask, no KV cache)
        attn_output = op.Attention(
            q,
            k,
            v,
            kv_num_heads=self.num_heads,
            q_num_heads=self.num_heads,
            scale=self.scale,
            _outputs=1,
        )
        return self.proj(op, attn_output)


class _InternVisionMLP(nn.Module):
    """InternViT MLP: fc1 → GELU → fc2.

    HF reference: ``InternMLP`` in ``modeling_intern_vit.py``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = _InternVisionLinear(hidden_size, intermediate_size)
        self.fc2 = _InternVisionLinear(intermediate_size, hidden_size)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        hidden_states = self.fc1(op, hidden_states)
        hidden_states = op.Gelu(hidden_states)
        return self.fc2(op, hidden_states)


class _InternVisionEncoderLayer(nn.Module):
    """InternViT encoder layer with layer scale.

    Structure: LayerNorm → Attention → LayerScale → Residual
             → LayerNorm → MLP → LayerScale → Residual

    Layer scale (``ls1``, ``ls2``) are bare parameter vectors that
    element-wise multiply the sub-layer output before the residual add.

    HF reference: ``InternVisionEncoderLayer`` in ``modeling_intern_vit.py``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn = _InternVisionAttention(hidden_size, num_heads)
        self.mlp = _InternVisionMLP(hidden_size, intermediate_size)
        self.norm1 = VisionLayerNorm(hidden_size, eps=norm_eps)
        self.norm2 = VisionLayerNorm(hidden_size, eps=norm_eps)
        # Layer scale — bare parameters (no .weight suffix in HF)
        self.ls1 = nn.Parameter([hidden_size])
        self.ls2 = nn.Parameter([hidden_size])

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # Pre-norm attention with layer scale
        residual = hidden_states
        hidden_states = self.norm1(op, hidden_states)
        hidden_states = self.attn(op, hidden_states)
        hidden_states = op.Mul(hidden_states, self.ls1)
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm MLP with layer scale
        residual = hidden_states
        hidden_states = self.norm2(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Mul(hidden_states, self.ls2)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class _InternVisionEncoder(nn.Module):
    """Stack of InternViT encoder layers.

    HF reference: ``InternVisionEncoder`` in ``modeling_intern_vit.py``.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _InternVisionEncoderLayer(hidden_size, intermediate_size, num_heads, norm_eps)
                for _ in range(num_layers)
            ]
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        for layer in self.layers:
            hidden_states = layer(op, hidden_states)
        return hidden_states


class _InternVisionModel(nn.Module):
    """InternViT vision model: embeddings + encoder (no post-layernorm).

    Unlike CLIP/SigLIP, InternViT does not apply a final layer norm after
    the encoder. The CLS token is included in embeddings but stripped by
    the caller (extract_feature) before pixel shuffle.

    HF reference: ``InternVisionModel`` in ``modeling_intern_vit.py``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None, "VisionConfig is required"
        self.embeddings = _InternVisionEmbeddings(
            image_size=vc.image_size,
            patch_size=vc.patch_size,
            hidden_size=vc.hidden_size,
        )
        self.encoder = _InternVisionEncoder(
            num_layers=vc.num_hidden_layers,
            hidden_size=vc.hidden_size,
            intermediate_size=vc.intermediate_size,
            num_heads=vc.num_attention_heads,
            norm_eps=vc.norm_eps,
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        hidden_states = self.embeddings(op, pixel_values)
        hidden_states = self.encoder(op, hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Pixel shuffle + MLP projector
# ---------------------------------------------------------------------------


class _GELUPlaceholder(nn.Module):
    """GELU activation as a module (fills index 2 in nn.Sequential).

    InternVL2's ``mlp1`` is ``nn.Sequential(LayerNorm, Linear, GELU, Linear)``.
    GELU has no parameters but occupies index 2, pushing the second Linear
    to index 3. This placeholder ensures correct sequential indexing so that
    weight names ``mlp1.1.*`` and ``mlp1.3.*`` match HuggingFace.
    """

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.Gelu(x)


# ---------------------------------------------------------------------------
# Three-model split
# ---------------------------------------------------------------------------


class _InternVL2DecoderModel(nn.Module):
    """InternVL2 text decoder taking inputs_embeds.

    Reuses the standard TextModel (Qwen2/LLaMA) decoder.
    Weight prefix: ``language_model.model.*`` / ``language_model.lm_head.*``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_decoder_weights(state_dict, tie=self.config.tie_word_embeddings)


class _InternVL2VisionEncoderModel(nn.Module):
    """InternVL2 vision encoder: InternViT + pixel shuffle + MLP projector.

    Pipeline: pixel_values → InternViT → strip CLS → pixel_shuffle(0.5)
    → mlp1(LayerNorm → Linear → GELU → Linear) → image_features.

    The pixel shuffle halves spatial dimensions and quadruples channels:
    ``(N, H*W, C) → (N, H/2 * W/2, C*4)``

    HF reference: ``InternVLChatModel.extract_feature`` +
    ``InternVLChatModel.pixel_shuffle``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None, "VisionConfig is required"
        self._downsample_ratio = 0.5
        vit_hidden = vc.hidden_size
        llm_hidden = config.hidden_size
        # Input dim after pixel shuffle: hidden_size * (1/ratio)^2 = hidden * 4
        proj_input_dim = vit_hidden * int((1 / self._downsample_ratio) ** 2)

        self.vision_model = _InternVisionModel(config)
        # mlp1: Sequential(LayerNorm, Linear, GELU, Linear)
        # Indices: 0=LayerNorm, 1=Linear, 2=GELU(no params), 3=Linear
        self.mlp1 = nn.Sequential(
            VisionLayerNorm(proj_input_dim),
            Linear(proj_input_dim, llm_hidden, bias=True),
            _GELUPlaceholder(),
            Linear(llm_hidden, llm_hidden, bias=True),
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # Run InternViT encoder
        vit_embeds = self.vision_model(op, pixel_values)
        # vit_embeds: [batch, num_patches+1, hidden_size] (includes CLS)

        # Strip CLS token (first position)
        # → [batch, num_patches, hidden_size]
        vit_embeds = op.Slice(
            vit_embeds,
            op.Constant(value_ints=[1]),  # start
            op.Constant(value_ints=[2147483647]),  # end (INT_MAX)
            op.Constant(value_ints=[1]),  # axes (seq dim)
        )

        # Pixel shuffle: (batch, num_patches, C) → (batch, num_patches/4, C*4)
        vit_embeds = self._pixel_shuffle(op, vit_embeds)

        # MLP projector
        image_features = self.mlp1(op, vit_embeds)
        return image_features

    def _pixel_shuffle(self, op, x):
        """Pixel shuffle downsampling (ps_version='v2').

        Reshapes (N, H*W, C) → spatial grid (N, W, H, C) → interleave
        spatial dims by scale factor → flatten back to (N, H'*W', C').

        With downsample_ratio=0.5:
        - (N, H*W, C) → (N, H, W, C) → (N, W, H/2, C*2) →
          (N, H/2, W, C*2) → (N, H/2, W/2, C*4) → (N, W/2, H/2, C*4)
          → (N, H/2*W/2, C*4)

        HF reference: ``InternVLChatModel.pixel_shuffle``.
        """
        scale = self._downsample_ratio  # 0.5
        batch = op.Shape(x, start=0, end=1)
        seq_len = op.Shape(x, start=1, end=2)
        channels = op.Shape(x, start=2, end=3)

        # Compute H = W = sqrt(num_patches).
        # InternVL2 tiles are always square (448x448 / patch_size^2),
        # so num_patches is a perfect square. Non-square grids would
        # need explicit h, w parameters.
        h = op.Sqrt(op.Cast(seq_len, to=1))  # FLOAT
        h = op.Cast(h, to=7)  # INT64
        w = h

        # Reshape to (N, H, W, C) — "spatial grid"
        shape_4d = op.Concat(batch, h, w, channels, axis=0)
        x = op.Reshape(x, shape_4d)

        # Step 1: view(N, W, H*scale, C/scale)
        h_scaled = op.Cast(
            op.Mul(op.Cast(h, to=1), op.Constant(value_float=scale)),
            to=7,
        )
        c_over_scale = op.Cast(
            op.Div(op.Cast(channels, to=1), op.Constant(value_float=scale)),
            to=7,
        )
        shape_step1 = op.Concat(batch, w, h_scaled, c_over_scale, axis=0)
        x = op.Reshape(x, shape_step1)

        # Step 2: permute(0, 2, 1, 3) → (N, H*scale, W, C/scale)
        x = op.Transpose(x, perm=[0, 2, 1, 3])

        # Step 3: view(N, H*scale, W*scale, C/(scale^2))
        w_scaled = op.Cast(
            op.Mul(op.Cast(w, to=1), op.Constant(value_float=scale)),
            to=7,
        )
        c_over_scale2 = op.Cast(
            op.Div(
                op.Cast(channels, to=1),
                op.Constant(value_float=scale * scale),
            ),
            to=7,
        )
        shape_step3 = op.Concat(batch, h_scaled, w_scaled, c_over_scale2, axis=0)
        x = op.Reshape(x, shape_step3)

        # Step 4 (ps_version='v2'): permute(0, 2, 1, 3)
        x = op.Transpose(x, perm=[0, 2, 1, 3])

        # Flatten spatial dims → (N, H'*W', C*4)
        flat_shape = op.Concat(batch, op.Constant(value_ints=[-1]), c_over_scale2, axis=0)
        x = op.Reshape(x, flat_shape)
        return x

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("vision_model.", "mlp1."))
        }


class _InternVL2EmbeddingModel(nn.Module):
    """InternVL2 embedding: token lookup + image feature fusion.

    Uses ``img_context_token_id`` as the image placeholder token.
    Same fusion pattern as LLaVA — scatter vision features at
    image token positions using cumulative sum indexing.

    HF reference: ``InternVLChatModel.forward`` (input embedding + fusion).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        if config.image_token_id is None:
            raise ValueError(
                "InternVL2 requires image_token_id in config. "
                "Set it to the token ID used for image placeholders "
                "(e.g. 151667 for InternVL2)."
            )
        self.image_token_id = config.image_token_id

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
    ):
        text_embeds = self.embed_tokens(op, input_ids)

        # Create mask where input_ids == image_token_id
        image_mask = op.Equal(
            input_ids,
            op.Constant(value_int=self.image_token_id),
        )
        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        # Compute indices into image_features via cumulative sum
        mask_int = op.Cast(image_mask, to=7)
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        gathered = op.Gather(image_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_embedding_weights(state_dict)


class InternVL2Model(nn.Module):
    """InternVL2 vision-language model (3-model split).

    Builds three separate ONNX models:
    - decoder: text decoder (Qwen2/LLaMA) taking inputs_embeds
    - vision_encoder: InternViT + pixel shuffle + MLP projector
    - embedding: token embedding + image feature fusion

    HF reference: ``InternVLChatModel`` (model_type ``internvl_chat``).
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _InternVL2DecoderModel(config)
        self.vision_encoder = _InternVL2VisionEncoderModel(config)
        self.embedding = _InternVL2EmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "InternVL2Model uses VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) "
            "separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF weights to the correct ONNX sub-model initializer names.

        HF prefixes → ONNX prefixes:
        - ``vision_model.*`` → ``vision_encoder.vision_model.*``
        - ``mlp1.*`` → ``vision_encoder.mlp1.*``
        - ``language_model.model.*`` → ``decoder.model.*``
        - ``language_model.lm_head.*`` → ``decoder.lm_head.*``
        - ``language_model.model.embed_tokens.*`` →
          ``embedding.embed_tokens.*`` (dual copy)
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(("vision_model.", "mlp1.")):
                renamed[f"vision_encoder.{key}"] = value
            elif key.startswith("language_model."):
                suffix = key[len("language_model.") :]
                renamed[f"decoder.{suffix}"] = value
                # Embedding model gets embed_tokens
                if suffix.startswith("model.embed_tokens."):
                    embed_suffix = suffix[len("model.") :]
                    renamed[f"embedding.{embed_suffix}"] = value
        # Weight tying: copy embed_tokens → lm_head
        if self.config.tie_word_embeddings:
            embed_key = "embedding.embed_tokens.weight"
            head_key = "decoder.lm_head.weight"
            if head_key not in renamed and embed_key in renamed:
                renamed[head_key] = renamed[embed_key]
        return renamed
