# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""VideoLLaMA3 multimodal model (vision + text) — 3-model split.

Splits the VideoLLaMA3 architecture into three ONNX models for
onnxruntime-genai:

- **decoder**: Qwen2 text decoder taking ``inputs_embeds``
- **vision**: SigLIP-like ViT + MLP projector
- **embedding**: token embedding + image feature fusion

VideoLLaMA3 uses a SigLIP2-style vision encoder (27 layers, hidden=1152,
patch=14, no absolute position embeddings) with a 2-layer MLP projector
feeding into a Qwen2 text decoder.

HuggingFace weight names:
- ``model.vision_encoder.vision_tower.*`` (patch embed + transformer blocks)
- ``model.mm_projector.readout.{0,2}.*`` (projector linear layers)
- ``model.embed_tokens.*`` / ``model.layers.*`` / ``lm_head.*`` (decoder)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import vlm_decoder_weights, vlm_embedding_weights
from mobius.components import Embedding, Linear
from mobius.components._vision import VisionAttention, VisionLayerNorm, VisionMLP
from mobius.models.base import TextModel

if TYPE_CHECKING:
    import onnx_ir as ir


class _PatchEmbedConv(nn.Module):
    """Conv2d weight container matching HF ``patch_embed.proj.*`` naming."""

    def __init__(self, hidden_size: int, patch_size: int, num_channels: int = 3):
        super().__init__()
        # Conv2d weight: [out_channels, in_channels, kH, kW]
        self.weight = nn.Parameter([hidden_size, num_channels, patch_size, patch_size])
        self.bias = nn.Parameter([hidden_size])
        self._patch_size = patch_size

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        return op.Conv(
            pixel_values,
            self.weight,
            self.bias,
            kernel_shape=[self._patch_size, self._patch_size],
            strides=[self._patch_size, self._patch_size],
        )


class _VideoLLaMA3PatchEmbed(nn.Module):
    """Conv2d patch tokenization without absolute position embeddings.

    VideoLLaMA3 uses 2D RoPE in the attention layers instead of learnable
    absolute position embeddings, so the patch embedding module only
    performs the spatial convolution.

    HF path: ``vision_encoder.vision_tower.patch_embed.*``
    """

    def __init__(self, hidden_size: int, patch_size: int = 14, num_channels: int = 3):
        super().__init__()
        self.proj = _PatchEmbedConv(hidden_size, patch_size, num_channels)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # pixel_values: [N, C, H, W]
        patches = self.proj(op, pixel_values)
        # patches: [N, hidden, H/p, W/p]
        # Flatten spatial dims → [N, hidden, num_patches], transpose → [N, num_patches, hidden]
        batch = op.Shape(patches, start=0, end=1)
        hidden = op.Shape(patches, start=1, end=2)
        minus_one = op.Constant(value_ints=[-1])
        hw_flat_shape = op.Concat(batch, hidden, minus_one, axis=0)
        patches = op.Reshape(patches, hw_flat_shape)  # [N, hidden, num_patches]
        patches = op.Transpose(patches, perm=[0, 2, 1])  # [N, num_patches, hidden]
        return patches


class _VideoLLaMA3EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer matching VideoLLaMA3 weight names.

    Uses ``norm1`` / ``attn`` / ``norm2`` / ``mlp`` attribute names to match
    the HuggingFace ``blocks.N.*`` weight naming convention (as opposed to
    VisionEncoderLayer which uses ``layer_norm1`` / ``self_attn``).

    HF path: ``vision_encoder.vision_tower.blocks.N.*``
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = VisionLayerNorm(hidden_size, eps=norm_eps)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.norm2 = VisionLayerNorm(hidden_size, eps=norm_eps)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # Pre-norm attention block
        residual = hidden_states
        hidden_states = self.norm1(op, hidden_states)
        hidden_states = self.attn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm MLP block
        residual = hidden_states
        hidden_states = self.norm2(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class _VideoLLaMA3VisionTower(nn.Module):
    """SigLIP2-like ViT backbone for VideoLLaMA3.

    Consists of Conv2d patch embedding, stacked transformer blocks, and
    a final LayerNorm (``norm``). No CLS token, no absolute position
    embeddings (2D RoPE is simplified away for ONNX).

    HF path: ``vision_encoder.vision_tower.*``
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None, "VisionConfig required for VideoLLaMA3"
        assert vc.hidden_size is not None
        assert vc.intermediate_size is not None
        assert vc.num_hidden_layers is not None
        assert vc.num_attention_heads is not None

        self.patch_embed = _VideoLLaMA3PatchEmbed(
            hidden_size=vc.hidden_size,
            patch_size=vc.patch_size or 14,
        )
        self.blocks = nn.ModuleList(
            [
                _VideoLLaMA3EncoderLayer(
                    hidden_size=vc.hidden_size,
                    intermediate_size=vc.intermediate_size,
                    num_heads=vc.num_attention_heads,
                    norm_eps=vc.norm_eps,
                )
                for _ in range(vc.num_hidden_layers)
            ]
        )
        # Final layer norm after all blocks
        self.norm = VisionLayerNorm(vc.hidden_size, eps=vc.norm_eps)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # pixel_values: [N, 3, H, W] → patches: [N, num_patches, hidden]
        hidden_states = self.patch_embed(op, pixel_values)
        for block in self.blocks:
            hidden_states = block(op, hidden_states)
        # Post-encoder normalization: [N, num_patches, hidden]
        hidden_states = self.norm(op, hidden_states)
        return hidden_states


class _VideoLLaMA3VisionEncoderInner(nn.Module):
    """Wraps vision_tower so parameter paths match ``vision_encoder.*``.

    HF path: ``model.vision_encoder.*``
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.vision_tower = _VideoLLaMA3VisionTower(config)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        return self.vision_tower(op, pixel_values)


class _ProjectorLinear(nn.Module):
    """Single linear layer for the MLP projector readout slots."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter([out_features, in_features])
        self.bias = nn.Parameter([out_features])

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        return op.Add(op.MatMul(x, weight_t), self.bias)


class _VideoLLaMA3Projector(nn.Module):
    """2-layer MLP projector: readout[0]=Linear → GELU → readout[1]=Linear.

    HuggingFace stores this as a ``nn.Sequential`` where index 0 is Linear,
    index 1 is GELU activation, and index 2 is Linear.  Our ``readout``
    ModuleList uses indices 0 and 1 (no activation module), and
    ``preprocess_weights`` in the parent renames ``readout.2.*`` →
    ``readout.1.*``.

    HF path: ``model.mm_projector.*``
    """

    def __init__(self, vision_hidden_size: int, text_hidden_size: int):
        super().__init__()
        self.readout = nn.ModuleList(
            [
                _ProjectorLinear(vision_hidden_size, text_hidden_size),
                _ProjectorLinear(text_hidden_size, text_hidden_size),
            ]
        )

    def forward(self, op: builder.OpBuilder, vision_features: ir.Value):
        # vision_features: [N, num_patches, vision_hidden]
        hidden = self.readout[0](op, vision_features)
        hidden = op.Gelu(hidden)
        hidden = self.readout[1](op, hidden)
        # Output: [N, num_patches, text_hidden]
        return hidden


class _VideoLLaMA3VisionEncoderModel(nn.Module):
    """VideoLLaMA3 vision branch: SigLIP-like ViT + MLP projector.

    Receives pixel_values and returns projected image features ready to be
    fused with text embeddings.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None
        self.vision_encoder = _VideoLLaMA3VisionEncoderInner(config)
        self.mm_projector = _VideoLLaMA3Projector(
            vision_hidden_size=vc.hidden_size,
            text_hidden_size=config.hidden_size,
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        vision_features = self.vision_encoder(op, pixel_values)
        return self.mm_projector(op, vision_features)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        renamed: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            # Strip leading "model." — VideoLLaMA3 nests everything under model.*
            new_k = k[len("model.") :] if k.startswith("model.") else k
            if not new_k.startswith(("vision_encoder.", "mm_projector.")):
                continue
            # HF Sequential skips index 1 (GELU), so readout.2 → readout.1
            if "mm_projector.readout.2." in new_k:
                new_k = new_k.replace("mm_projector.readout.2.", "mm_projector.readout.1.")
            renamed[new_k] = v
        return renamed


class _VideoLLaMA3DecoderModel(nn.Module):
    """VideoLLaMA3 Qwen2 text decoder taking inputs_embeds."""

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
        # Decoder weights live directly under "model.*" (no "language_model." prefix)
        return vlm_decoder_weights(
            state_dict, prefix="model.", tie=self.config.tie_word_embeddings
        )


class _VideoLLaMA3EmbeddingModel(nn.Module):
    """VideoLLaMA3 embedding: token lookup + image feature fusion.

    Replaces image-token placeholder positions in text_embeds with the
    corresponding projected vision features.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.image_token_id = config.image_token_id or 0

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value, image_features: ir.Value):
        text_embeds = self.embed_tokens(op, input_ids)

        # Build a boolean mask where input token == image_token_id
        image_mask = op.Equal(
            input_ids,
            op.Constant(value_int=self.image_token_id),
        )
        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        # Map each image-token position to the corresponding vision feature row
        mask_int = op.Cast(image_mask, to=7)
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        gathered = op.Gather(image_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # embed_tokens lives at model.embed_tokens.* — strip "model." prefix
        return vlm_embedding_weights(state_dict, keyword="embed_tokens", prefixes=("model.",))


class VideoLLaMA3Model(nn.Module):
    """VideoLLaMA3 vision-language model (3-model split).

    Builds three separate ONNX models:

    - **decoder**: Qwen2 text decoder taking ``inputs_embeds``
    - **vision**: SigLIP-like ViT (27 layers, hidden=1152) + 2-layer MLP projector
    - **embedding**: token embedding + image feature fusion

    model_type: ``videollama3_qwen2``

    HuggingFace reference: ``DAMO-NLP-SG/VideoLLaMA3-7B``
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _VideoLLaMA3DecoderModel(config)
        self.vision_encoder = _VideoLLaMA3VisionEncoderModel(config)
        self.embedding = _VideoLLaMA3EmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "VideoLLaMA3Model uses VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Handle tied word embeddings: copy embed_tokens → lm_head if absent
        if self.config.tie_word_embeddings:
            embed_key = "model.embed_tokens.weight"
            head_key = "lm_head.weight"
            if head_key not in state_dict and embed_key in state_dict:
                state_dict[head_key] = state_dict[embed_key]
        return state_dict
