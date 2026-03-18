# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    InputMixer,
    Qwen3VLVisionModel,
    Qwen25VLVisionModel,
)
from mobius.components._common import (
    Embedding,
    Linear,
    create_attention_bias,
)
from mobius.components._decoder import DecoderLayer
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import initialize_rope
from mobius.models.base import CausalLMModel, TextModel

if TYPE_CHECKING:
    import onnx_ir as ir

# Text-only decoders — extract the language model from multimodal weights.
# These strip ``language_model.`` prefixes and drop ``visual.`` keys.


class _QwenVLTextMixin:
    """Shared weight preprocessing for Qwen VL text decoders."""

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        for key in list(state_dict.keys()):
            if "language_model." in key:
                new_key = key.replace("language_model.", "")
                state_dict[new_key] = state_dict.pop(key)
            elif "visual." in key:
                state_dict.pop(key)
        return super().preprocess_weights(state_dict)


class Qwen25VLTextModel(_QwenVLTextMixin, CausalLMModel):
    """Qwen2.5-VL text-only decoder.

    Extracts the text backbone from the Qwen2.5-VL multimodal model.
    Strips ``language_model.`` weight prefixes and drops ``visual.`` keys.
    For text-only inference the standard 1D RoPE is equivalent to MRoPE
    (all three dimensions are identical for text tokens).
    """


class Qwen3VLTextModel(_QwenVLTextMixin, CausalLMModel):
    """Qwen3-VL text-only decoder.

    Extends Qwen2.5-VL text decoder with Q/K normalization (RMSNorm on
    query and key projections), configured via ``attn_qk_norm=True``.
    """


# Full VL models — multimodal (text + vision).


class Qwen25VLCausalLMModel(nn.Module):
    """Qwen2.5-VL vision-language model (3-model split).

    Builds three separate ONNX models for onnxruntime-genai:

    - ``decoder``: text decoder taking ``inputs_embeds`` (MRoPE position_ids)
    - ``vision_encoder``: vision ViT with windowed/full attention
    - ``embedding``: token embedding + image feature fusion

    The :class:`~mobius.tasks.Qwen25VL3ModelTask` calls each
    sub-module separately to produce 3 ONNX graphs.
    """

    default_task: str = "qwen-vl"
    category: str = "Multimodal"
    config_class: type = ArchitectureConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = Qwen25VLDecoderModel(config)
        self.vision_encoder = Qwen25VLVisionEncoderModel(config)
        self.embedding = Qwen25VLEmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "Qwen25VLCausalLMModel uses Qwen25VL3ModelTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF weights to the correct sub-model ONNX initializer names.

        ONNX initializer names include the composite module's attribute
        prefixes (``decoder.``, ``vision_encoder.``, ``embedding.``) because
        onnxscript qualifies parameter names via the parent-child hierarchy.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("visual."):
                new_key = f"vision_encoder.{key}"
                # Merger uses mlp_0/mlp_2 attributes; HF uses mlp.0/mlp.2
                new_key = new_key.replace(".merger.mlp.0.", ".merger.mlp_0.")
                new_key = new_key.replace(".merger.mlp.2.", ".merger.mlp_2.")
                renamed[new_key] = value
            elif key.startswith("model.embed_tokens."):
                # Shared embedding → both decoder (TextModel) and embedding model
                renamed[f"decoder.{key}"] = value
                stripped = key[len("model.") :]
                renamed[f"embedding.{stripped}"] = value
                # Weight tying: also use as lm_head
                if self.config.tie_word_embeddings and key == "model.embed_tokens.weight":
                    renamed["decoder.lm_head.weight"] = value
            elif key.startswith("model."):
                renamed[f"decoder.{key}"] = value
            elif key.startswith("lm_head."):
                renamed[f"decoder.{key}"] = value
        return renamed


class Qwen25VLDecoderModel(nn.Module):
    """Qwen2.5-VL text decoder that takes ``inputs_embeds`` instead of ``input_ids``.

    This is the decoder component of the 3-model split for onnxruntime-genai.
    It receives pre-computed embeddings (from the embedding model) and uses
    MRoPE with 3D ``position_ids`` of shape ``(3, batch, seq_len)``.

    Weight prefix: ``language_model.``
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
        """Route language_model weights for standalone decoder build.

        When built standalone (not via composite), the decoder ONNX
        initializers are ``model.layers.*``, ``model.norm.*``,
        ``model.embed_tokens.*``, ``lm_head.*``.  HF keys already match.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Drop vision weights
            if key.startswith("visual."):
                continue
            renamed[key] = value

        # Handle weight tying
        if self.config.tie_word_embeddings:
            if "lm_head.weight" not in renamed and "model.embed_tokens.weight" in renamed:
                renamed["lm_head.weight"] = renamed["model.embed_tokens.weight"]
        return renamed


class Qwen25VLVisionEncoderModel(nn.Module):
    """Qwen2.5-VL vision encoder for the 3-model split.

    Processes image patches through ViT with windowed/full attention,
    spatial merge, and outputs image features.

    Inputs:
        - pixel_values: (total_patches, C*T_p*P*P) — flattened patches
        - image_grid_thw: (num_images, 3) INT64 — [T, H, W] per image
    Output:
        - image_features: (num_merged_patches, out_hidden_size)

    Weight prefix: ``visual.``
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None and vc.hidden_size is not None
        assert vc.num_hidden_layers is not None
        assert vc.num_attention_heads is not None
        self.visual = Qwen25VLVisionModel(
            depth=vc.num_hidden_layers,
            hidden_size=vc.hidden_size,
            intermediate_size=vc.intermediate_size or vc.hidden_size * 4,
            num_heads=vc.num_attention_heads,
            patch_size=vc.patch_size or 14,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=vc.in_channels,
            out_hidden_size=vc.out_hidden_size or config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            fullatt_block_indexes=config.fullatt_block_indexes,
            window_size=config.window_size,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ):
        image_features = self.visual(
            op,
            pixel_values,
            image_grid_thw,
        )
        return image_features

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Keep only visual.* weights for standalone vision encoder build.

        Also maps merger ``mlp.0``/``mlp.2`` → ``mlp_0``/``mlp_2``.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("visual."):
                new_key = key.replace(".merger.mlp.0.", ".merger.mlp_0.")
                new_key = new_key.replace(".merger.mlp.2.", ".merger.mlp_2.")
                renamed[new_key] = value
        return renamed


class Qwen25VLEmbeddingModel(nn.Module):
    """Qwen2.5-VL embedding model for the 3-model split.

    Fuses token embeddings with image features at image token positions.

    Inputs:
        - input_ids: (batch, seq_len) INT64
        - image_features: (num_image_tokens, hidden_size) FLOAT
    Output:
        - inputs_embeds: (batch, seq_len, hidden_size) FLOAT
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.image_token_id = config.image_token_id or 151655

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
    ):
        # Token embedding lookup
        text_embeds = self.embed_tokens(op, input_ids)

        # Create mask for image token positions
        image_mask = op.Equal(
            input_ids,
            op.Constant(value_int=self.image_token_id),
        )
        # Expand mask to 3D for broadcasting: (batch, seq, 1)
        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        # Cumulative sum to map flat image_features indices
        # image_mask is (batch, seq), cast to int
        mask_int = op.Cast(image_mask, to=7)  # INT64
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        # Zero-based index: subtract 1, clip to 0
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        # Pad image_features with one zero row so Gather is valid even when
        # image_features is empty (text-only input: num_image_tokens == 0).
        # The Where mask ensures the padding row is never used in the output.
        pad_row = op.Expand(
            op.CastLike(op.Constant(value_float=0.0), image_features),
            op.Concat(
                op.Constant(value_ints=[1]),
                op.Shape(image_features, start=1, end=2),
                axis=0,
            ),
        )
        padded_features = op.Concat(image_features, pad_row, axis=0)

        # Gather from padded_features using indices
        # padded_features: (num_image_tokens + 1, hidden)
        # indices: (batch, seq) → gather → (batch, seq, hidden)
        gathered = op.Gather(padded_features, indices, axis=0)

        # Where image_mask → use gathered features, else text_embeds
        inputs_embeds = op.Where(image_mask_3d, gathered, text_embeds)

        return inputs_embeds

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Keep only embed_tokens weights for standalone embedding build.

        HF key ``model.embed_tokens.weight`` → ``embed_tokens.weight``.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if "embed_tokens" in key:
                new_key = key
                if new_key.startswith("model."):
                    new_key = new_key[len("model.") :]
                renamed[new_key] = value
        return renamed


class _Qwen3VLTextModel(nn.Module):
    """Qwen3-VL text model with DeepStack injection and MRoPE.

    After specific decoder layers, adds visual features from DeepStack at
    positions corresponding to image tokens.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.image_token_id = config.image_token_id or 0
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self.input_mixer = InputMixer(
            image_token_id=config.image_token_id or 0,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
        vision_embeddings: ir.Value | None = None,
        deepstack_visual_embeds: list | None = None,
    ):
        if vision_embeddings is not None and inputs_embeds is None:
            text_embeddings = self.embed_tokens(op, input_ids)
            hidden_states = self.input_mixer(
                op,
                text_embeddings,
                vision_embeddings,
                input_ids,
            )
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)

        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        # Visual position mask for DeepStack injection
        if deepstack_visual_embeds is not None:
            visual_mask = op.Equal(
                input_ids,
                op.Constant(value_int=self.image_token_id),
            )
            # Expand to [batch, seq, 1] for broadcasting
            visual_mask_3d = op.Unsqueeze(visual_mask, [-1])

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer_idx, (layer, past_kv) in enumerate(zip(self.layers, past_kvs)):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

            # DeepStack: add intermediate vision features at image positions
            if deepstack_visual_embeds is not None and layer_idx < len(
                deepstack_visual_embeds
            ):
                ds_embeds = deepstack_visual_embeds[layer_idx]
                # Scatter vision features at visual token positions
                # ds_embeds: (num_visual_tokens, hidden_size)
                # Use cumsum of mask to index into ds_embeds
                mask_int = op.Cast(visual_mask, to=7)  # INT64
                cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
                indices = op.Sub(cumsum, op.Constant(value_int=1))
                indices = op.Clip(indices, op.Constant(value_int=0))
                # Expand ds_embeds with batch dim: (1, num_visual_tokens, hidden_size)
                ds_embeds_3d = op.Unsqueeze(ds_embeds, [0])
                # Gather at computed indices
                indices_3d = op.Unsqueeze(indices, [-1])
                hidden_dim = op.Shape(ds_embeds_3d, start=2, end=3)
                ones_shape = op.Concat(
                    op.Constant(value_ints=[1, 1]),
                    hidden_dim,
                    axis=0,
                )
                gather_idx = op.Expand(indices_3d, ones_shape)
                scattered_ds = op.GatherElements(ds_embeds_3d, gather_idx, axis=1)
                # Add at visual positions only
                hidden_states = op.Add(
                    hidden_states,
                    op.Where(
                        visual_mask_3d,
                        scattered_ds,
                        op.CastLike(op.Constant(value_float=0.0), hidden_states),
                    ),
                )

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class _Qwen3VLForMultimodalLM(nn.Module):
    """Qwen3-VL causal LM with DeepStack-aware text decoder.

    Accepts ``vision_embeddings`` for input mixing and
    ``deepstack_visual_embeds`` for intermediate layer injection.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = _Qwen3VLTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        vision_embeddings: ir.Value | None = None,
        deepstack_visual_embeds: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            vision_embeddings=vision_embeddings,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values


class Qwen3VLCausalLMModel(nn.Module):
    """Qwen3-VL vision-language model with packed-attention vision encoder.

    Combines a Qwen3-VL ViT vision encoder with a Qwen3-VL text decoder.
    The vision encoder processes packed image/video patches through Conv3d
    embedding, rotary-embedded transformer blocks with packed attention,
    spatial merge, and DeepStack intermediate feature extraction.

    The text decoder uses interleaved MRoPE for 3D positional encoding
    (temporal, height, width) and injects DeepStack vision features into
    early decoder layers at image token positions.

    Weight names match HuggingFace convention:
    ``visual.*`` for vision encoder, ``language_model.*`` for text decoder.
    """

    default_task: str = "qwen3-vl-vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config

        # Vision encoder — uses ``visual.*`` weight prefix
        vc = config.vision
        assert vc is not None and vc.hidden_size is not None
        assert vc.num_hidden_layers is not None
        assert vc.num_attention_heads is not None
        self.visual = Qwen3VLVisionModel(
            depth=vc.num_hidden_layers,
            hidden_size=vc.hidden_size,
            intermediate_size=vc.intermediate_size or vc.hidden_size * 4,
            num_heads=vc.num_attention_heads,
            patch_size=vc.patch_size or 16,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=vc.in_channels,
            out_hidden_size=vc.out_hidden_size or config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            num_position_embeddings=vc.num_position_embeddings or 2304,
            deepstack_visual_indexes=config.deepstack_visual_indexes or [],
        )

        # Text decoder — uses ``language_model.*`` weight prefix
        self.language_model = _Qwen3VLForMultimodalLM(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        pixel_values: ir.Value,
        grid_thw: ir.Value,
        past_key_values: list | None = None,
    ):
        """Full vision-language forward pass.

        Args:
            input_ids: Text input token IDs ``(batch, seq_len)``.
            attention_mask: ``(batch, past_seq_len + seq_len)``.
            position_ids: MRoPE 3D positions ``(3, batch, seq_len)``.
            pixel_values: Flattened image patches
                ``(total_patches, C * T_p * P * P)``.
            grid_thw: ``(num_images, 3)`` INT64 with ``[T, H, W]`` per
                image, used for position embedding interpolation and
                computing cu_seqlens and rotary position IDs.
            past_key_values: Optional KV cache.

        Returns:
            Tuple of ``(logits, present_key_values)``.
        """
        # Vision encoding
        vision_outputs = self.visual(
            op,
            hidden_states=pixel_values,
            grid_thw=grid_thw,
        )

        # Separate merged features from deepstack features
        if isinstance(vision_outputs, tuple):
            vision_embeddings = vision_outputs[0]
            deepstack_features = list(vision_outputs[1:]) if len(vision_outputs) > 1 else None
        else:
            vision_embeddings = vision_outputs
            deepstack_features = None

        # Add batch dim to vision embeddings for InputMixer:
        # (num_merged_tokens, hidden) → (1, num_merged_tokens, hidden)
        vision_embeddings = op.Unsqueeze(vision_embeddings, [0])

        logits, present_key_values = self.language_model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            vision_embeddings=vision_embeddings,
            deepstack_visual_embeds=deepstack_features,
        )
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace weight names to the ONNX model's parameter names.

        HF keys use ``model.`` prefix and flatten the language model:
        ``model.language_model.layers.*``, ``model.visual.*``.
        Our model uses ``language_model.model.layers.*``, ``visual.*``.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            # Strip outer ``model.`` prefix
            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            # HF flattens language_model.model → language_model; restore it
            if new_key.startswith("language_model.") and not new_key.startswith(
                "language_model.lm_head"
            ):
                new_key = new_key.replace("language_model.", "language_model.model.", 1)
            renamed[new_key] = value

        # Handle weight tying
        config = self.config
        if config.tie_word_embeddings:
            embed_key = "language_model.model.embed_tokens.weight"
            head_key = "language_model.lm_head.weight"
            if head_key not in renamed and embed_key in renamed:
                renamed[head_key] = renamed[embed_key]
            elif embed_key not in renamed and head_key in renamed:
                renamed[embed_key] = renamed[head_key]
        return renamed


# ---------------------------------------------------------------------------
# Qwen3-VL 3-model split for onnxruntime-genai
# ---------------------------------------------------------------------------


class Qwen3VL3ModelCausalLMModel(nn.Module):
    """Qwen3-VL vision-language model (3-model split).

    Builds three separate ONNX models for onnxruntime-genai:

    - ``decoder``: text decoder taking ``inputs_embeds`` (interleaved MRoPE)
    - ``vision_encoder``: packed-attention ViT outputting merged features
    - ``embedding``: token embedding + image feature fusion

    .. note::
       DeepStack intermediate feature injection is not used in the
       3-model split; only the final merged vision features are passed
       to the embedding model.
    """

    default_task: str = "qwen-vl"
    category: str = "Multimodal"
    config_class: type = ArchitectureConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = Qwen3VLDecoderModel(config)
        self.vision_encoder = Qwen3VLVisionEncoderModel(config)
        self.embedding = Qwen3VLEmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "Qwen3VL3ModelCausalLMModel uses QwenVLTask "
            "which calls each sub-module separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF weights to the correct sub-model ONNX initializer names.

        ONNX initializer names include composite attribute prefixes
        (``decoder.``, ``vision_encoder.``, ``embedding.``).

        HF keys: ``model.visual.*``, ``model.language_model.*``.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]

            if stripped.startswith("visual."):
                renamed[f"vision_encoder.{stripped}"] = value
            elif stripped.startswith("language_model.embed_tokens."):
                # Shared embedding → both decoder and embedding model
                suffix = stripped[len("language_model.") :]
                renamed[f"decoder.model.{suffix}"] = value
                renamed[f"embedding.{suffix}"] = value
                # Weight tying
                if (
                    self.config.tie_word_embeddings
                    and stripped == "language_model.embed_tokens.weight"
                ):
                    renamed["decoder.lm_head.weight"] = value
            elif stripped.startswith("language_model.lm_head."):
                renamed[f"decoder.{stripped[len('language_model.') :]}"] = value
            elif stripped.startswith("language_model."):
                # language_model.layers.* → decoder.model.layers.*
                suffix = stripped[len("language_model.") :]
                renamed[f"decoder.model.{suffix}"] = value
        return renamed


class Qwen3VLDecoderModel(nn.Module):
    """Qwen3-VL text decoder taking ``inputs_embeds`` (3-model split).

    Uses interleaved MRoPE with 3D ``position_ids`` of shape
    ``(3, batch, seq_len)``. QK normalization is enabled.
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
        """Route language_model weights for standalone decoder build."""
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]
            if stripped.startswith("visual."):
                continue
            # language_model.layers.* → model.layers.*
            if stripped.startswith("language_model."):
                stripped = stripped[len("language_model.") :]
            renamed[stripped] = value

        if self.config.tie_word_embeddings:
            if "lm_head.weight" not in renamed and "model.embed_tokens.weight" in renamed:
                renamed["lm_head.weight"] = renamed["model.embed_tokens.weight"]
        return renamed


class Qwen3VLVisionEncoderModel(nn.Module):
    """Qwen3-VL vision encoder for the 3-model split.

    Processes packed image patches through the Qwen3-VL ViT and outputs
    merged features (DeepStack intermediate features are not exported).

    Inputs:
        - pixel_values: (total_patches, C*T_p*P*P)
        - image_grid_thw: (num_images, 3) INT64
    Output:
        - image_features: (num_merged_patches, out_hidden_size)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None and vc.hidden_size is not None
        assert vc.num_hidden_layers is not None
        assert vc.num_attention_heads is not None
        self.visual = Qwen3VLVisionModel(
            depth=vc.num_hidden_layers,
            hidden_size=vc.hidden_size,
            intermediate_size=vc.intermediate_size or vc.hidden_size * 4,
            num_heads=vc.num_attention_heads,
            patch_size=vc.patch_size or 16,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=vc.in_channels,
            out_hidden_size=vc.out_hidden_size or config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            num_position_embeddings=vc.num_position_embeddings or 2304,
            deepstack_visual_indexes=config.deepstack_visual_indexes or [],
        )

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ):
        outputs = self.visual(
            op,
            hidden_states=pixel_values,
            grid_thw=image_grid_thw,
        )
        # Only return merged features (first element), drop DeepStack
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Keep only visual.* weights for standalone vision encoder build."""
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]
            if stripped.startswith("visual."):
                renamed[stripped] = value
        return renamed


class Qwen3VLEmbeddingModel(Qwen25VLEmbeddingModel):
    """Qwen3-VL embedding model for the 3-model split.

    Identical to Qwen2.5-VL embedding — scatters image features at
    image token positions using cumsum + Gather + Where.
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Keep only embed_tokens weights.

        HF key: ``model.language_model.embed_tokens.weight`` → ``embed_tokens.weight``.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if "embed_tokens" in key:
                new_key = key
                for prefix in ("model.", "language_model."):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix) :]
                renamed[new_key] = value
        return renamed
