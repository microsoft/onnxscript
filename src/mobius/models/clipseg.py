# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""CLIPSeg: text-guided image segmentation.

Architecture: CLIP vision+text encoder + lightweight FiLM-conditioned decoder
that produces per-pixel segmentation masks.

Reference: https://huggingface.co/docs/transformers/model_doc/clipseg
HuggingFace class: CLIPSegForImageSegmentation

Pipeline:
    1. Vision encoder processes the query image, collecting intermediate
       hidden states at ``extract_layers`` (e.g. [3, 6, 9]).
    2. Text encoder processes the prompt, pools to the EOS token embedding,
       and projects to ``projection_dim`` via ``text_projection``.
    3. Decoder fuses vision features with text conditioning via FiLM
       (Feature-wise Linear Modulation), processes through transformer
       layers, then upsamples through transposed convolutions to produce
       a segmentation logit map.

Inputs:
    pixel_values: [batch, channels, height, width] — query image
    input_ids: [batch, text_seq_len] — text prompt token ids
    attention_mask: [batch, text_seq_len] — text attention mask

Output:
    logits: [batch, height_out, width_out] — per-pixel segmentation logits
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import FCMLP
from mobius.components._common import Embedding, LayerNorm, Linear
from mobius.components._conv import Conv2d, ConvTranspose2d
from mobius.components._encoder import EncoderAttention

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Vision encoder components
# ---------------------------------------------------------------------------


class _CLIPSegPatchEmbedding(nn.Module):
    """Conv2d patch embedding → flatten → transpose."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.projection = Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # [batch, channels, H, W] → [batch, hidden, H', W']
        conv_out = self.projection(op, x)
        batch_size = op.Shape(conv_out, start=0, end=1)
        out_channels = op.Shape(conv_out, start=1, end=2)
        # Flatten spatial: [batch, hidden, num_patches]
        conv_out = op.Reshape(conv_out, op.Concat(batch_size, out_channels, [-1], axis=0))
        # Transpose to [batch, num_patches, hidden]
        return op.Transpose(conv_out, perm=[0, 2, 1])


class _CLIPSegVisionEmbeddings(nn.Module):
    """CLS token + patch embeddings + learned position embeddings."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        v = config.vision
        assert v is not None
        hidden_size = v.hidden_size or config.hidden_size
        patch_size = v.patch_size or config.patch_size
        image_size = v.image_size or config.image_size
        in_channels = v.in_channels

        self.class_embedding = nn.Parameter((hidden_size,))
        self.patch_embedding = _CLIPSegPatchEmbedding(in_channels, hidden_size, patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = Embedding(num_patches + 1, hidden_size)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        patch_embeds = self.patch_embedding(op, pixel_values)
        batch_size = op.Shape(pixel_values, start=0, end=1)

        # Expand CLS token: [1, 1, hidden] → [batch, 1, hidden]
        cls_tokens = op.Unsqueeze(self.class_embedding, [0, 1])
        cls_tokens = op.Expand(
            cls_tokens,
            op.Concat(batch_size, [1], [self.class_embedding.shape[0]], axis=0),
        )
        # [batch, 1 + num_patches, hidden]
        embeddings = op.Concat(cls_tokens, patch_embeds, axis=1)

        # Add position embeddings
        seq_len = op.Shape(embeddings, start=1, end=2)
        position_ids = op.Range(op.Constant(value_int=0), seq_len, op.Constant(value_int=1))
        position_ids = op.Cast(position_ids, to=7)  # INT64
        position_ids = op.Unsqueeze(position_ids, [0])
        embeddings = op.Add(embeddings, self.position_embedding(op, position_ids))
        return embeddings


class _CLIPSegEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer (shared by vision and decoder).

    Structure: LayerNorm → Attention → residual → LayerNorm → MLP → residual
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        activation: str,
        eps: float,
    ):
        super().__init__()
        self.self_attn = EncoderAttention(hidden_size, num_attention_heads)
        self.layer_norm1 = LayerNorm(hidden_size, eps=eps)
        self.mlp = FCMLP(hidden_size, intermediate_size, activation=activation)
        self.layer_norm2 = LayerNorm(hidden_size, eps=eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm1(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states, attention_mask)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layer_norm2(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class _CLIPSegVisionEncoder(nn.Module):
    """Vision encoder that collects intermediate hidden states.

    Returns (final_hidden_state, [extract_layer_outputs]).
    Hidden states at ``extract_layers`` indices are collected for the decoder.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        v = config.vision
        assert v is not None
        hidden_size = v.hidden_size or config.hidden_size
        num_heads = v.num_attention_heads or config.num_attention_heads
        num_layers = v.num_hidden_layers or config.num_hidden_layers
        intermediate = v.intermediate_size or config.intermediate_size
        activation = config.hidden_act or "quick_gelu"
        eps = v.norm_eps

        self.embeddings = _CLIPSegVisionEmbeddings(config)
        self.pre_layrnorm = LayerNorm(hidden_size, eps=eps)
        self.encoder = nn.ModuleList(
            [
                _CLIPSegEncoderLayer(hidden_size, num_heads, intermediate, activation, eps)
                for _ in range(num_layers)
            ]
        )
        self.post_layernorm = LayerNorm(hidden_size, eps=eps)

        # 1-indexed layer indices to collect (e.g. [3, 6, 9])
        self._extract_layers = config.extract_layers or [3, 6, 9]

    def forward(
        self, op: builder.OpBuilder, pixel_values: ir.Value
    ) -> tuple[ir.Value, list[ir.Value]]:
        hidden_states = self.embeddings(op, pixel_values)
        hidden_states = self.pre_layrnorm(op, hidden_states)

        intermediate_states: list[ir.Value] = []
        for i, layer in enumerate(self.encoder):
            hidden_states = layer(op, hidden_states)
            # extract_layers uses 1-indexed positions: layer output after
            # the i-th layer corresponds to index i+1
            if (i + 1) in self._extract_layers:
                intermediate_states.append(hidden_states)

        hidden_states = self.post_layernorm(op, hidden_states)
        return hidden_states, intermediate_states


# ---------------------------------------------------------------------------
# Text encoder components
# ---------------------------------------------------------------------------


class _CLIPSegTextEmbeddings(nn.Module):
    """Token + learned absolute position embeddings."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value):
        token_embeds = self.word_embeddings(op, input_ids)
        seq_len = op.Shape(input_ids, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(seq_len),
            op.Constant(value_int=1),
        )
        position_ids = op.Cast(position_ids, to=7)  # INT64
        position_ids = op.Unsqueeze(position_ids, [0])
        return op.Add(token_embeds, self.position_embedding(op, position_ids))


class _CLIPSegTextEncoder(nn.Module):
    """Text encoder with causal attention and EOS-token pooling.

    Returns pooled text embeddings projected to ``projection_dim``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        hidden_size = config.hidden_size
        activation = config.hidden_act or "quick_gelu"
        eps = config.rms_norm_eps

        self.embeddings = _CLIPSegTextEmbeddings(config)
        self.encoder = nn.ModuleList(
            [
                _CLIPSegEncoderLayer(
                    hidden_size,
                    config.num_attention_heads,
                    config.intermediate_size,
                    activation,
                    eps,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = LayerNorm(hidden_size, eps=eps)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
    ) -> ir.Value:
        """Run text through encoder and return last_hidden_state.

        Returns (batch, seq_len, hidden_size).
        """
        hidden_states = self.embeddings(op, input_ids)

        # Build causal attention mask (lower-triangular)
        seq_len = op.Shape(input_ids, start=1, end=2)
        # Build causal mask: upper triangle = -10000, lower+diag = 0
        neg_inf = op.Expand(
            op.Constant(value_float=-10000.0),
            op.Concat(seq_len, seq_len, axis=0),
        )
        upper_tri = op.Sub(op.Trilu(neg_inf, upper=1), op.Trilu(neg_inf, upper=0))
        causal_bias = op.Unsqueeze(upper_tri, [0, 1])  # [1, 1, seq, seq]

        for layer in self.encoder:
            hidden_states = layer(op, hidden_states, causal_bias)

        return self.final_layer_norm(op, hidden_states)


# ---------------------------------------------------------------------------
# Segmentation decoder
# ---------------------------------------------------------------------------


class _CLIPSegDecoder(nn.Module):
    """Lightweight decoder: FiLM conditioning + transformer layers + upsampling.

    Steps:
        1. Reverse the extracted vision hidden states.
        2. For each layer: reduce activation (768→reduce_dim), add to running
           output, optionally apply FiLM conditioning, run through a
           transformer decoder layer.
        3. Remove CLS token, reshape to spatial 2D grid.
        4. Upsample through transposed convolutions → segmentation logits.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        v = config.vision
        assert v is not None
        vision_hidden = v.hidden_size or config.hidden_size
        projection_dim = config.projection_dim or 512
        reduce_dim = config.reduce_dim
        n_extract = len(config.extract_layers or [3, 6, 9])
        decoder_heads = config.decoder_num_attention_heads
        decoder_intermediate = config.decoder_intermediate_size or 2048
        decoder_act = config.decoder_hidden_act
        eps = config.rms_norm_eps

        # FiLM conditioning layers: project text embedding → per-channel scale/bias
        self.film_mul = Linear(projection_dim, reduce_dim, bias=True)
        self.film_add = Linear(projection_dim, reduce_dim, bias=True)

        # Reduce vision hidden dim → decoder dim for each extract layer
        self.reduces = nn.ModuleList(
            [Linear(vision_hidden, reduce_dim, bias=True) for _ in range(n_extract)]
        )

        # Transformer decoder layers (pre-norm attention + MLP)
        self.layers = nn.ModuleList(
            [
                _CLIPSegEncoderLayer(
                    reduce_dim, decoder_heads, decoder_intermediate, decoder_act, eps
                )
                for _ in range(n_extract)
            ]
        )

        # Transposed convolution chain for spatial upsampling
        # Conv2d(reduce_dim, reduce_dim, 3, padding=1) → ReLU
        # → ConvTranspose2d(reduce_dim, reduce_dim//2, 4, stride=4) → ReLU
        # → ConvTranspose2d(reduce_dim//2, 1, 4, stride=4)
        self.transposed_convolution = nn.ModuleList(
            [
                Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1),
                ConvTranspose2d(reduce_dim, reduce_dim // 2, kernel_size=4, stride=4),
                ConvTranspose2d(reduce_dim // 2, 1, kernel_size=4, stride=4),
            ]
        )

        self._conditional_layer = config.conditional_layer

    def forward(
        self,
        op: builder.OpBuilder,
        vision_hidden_states: list[ir.Value],
        conditional_embeddings: ir.Value,
    ) -> ir.Value:
        """Produce segmentation logits from vision features + text conditioning.

        Args:
            vision_hidden_states: List of (batch, seq, vision_hidden) tensors
                from extract_layers, in forward order (will be reversed).
            conditional_embeddings: (batch, projection_dim) text embedding.

        Returns:
            logits: (batch, H_out, W_out) segmentation map.
        """
        # Process extract layers in reverse order (deepest first)
        activations = list(reversed(vision_hidden_states))

        output = None
        for i, (activation, layer, reduce) in enumerate(
            zip(activations, self.layers, self.reduces)
        ):
            reduced = reduce(op, activation)  # (batch, seq, reduce_dim)
            if output is not None:
                output = op.Add(reduced, output)
            else:
                output = reduced

            # Apply FiLM at the conditional layer
            if i == self._conditional_layer:
                # film_scale/bias: (batch, reduce_dim) → (batch, 1, reduce_dim)
                film_scale = op.Unsqueeze(self.film_mul(op, conditional_embeddings), [1])
                film_bias = op.Unsqueeze(self.film_add(op, conditional_embeddings), [1])
                # Element-wise affine: scale * output + bias
                output = op.Add(op.Mul(film_scale, output), film_bias)

            output = layer(op, output)

        assert output is not None

        # Remove CLS token (first position)
        # output: (batch, 1+num_patches, reduce_dim) → (batch, num_patches, reduce_dim)
        output = op.Slice(
            output,
            starts=[1],
            ends=[9999999],
            axes=[1],
        )

        # Reshape to spatial: (batch, num_patches, reduce_dim) →
        # (batch, reduce_dim, grid_h, grid_w)
        batch_size = op.Shape(output, start=0, end=1)
        reduce_dim = op.Shape(output, start=2, end=3)
        num_patches = op.Shape(output, start=1, end=2)
        grid_size = op.Sqrt(op.Cast(num_patches, to=1))  # to=1: FLOAT
        grid_size = op.Cast(grid_size, to=7)  # to=7: INT64

        # (batch, seq, reduce_dim) → (batch, reduce_dim, seq)
        output = op.Transpose(output, perm=[0, 2, 1])
        # (batch, reduce_dim, grid_h, grid_w)
        output = op.Reshape(
            output,
            op.Concat(batch_size, reduce_dim, grid_size, grid_size, axis=0),
        )

        # Transposed convolution upsampling: Conv2d → ReLU → Deconv → ReLU → Deconv
        output = self.transposed_convolution[0](op, output)
        output = op.Relu(output)
        output = self.transposed_convolution[1](op, output)
        output = op.Relu(output)
        output = self.transposed_convolution[2](op, output)

        # Squeeze channel dim: (batch, 1, H, W) → (batch, H, W)
        output = op.Squeeze(output, [1])
        return output


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class CLIPSegModel(nn.Module):
    """CLIPSeg for text-guided image segmentation.

    Composes a CLIP vision+text backbone with a FiLM-conditioned segmentation
    decoder. Given an image and a text prompt, produces per-pixel logits.

    HuggingFace: ``CLIPSegForImageSegmentation``
    """

    default_task = "image-segmentation"
    category = "multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        v = config.vision
        assert v is not None
        vision_hidden = v.hidden_size or config.hidden_size
        text_hidden = config.hidden_size
        projection_dim = config.projection_dim or 512

        # CLIP backbone
        self.clip = _CLIPSegClip(config)

        # Segmentation decoder
        self.decoder = _CLIPSegDecoder(config)

        # Projection heads (no bias, as in HF CLIP)
        self._visual_projection = Linear(vision_hidden, projection_dim, bias=False)
        self._text_projection = Linear(text_hidden, projection_dim, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
        input_ids: ir.Value,
        attention_mask: ir.Value,
    ) -> ir.Value:
        """Forward pass: image + text → segmentation logits.

        Args:
            pixel_values: (batch, channels, H, W)
            input_ids: (batch, text_seq_len)
            attention_mask: (batch, text_seq_len)

        Returns:
            logits: (batch, H_out, W_out) segmentation map.
        """
        # Step 1: Vision encoder → hidden states + intermediate features
        _vision_output, intermediate_states = self.clip.vision_model(op, pixel_values)

        # Step 2: Text encoder → pooled text embedding
        text_output = self.clip.text_model(op, input_ids, attention_mask)
        # Pool: gather the EOS token position (argmax of input_ids)
        # CLIP convention: eos_token_id=2 is the highest-valued token
        input_ids_i32 = op.Cast(input_ids, to=6)  # to=6: INT32 for ArgMax
        eos_positions = op.ArgMax(input_ids_i32, axis=1, keepdims=True)  # (batch, 1)
        eos_positions = op.Cast(eos_positions, to=7)  # to=7: INT64 for GatherND
        # GatherND with batch_dims=1: for each batch, select the token at eos_positions
        pooled_text = op.GatherND(
            text_output, eos_positions, batch_dims=1
        )  # (batch, hidden_size)

        # Project text to conditioning space
        conditional_embeddings = self._text_projection(
            op, pooled_text
        )  # (batch, projection_dim)

        # Step 3: Decoder fuses vision features with text conditioning
        logits = self.decoder(op, intermediate_states, conditional_embeddings)
        return logits

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_dict: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            new_name = _rename_clipseg_weight(name)
            if new_name is not None:
                new_dict[new_name] = tensor
        return new_dict


class _CLIPSegClip(nn.Module):
    """Container for the CLIP backbone (vision + text encoders)."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.vision_model = _CLIPSegVisionEncoder(config)
        self.text_model = _CLIPSegTextEncoder(config)


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------

# Shared layer-level renames (attention projections pass through)
_LAYER_RENAMES = {
    "self_attn.q_proj.": "self_attn.q_proj.",
    "self_attn.k_proj.": "self_attn.k_proj.",
    "self_attn.v_proj.": "self_attn.v_proj.",
    "self_attn.out_proj.": "self_attn.out_proj.",
}


def _rename_encoder_layer(remainder: str, layer_idx: str) -> str | None:
    """Rename an encoder layer weight: encoder.layers.{i}.X → encoder.{i}.X."""
    for old, new in _LAYER_RENAMES.items():
        if remainder.startswith(old):
            suffix = remainder[len(old) :]
            return f"encoder.{layer_idx}.{new}{suffix}"

    # MLP: fc1 → up_proj, fc2 → down_proj (FCMLP naming)
    remainder = remainder.replace("mlp.fc1.", "mlp.up_proj.")
    remainder = remainder.replace("mlp.fc2.", "mlp.down_proj.")
    return f"encoder.{layer_idx}.{remainder}"


def _rename_clipseg_weight(name: str) -> str | None:
    """Map a HuggingFace CLIPSeg weight name to our naming convention."""
    # Skip position_ids buffers
    if "position_ids" in name:
        return None

    # --- CLIP vision weights: clip.vision_model.* ---
    if name.startswith("clip.vision_model."):
        suffix = name[len("clip.vision_model.") :]

        if suffix.startswith("embeddings."):
            return f"clip.vision_model.{suffix}"

        if suffix.startswith(("pre_layrnorm.", "post_layernorm.")):
            return f"clip.vision_model.{suffix}"

        if suffix.startswith("encoder.layers."):
            parts = suffix.split(".", 3)  # encoder, layers, idx, rest
            if len(parts) < 4:
                return None
            renamed = _rename_encoder_layer(parts[3], parts[2])
            return f"clip.vision_model.{renamed}" if renamed else None

        return None

    # --- CLIP text weights: clip.text_model.* ---
    if name.startswith("clip.text_model."):
        suffix = name[len("clip.text_model.") :]

        if suffix.startswith("embeddings."):
            # HF: token_embedding → word_embeddings
            suffix = suffix.replace(
                "embeddings.token_embedding.", "embeddings.word_embeddings."
            )
            return f"clip.text_model.{suffix}"

        if suffix.startswith("final_layer_norm."):
            return f"clip.text_model.{suffix}"

        if suffix.startswith("encoder.layers."):
            parts = suffix.split(".", 3)
            if len(parts) < 4:
                return None
            renamed = _rename_encoder_layer(parts[3], parts[2])
            return f"clip.text_model.{renamed}" if renamed else None

        return None

    # --- Projection weights: clip.visual_projection.*, clip.text_projection.* ---
    if name.startswith("clip.visual_projection."):
        # Map to our _visual_projection (named with underscore to avoid
        # nesting under the clip sub-module)
        return name.replace("clip.visual_projection.", "_visual_projection.")

    if name.startswith("clip.text_projection."):
        return name.replace("clip.text_projection.", "_text_projection.")

    # --- Decoder weights: decoder.* ---
    if name.startswith("decoder."):
        suffix = name[len("decoder.") :]

        # film_mul, film_add — pass through
        if suffix.startswith(("film_mul.", "film_add.")):
            return name

        # reduces.{i}.* — pass through (ModuleList indices match)
        if suffix.startswith("reduces."):
            return name

        # layers.{i}.* — rename MLP fc1/fc2 → up_proj/down_proj
        if suffix.startswith("layers."):
            parts = suffix.split(".", 2)  # layers, idx, rest
            if len(parts) < 3:
                return None
            rest = parts[2]
            rest = rest.replace("mlp.fc1.", "mlp.up_proj.")
            rest = rest.replace("mlp.fc2.", "mlp.down_proj.")
            return f"decoder.layers.{parts[1]}.{rest}"

        # transposed_convolution.{idx}.* — remap HF sequential indices
        # HF uses [0]=Conv2d, [1]=ReLU, [2]=ConvTranspose2d, [3]=ReLU, [4]=ConvTranspose2d
        # We use [0]=Conv2d, [1]=ConvTranspose2d, [2]=ConvTranspose2d
        if suffix.startswith("transposed_convolution."):
            tc_rest = suffix[len("transposed_convolution.") :]
            # Map HF index → our index
            for hf_idx, our_idx in [("0.", "0."), ("2.", "1."), ("4.", "2.")]:
                if tc_rest.startswith(hf_idx):
                    return f"decoder.transposed_convolution.{our_idx}{tc_rest[len(hf_idx) :]}"
            return None  # Skip ReLU indices (no weights)

        return None

    # logit_scale — not needed for segmentation
    if name.startswith("logit_scale"):
        return None

    return None
