# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Moondream vision-language model — 3-model split.

Moondream is a lightweight VL model with:
- **Vision encoder**: ViT (linear patch embedding, bidirectional attention)
  + 2-layer projection MLP
- **Text decoder**: Parallel pre-norm transformer (single LayerNorm feeds
  both attention and MLP branches, like Cohere)
- **Embedding**: Token lookup + image feature fusion

Split into three ONNX models for onnxruntime-genai:
- ``decoder``: text decoder taking ``inputs_embeds``
- ``vision``: vision encoder + projection MLP
- ``embedding``: token embedding + image feature scatter

HuggingFace weight prefix: ``model.text.*``, ``model.vision.*``.

Replicates HuggingFace ``vikhyatk/moondream2`` (model_type: moondream1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import Attention, Embedding, LayerNorm, Linear, StaticCacheState
from mobius.components._vision import (
    VisionEncoder,
    VisionLayerNorm,
    VisionMLP,
    _VisionLinear,
)

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Text decoder components
# ---------------------------------------------------------------------------


class _SimpleMLP(nn.Module):
    """Two-layer MLP: Linear → GELU(tanh) → Linear.

    Moondream uses a non-gated MLP (unlike SwiGLU in Llama/Qwen).
    Attribute names ``fc1``/``fc2`` match HF weight naming.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.fc1(op, x)
        x = op.Gelu(x, approximate="tanh")
        return self.fc2(op, x)


class _MoondreamDecoderLayer(nn.Module):
    """Parallel pre-norm decoder layer.

    A single ``ln`` feeds both attention and MLP; their outputs
    are summed before the residual addition::

        residual = x
        normed = LayerNorm(x)
        x = residual + attention(normed) + mlp(normed)

    This is the same pattern as Cohere's parallel transformer.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.ln = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.mlp = _SimpleMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | StaticCacheState | None,
    ) -> tuple[ir.Value, tuple]:
        if isinstance(past_key_value, StaticCacheState):
            static_cache = past_key_value
            past_key_value = None
        else:
            static_cache = None

        residual = hidden_states
        normed = self.ln(op, hidden_states)

        # Parallel branches: attention and MLP share the same normed input
        attn_out, present_kv = self.self_attn(
            op,
            normed,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            static_cache=static_cache,
        )
        mlp_out = self.mlp(op, normed)

        # x = residual + attn + mlp
        hidden_states = op.Add(residual, op.Add(attn_out, mlp_out))
        return hidden_states, present_kv


# ---------------------------------------------------------------------------
# Text decoder model (3-model split: "decoder" graph)
# ---------------------------------------------------------------------------


class _MoondreamTextModel(nn.Module):
    """Moondream text transformer (no lm_head).

    Takes inputs_embeds and returns hidden_states + present KV cache.
    Uses parallel pre-norm decoder layers.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [_MoondreamDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.post_ln = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        from mobius.components._rotary_embedding import initialize_rope

        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        cos, sin = self.rotary_emb(op, position_ids)
        position_embeddings = (cos, sin)

        # Causal attention bias from mask
        attention_bias = op.CastLike(
            op.Where(
                op.Equal(attention_mask, op.Constant(value_int=0)),
                op.Constant(value_float=-10000.0),
                op.Constant(value_float=0.0),
            ),
            inputs_embeds,
        )

        hidden_states = inputs_embeds
        present_key_values: list = []

        for i, layer in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, present_kv = layer(
                op, hidden_states, attention_bias, position_embeddings, past_kv
            )
            present_key_values.append(present_kv)

        hidden_states = self.post_ln(op, hidden_states)
        return hidden_states, present_key_values


class _MoondreamDecoderModel(nn.Module):
    """Moondream text decoder: transformer + lm_head.

    This is the "decoder" sub-model in the 3-model split.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = _MoondreamTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op, inputs_embeds, attention_mask, position_ids, past_key_values
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _rename_decoder_weights(state_dict, self.config)


# ---------------------------------------------------------------------------
# Vision encoder model (3-model split: "vision" graph)
# ---------------------------------------------------------------------------


class _LinearPatchEmbedding(nn.Module):
    """Linear patch embedding for images.

    Unlike Conv2d patch embedding, this reshapes patches explicitly
    then applies a linear projection. Matches Moondream's approach.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        num_channels: int = 3,
    ):
        super().__init__()
        self._patch_size = patch_size
        self._num_channels = num_channels
        patch_dim = patch_size * patch_size * num_channels
        num_patches = (image_size // patch_size) ** 2
        self.patch_emb = _VisionLinear(patch_dim, hidden_size)
        self.pos_emb = nn.Parameter([1, num_patches, hidden_size])

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # pixel_values: [B, C, H, W]
        # Reshape to patches: [B, num_patches, C*P*P]
        batch = op.Shape(pixel_values, start=0, end=1)
        p = self._patch_size
        c = self._num_channels
        # [B, C, H, W] → [B, C, H/P, P, W/P, P]
        h_div_p = op.Div(
            op.Shape(pixel_values, start=2, end=3),
            op.Constant(value_ints=[p]),
        )
        w_div_p = op.Div(
            op.Shape(pixel_values, start=3, end=4),
            op.Constant(value_ints=[p]),
        )
        shape_6d = op.Concat(
            batch,
            op.Constant(value_ints=[c]),
            h_div_p,
            op.Constant(value_ints=[p]),
            w_div_p,
            op.Constant(value_ints=[p]),
            axis=0,
        )
        x = op.Reshape(pixel_values, shape_6d)
        # [B, C, H/P, P, W/P, P] → [B, H/P, W/P, C, P, P]
        x = op.Transpose(x, perm=[0, 2, 4, 1, 3, 5])
        # [B, H/P, W/P, C, P, P] → [B, num_patches, C*P*P]
        num_patches = op.Mul(h_div_p, w_div_p)
        shape_3d = op.Concat(batch, num_patches, op.Constant(value_ints=[c * p * p]), axis=0)
        x = op.Reshape(x, shape_3d)

        # Linear projection + position embedding
        x = self.patch_emb(op, x)
        x = op.Add(x, self.pos_emb)
        return x


class _MoondreamVisionEncoderModel(nn.Module):
    """Moondream vision encoder + projection MLP.

    Architecture:
        pixel_values → linear patch embed + pos_emb
        → N x (LN→Attn + LN→MLP) vision blocks
        → post-LN
        → duplicate-concat (placeholder for multi-crop reconstruction)
        → 2-layer projection MLP → image_features
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None, "VisionConfig required"

        self.patch_embed = _LinearPatchEmbedding(
            image_size=vc.image_size,
            patch_size=vc.patch_size,
            hidden_size=vc.hidden_size,
            num_channels=getattr(vc, "num_channels", 3),
        )
        self.encoder = VisionEncoder(
            num_layers=vc.num_hidden_layers,
            hidden_size=vc.hidden_size,
            intermediate_size=vc.intermediate_size,
            num_heads=vc.num_attention_heads,
            norm_eps=vc.norm_eps,
        )
        self.post_ln = VisionLayerNorm(vc.hidden_size, eps=vc.norm_eps)

        # Projection MLP: 2*enc_dim → proj_inner → text_dim
        # (2*enc_dim because moondream concatenates global + spatial features)
        proj_inner = getattr(vc, "proj_inner_dim", None) or (vc.intermediate_size * 2)
        self.proj_mlp = VisionMLP(vc.hidden_size * 2, proj_inner)
        self._proj_out_dim = config.hidden_size

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # Vision encoder: [B, C, H, W] → [B, num_patches, enc_dim]
        x = self.patch_embed(op, pixel_values)
        x = self.encoder(op, x)
        x = self.post_ln(op, x)

        # Moondream concatenates global features + spatial reconstruction.
        # Phase 1: duplicate global features as placeholder for spatial.
        x = op.Concat(x, x, axis=-1)  # [B, patches, 2*enc_dim]

        # Project to text dimension: [B, patches, text_dim]
        x = self.proj_mlp(op, x)
        return x

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _rename_vision_weights(state_dict)


# ---------------------------------------------------------------------------
# Embedding model (3-model split: "embedding" graph)
# ---------------------------------------------------------------------------


class _MoondreamEmbeddingModel(nn.Module):
    """Token embedding + image feature scatter.

    Replaces image placeholder tokens in the input sequence with
    projected vision features. Same pattern as LLaVA.
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

        # Mask where input_ids == image_token_id
        image_mask = op.Equal(input_ids, op.Constant(value_int=self.image_token_id))
        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        # CumSum to map each image token to the right feature index
        mask_int = op.Cast(image_mask, to=7)
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        gathered = op.Gather(image_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _rename_embedding_weights(state_dict)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class MoondreamModel(nn.Module):
    """Moondream vision-language model (3-model split).

    Produces three ONNX models:
    - ``decoder``: parallel-residual text decoder
    - ``vision_encoder``: ViT + projection MLP
    - ``embedding``: token embedding + image feature fusion

    Replicates ``vikhyatk/moondream2`` (model_type: moondream1).
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _MoondreamDecoderModel(config)
        self.vision_encoder = _MoondreamVisionEncoderModel(config)
        self.embedding = _MoondreamEmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "MoondreamModel uses VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return state_dict


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------


def _split_fused_qkv(
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor | None,
    n_heads: int,
    n_kv_heads: int,
) -> dict[str, torch.Tensor]:
    """Split fused QKV weight/bias into separate Q, K, V tensors."""
    head_dim = qkv_weight.shape[0] // (n_heads + 2 * n_kv_heads)
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    q_w, k_w, v_w = qkv_weight.split([q_dim, kv_dim, kv_dim], dim=0)
    result = {
        "q_proj.weight": q_w,
        "k_proj.weight": k_w,
        "v_proj.weight": v_w,
    }
    if qkv_bias is not None:
        q_b, k_b, v_b = qkv_bias.split([q_dim, kv_dim, kv_dim], dim=0)
        result["q_proj.bias"] = q_b
        result["k_proj.bias"] = k_b
        result["v_proj.bias"] = v_b
    return result


def _rename_decoder_weights(
    state_dict: dict[str, torch.Tensor],
    config: ArchitectureConfig,
) -> dict[str, torch.Tensor]:
    """Rename HF moondream text weights to match ONNX parameter names.

    HF: ``model.text.blocks.N.attn.qkv.weight``
    ONNX: ``model.blocks.N.self_attn.q_proj.weight``
    """
    result: dict[str, torch.Tensor] = {}
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads

    for name, tensor in state_dict.items():
        if not name.startswith("model.text."):
            continue
        # Strip "model.text." prefix
        key = name[len("model.text.") :]

        # Skip embedding (goes to embedding model)
        if key == "wte":
            continue

        # Fused QKV → split into separate Q, K, V
        if ".attn.qkv." in key:
            # Find the layer prefix: "blocks.N."
            prefix = key[: key.index(".attn.qkv.")]
            suffix = key[key.index(".attn.qkv.") + len(".attn.qkv.") :]
            if suffix == "weight":
                bias_key = f"model.text.{prefix}.attn.qkv.bias"
                bias = state_dict.get(bias_key)
                split = _split_fused_qkv(tensor, bias, n_heads, n_kv_heads)
                for k, v in split.items():
                    result[f"model.{prefix}.self_attn.{k}"] = v
            # Skip bias (handled with weight above)
            continue

        # attn.proj → self_attn.o_proj
        key = key.replace(".attn.proj.", ".self_attn.o_proj.")
        # ln → ln (matches attribute name)
        # mlp.fc1/fc2 → mlp.fc1/fc2 (matches)
        # post_ln → post_ln (matches)
        # lm_head → lm_head (matches)
        result[f"model.{key}"] = tensor

    return result


def _rename_vision_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rename HF moondream vision weights to match ONNX parameter names.

    HF: ``model.vision.blocks.N.attn.qkv.weight``
    ONNX: ``encoder.layers.N.self_attn.q_proj.weight``

    Vision uses equal-size Q/K/V heads (no GQA).
    """
    result: dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        if not name.startswith("model.vision."):
            continue
        key = name[len("model.vision.") :]

        # patch_emb → patch_embed.patch_emb
        if key.startswith("patch_emb."):
            result[f"patch_embed.{key}"] = tensor
            continue

        # pos_emb → patch_embed.pos_emb
        if key == "pos_emb":
            result["patch_embed.pos_emb"] = tensor
            continue

        # post_ln → post_ln (matches)
        if key.startswith("post_ln."):
            result[key] = tensor
            continue

        # proj_mlp → proj_mlp (matches)
        if key.startswith("proj_mlp."):
            result[key] = tensor
            continue

        # Encoder blocks: blocks.N.* → encoder.layers.N.*
        if key.startswith("blocks."):
            # blocks.N.attn.qkv.weight → encoder.layers.N.self_attn.q/k/v_proj
            if ".attn.qkv." in key:
                prefix = key[: key.index(".attn.qkv.")]
                layer_idx = prefix.split(".")[1]
                suffix = key[key.index(".attn.qkv.") + len(".attn.qkv.") :]
                if suffix == "weight":
                    # Vision: equal Q/K/V (enc_dim each)
                    dim = tensor.shape[0] // 3
                    q_w, k_w, v_w = tensor.split([dim, dim, dim], dim=0)
                    result[f"encoder.layers.{layer_idx}.self_attn.q_proj.weight"] = q_w
                    result[f"encoder.layers.{layer_idx}.self_attn.k_proj.weight"] = k_w
                    result[f"encoder.layers.{layer_idx}.self_attn.v_proj.weight"] = v_w
                elif suffix == "bias":
                    dim = tensor.shape[0] // 3
                    q_b, k_b, v_b = tensor.split([dim, dim, dim], dim=0)
                    result[f"encoder.layers.{layer_idx}.self_attn.q_proj.bias"] = q_b
                    result[f"encoder.layers.{layer_idx}.self_attn.k_proj.bias"] = k_b
                    result[f"encoder.layers.{layer_idx}.self_attn.v_proj.bias"] = v_b
                continue

            # blocks.N.attn.proj → encoder.layers.N.self_attn.out_proj
            if ".attn.proj." in key:
                layer_idx = key.split(".")[1]
                suffix = key.split(".attn.proj.")[1]
                result[f"encoder.layers.{layer_idx}.self_attn.out_proj.{suffix}"] = tensor
                continue

            # blocks.N.ln1 → encoder.layers.N.layer_norm1
            # blocks.N.ln2 → encoder.layers.N.layer_norm2
            layer_idx = key.split(".")[1]
            rest = ".".join(key.split(".")[2:])
            rest = rest.replace("ln1.", "layer_norm1.")
            rest = rest.replace("ln2.", "layer_norm2.")
            result[f"encoder.layers.{layer_idx}.{rest}"] = tensor
            continue

    return result


def _rename_embedding_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Extract token embedding weight for the embedding sub-model.

    HF: ``model.text.wte`` → ONNX: ``embed_tokens.weight``
    """
    result: dict[str, torch.Tensor] = {}
    wte = state_dict.get("model.text.wte")
    if wte is not None:
        result["embed_tokens.weight"] = wte
    return result
