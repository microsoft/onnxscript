# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-OCR-2 vision-language model.

Architecture:
- Vision: SAM ViT-B → Qwen2 decoder-as-encoder → Linear projector
- Embedding: embed_tokens + scatter image features at placeholder tokens
- Decoder: DeepSeek-V2 LLM (standard attention + MoE, no MLA)

The vision pipeline processes images through SAM (1024x1024 -> 896-dim
features), then a Qwen2 transformer with dual-mask attention (non-causal
for visual tokens, causal for learned queries), and a linear projector
to the LLM hidden dimension.

Reference: deepseek-ai/DeepSeek-OCR-2 (HuggingFace, trust_remote_code=True)
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    Linear,
    RMSNorm,
    initialize_rope,
)
from mobius.components._sam_vision import SAMVisionEncoder
from mobius.models.deepseek import (
    DeepSeekV3TextModel,
)

# ──────────────────────────────────────────────────────────
# Qwen2 Encoder-as-Encoder (vision feature encoder)
# ──────────────────────────────────────────────────────────


class _Qwen2EncoderLayer(nn.Module):
    """Standard Qwen2 decoder layer used as encoder (no KV cache output).

    Same architecture as a standard decoder layer: pre-norm attention + MLP.
    Used in the vision encoder pipeline, not the LLM decoder.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        # Build a minimal config-like for Attention
        config = ArchitectureConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            hidden_act="silu",
            attn_qkv_bias=True,
            attn_o_bias=False,
        )
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
    ):
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        # No KV cache for encoder
        hidden_states, _ = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=None,
        )
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        return op.Add(residual, hidden_states)


class _Qwen2DecoderAsEncoder(nn.Module):
    """Qwen2 transformer used as vision feature encoder.

    Takes SAM features (flattened to 2D) and learned query embeddings,
    concatenates them, and runs through a standard Qwen2 transformer with
    a dual-mask: non-causal attention for visual features, causal attention
    for learned queries.

    Output: only the query portion of the output hidden states.

    Architecture:
    - 24 layers, hidden=896, 14 heads, 2 KV heads, intermediate=4864
    - No embed_tokens (takes inputs_embeds directly)
    - query_768: 144 learned queries (for 768x768 images)
    - query_1024: 256 learned queries (for 1024x1024 images)
    - Dual attention mask via token_type_ids
    """

    def __init__(
        self,
        num_layers: int = 24,
        hidden_size: int = 896,
        num_heads: int = 14,
        num_kv_heads: int = 2,
        intermediate_size: int = 4864,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 131072,
        num_queries_1024: int = 256,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._num_queries = num_queries_1024

        # Transformer layers (as ModuleList for weight name alignment)
        # HF weight path: model.model.layers.N.{self_attn,mlp,*_layernorm}
        head_dim = hidden_size // num_heads
        self.layers = nn.ModuleList(
            [
                _Qwen2EncoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Learned query embeddings (for 1024x1024 images)
        self.query_1024 = Embedding(num_queries_1024, hidden_size)

        # RoPE for position encoding
        rope_config = ArchitectureConfig(
            head_dim=head_dim,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )
        self.rotary_emb = initialize_rope(rope_config)

    def forward(
        self,
        op: builder.OpBuilder,
        sam_features: ir.Value,
    ):
        # sam_features: (B, 896, H_sam, W_sam) — e.g., (B, 896, 16, 16)
        # Flatten spatial dims: (B, 896, 16, 16) → (B, 256, 896)
        B = op.Shape(sam_features, start=0, end=1)  # noqa: N806
        x = op.Reshape(
            sam_features,
            op.Concat(B, [self._hidden_size, -1], axis=0),
        )  # (B, 896, 256)
        x = op.Transpose(x, perm=[0, 2, 1])  # (B, 256, 896)

        n_visual = self._num_queries  # 256 visual features

        # Learned query embeddings: (num_queries, hidden) → (1, num_queries, hidden)
        query_weight = self.query_1024(
            op, op.Constant(value=ir.tensor(np.arange(self._num_queries, dtype=np.int64)))
        )  # (num_queries, hidden)
        query_weight = op.Unsqueeze(query_weight, [0])  # (1, Q, H)
        # Broadcast to batch: (B, Q, H)
        batch_queries = op.Expand(
            query_weight,
            op.Concat(B, [1, 1], axis=0),
        )

        # Concatenate visual features and queries
        # x_combined: (B, 2*num_queries, hidden) = (B, 512, 896)
        x_combined = op.Concat(x, batch_queries, axis=1)
        total_len = 2 * n_visual

        # Position IDs: [0, 1, ..., total_len-1]
        position_ids = op.Constant(
            value=ir.tensor(np.arange(total_len, dtype=np.int64).reshape(1, -1))
        )
        position_ids = op.Expand(position_ids, op.Concat(B, [1], axis=0))
        position_embeddings = self.rotary_emb(op, position_ids)

        # Build dual attention mask:
        # - Non-causal (full attention) for visual tokens (first n_visual)
        # - Causal attention for query tokens (last n_visual)
        # - Query tokens can attend to all visual tokens
        # Precompute as a static constant:
        attention_bias = self._make_dual_mask(n_visual, total_len)
        attention_bias_const = op.Constant(value=ir.tensor(attention_bias))
        # Expand to (B, 1, S, S) for batch broadcasting
        attention_bias_4d = op.Expand(
            attention_bias_const,
            op.Concat(B, [1, 1, 1], axis=0),
        )

        # Run transformer layers
        hidden_states = x_combined
        for layer in self.layers:
            hidden_states = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias_4d,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(op, hidden_states)

        # Extract query portion only: (B, total_len, H) → (B, n_visual, H)
        hidden_states = op.Slice(
            hidden_states,
            [n_visual],
            [total_len],
            [1],  # axis=1
        )

        return hidden_states  # (B, num_queries, hidden)

    @staticmethod
    def _make_dual_mask(n_visual: int, total_len: int) -> np.ndarray:
        """Create the dual attention mask.

        Non-causal for image tokens, causal for query tokens.
        Query tokens can attend to all image tokens.

        Returns: (1, 1, total_len, total_len) float32 mask with
        0 = attend, -10000 = block.
        """
        mask = np.full((total_len, total_len), -10000.0, dtype=np.float32)

        # Image tokens (0..n_visual-1): full bidirectional attention
        mask[:n_visual, :n_visual] = 0.0

        # Query tokens (n_visual..total_len-1):
        for i in range(n_visual, total_len):
            # Can attend to all image tokens
            mask[i, :n_visual] = 0.0
            # Causal: can attend to query tokens up to and including self
            mask[i, n_visual : i + 1] = 0.0

        return mask.reshape(1, 1, total_len, total_len)


# ──────────────────────────────────────────────────────────
# OCR-2 Vision Encoder (SAM + Qwen2 + Projector)
# ──────────────────────────────────────────────────────────


class DeepSeekOCR2VisionEncoderModel(nn.Module):
    """OCR-2 vision encoder: SAM ViT-B → Qwen2 encoder → Linear projector.

    Input: pixel_values (B, 3, 1024, 1024)
    Output: image_features (B, 256, 1280) — projected to LLM hidden dim
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        # SAM ViT-B image encoder
        self.sam_model = SAMVisionEncoder(
            img_size=1024,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            out_chans=256,
            window_size=14,
            global_attn_indexes=(2, 5, 8, 11),
            downsample_channels=(512, 896),
        )

        # Qwen2 decoder-as-encoder
        self.qwen2_model = _Qwen2DecoderAsEncoder(
            num_layers=24,
            hidden_size=896,
            num_heads=14,
            num_kv_heads=2,
            intermediate_size=4864,
            num_queries_1024=256,
        )

        # Linear projector: 896 → LLM hidden size
        self.projector = Linear(896, config.hidden_size, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
    ):
        # SAM: (B, 3, 1024, 1024) → (B, 896, 16, 16)
        sam_features = self.sam_model(op, pixel_values)

        # Qwen2 encoder: (B, 896, 16, 16) → (B, 256, 896)
        query_features = self.qwen2_model(op, sam_features)

        # Project to LLM hidden dim: (B, 256, 896) → (B, 256, 1280)
        return self.projector(op, query_features)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF weight names to ONNX parameter names.

        HF prefix: model.sam_model.*, model.qwen2_model.*, model.projector.*
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Vision weights have model. prefix from HF composite
            if not key.startswith("model."):
                continue

            new_key = key[len("model.") :]

            if new_key.startswith("sam_model."):
                # SAM weights map directly
                renamed[new_key] = value
            elif new_key.startswith("qwen2_model."):
                # HF: qwen2_model.model.model.layers.N.* → qwen2_model.layers.N.*
                # The double "model.model." is from Qwen2Decoder2Encoder→CustomQwen2Decoder→model
                new_key = new_key.replace("qwen2_model.model.model.", "qwen2_model.")
                renamed[new_key] = value
            elif new_key.startswith("projector."):
                # HF: projector.layers.weight → projector.weight
                new_key = new_key.replace("projector.layers.", "projector.")
                renamed[new_key] = value

        return renamed


# ──────────────────────────────────────────────────────────
# OCR-2 Embedding Model
# ──────────────────────────────────────────────────────────


class DeepSeekOCR2EmbeddingModel(nn.Module):
    """OCR-2 embedding model: embed_tokens + scatter image features.

    Inputs:
        - input_ids: (batch, seq_len) INT64
        - image_features: (num_image_tokens, hidden_size) FLOAT
    Output:
        - inputs_embeds: (batch, seq_len, hidden_size) FLOAT

    Uses image placeholder token ID to identify where to insert
    image features into the text embeddings.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        # OCR-2 uses a special image token (from tokenizer config)
        self.image_token_id = config.image_token_id or 100015

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
        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        # Cumulative sum to map flat image_features indices
        mask_int = op.Cast(image_mask, to=7)  # INT64
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        # Pad image_features for text-only safety
        pad_row = op.Expand(
            op.Constant(value_float=0.0),
            op.Concat(
                op.Constant(value_ints=[1]),
                op.Shape(image_features, start=1, end=2),
                axis=0,
            ),
        )
        padded_features = op.Concat(image_features, pad_row, axis=0)

        # Scatter image features at placeholder positions
        gathered = op.Gather(padded_features, indices, axis=0)
        inputs_embeds = op.Where(image_mask_3d, gathered, text_embeds)

        return inputs_embeds

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Keep only embed_tokens weights."""
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if "embed_tokens" in key:
                new_key = key
                if new_key.startswith("model."):
                    new_key = new_key[len("model.") :]
                renamed[new_key] = value
        return renamed


# ──────────────────────────────────────────────────────────
# OCR-2 Decoder Model
# ──────────────────────────────────────────────────────────


class DeepSeekOCR2DecoderModel(nn.Module):
    """OCR-2 text decoder: DeepSeek-V2 with MoE (no MLA).

    Takes inputs_embeds (from embedding model) and generates logits.
    Standard attention with MoE layers, KV cache for autoregressive generation.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        # Reuse the DeepSeek text model (auto-detects non-MLA)
        self.model = DeepSeekV3TextModel(config)
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
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF weight names. LLM weights don't have extra prefixes."""
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Skip vision/embedding weights
            if key.startswith("model.sam_model."):
                continue
            if key.startswith("model.qwen2_model."):
                continue
            if key.startswith("model.projector."):
                continue
            if key.startswith("model.view_seperator"):
                continue

            new_key = key
            # MoE layer renames
            new_key = new_key.replace(".mlp.gate.", ".mlp.moe.gate.")
            new_key = new_key.replace(".mlp.experts.", ".mlp.moe.experts.")

            renamed[new_key] = value

        return renamed


# ──────────────────────────────────────────────────────────
# OCR-2 Composite Model (3-model VL split)
# ──────────────────────────────────────────────────────────


class DeepSeekOCR2CausalLMModel(nn.Module):
    """DeepSeek-OCR-2 vision-language model (3-model split).

    model_type: deepseek_vl_v2

    Components:
    - vision_encoder: SAM ViT-B + Qwen2 encoder + Linear projector
    - embedding: embed_tokens + image feature scatter
    - decoder: DeepSeek-V2 LLM (standard attention + MoE)
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = DeepSeekOCR2DecoderModel(config)
        self.vision_encoder = DeepSeekOCR2VisionEncoderModel(config)
        self.embedding = DeepSeekOCR2EmbeddingModel(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF weights to the appropriate sub-model.

        HF weight prefixes:
        - model.sam_model.* → vision_encoder
        - model.qwen2_model.* → vision_encoder
        - model.projector.* → vision_encoder
        - model.embed_tokens.* → embedding
        - model.layers.*, model.norm.*, lm_head.* → decoder
        """
        vision_sd: dict[str, torch.Tensor] = {}
        embedding_sd: dict[str, torch.Tensor] = {}
        decoder_sd: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            if key.startswith("model.sam_model."):
                vision_sd[key] = value
            elif key.startswith("model.qwen2_model."):
                vision_sd[key] = value
            elif key.startswith("model.projector."):
                vision_sd[key] = value
            elif "embed_tokens" in key:
                embedding_sd[key] = value
            else:
                decoder_sd[key] = value

        # Delegate to each sub-model's preprocess_weights
        result: dict[str, torch.Tensor] = {}

        vision_renamed = self.vision_encoder.preprocess_weights(vision_sd)
        for k, v in vision_renamed.items():
            result[f"vision_encoder.{k}"] = v

        embed_renamed = self.embedding.preprocess_weights(embedding_sd)
        for k, v in embed_renamed.items():
            result[f"embedding.{k}"] = v

        decoder_renamed = self.decoder.preprocess_weights(decoder_sd)
        for k, v in decoder_renamed.items():
            result[f"decoder.{k}"] = v

        return result
