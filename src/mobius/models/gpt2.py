# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""GPT-2 model with absolute positional embeddings and pre-norm LayerNorm.

Replicates HuggingFace's ``GPT2LMHeadModel``. Conv1D weights are transposed
during ``preprocess_weights`` and fused QKV projections are split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    FCMLP,
    Embedding,
    LayerNorm,
    Linear,
    create_padding_mask,
)
from mobius.components._attention import Attention
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class GPT2CausalLMModel(CausalLMModel):
    """GPT-2 causal language model.

    Differences from Llama-style:
    - Pre-norm with LayerNorm (not RMSNorm)
    - Absolute positional embeddings (not RoPE)
    - Combined QKV projection in HF weights (split during preprocess)
    - Conv1D weights are transposed vs Linear
    - Tied word embeddings (lm_head = wte)

    Replicates HuggingFace's ``GPT2LMHeadModel``.
    """

    default_task = "text-generation"
    category = "causal-lm"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        # OpenAI-GPT uses post-norm (attn → residual+norm → mlp → residual+norm)
        # while standard GPT-2 uses pre-norm. Detect via 'post_norm' config flag.
        post_norm = getattr(config, "post_norm", False)
        self.transformer = _GPT2TextModel(config, post_norm=post_norm)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.transformer(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF GPT-2-family weight names to our ONNX attribute names.

        Handles four distinct naming conventions found across the family:

        1. **GPT-2 / OpenAI-GPT / GPT-SW3** (Conv1D layout ``[in, out]``):
           - ``c_attn.weight`` → transpose + split equally → ``q/k/v_proj``
           - ``c_proj.weight`` (attn) → transpose → ``o_proj``
           - ``c_fc/c_proj.weight`` (mlp) → transpose → ``up/down_proj``
           - OpenAI-GPT uses ``tokens_embed``/``positions_embed`` instead of
             ``wte``/``wpe``.

        2. **GPT-Neo** (``nn.Linear``, extra ``attn.attention.*`` nesting):
           - Strips the extra ``.attn.attention.`` path segment.
           - MLP weights already have Linear layout; rename only.

        3. **GPT-BigCode** (``nn.Linear``, optional MQA):
           - ``c_attn.weight [3*H, H]`` → split equally → ``q/k/v_proj``
           - MQA ``c_attn.weight [H+kv+kv, H]`` → split by sizes.

        4. **BioGPT / XGLM** (``nn.Linear``, different module prefixes):
           - ``biogpt.layers.N.*`` / ``model.layers.N.*`` → ``transformer.h.N.*``
           - ``self_attn.*`` → ``attn.*``; ``fc1/fc2`` → ``up/down_proj``
           - Top-level embeddings / norms remapped to ``transformer.*``.
        """
        new: dict[str, torch.Tensor] = {}
        hidden = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = hidden // num_heads

        # Detect Conv1D variant: GPT2/OpenAI-GPT/GPT-SW3 store c_attn as [in, out]
        # (shape[0] < shape[1]), while GPT-BigCode stores it as nn.Linear [out, in].
        # This flag determines whether to transpose c_proj.weight (square matrices
        # can't be distinguished by shape alone).
        _conv1d_attn = any(
            v.ndim == 2 and v.shape[0] < v.shape[1]
            for k, v in state_dict.items()
            if "attn.c_attn.weight" in k
        )

        for orig_name, tensor in state_dict.items():
            name = orig_name

            # ── 1. Module-prefix normalisation ──────────────────────────────
            if name.startswith("biogpt.layers."):
                name = "transformer.h." + name[len("biogpt.layers.") :]
            elif name.startswith("model.layers."):
                name = "transformer.h." + name[len("model.layers.") :]

            # ── 2. Top-level embedding / norm renames ────────────────────────
            # OpenAI-GPT
            if name == "transformer.tokens_embed.weight":
                new["transformer.wte.weight"] = tensor
                continue
            if name == "transformer.positions_embed.weight":
                new["transformer.wpe.weight"] = tensor
                continue
            # BioGPT token embedding — scale by sqrt(hidden_size) to match
            # BioGptScaledWordEmbedding which multiplies embeddings at runtime.
            if name == "biogpt.embed_tokens.weight":
                embed_scale = float(tensor.shape[1]) ** 0.5
                new["transformer.wte.weight"] = tensor * embed_scale
                continue
            # BioGPT position embedding — first 2 rows are padding slots
            if name == "biogpt.embed_positions.weight":
                new["transformer.wpe.weight"] = tensor[2:]
                continue
            if name.startswith("biogpt.layer_norm."):
                new["transformer.ln_f." + name[len("biogpt.layer_norm.") :]] = tensor
                continue
            if name == "output_projection.weight":
                new["lm_head.weight"] = tensor
                continue
            # XGLM token embedding — scale by sqrt(d_model) to match
            # XGLMScaledWordEmbedding which multiplies embeddings at runtime.
            if name == "model.embed_tokens.weight":
                embed_scale = float(tensor.shape[1]) ** 0.5
                new["transformer.wte.weight"] = tensor * embed_scale
                continue
            if name.startswith("model.layer_norm."):
                new["transformer.ln_f." + name[len("model.layer_norm.") :]] = tensor
                continue

            # ── 3. Intra-layer structural renames ────────────────────────────
            # GPT-Neo: strip the extra .attn.attention. nesting level
            name = name.replace(".attn.attention.", ".attn.")
            # BioGPT / XGLM: align attn sub-module names
            name = name.replace(".self_attn.", ".attn.")
            name = name.replace(".self_attn_layer_norm.", ".ln_1.")
            name = name.replace(".final_layer_norm.", ".ln_2.")
            # BioGPT / XGLM MLP
            name = name.replace(".fc1.", ".mlp.up_proj.")
            name = name.replace(".fc2.", ".mlp.down_proj.")
            # GPT-Neo / BioGPT / XGLM: out_proj → o_proj
            name = name.replace(".attn.out_proj.", ".attn.o_proj.")

            # ── 4. Fused QKV projection (c_attn) ────────────────────────────
            if "attn.c_attn.weight" in name:
                prefix = name.replace("attn.c_attn.weight", "attn.")
                if tensor.shape[0] < tensor.shape[1]:
                    # Conv1D layout [in, out]: GPT-2 / OpenAI-GPT / GPT-SW3
                    tensor = tensor.t()  # → [3*hidden, hidden]
                    q, k, v = tensor.split(hidden, dim=0)
                else:
                    # nn.Linear layout [out=3*hidden, in]: GPT-BigCode MHA.
                    # HF reshapes output as [B, S, num_heads, 3*head_dim] and
                    # splits along the last dim, so rows are interleaved per head:
                    # [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...].
                    # Reorder to standard [Q_all, K_all, V_all] before splitting.
                    q_size = hidden
                    kv_size = (tensor.shape[0] - q_size) // 2
                    if kv_size == hidden:
                        # MHA only: reorder interleaved [num_heads, 3, head_dim] → [3, num_heads, head_dim].
                        # NOTE: Speculative branch — GPT-BigCode in practice always uses MQA
                        # (multi_query=True). This path is untested with real weights.
                        in_features = tensor.shape[1]
                        t = tensor.view(num_heads, 3, head_dim, in_features)
                        t = t.permute(1, 0, 2, 3).reshape(3 * hidden, in_features)
                        q, k, v = t.split(hidden, dim=0)
                    else:
                        # MQA: output is [Q_all_heads | K_single | V_single] — no reorder needed
                        q = tensor[:q_size]
                        k = tensor[q_size : q_size + kv_size]
                        v = tensor[q_size + kv_size :]
                new[f"{prefix}q_proj.weight"] = q
                new[f"{prefix}k_proj.weight"] = k
                new[f"{prefix}v_proj.weight"] = v
                continue

            if "attn.c_attn.bias" in name:
                prefix = name.replace("attn.c_attn.bias", "attn.")
                q_size = hidden
                kv_size = (tensor.shape[0] - q_size) // 2
                if kv_size == hidden and not _conv1d_attn:
                    # GPT-BigCode MHA (multi_query=False): interleaved per-head layout
                    # [Q_h0, K_h0, V_h0, Q_h1, ...] → reorder to standard [Q_all, K_all, V_all]
                    t = tensor.view(num_heads, 3, head_dim)
                    t = t.permute(1, 0, 2).reshape(3 * hidden)
                    q, k, v = t.split(hidden, dim=0)
                else:
                    q = tensor[:q_size]
                    k = tensor[q_size : q_size + kv_size]
                    v = tensor[q_size + kv_size :]
                new[f"{prefix}q_proj.bias"] = q
                new[f"{prefix}k_proj.bias"] = k
                new[f"{prefix}v_proj.bias"] = v
                continue

            # ── 5. Attention output projection ──────────────────────────────
            if "attn.c_proj.weight" in name:
                # c_proj is Conv1D in GPT-2-family (always transpose) but
                # nn.Linear in GPT-BigCode (never transpose). Use the c_attn
                # layout detected above since square shapes are ambiguous.
                new[name.replace("attn.c_proj.", "attn.o_proj.")] = (
                    tensor.t() if _conv1d_attn else tensor
                )
                continue
            if "attn.c_proj.bias" in name:
                new[name.replace("attn.c_proj.", "attn.o_proj.")] = tensor
                continue

            # ── 6. MLP projections ───────────────────────────────────────────
            # c_fc maps hidden → intermediate
            if name.endswith(".weight") and ".mlp.c_fc." in name:
                # Conv1D: shape [in=hidden, out=intermediate] → in < out → transpose
                if tensor.shape[0] < tensor.shape[1]:
                    tensor = tensor.t()
                new[name.replace(".c_fc.", ".up_proj.")] = tensor
                continue
            if name.endswith(".bias") and ".mlp.c_fc." in name:
                new[name.replace(".c_fc.", ".up_proj.")] = tensor
                continue

            # c_proj maps intermediate → hidden
            if name.endswith(".weight") and ".mlp.c_proj." in name:
                # Conv1D: shape [in=intermediate, out=hidden] → in > out → transpose
                if tensor.shape[0] > tensor.shape[1]:
                    tensor = tensor.t()
                new[name.replace(".c_proj.", ".down_proj.")] = tensor
                continue
            if name.endswith(".bias") and ".mlp.c_proj." in name:
                new[name.replace(".c_proj.", ".down_proj.")] = tensor
                continue

            new[name] = tensor

        # ── 7. XGLM sinusoidal position embeddings ───────────────────────────
        # XGLM uses XGLMSinusoidalPositionalEmbedding which is not in state_dict
        # (persistent=False). Pre-compute and store so wpe can load them.
        # Detected by the model.embed_tokens prefix (unique to XGLM in this family).
        if "transformer.wpe.weight" not in new and any(
            k.startswith("model.layers.") for k in state_dict
        ):
            wte_w = new.get("transformer.wte.weight")
            if wte_w is not None:
                hidden_size = wte_w.shape[1]
                num_positions = self.config.max_position_embeddings
                offset = 2  # XGLMSinusoidalPositionalEmbedding.offset
                total = num_positions + offset
                half_dim = hidden_size // 2
                import math as _math

                freq = _math.log(10000) / (half_dim - 1)
                freq_t = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -freq)
                pos_t = torch.arange(total, dtype=torch.float32).unsqueeze(1)
                emb = pos_t * freq_t.unsqueeze(0)
                emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
                if hidden_size % 2 == 1:
                    emb = torch.cat([emb, torch.zeros(total, 1)], dim=1)
                # Slice off the first `offset` rows to align position_id 0 → row 2
                new["transformer.wpe.weight"] = emb[offset:]

        # ── 8. Tied word embeddings ──────────────────────────────────────────
        if "lm_head.weight" not in new and "transformer.wte.weight" in new:
            new["lm_head.weight"] = new["transformer.wte.weight"]
        return super().preprocess_weights(new)


class _GPT2TextModel(nn.Module):
    """GPT-2 text backbone with absolute positional embeddings.

    Attribute names match HF GPT-2 naming (wte, wpe, h, ln_f).
    """

    def __init__(self, config: ArchitectureConfig, post_norm: bool = False):
        super().__init__()
        self.post_norm = post_norm
        self.wte = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.wpe = Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList(
            [
                _GPT2DecoderLayer(config, post_norm=post_norm)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.wte(op, input_ids)

        position_embeds = self.wpe(op, position_ids)
        hidden_states = op.Add(hidden_states, position_embeds)

        attention_bias = create_padding_mask(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.h)
        for layer, past_kv in zip(self.h, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        # OpenAI-GPT uses post-norm layers; ln_f would apply LayerNorm normalization
        # (mean subtraction + std division) even with identity weights, so skip it.
        if not self.post_norm:
            hidden_states = self.ln_f(op, hidden_states)
        return hidden_states, present_key_values


class _GPT2DecoderLayer(nn.Module):
    """GPT-2 decoder layer with LayerNorm.

    Supports both pre-norm (standard GPT-2) and post-norm (OpenAI-GPT) structures.
    Pre-norm:  norm → attn → residual → norm → mlp → residual.
    Post-norm: attn → residual → norm → mlp → residual → norm.
    Attribute names match HF GPT-2 naming (ln_1, ln_2, attn).
    """

    def __init__(self, config: ArchitectureConfig, post_norm: bool = False):
        super().__init__()
        self.post_norm = post_norm
        self.ln_1 = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config, scale=getattr(config, "attention_multiplier", None))
        self.ln_2 = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act,
            bias=True,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        past_key_value: tuple | None = None,
    ):
        if self.post_norm:
            # OpenAI-GPT post-norm: attn → residual+norm → mlp → residual+norm
            attn_out, present_kv = self.attn(
                op, hidden_states, attention_bias, past_key_value=past_key_value
            )
            hidden_states = self.ln_1(op, op.Add(hidden_states, attn_out))
            mlp_out = self.mlp(op, hidden_states)
            hidden_states = self.ln_2(op, op.Add(hidden_states, mlp_out))
        else:
            # Standard GPT-2 pre-norm: norm → attn → residual → norm → mlp → residual
            residual = hidden_states
            hidden_states = self.ln_1(op, hidden_states)
            hidden_states, present_kv = self.attn(
                op, hidden_states, attention_bias, past_key_value=past_key_value
            )
            hidden_states = op.Add(residual, hidden_states)

            residual = hidden_states
            hidden_states = self.ln_2(op, hidden_states)
            hidden_states = self.mlp(op, hidden_states)
            hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv
