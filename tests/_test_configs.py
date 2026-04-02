# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared test configurations for all test tiers.

Each config entry is a 3-tuple:
    (model_type, config_overrides, is_representative)

``is_representative=True`` means the model exercises unique behaviour
(custom model class, softcapping, parallel attention, ALiBi, MoE, etc.)
and should always be tested — including in ``--fast`` mode.  Models that
are simple aliases of a base class with ``{}`` overrides are marked
``False``.
"""

from __future__ import annotations

import dataclasses

from mobius._configs import (
    ArchitectureConfig,
    AudioConfig,
    BambaConfig,
    CodecDecoderConfig,
    CodecEncoderConfig,
    DepthAnythingConfig,
    Gemma2Config,
    Gemma3nConfig,
    GraniteMoeHybridConfig,
    JambaConfig,
    JetMoeConfig,
    LongcatFlashConfig,
    Mamba2Config,
    MambaConfig,
    MllamaConfig,
    MoondreamConfig,
    NanoChatConfig,
    NemotronHConfig,
    ResNetConfig,
    Sam2Config,
    SegformerConfig,
    VisionConfig,
    WhisperConfig,
    YolosConfig,
)

# ---------------------------------------------------------------------------
# Tiny model dimensions shared by all configs
# ---------------------------------------------------------------------------
TINY_HIDDEN = 64
TINY_INTERMEDIATE = 128
TINY_HEADS = 4
TINY_KV_HEADS = 2
TINY_HEAD_DIM = TINY_HIDDEN // TINY_HEADS
TINY_LAYERS = 2
TINY_VOCAB = 256

LONGROPE_FACTORS = [1.0] * (int(TINY_HEAD_DIM * 0.5) // 2)


def _base_config(config_cls=None, **overrides) -> ArchitectureConfig:
    """Create a tiny ArchitectureConfig for graph-build and parity tests.

    Applies a set of small defaults (hidden_size=64, 2 layers, etc.) and
    merges caller-supplied *overrides*.  Unknown fields are filtered out for
    dataclass-based config classes so that specialised configs (e.g.
    MambaConfig) that lack ``rope_*`` fields don't fail on construction.
    """
    if config_cls is None:
        config_cls = overrides.pop("_config_cls", ArchitectureConfig)
    else:
        overrides.pop("_config_cls", None)
    defaults = dict(
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_INTERMEDIATE,
        num_attention_heads=TINY_HEADS,
        num_key_value_heads=TINY_KV_HEADS,
        head_dim=TINY_HEAD_DIM,
        num_hidden_layers=TINY_LAYERS,
        vocab_size=TINY_VOCAB,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        pad_token_id=0,
    )
    defaults.update(overrides)
    # Filter out fields not accepted by the config class (e.g. MambaConfig
    # doesn't have max_position_embeddings or rope_* fields).
    if dataclasses.is_dataclass(config_cls):
        valid_fields = {f.name for f in dataclasses.fields(config_cls)}
        defaults = {k: v for k, v in defaults.items() if k in valid_fields}
    return config_cls(**defaults)


# ---------------------------------------------------------------------------
# Causal LM configs  (task: text-generation / hybrid-text-generation)
# ---------------------------------------------------------------------------
CAUSAL_LM_CONFIGS: list[tuple[str, dict, bool]] = [
    # === Text Generation (Llama-compatible) ===
    ("llama", {}, True),
    ("mistral", {}, False),
    ("qwen2", {}, True),
    ("cohere", {"tie_word_embeddings": True, "logit_scale": 0.0625}, True),
    ("cohere2", {"tie_word_embeddings": True, "logit_scale": 0.0625}, False),
    ("diffllama", {}, False),
    ("doge", {}, False),
    ("dots1", {}, False),
    ("exaone4", {}, False),
    (
        "glm",
        {"attn_qkv_bias": True},
        False,
    ),
    (
        "glm4",
        {"attn_qkv_bias": True},
        False,
    ),
    ("helium", {}, False),
    ("hunyuan_v1_dense", {}, False),
    ("llama4_text", {}, False),
    ("ministral", {}, False),
    ("ministral3", {}, False),
    (
        "nanochat",
        {
            "_config_cls": NanoChatConfig,
            "hidden_act": "relu2",
            "final_logit_softcapping": 15.0,
        },
        True,
    ),
    (
        "olmo2",
        {"attn_qk_norm": True, "attn_qk_norm_full": True},
        True,
    ),
    (
        "olmo3",
        {"attn_qk_norm": True, "attn_qk_norm_full": True},
        False,
    ),
    (
        "qwen3_5_text",
        {
            "partial_rotary_factor": 0.5,
            "layer_types": ["full_attention", "linear_attention"],
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
        False,
    ),
    (
        "qwen3_next",
        {
            "hidden_act": "silu",
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "num_hidden_layers": 4,
            "partial_rotary_factor": 0.25,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "norm_topk_prob": True,
            "attn_qk_norm": True,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
        True,
    ),
    ("solar_open", {}, False),
    ("stablelm", {"partial_rotary_factor": 0.25}, True),
    (
        "starcoder2",
        {"hidden_act": "gelu_pytorch_tanh", "tie_word_embeddings": True},
        True,
    ),
    ("youtu", {"tie_word_embeddings": True}, False),
    # === Absolute positional embeddings (non-RoPE) ===
    (
        "ctrl",
        {
            # HF CTRL FFN uses ReLU (not gelu_new); token embeds scaled by sqrt(n_embd)
            "hidden_act": "relu",
            "tie_word_embeddings": True,
            "num_key_value_heads": TINY_HEADS,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
        },
        True,
    ),
    (
        "gpt2",
        {
            "hidden_act": "gelu_new",
            "tie_word_embeddings": True,
            "num_key_value_heads": TINY_HEADS,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            "rms_norm_eps": 1e-5,
        },
        True,
    ),
    (
        "imagegpt",
        {"hidden_act": "gelu_new", "tie_word_embeddings": True},
        False,
    ),
    (
        "opt",
        {
            "hidden_act": "relu",
            "tie_word_embeddings": True,
            "num_key_value_heads": TINY_HEADS,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            # HF OPT uses eps=1e-5 for its LayerNorms (nn.LayerNorm default)
            "rms_norm_eps": 1e-5,
        },
        True,
    ),
    (
        "xlm",
        {
            # HF XLM uses standard GELU (erf-based) and eps=1e-12 for LayerNorms
            "hidden_act": "gelu",
            "tie_word_embeddings": True,
            "num_key_value_heads": TINY_HEADS,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            # XLM uses 4*emb_dim for feedforward (hardcoded in HF)
            "intermediate_size": 4 * TINY_HIDDEN,
            # HF XLM uses eps=1e-12 for all LayerNorms
            "rms_norm_eps": 1e-12,
        },
        False,
    ),
    # === Other Llama-compatible ===
    # ModernBERT-Decoder uses MHA only (HF sets kv_heads=num_heads internally);
    # set num_key_value_heads=TINY_HEADS to match.
    ("modernbert-decoder", {"num_key_value_heads": TINY_HEADS}, False),
    # === Text Generation (architecture-specific) ===
    (
        "gemma",
        {"attn_qkv_bias": False, "attn_o_bias": False},
        True,
    ),
    (
        "gemma2",
        {
            "_config_cls": Gemma2Config,
            "attn_qkv_bias": False,
            "attn_o_bias": False,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": TINY_HEAD_DIM,
        },
        True,
    ),
    (
        "shieldgemma2",
        {
            "_config_cls": Gemma2Config,
            "attn_qkv_bias": False,
            "attn_o_bias": False,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": TINY_HEAD_DIM,
        },
        False,
    ),
    (
        "gemma3_text",
        {
            "attn_qk_norm": True,
            "rope_local_base_freq": 10_000.0,
            "layer_types": ["full_attention", "sliding_attention"],
        },
        True,
    ),
    (
        "gemma3n_text",
        {
            "_config_cls": Gemma3nConfig,
            "attn_qk_norm": True,
            "hidden_act": "gelu_pytorch_tanh",
            "rope_local_base_freq": 10_000.0,
            "layer_types": ["full_attention", "sliding_attention"],
            "altup_num_inputs": 2,
            "altup_active_idx": 0,
            "altup_correct_scale": True,
            "laurel_rank": 16,
            "hidden_size_per_layer_input": 32,
            "vocab_size_per_layer_input": 256,
        },
        True,
    ),
    (
        "gemma3n",
        {
            "_config_cls": Gemma3nConfig,
            "attn_qk_norm": True,
            "hidden_act": "gelu_pytorch_tanh",
            "rope_local_base_freq": 10_000.0,
            "layer_types": ["full_attention", "sliding_attention"],
            "altup_num_inputs": 2,
            "altup_active_idx": 0,
            "altup_correct_scale": True,
            "laurel_rank": 16,
            "hidden_size_per_layer_input": 32,
            "vocab_size_per_layer_input": 256,
        },
        False,
    ),
    ("granite", {}, True),
    ("olmo", {}, False),
    ("internlm2", {"attn_qkv_bias": True}, True),
    (
        "nemotron",
        {
            "hidden_act": "relu2",
            "partial_rotary_factor": 0.5,
            "rms_norm_eps": 1e-5,
        },
        True,
    ),
    (
        "phi3",
        {
            "partial_rotary_factor": 0.5,
            "rope_type": "longrope",
            "rope_scaling": {
                "short_factor": LONGROPE_FACTORS,
                "long_factor": LONGROPE_FACTORS,
            },
            "original_max_position_embeddings": 128,
        },
        True,
    ),
    ("qwen3", {"attn_qk_norm": True}, True),
    (
        "qwen3_5_text",
        {
            "partial_rotary_factor": 0.5,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
        True,
    ),
    ("qwen3_vl_text", {"attn_qk_norm": True}, False),
    ("smollm3", {}, False),
    # === Mixture of Experts ===
    (
        "phimoe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "partial_rotary_factor": 0.5,
            "rope_type": "longrope",
            "rope_scaling": {
                "short_factor": LONGROPE_FACTORS,
                "long_factor": LONGROPE_FACTORS,
            },
            "original_max_position_embeddings": 128,
        },
        True,
    ),
    (
        "granitemoe",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        True,
    ),
    (
        "mixtral",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        True,
    ),
    (
        "olmoe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "attn_qk_norm": True,
            "attn_qk_norm_full": True,
        },
        False,
    ),
    (
        "qwen2_moe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 64,
            "attn_qkv_bias": True,
        },
        True,
    ),
    (
        "qwen3_moe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 128,
            "attn_qk_norm": True,
        },
        True,
    ),
    (
        "qwen3_5_moe",
        {
            "hidden_act": "silu",
            "layer_types": ["linear_attention", "full_attention"],
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
        True,
    ),
    # === Falcon and Bloom ===
    # dual_ln=True: Falcon with new_decoder_architecture uses separate ln_attn + ln_mlp.
    ("falcon", {"parallel_attn": True, "dual_ln": True}, True),
    (
        "falcon_h1",
        # ALiBi bias shape (1, num_heads, q, total) requires kv_num_heads == num_heads
        # in ORT Attention (GQA is incompatible with ALiBi). Use MHA (kv_heads=num_heads).
        # dual_ln=True: new_decoder_architecture uses separate ln_attn + ln_mlp.
        {
            "alibi": True,
            "attn_qkv_bias": True,
            "num_key_value_heads": TINY_HEADS,
            "dual_ln": True,
        },
        True,
    ),
    (
        "bloom",
        {
            "alibi": True,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            "mlp_bias": True,
            "num_key_value_heads": TINY_HEADS,
            "intermediate_size": 4 * TINY_HIDDEN,
        },
        True,
    ),
    # === Additional Llama-compatible aliases ===
    ("baichuan", {}, False),
    ("codegen2", {}, False),
    ("command_r", {}, False),
    # === DeepSeek (MLA + MoE) ===
    (
        "deepseek_v3",
        {
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "n_group": 2,
            "topk_group": 1,
            "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "first_k_dense_replace": 1,
            "n_shared_experts": 1,
            "rope_interleave": True,
            "rope_type": "yarn",
            "rope_scaling": {
                "factor": 4.0,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 128,
            },
        },
        True,
    ),
    (
        "deepseek_v2",
        {
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "n_group": 2,
            "topk_group": 1,
            "routed_scaling_factor": 1.0,
            "scoring_func": "softmax",
            "topk_method": "group_limited_greedy",
            "first_k_dense_replace": 1,
            "n_shared_experts": 1,
        },
        True,
    ),
    ("exaone", {}, False),
    # DeepSeek-V2 without MLA (standard attention + MoE, like OCR-2 LLM)
    (
        "deepseek_v2",
        {
            "qk_nope_head_dim": 0,
            "qk_rope_head_dim": 0,
            "v_head_dim": 0,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 1.0,
            "scoring_func": "softmax",
            "topk_method": "greedy",
            "first_k_dense_replace": 1,
            "n_shared_experts": 2,
        },
        True,
    ),
    ("minicpm", {}, True),
    ("minicpm3", {}, True),
    ("mistral3", {}, False),
    ("openelm", {}, True),
    (
        "persimmon",
        {
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            "mlp_bias": True,
            "num_key_value_heads": TINY_HEADS,
            "partial_rotary_factor": 0.5,
            "hidden_act": "relu2",
            # HF Persimmon uses layer_norm_eps=1e-5
            "rms_norm_eps": 1e-5,
        },
        True,
    ),
    ("yi", {}, False),
    ("zamba", {}, True),
    # === Architecture-specific (untested classes) ===
    ("chatglm", {}, True),
    ("ernie4_5", {}, True),
    (
        "gemma3",
        {
            "attn_qk_norm": True,
            "rope_local_base_freq": 10_000.0,
            "layer_types": ["full_attention", "sliding_attention"],
        },
        False,
    ),
    (
        "phi",
        {
            "partial_rotary_factor": 0.5,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            "mlp_bias": True,
            "hidden_act": "gelu_new",
        },
        True,
    ),
    ("phi3small", {"partial_rotary_factor": 0.5}, True),
    ("qwen", {}, True),
    # === MoE aliases ===
    (
        "arctic",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "dbrx",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "jetmoe",
        {
            "_config_cls": JetMoeConfig,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "head_dim": TINY_HEAD_DIM,  # kv_channels = head_dim; not hidden/num_heads
            # num_attention_heads = num_experts_per_tok * num_kv_heads = 2 * 2 = 4 (TINY_HEADS)
        },
        False,
    ),
    # === Additional CausalLM aliases ===
    ("apertus", {}, False),
    ("arcee", {"hidden_act": "relu2"}, False),
    ("code_llama", {}, False),
    (
        "codegen",
        {
            "partial_rotary_factor": 0.5,
            "mlp_bias": True,
            "num_key_value_heads": TINY_HEADS,
            "hidden_act": "gelu_new",
            # HF CodeGen defaults to layer_norm_epsilon=1e-5
            "rms_norm_eps": 1e-5,
        },
        False,
    ),
    ("csm", {}, False),
    ("evolla", {}, False),
    (
        "gpt_neox",
        {
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            "mlp_bias": True,
            "num_key_value_heads": TINY_HEADS,
            "hidden_act": "gelu",
            # HF GPT-NeoX defaults to partial_rotary_factor=0.25
            "partial_rotary_factor": 0.25,
        },
        False,
    ),
    (
        "gpt_neox_japanese",
        {
            # GPT-NeoX-Japanese has NO QKV bias (only a separate dense_bias for output proj)
            "attn_qkv_bias": False,
            "attn_o_bias": True,
            "mlp_bias": False,
            "num_key_value_heads": TINY_HEADS,
            "hidden_act": "gelu",
            # NOTE: HF GPT-NeoX-Japanese reads partial_rotary_factor from config.rope_parameters,
            # not from a top-level field. HF default is 1.0 (full rotary). Use 1.0 here so both
            # ONNX and HF apply rotary to all head_dim dimensions.
            # GPT-NeoX-Japanese uses intermediate_multiple_size (default 4) not intermediate_size
            "intermediate_size": 4 * TINY_HIDDEN,
            # HF GPT-NeoX-Japanese uses layer_norm_eps=1e-5 by default; match it
            "rms_norm_eps": 1e-5,
        },
        False,
    ),
    (
        "gptj",
        {
            "partial_rotary_factor": 0.25,
            "mlp_bias": True,
            "num_key_value_heads": TINY_HEADS,
            "hidden_act": "gelu_new",
            # HF GPT-J defaults to layer_norm_epsilon=1e-5
            "rms_norm_eps": 1e-5,
        },
        False,
    ),
    (
        "longcat_flash",
        {
            "_config_cls": LongcatFlashConfig,
            "q_lora_rank": 16,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "v_head_dim": 8,
            "num_local_experts": 4,
            "zero_expert_num": 2,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "routed_scaling_factor": 1.0,
            "intermediate_size": TINY_INTERMEDIATE,
            "num_hidden_layers": TINY_LAYERS * 2,
            "rope_interleave": True,
        },
        True,
    ),
    ("open-llama", {}, False),
    # seed_oss uses attention_bias=True by default (HF always has q/k/v biases).
    # Set attn_qkv_bias=True so our ONNX model also has these biases for parity.
    ("seed_oss", {"attn_qkv_bias": True}, False),
    ("zamba2", {}, False),
    # === Additional GPT2 aliases ===
    # num_key_value_heads=TINY_HEADS: GPT-2 family never uses GQA; setting
    # kv_heads = num_heads ensures ONNX KV-cache and HF weight shapes agree.
    # attn_qkv_bias / attn_o_bias: all GPT-2 family models use attention biases.
    (
        "biogpt",
        {
            "hidden_act": "gelu_new",
            "tie_word_embeddings": True,
            "num_key_value_heads": TINY_HEADS,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
        },
        False,
    ),
    (
        "gpt-sw3",
        {
            "hidden_act": "gelu_new",
            "tie_word_embeddings": True,
            "num_key_value_heads": TINY_HEADS,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
        },
        False,
    ),
    (
        "gpt_bigcode",
        {
            "hidden_act": "gelu_new",
            "tie_word_embeddings": True,
            "num_key_value_heads": TINY_HEADS,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            "rms_norm_eps": 1e-5,
        },
        False,
    ),
    (
        "gpt_neo",
        # GPT-Neo does not scale attention (no 1/sqrt(head_dim)), so set attention_multiplier=1.0.
        {
            "hidden_act": "gelu_new",
            "tie_word_embeddings": True,
            "num_key_value_heads": TINY_HEADS,
            "attn_qkv_bias": False,
            "attn_o_bias": True,
            "rms_norm_eps": 1e-5,
            "attention_multiplier": 1.0,
        },
        False,
    ),
    (
        "openai-gpt",
        # OpenAI-GPT uses post-norm (attn → residual+norm) instead of GPT-2 pre-norm.
        # Also: always uses 4 * n_embd for MLP; no n_inner config option.
        # No final LayerNorm (ln_f injected as identity in preprocess_weights).
        {
            "hidden_act": "gelu_new",
            "tie_word_embeddings": True,
            "num_key_value_heads": TINY_HEADS,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            "intermediate_size": 4 * TINY_HIDDEN,
            "post_norm": True,
            "rms_norm_eps": 1e-5,
        },
        False,
    ),
    (
        "xglm",
        {
            "hidden_act": "gelu_new",
            "tie_word_embeddings": True,
            "num_key_value_heads": TINY_HEADS,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
        },
        False,
    ),
    # === Additional Falcon aliases ===
    (
        "mpt",
        {
            "num_key_value_heads": TINY_HEADS,
            "hidden_act": "gelu",
            # MPT hardcodes 4*hidden_size; set our intermediate_size to match
            "intermediate_size": 4 * TINY_HIDDEN,
        },
        False,
    ),
    # === Additional MoE aliases ===
    (
        "ernie4_5_moe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 128,
        },
        False,
    ),
    (
        "flex_olmo",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "attn_qk_norm": True,
            "attn_qk_norm_full": True,
            "post_feedforward_norm": True,
        },
        False,
    ),
    (
        "glm4_moe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 128,
        },
        False,
    ),
    (
        "glm4v_text",
        {},
        False,
    ),
    (
        "glm4v_moe_text",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        },
        False,
    ),
    (
        "qwen3_omni_moe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        },
        False,
    ),
    (
        "qwen3_vl_moe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        },
        False,
    ),
    (
        "granitemoehybrid",
        {
            "_config_cls": GraniteMoeHybridConfig,
            "layer_types": ["mamba2", "full_attention"],
            "mamba_n_heads": 4,
            "mamba_d_head": 32,
            "mamba_d_state": 8,
            "mamba_n_groups": 1,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
            "shared_intermediate_size": 32,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        },
        True,
    ),
    (
        "granitemoeshared",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "hunyuan_v1_moe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "attn_qk_norm": True,
            "shared_expert_intermediate_size": TINY_INTERMEDIATE,
        },
        True,
    ),
    (
        "minimax",
        {
            "layer_types": ["full_attention", "lightning_attention"],
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "head_dim": TINY_HIDDEN // TINY_HEADS,
        },
        True,
    ),
    (
        "gpt_oss",
        {
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 64,
            "head_dim": TINY_HEAD_DIM,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        },
        True,
    ),
    # --- Variant coverage: architecture code-path variants ---
    # qwen3_next: mostly full-attention (1 linear + 1 full) — exercises full-attention path
    # while satisfying HF Qwen3NextDynamicCache's requirement for at least one linear layer.
    (
        "qwen3_next",
        {
            "hidden_act": "silu",
            "layer_types": ["full_attention", "linear_attention"],
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "norm_topk_prob": True,
            "attn_qk_norm": True,
            "partial_rotary_factor": 0.25,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
        False,
    ),
    # qwen3_next: mostly linear-attention (1 linear + 1 full) — exercises DeltaNet path
    # while satisfying HF Qwen3NextDynamicCache's requirement for at least one full-attn layer.
    (
        "qwen3_next",
        {
            "hidden_act": "silu",
            "layer_types": ["linear_attention", "full_attention"],
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "norm_topk_prob": True,
            "attn_qk_norm": True,
            "partial_rotary_factor": 0.25,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
        False,
    ),
    # deepseek_v2: MLA with dense MLP (no MoE, all layers dense)
    (
        "deepseek_v2",
        {
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "num_local_experts": 0,
            "num_experts_per_tok": 0,
            "moe_intermediate_size": 32,
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 1.0,
            "scoring_func": "softmax",
            "topk_method": "greedy",
            "first_k_dense_replace": TINY_LAYERS,
            "n_shared_experts": 0,
        },
        False,
    ),
    # falcon_h1: ALiBi with parallel attention+MLP
    (
        "falcon_h1",
        # ALiBi bias shape (1, num_heads, q, total) requires kv_num_heads == num_heads
        # in ORT Attention (GQA is incompatible with ALiBi). Use MHA (kv_heads=num_heads).
        # dual_ln=True: new_decoder_architecture uses separate ln_attn + ln_mlp.
        {
            "alibi": True,
            "attn_qkv_bias": True,
            "parallel_attn": True,
            "num_key_value_heads": TINY_HEADS,
            "dual_ln": True,
        },
        False,
    ),
    # jamba: hybrid Mamba+Attention with MoE (requires JambaConfig)
    (
        "jamba",
        {
            "_config_cls": JambaConfig,
            "num_hidden_layers": 4,
            "layer_types": [
                "mamba",
                "full_attention",
                "mamba",
                "full_attention",
            ],
            "mamba_d_state": 8,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
            "mamba_dt_rank": 4,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
        },
        True,
    ),
    # jamba: all attention layers (no Mamba SSM)
    (
        "jamba",
        {
            "_config_cls": JambaConfig,
            "layer_types": ["full_attention"] * TINY_LAYERS,
            "mamba_d_state": 8,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
            "mamba_dt_rank": 4,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "expert_layer_period": 2,
            "expert_layer_offset": 1,
        },
        False,
    ),
    # bamba: hybrid Mamba2+Attention (requires BambaConfig)
    (
        "bamba",
        {
            "_config_cls": BambaConfig,
            "num_hidden_layers": 4,
            "layer_types": [
                "mamba2",
                "full_attention",
                "mamba2",
                "mamba2",
            ],
            "mamba_n_heads": 4,
            "mamba_d_head": 32,
            "mamba_d_state": 8,
            "mamba_n_groups": 1,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
        },
        True,
    ),
    # bamba: all attention layers (no Mamba2)
    (
        "bamba",
        {
            "_config_cls": BambaConfig,
            "layer_types": ["full_attention"] * TINY_LAYERS,
            "mamba_n_heads": 4,
            "mamba_d_head": 32,
            "mamba_d_state": 8,
            "mamba_n_groups": 1,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
        },
        False,
    ),
    # nemotron_h: hybrid Mamba2+Attention+MLP (requires NemotronHConfig)
    (
        "nemotron_h",
        {
            "hidden_act": "relu2",
            "layer_types": ["mamba2", "mlp", "full_attention", "mlp"],
            "_config_cls": NemotronHConfig,
            "num_hidden_layers": 4,
            "mamba_n_heads": TINY_KV_HEADS,
            "mamba_d_head": TINY_HEAD_DIM,
            "mamba_d_state": 16,
            "mamba_n_groups": 1,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
        },
        True,
    ),
    # gemma3n_text: all full attention (no sliding window)
    (
        "gemma3n_text",
        {
            "_config_cls": Gemma3nConfig,
            "attn_qk_norm": True,
            "hidden_act": "gelu_pytorch_tanh",
            "rope_local_base_freq": 10_000.0,
            "layer_types": ["full_attention"] * TINY_LAYERS,
            "altup_num_inputs": 2,
            "altup_active_idx": 0,
            "altup_correct_scale": True,
            "laurel_rank": 16,
            "hidden_size_per_layer_input": 32,
            "vocab_size_per_layer_input": 256,
        },
        False,
    ),
    # granite: with non-default scaling multipliers
    (
        "granite",
        {
            "embedding_multiplier": 12.0,
            "attention_multiplier": 0.0625,
            "logits_scaling": 0.125,
            "residual_multiplier": 0.5,
        },
        False,
    ),
    # phi3small: different partial_rotary_factor
    (
        "phi3small",
        {"partial_rotary_factor": 0.25},
        False,
    ),
]


# ---------------------------------------------------------------------------
# Encoder-only configs  (task: feature-extraction)
# ---------------------------------------------------------------------------
ENCODER_CONFIGS: list[tuple[str, dict, bool]] = [
    ("bert", {"hidden_act": "gelu", "type_vocab_size": 2}, True),
    ("albert", {"hidden_act": "gelu", "type_vocab_size": 2}, True),
    (
        "camembert",
        {"hidden_act": "gelu", "type_vocab_size": 1},
        False,
    ),
    (
        "clip_text_model",
        {"hidden_act": "gelu", "type_vocab_size": 0},
        True,
    ),
    (
        "data2vec-text",
        {"hidden_act": "gelu", "type_vocab_size": 1},
        False,
    ),
    ("deberta", {"hidden_act": "gelu", "type_vocab_size": 2}, True),
    (
        "deberta-v2",
        {"hidden_act": "gelu", "type_vocab_size": 2},
        True,
    ),
    (
        "distilbert",
        {
            "hidden_act": "gelu",
            "type_vocab_size": 0,
            "max_position_embeddings": 64,
        },
        True,
    ),
    ("electra", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    ("ernie", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    ("ernie_m", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    ("esm", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    ("flaubert", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    ("ibert", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    (
        "megatron-bert",
        {"hidden_act": "gelu", "type_vocab_size": 2},
        False,
    ),
    (
        "mobilebert",
        {"hidden_act": "gelu", "type_vocab_size": 2},
        False,
    ),
    ("nezha", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    ("qdqbert", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    ("rembert", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    (
        "roberta",
        {"hidden_act": "gelu", "type_vocab_size": 1},
        False,
    ),
    (
        "roberta-prelayernorm",
        {"hidden_act": "gelu", "type_vocab_size": 1},
        False,
    ),
    ("roc_bert", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    ("roformer", {"hidden_act": "gelu", "type_vocab_size": 2}, True),
    (
        "splinter",
        {"hidden_act": "gelu", "type_vocab_size": 2},
        False,
    ),
    (
        "squeezebert",
        {"hidden_act": "gelu", "type_vocab_size": 2},
        False,
    ),
    (
        "xlm-roberta",
        {"hidden_act": "gelu", "type_vocab_size": 1},
        False,
    ),
    (
        "xlm-roberta-xl",
        {"hidden_act": "gelu", "type_vocab_size": 1},
        False,
    ),
    ("xlnet", {"hidden_act": "gelu", "type_vocab_size": 2}, True),
    ("xmod", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    ("bros", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    ("layoutlm", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    (
        "layoutlmv2",
        {"hidden_act": "gelu", "type_vocab_size": 2},
        False,
    ),
    (
        "layoutlmv3",
        {"hidden_act": "gelu", "type_vocab_size": 2},
        False,
    ),
    ("modernbert", {"hidden_act": "gelu"}, True),
    (
        "nomic_bert",
        {
            "hidden_act": "swiglu",
            "type_vocab_size": 2,
            "rope_theta": 1000.0,
        },
        True,
    ),
    ("lilt", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    ("markuplm", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    ("mega", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    ("mra", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    (
        "nystromformer",
        {"hidden_act": "gelu", "type_vocab_size": 2},
        False,
    ),
    ("yoso", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
    # === Additional encoder aliases ===
    ("mpnet", {"hidden_act": "gelu", "type_vocab_size": 2}, False),
]


# ---------------------------------------------------------------------------
# Seq2Seq (encoder-decoder) configs  (task: seq2seq)
# ---------------------------------------------------------------------------
SEQ2SEQ_CONFIGS: list[tuple[str, dict, bool]] = [
    (
        "bart",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        True,
    ),
    (
        "bigbird_pegasus",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        True,
    ),
    (
        "blenderbot",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "blenderbot-small",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "fsmt",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "led",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "longt5",
        {"hidden_act": "relu", "num_decoder_layers": 2},
        True,
    ),
    (
        "m2m_100",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "marian",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "mbart",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "mt5",
        {"hidden_act": "gelu_new", "num_decoder_layers": 2, "is_gated_act": True},
        True,
    ),
    (
        "nllb_moe",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        True,
    ),
    (
        "pegasus",
        {
            "hidden_act": "relu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "pegasus_x",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "plbart",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "prophetnet",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "t5",
        {"hidden_act": "relu", "num_decoder_layers": 2},
        True,
    ),
    (
        "switch_transformers",
        {"hidden_act": "relu", "num_decoder_layers": 2},
        True,
    ),
    (
        "trocr",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "xlm-prophetnet",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    # === Additional seq2seq aliases ===
    (
        "mvp",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "nllb-moe",
        {
            "hidden_act": "gelu",
            "num_decoder_layers": 2,
            "max_position_embeddings": 64,
        },
        False,
    ),
    (
        "umt5",
        {"hidden_act": "gelu_new", "num_decoder_layers": 2, "is_gated_act": True},
        False,
    ),
]


# ---------------------------------------------------------------------------
# Vision (image classification / feature extraction) configs
# ---------------------------------------------------------------------------
VISION_CONFIGS: list[tuple[str, dict, bool]] = [
    (
        "beit",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        True,
    ),
    (
        "blip",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "clip_vision_model",
        {
            "hidden_act": "quick_gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        True,
    ),
    (
        "cvt",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        True,
    ),
    (
        "depth_anything",
        {
            "_config_cls": DepthAnythingConfig,
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
            "backbone_out_indices": [1, 2],
            "neck_hidden_sizes": [16, 32],
            "reassemble_factors": [2.0, 1.0],
            "fusion_hidden_size": 16,
            "head_hidden_size": 8,
        },
        True,
    ),
    (
        "data2vec-vision",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "deit",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "dinov2",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        True,
    ),
    (
        "dinov2_with_registers",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "dinov3_vit",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "hiera",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        True,
    ),
    (
        "ijepa",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "mobilevit",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        True,
    ),
    (
        "mobilevitv2",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "pvt",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        True,
    ),
    (
        "pvt_v2",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "resnet",
        {
            "_config_cls": ResNetConfig,
            "hidden_act": "relu",
            "num_channels": 3,
            "embedding_size": 16,
            "hidden_sizes": [32, 64, 128, 256],
            "depths": [1, 1, 1, 1],
            "layer_type": "bottleneck",
            "image_size": 32,
        },
        True,
    ),
    (
        "siglip_vision_model",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        True,
    ),
    (
        "siglip2_vision_model",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "siglip",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "siglip2",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "swin",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        True,
    ),
    (
        "swin2sr",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "swinv2",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "vit",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        True,
    ),
    (
        "vit_hybrid",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "vit_mae",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "vit_msn",
        {
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
        },
        False,
    ),
    (
        "sam2",
        {
            "_config_cls": Sam2Config,
            "hidden_act": "gelu",
            "image_size": 64,
            "num_channels": 3,
            "sam2_embed_dims": [16, 32, 64, 128],
            "sam2_blocks_per_stage": [1, 1, 2, 1],
            "sam2_num_heads_per_stage": [1, 2, 4, 8],
            "sam2_mlp_ratio": 4.0,
            "sam2_fpn_hidden_size": 32,
        },
        True,
    ),
    (
        "segformer",
        {
            "_config_cls": SegformerConfig,
            "hidden_act": "gelu",
            "image_size": 32,
            "num_channels": 3,
            "segformer_hidden_sizes": [16, 32],
            "segformer_num_attention_heads": [1, 2],
            "segformer_depths": [1, 1],
            "segformer_sr_ratios": [2, 1],
            "segformer_mlp_ratios": [4, 4],
            "segformer_patch_sizes": [3, 3],
            "segformer_strides": [2, 2],
            "decoder_hidden_size": 16,
            "num_labels": 5,
        },
        True,
    ),
]


# ---------------------------------------------------------------------------
# Object detection configs
# ---------------------------------------------------------------------------
DETECTION_CONFIGS: list[tuple[str, dict, bool]] = [
    (
        "yolos",
        {
            "_config_cls": YolosConfig,
            "hidden_act": "gelu",
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
            "num_detection_tokens": 10,
            "num_labels": 5,
        },
        True,
    ),
]


# ---------------------------------------------------------------------------
# SSM (State Space Model) configs — pure Mamba/Mamba2, no attention
# ---------------------------------------------------------------------------
SSM_CONFIGS: list[tuple[str, dict, bool]] = [
    # mamba: pure Mamba SSM (no attention) — requires MambaConfig
    (
        "mamba",
        {
            "_config_cls": MambaConfig,
            "state_size": 8,
            "conv_kernel": 4,
            "expand": 2,
            "time_step_rank": 4,
        },
        True,
    ),
    # mamba2: pure Mamba2/SSD (no attention) — requires Mamba2Config
    (
        "mamba2",
        {
            "_config_cls": Mamba2Config,
            "num_heads": 4,
            "head_dim": 16,
            "state_size": 8,
            "n_groups": 2,
            "conv_kernel": 4,
            "expand": 2,
        },
        True,
    ),
]


# ---------------------------------------------------------------------------
# Shared VL tiny vision sub-config (SigLIP-style: 28x28, patch 14)
# ---------------------------------------------------------------------------
_TINY_VISION = VisionConfig(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    image_size=28,
    patch_size=14,
    norm_eps=1e-6,
)

_TINY_QWEN_VL_VISION = VisionConfig(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    image_size=28,
    patch_size=14,
    norm_eps=1e-6,
    spatial_merge_size=2,
    temporal_patch_size=2,
    out_hidden_size=64,
)

_TINY_QWEN3_VL_VISION = VisionConfig(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    image_size=28,
    patch_size=14,
    norm_eps=1e-6,
    spatial_merge_size=2,
    temporal_patch_size=2,
    out_hidden_size=64,
    fullatt_block_indexes=[0],
    window_size=4,
)


# ---------------------------------------------------------------------------
# Vision-Language configs  (task: vision-language and variants)
# ---------------------------------------------------------------------------
# NOTE: These models build multi-model packages (decoder + vision + embedding).
# The test parametrization in build_graph_test.py uses specialised test
# methods that invoke the correct task and assert the right output models.
VL_CONFIGS: list[tuple[str, dict, bool]] = [
    # --- LLaVA family (vision-language, 3-model split) ---
    ("llava", {"vision": _TINY_VISION, "image_token_id": 32000}, True),
    ("aya_vision", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("chameleon", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("cohere2_vision", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("deepseek_vl", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("deepseek_vl_hybrid", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("florence2", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("fuyu", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("glm4v", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("glm4v_moe", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("got_ocr2", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("idefics2", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("idefics3", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("instructblip", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("instructblipvideo", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("janus", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("llava_next", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("llava_next_video", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("llava_onevision", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("molmo", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    (
        "moondream1",
        {
            "_config_cls": MoondreamConfig,
            "vision": _TINY_VISION,
            "image_token_id": 50256,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            "hidden_act": "gelu_tanh",
        },
        True,
    ),
    ("ovis2", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("paligemma", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("pixtral", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("smolvlm", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("video_llava", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("vipllava", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    # --- InternVL family ---
    ("internvl_chat", {"vision": _TINY_VISION, "image_token_id": 32000}, True),
    ("internvl2", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    ("internvl", {"vision": _TINY_VISION, "image_token_id": 32000}, False),
    # --- Gemma3 multimodal (requires rope_local_base_freq, layer_types) ---
    (
        "gemma3_multimodal",
        {
            "attn_qk_norm": True,
            "rope_local_base_freq": 10_000.0,
            "layer_types": ["full_attention", "sliding_attention"],
            "vision": VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
                mm_tokens_per_image=4,
            ),
            "mm_tokens_per_image": 4,
            "image_token_id": 255999,
        },
        True,
    ),
    # --- Blip2 (ViT + Q-Former + LLM) ---
    (
        "blip-2",
        {
            "vision": _TINY_VISION,
            "image_token_id": 50265,
            "num_query_tokens": 4,
            "qformer_hidden_size": 32,
            "qformer_num_hidden_layers": 1,
            "qformer_num_attention_heads": 2,
            "qformer_intermediate_size": 64,
        },
        True,
    ),
    # --- DeepSeek-VL-V2 (SAM ViT + Qwen2 encoder + MoE decoder) ---
    (
        "deepseek_vl_v2",
        {
            "vision": _TINY_VISION,
            "image_token_id": 32000,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "moe_layer_frequency": 1,
            "first_k_dense_replace": 0,
            "n_shared_experts": 1,
        },
        True,
    ),
    # --- Mllama (cross-attention VL, requires MllamaConfig) ---
    (
        "mllama",
        {
            "_config_cls": MllamaConfig,
            "num_hidden_layers": 3,
            "vision": _TINY_VISION,
            "image_token_id": 32000,
            "cross_attention_layers": [1],
        },
        True,
    ),
    # --- Qwen VL family (packed-attention, MRoPE) ---
    (
        "qwen2_vl",
        {
            "vision": _TINY_QWEN_VL_VISION,
            "image_token_id": 32000,
            "temporal_patch_size": 2,
            "mrope_section": [16, 24, 24],
        },
        True,
    ),
    (
        "qwen2_5_vl",
        {
            "vision": _TINY_QWEN_VL_VISION,
            "image_token_id": 32000,
            "temporal_patch_size": 2,
            "mrope_section": [16, 24, 24],
        },
        False,
    ),
    (
        "qwen3_vl",
        {
            "vision": _TINY_QWEN3_VL_VISION,
            "image_token_id": 32000,
            "temporal_patch_size": 2,
            "mrope_section": [16, 24, 24],
            "attn_qk_norm": True,
        },
        True,
    ),
    (
        "qwen3_vl_single",
        {
            "vision": _TINY_QWEN3_VL_VISION,
            "image_token_id": 32000,
            "temporal_patch_size": 2,
            "mrope_section": [16, 24, 24],
            "attn_qk_norm": True,
        },
        False,
    ),
    (
        "qwen3_5_vl",
        {
            "vision": _TINY_QWEN3_VL_VISION,
            "image_token_id": 32000,
            "temporal_patch_size": 2,
            "mrope_section": [16, 24, 24],
            "attn_qk_norm": True,
        },
        True,
    ),
]


# ---------------------------------------------------------------------------
# Speech / TTS / Codec configs
# ---------------------------------------------------------------------------
SPEECH_CONFIGS: list[tuple[str, dict, bool]] = [
    # --- Whisper (speech-to-text, encoder-decoder) ---
    (
        "whisper",
        {
            "_config_cls": WhisperConfig,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            "tie_word_embeddings": True,
            "encoder_layers": TINY_LAYERS,
            "encoder_attention_heads": TINY_HEADS,
            "encoder_ffn_dim": TINY_INTERMEDIATE,
            "num_mel_bins": 16,
            "max_source_positions": 100,
            "max_target_positions": 50,
            "scale_embedding": True,
        },
        True,
    ),
    # --- Qwen3-ASR (speech-language, 3-model split) ---
    (
        "qwen3_asr",
        {
            "attn_qk_norm": True,
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
            "audio": AudioConfig(
                d_model=64,
                encoder_layers=2,
                encoder_attention_heads=4,
                encoder_ffn_dim=128,
                num_mel_bins=128,
                max_source_positions=256,
                downsample_hidden_size=32,
                output_dim=64,
                audio_token_id=100,
            ),
        },
        True,
    ),
    # --- Qwen3-ForcedAligner (speech-language, same class as ASR) ---
    (
        "qwen3_forced_aligner",
        {
            "attn_qk_norm": True,
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
            "audio": AudioConfig(
                d_model=64,
                encoder_layers=2,
                encoder_attention_heads=4,
                encoder_ffn_dim=128,
                num_mel_bins=128,
                max_source_positions=256,
                downsample_hidden_size=32,
                output_dim=64,
                audio_token_id=100,
                classify_num=10,
            ),
        },
        False,
    ),
    # --- Qwen3-TTS Codec Tokenizer (codec, 2-model split) ---
    (
        "qwen3_tts_tokenizer_12hz",
        {
            "codec_decoder": CodecDecoderConfig(
                codebook_dim=32,
                codebook_size=64,
                latent_dim=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=8,
                rms_norm_eps=1e-5,
                rope_theta=10000.0,
                max_position_embeddings=128,
                decoder_dim=96,
                num_quantizers=4,
                upsample_rates=[2, 2, 2, 2],
                upsampling_ratios=[2, 2],
            ),
            "codec_encoder": CodecEncoderConfig(
                codebook_dim=16,
                codebook_size=64,
                hidden_size=32,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=8,
                rope_theta=10000.0,
                max_position_embeddings=128,
                num_quantizers=8,
                num_semantic_quantizers=1,
            ),
        },
        True,
    ),
]


# ---------------------------------------------------------------------------
# Aggregate lists
# ---------------------------------------------------------------------------
ALL_CONFIGS: list[tuple[str, dict, bool]] = (
    CAUSAL_LM_CONFIGS
    + ENCODER_CONFIGS
    + SEQ2SEQ_CONFIGS
    + VISION_CONFIGS
    + DETECTION_CONFIGS
    + SSM_CONFIGS
    + VL_CONFIGS
    + SPEECH_CONFIGS
)

# Model types explicitly declared in configs above (may have duplicates —
# a model_type can appear more than once with different overrides).
_EXPLICIT_MODEL_TYPES: set[str] = {mt for mt, _, _ in ALL_CONFIGS}

# Internal aliases removed from test configs — they are still registered in
# the registry but should not appear in any test parametrization.  Their real
# HF model_type counterpart (or the underlying model class) is already tested.
_EXCLUDED_ALIASES: set[str] = {
    "qwen3_5_vl_text",  # VL text decoder; real type is qwen3_5_text
    "qwen3_omni_moe",  # VL MoE; no HF AutoModelForCausalLM support
    "qwen3_vl_moe",  # VL MoE; no HF AutoModelForCausalLM support
    "glm4v_moe_text",  # VL MoE text; no HF AutoModelForCausalLM support
    "glm4v_text",  # VL text; GLM architecture incompatible with CausalLMModel
    "deepseek_v2_moe",  # our custom alias; real type is deepseek_v2
    # VL text-only submodels (tested via VL parent model)
    "qwen2_vl_text",
    "qwen2_5_vl_text",
    "qwen3_vl_text",
    # Duplicate VL alias
    "qwen3_5",  # same as qwen3_5_vl (Qwen35VL3ModelCausalLMModel)
}


# ---------------------------------------------------------------------------
# Auto-generated entries for registered model types not covered above
# ---------------------------------------------------------------------------
def _auto_generated_configs() -> list[tuple[str, dict, bool]]:
    """Return (model_type, {}, False) for registered types without explicit entries.

    This ensures new registrations get basic graph-build coverage
    automatically.

    Only model types with the ``text-generation`` or
    ``hybrid-text-generation`` task are auto-generated, since other tasks
    (vision-language, speech, diffusion, etc.) require specialised configs
    that cannot be guessed.
    """
    try:
        from mobius._config_resolver import _default_task_for_model
        from mobius._registry import registry
    except Exception:
        return []

    auto_tasks = {"text-generation", "hybrid-text-generation"}
    auto: list[tuple[str, dict, bool]] = []
    for model_type in sorted(registry.architectures()):
        if model_type in _EXPLICIT_MODEL_TYPES:
            continue
        if model_type in _EXCLUDED_ALIASES:
            continue
        task = _default_task_for_model(model_type)
        if task in auto_tasks:
            auto.append((model_type, {}, False))
    return auto


AUTO_GENERATED_CONFIGS: list[tuple[str, dict, bool]] = _auto_generated_configs()

ALL_CAUSAL_LM_CONFIGS: list[tuple[str, dict, bool]] = (
    CAUSAL_LM_CONFIGS + AUTO_GENERATED_CONFIGS
)

FAST_CAUSAL_LM_CONFIGS: list[tuple[str, dict]] = [
    (mt, ov) for mt, ov, rep in CAUSAL_LM_CONFIGS if rep
]
FAST_ENCODER_CONFIGS: list[tuple[str, dict]] = [
    (mt, ov) for mt, ov, rep in ENCODER_CONFIGS if rep
]
FAST_SEQ2SEQ_CONFIGS: list[tuple[str, dict]] = [
    (mt, ov) for mt, ov, rep in SEQ2SEQ_CONFIGS if rep
]
FAST_VISION_CONFIGS: list[tuple[str, dict]] = [
    (mt, ov) for mt, ov, rep in VISION_CONFIGS if rep
]
FAST_DETECTION_CONFIGS: list[tuple[str, dict]] = [
    (mt, ov) for mt, ov, rep in DETECTION_CONFIGS if rep
]
FAST_SSM_CONFIGS: list[tuple[str, dict]] = [(mt, ov) for mt, ov, rep in SSM_CONFIGS if rep]
FAST_VL_CONFIGS: list[tuple[str, dict]] = [(mt, ov) for mt, ov, rep in VL_CONFIGS if rep]
FAST_SPEECH_CONFIGS: list[tuple[str, dict]] = [
    (mt, ov) for mt, ov, rep in SPEECH_CONFIGS if rep
]
