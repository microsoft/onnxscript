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

from mobius._configs import (
    BambaConfig,
    DepthAnythingConfig,
    Gemma2Config,
    Gemma3nConfig,
    JambaConfig,
    Mamba2Config,
    MambaConfig,
    Sam2Config,
    SegformerConfig,
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


# ---------------------------------------------------------------------------
# Causal LM configs  (task: text-generation / hybrid-text-generation)
# ---------------------------------------------------------------------------
CAUSAL_LM_CONFIGS: list[tuple[str, dict, bool]] = [
    # === Text Generation (Llama-compatible) ===
    ("llama", {}, True),
    ("mistral", {}, False),
    ("qwen2", {}, True),
    ("cohere", {"tie_word_embeddings": True}, True),
    ("cohere2", {"tie_word_embeddings": True}, False),
    ("diffllama", {}, False),
    ("doge", {}, False),
    ("dots1", {}, False),
    ("exaone4", {}, False),
    (
        "glm",
        {"attn_qkv_bias": True, "attn_o_bias": True},
        True,
    ),
    (
        "glm4",
        {"attn_qkv_bias": True, "attn_o_bias": True},
        False,
    ),
    ("helium", {}, False),
    ("hunyuan_v1_dense", {}, False),
    ("llama4_text", {}, False),
    ("ministral", {}, False),
    ("ministral3", {}, False),
    ("nanochat", {"hidden_act": "relu2"}, True),
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
    ("ctrl", {"hidden_act": "gelu_new", "tie_word_embeddings": True}, True),
    (
        "gpt2",
        {"hidden_act": "gelu_new", "tie_word_embeddings": True},
        True,
    ),
    (
        "imagegpt",
        {"hidden_act": "gelu_new", "tie_word_embeddings": True},
        False,
    ),
    ("opt", {"hidden_act": "relu", "tie_word_embeddings": True}, True),
    (
        "xlm",
        {"hidden_act": "gelu_new", "tie_word_embeddings": True},
        False,
    ),
    # === Other Llama-compatible ===
    ("modernbert-decoder", {}, False),
    # === Text Generation (architecture-specific) ===
    (
        "gemma",
        {"attn_qkv_bias": True, "attn_o_bias": True},
        True,
    ),
    (
        "gemma2",
        {
            "_config_cls": Gemma2Config,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": 256,
        },
        True,
    ),
    (
        "shieldgemma2",
        {
            "_config_cls": Gemma2Config,
            "attn_qkv_bias": True,
            "attn_o_bias": True,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": 256,
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
    ("qwen3_5_vl_text", {"attn_qk_norm": True}, False),
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
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "qwen2_moe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "attn_qkv_bias": True,
        },
        True,
    ),
    (
        "qwen3_moe",
        {
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
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
    ("falcon", {"parallel_attn": True}, True),
    (
        "falcon_h1",
        {"alibi": True, "attn_qkv_bias": True},
        True,
    ),
    (
        "bloom",
        {"alibi": True, "attn_qkv_bias": True, "mlp_bias": True},
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
    (
        "deepseek_v2_moe",
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
        False,
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
    ("persimmon", {}, True),
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
    ("phi", {"partial_rotary_factor": 0.5}, True),
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
        "gptoss",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "gpt_oss",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "jetmoe",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    # === Additional CausalLM aliases ===
    ("apertus", {}, False),
    ("arcee", {}, False),
    ("code_llama", {}, False),
    ("codegen", {"partial_rotary_factor": 0.5}, False),
    ("csm", {}, False),
    ("evolla", {}, False),
    ("gpt_neox", {}, False),
    ("gpt_neox_japanese", {}, False),
    ("gptj", {"partial_rotary_factor": 0.25}, False),
    ("longcat_flash", {}, False),
    ("open-llama", {}, False),
    ("seed_oss", {}, False),
    (
        "glm4v_text",
        {"attn_qkv_bias": True, "attn_o_bias": True},
        False,
    ),
    ("zamba2", {}, False),
    # === Additional GPT2 aliases ===
    (
        "biogpt",
        {"hidden_act": "gelu_new", "tie_word_embeddings": True},
        False,
    ),
    (
        "gpt-sw3",
        {"hidden_act": "gelu_new", "tie_word_embeddings": True},
        False,
    ),
    (
        "gpt_bigcode",
        {"hidden_act": "gelu_new", "tie_word_embeddings": True},
        False,
    ),
    (
        "gpt_neo",
        {"hidden_act": "gelu_new", "tie_word_embeddings": True},
        False,
    ),
    (
        "openai-gpt",
        {"hidden_act": "gelu_new", "tie_word_embeddings": True},
        False,
    ),
    (
        "xglm",
        {"hidden_act": "gelu_new", "tie_word_embeddings": True},
        False,
    ),
    # === Additional Falcon aliases ===
    ("mpt", {"attn_qkv_bias": True}, False),
    # === Additional MoE aliases ===
    (
        "ernie4_5_moe",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "flex_olmo",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "glm4_moe",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "glm4v_moe_text",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "granitemoehybrid",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "granitemoeshared",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "hunyuan_v1_moe",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "minimax",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "qwen3_omni_moe",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    (
        "qwen3_vl_moe",
        {"num_local_experts": 4, "num_experts_per_tok": 2},
        False,
    ),
    # --- Variant coverage: architecture code-path variants ---
    # qwen3_next: all full-attention layers (no linear_attention/DeltaNet)
    (
        "qwen3_next",
        {
            "hidden_act": "silu",
            "layer_types": ["full_attention"] * TINY_LAYERS,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "norm_topk_prob": True,
            "attn_qk_norm": True,
        },
        False,
    ),
    # qwen3_next: all linear-attention layers (DeltaNet only, no full attn)
    (
        "qwen3_next",
        {
            "hidden_act": "silu",
            "layer_types": ["linear_attention"] * TINY_LAYERS,
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
        {"alibi": True, "attn_qkv_bias": True, "parallel_attn": True},
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
    # gemma3n_text: all full attention (no sliding window)
    (
        "gemma3n_text",
        {
            "_config_cls": Gemma3nConfig,
            "attn_qk_norm": True,
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
# Aggregate lists
# ---------------------------------------------------------------------------
ALL_CONFIGS: list[tuple[str, dict, bool]] = (
    CAUSAL_LM_CONFIGS
    + ENCODER_CONFIGS
    + SEQ2SEQ_CONFIGS
    + VISION_CONFIGS
    + DETECTION_CONFIGS
    + SSM_CONFIGS
)

# Model types explicitly declared in configs above (may have duplicates —
# a model_type can appear more than once with different overrides).
_EXPLICIT_MODEL_TYPES: set[str] = {mt for mt, _, _ in ALL_CONFIGS}


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
