# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for quantized (GPTQ/AWQ) model pipelines.

These tests build a tiny quantized Llama model, populate it with synthetic
weights, and run ORT inference to verify the full pipeline works:
graph construction → weight loading → MatMulNBits execution.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig, QuantizationConfig
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius._weight_loading import apply_weights

# Tiny model dimensions — small enough for fast tests, large enough
# to exercise quantization (K must be divisible by group_size).
HIDDEN = 64
INTERMEDIATE = 128
HEADS = 4
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS  # 16
LAYERS = 1
VOCAB = 256
GROUP_SIZE = 32
BITS = 4


def _make_quantized_config(quant_method: str, sym: bool = False) -> ArchitectureConfig:
    qc = QuantizationConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        quant_method=quant_method,
        sym=sym,
    )
    return ArchitectureConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_hidden_layers=LAYERS,
        vocab_size=VOCAB,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        pad_token_id=0,
        quantization=qc,
    )


def _generate_synthetic_weights(
    model: ir.Model,
) -> dict[str, torch.Tensor]:
    """Create random tensors matching every initializer in the model.

    Uses the initializer's declared shape and dtype.  Constants (names
    starting with ``const_``) and RoPE caches are skipped — they are
    already set during graph construction.
    """
    onnx_to_torch = {
        ir.DataType.FLOAT: torch.float32,
        ir.DataType.FLOAT16: torch.float16,
        ir.DataType.BFLOAT16: torch.bfloat16,
        ir.DataType.INT64: torch.int64,
        ir.DataType.INT32: torch.int32,
        ir.DataType.INT8: torch.int8,
        ir.DataType.UINT8: torch.uint8,
    }
    state_dict: dict[str, torch.Tensor] = {}
    for name, init in model.graph.initializers.items():
        # Skip graph-construction constants and RoPE caches
        if name.startswith("const_") or "cache" in name:
            continue
        shape = [int(d) for d in init.shape]
        dtype = onnx_to_torch.get(init.dtype, torch.float32)
        if dtype in (torch.uint8, torch.int8, torch.int32, torch.int64):
            state_dict[name] = torch.randint(0, 15, shape, dtype=dtype)
        else:
            state_dict[name] = torch.randn(shape, dtype=dtype) * 0.02
    return state_dict


def _make_prefill_feeds(
    config: ArchitectureConfig,
    seq_len: int = 4,
) -> dict[str, np.ndarray]:
    """Build input feeds for a prefill (first-pass) forward."""
    input_ids = np.random.randint(1, VOCAB, (1, seq_len)).astype(np.int64)
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    feeds: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    # Empty KV cache for prefill
    for i in range(config.num_hidden_layers):
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )
    return feeds


@pytest.mark.integration
class TestGptqEndToEnd:
    """Full pipeline: build → load weights → ORT inference for GPTQ."""

    def test_gptq_asymmetric_inference(self):
        """GPTQ asymmetric (has zero points) produces valid logits."""
        config = _make_quantized_config("gptq", sym=False)
        model_cls = registry.get("llama")
        module = model_cls(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        state_dict = _generate_synthetic_weights(model)
        apply_weights(model, state_dict)

        session = OnnxModelSession(model)
        feeds = _make_prefill_feeds(config)
        outputs = session.run(feeds)
        session.close()

        logits = outputs["logits"]
        # Shape: [batch=1, seq_len, vocab_size]
        assert logits.shape == (1, 4, VOCAB)
        assert np.isfinite(logits).all(), "Logits contain NaN/Inf"
        # Softmax should sum to ~1 along vocab axis
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(
            probs.sum(axis=-1),
            1.0,
            atol=1e-5,
            err_msg="Softmax of logits should sum to 1",
        )

    def test_gptq_symmetric_inference(self):
        """GPTQ symmetric (no zero points) produces valid logits."""
        config = _make_quantized_config("gptq", sym=True)
        model_cls = registry.get("llama")
        module = model_cls(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        state_dict = _generate_synthetic_weights(model)
        apply_weights(model, state_dict)

        session = OnnxModelSession(model)
        feeds = _make_prefill_feeds(config)
        outputs = session.run(feeds)
        session.close()

        logits = outputs["logits"]
        assert logits.shape == (1, 4, VOCAB)
        assert np.isfinite(logits).all(), "Logits contain NaN/Inf"

    def test_gptq_preprocess_weights_pipeline(self):
        """Full HF-format GPTQ → preprocess → load → infer pipeline."""
        config = _make_quantized_config("gptq", sym=False)
        model_cls = registry.get("llama")
        module = model_cls(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        # Create HF-format GPTQ state_dict and preprocess
        hf_state_dict = _make_hf_gptq_state_dict(config)
        processed = module.preprocess_weights(hf_state_dict)

        apply_weights(model, processed)

        session = OnnxModelSession(model)
        feeds = _make_prefill_feeds(config)
        outputs = session.run(feeds)
        session.close()

        logits = outputs["logits"]
        assert logits.shape == (1, 4, VOCAB)
        assert np.isfinite(logits).all(), "Logits contain NaN/Inf"


@pytest.mark.integration
class TestAwqEndToEnd:
    """Full pipeline: build → load weights → ORT inference for AWQ."""

    def test_awq_inference(self):
        """AWQ quantized model produces valid logits."""
        config = _make_quantized_config("awq", sym=False)
        model_cls = registry.get("llama")
        module = model_cls(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        state_dict = _generate_synthetic_weights(model)
        apply_weights(model, state_dict)

        session = OnnxModelSession(model)
        feeds = _make_prefill_feeds(config)
        outputs = session.run(feeds)
        session.close()

        logits = outputs["logits"]
        assert logits.shape == (1, 4, VOCAB)
        assert np.isfinite(logits).all(), "Logits contain NaN/Inf"

    def test_awq_preprocess_weights_pipeline(self):
        """Full HF-format AWQ → preprocess → load → infer pipeline."""
        config = _make_quantized_config("awq", sym=False)
        model_cls = registry.get("llama")
        module = model_cls(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        hf_state_dict = _make_hf_awq_state_dict(config)
        processed = module.preprocess_weights(hf_state_dict)

        apply_weights(model, processed)

        session = OnnxModelSession(model)
        feeds = _make_prefill_feeds(config)
        outputs = session.run(feeds)
        session.close()

        logits = outputs["logits"]
        assert logits.shape == (1, 4, VOCAB)
        assert np.isfinite(logits).all(), "Logits contain NaN/Inf"

    def test_awq_zero_point_offset_applied(self):
        """Verify the AWQ -1 zero-point offset is correctly applied E2E.

        AWQ stores zero points with an implicit +1 offset. The pipeline
        must subtract 1 so MatMulNBits receives raw values.  This test
        sets all qzeros bytes to 0x05 (=5) and verifies the resulting
        zero_points are 4 (5 - 1), then runs inference to confirm the
        model works end-to-end with these weights.
        """
        config = _make_quantized_config("awq", sym=False)
        model_cls = registry.get("llama")
        module = model_cls(config)

        # Create AWQ weights with known zero-point values
        hf_state_dict = _make_hf_awq_state_dict(config)
        for key in list(hf_state_dict.keys()):
            if key.endswith(".qzeros"):
                hf_state_dict[key] = torch.full_like(hf_state_dict[key], 0x05050505)

        processed = module.preprocess_weights(hf_state_dict)

        # Verify zero points are 5 - 1 = 4
        zp_keys = [k for k in processed if k.endswith(".zero_points")]
        assert len(zp_keys) > 0, "No zero_points found after preprocessing"
        for key in zp_keys:
            zp = processed[key]
            assert zp.dtype == torch.uint8, f"{key}: expected uint8"
            assert (zp == 4).all(), (
                f"{key}: expected all zero_points == 4 (5 - 1), "
                f"got unique values {zp.unique().tolist()}"
            )

        # Full E2E: build model, apply weights, run inference
        pkg = build_from_module(module, config)
        model = pkg["model"]
        apply_weights(model, processed)

        session = OnnxModelSession(model)
        feeds = _make_prefill_feeds(config)
        outputs = session.run(feeds)
        session.close()

        logits = outputs["logits"]
        assert logits.shape == (1, 4, VOCAB)
        assert np.isfinite(logits).all(), "Logits contain NaN/Inf"


# ---------------------------------------------------------------------------
# Helpers: synthetic HF-format GPTQ/AWQ state dicts
# ---------------------------------------------------------------------------

# Projection names for a single decoder layer
_ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]
_MLP_PROJS = ["gate_proj", "up_proj", "down_proj"]


def _packed_dim(k: int, bits: int) -> int:
    """Number of int32 elements to pack k values at the given bit-width."""
    return k * bits // 32


def _make_hf_gptq_state_dict(
    config: ArchitectureConfig,
) -> dict[str, torch.Tensor]:
    """Build a synthetic HF GPTQ state dict with correct shapes.

    GPTQ HF layout per linear layer:
      qweight: [K_packed, N] int32
      qzeros:  [n_groups_packed, N] int32
      scales:  [n_groups, N] float32
      g_idx:   [K] int32 (trivial: arange)
    """
    bits = config.quantization.bits
    group_size = config.quantization.group_size
    sd: dict[str, torch.Tensor] = {}

    # Non-quantized params
    sd["model.embed_tokens.weight"] = torch.randn(config.vocab_size, config.hidden_size)
    sd["lm_head.weight"] = torch.randn(config.vocab_size, config.hidden_size)
    sd["model.norm.weight"] = torch.randn(config.hidden_size)

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        sd[f"{prefix}.input_layernorm.weight"] = torch.randn(config.hidden_size)
        sd[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(config.hidden_size)

        # Build quantized projections
        proj_shapes: dict[str, tuple[int, int]] = {}
        for proj in _ATTN_PROJS:
            if proj in ("k_proj", "v_proj"):
                n = config.num_key_value_heads * config.head_dim
            else:
                n = config.hidden_size
            k = config.hidden_size
            if proj == "o_proj":
                k = config.hidden_size
            proj_shapes[f"{prefix}.self_attn.{proj}"] = (k, n)

        for proj in _MLP_PROJS:
            if proj == "down_proj":
                k, n = config.intermediate_size, config.hidden_size
            else:
                k, n = config.hidden_size, config.intermediate_size
            proj_shapes[f"{prefix}.mlp.{proj}"] = (k, n)

        for name, (k, n) in proj_shapes.items():
            k_packed = _packed_dim(k, bits)
            n_groups = k // group_size
            n_groups_packed = _packed_dim(n_groups, bits)
            # Ensure at least 1 for the packed dimension
            n_groups_packed = max(1, n_groups_packed)

            sd[f"{name}.qweight"] = torch.randint(
                0, 2**31 - 1, (k_packed, n), dtype=torch.int32
            )
            sd[f"{name}.qzeros"] = torch.randint(
                0, 2**31 - 1, (n_groups_packed, n), dtype=torch.int32
            )
            sd[f"{name}.scales"] = torch.randn(n_groups, n)
            # Trivial g_idx (no desc_act reordering)
            sd[f"{name}.g_idx"] = torch.arange(k, dtype=torch.int32)

    return sd


def _make_hf_awq_state_dict(
    config: ArchitectureConfig,
) -> dict[str, torch.Tensor]:
    """Build a synthetic HF AWQ state dict with correct shapes.

    AWQ uses the same key names as GPTQ but without g_idx.
    Zero points have an implicit +1 offset.
    """
    bits = config.quantization.bits
    group_size = config.quantization.group_size
    sd: dict[str, torch.Tensor] = {}

    sd["model.embed_tokens.weight"] = torch.randn(config.vocab_size, config.hidden_size)
    sd["lm_head.weight"] = torch.randn(config.vocab_size, config.hidden_size)
    sd["model.norm.weight"] = torch.randn(config.hidden_size)

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        sd[f"{prefix}.input_layernorm.weight"] = torch.randn(config.hidden_size)
        sd[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(config.hidden_size)

        proj_shapes: dict[str, tuple[int, int]] = {}
        for proj in _ATTN_PROJS:
            if proj in ("k_proj", "v_proj"):
                n = config.num_key_value_heads * config.head_dim
            else:
                n = config.hidden_size
            proj_shapes[f"{prefix}.self_attn.{proj}"] = (config.hidden_size, n)

        for proj in _MLP_PROJS:
            if proj == "down_proj":
                k, n = config.intermediate_size, config.hidden_size
            else:
                k, n = config.hidden_size, config.intermediate_size
            proj_shapes[f"{prefix}.mlp.{proj}"] = (k, n)

        for name, (k, n) in proj_shapes.items():
            k_packed = _packed_dim(k, bits)
            n_groups = k // group_size
            n_groups_packed = max(1, _packed_dim(n_groups, bits))

            sd[f"{name}.qweight"] = torch.randint(
                0, 2**31 - 1, (k_packed, n), dtype=torch.int32
            )
            # AWQ zero points have +1 offset; use values >= 1
            sd[f"{name}.qzeros"] = torch.randint(
                1, 2**31 - 1, (n_groups_packed, n), dtype=torch.int32
            )
            sd[f"{name}.scales"] = torch.randn(n_groups, n)

    return sd
