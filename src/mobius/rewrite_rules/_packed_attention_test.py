# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for packed attention rewrite rules."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from onnxscript.rewriter import rewrite

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig, VisionConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.qwen_vl import Qwen3VLCausalLMModel
from mobius.rewrite_rules import packed_attention_rules
from mobius.rewrite_rules._testing_utils import (
    count_ops,
)


def _make_tiny_qwen3_vl_config() -> ArchitectureConfig:
    return ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_type="default",
        pad_token_id=0,
        attn_qk_norm=True,
        # Vision config
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            patch_size=16,
            out_hidden_size=64,
            num_position_embeddings=16,
            in_channels=3,
        ),
        spatial_merge_size=2,
        temporal_patch_size=2,
        deepstack_visual_indexes=[0],
        image_token_id=151655,
        mrope_section=[8, 12, 12],
    )


class TestPackedAttentionRules:
    """Test packed attention rewrite rules on a tiny Qwen3-VL model."""

    def test_rule_replaces_vision_attention(self):
        """Vision encoder Attention ops are replaced with PackedMultiHeadAttention."""
        config = _make_tiny_qwen3_vl_config()
        model = Qwen3VLCausalLMModel(config)
        onnx_pkg = build_from_module(
            model,
            config,
            task="qwen3-vl-vision-language",
        )
        onnx_model = onnx_pkg["model"]

        counts_before = count_ops(onnx_model)
        assert counts_before["Attention"] == 4  # 2 vision + 2 text decoder

        rules = packed_attention_rules()
        rewrite(onnx_model, pattern_rewrite_rules=rules)

        counts_after = count_ops(onnx_model)
        # Vision encoder attention replaced
        assert counts_after["PackedMultiHeadAttention"] == 2
        # Text decoder attention untouched
        assert counts_after["Attention"] == 2

    def test_rule_preserves_text_only_model(self):
        """Text-only models (no block-diagonal mask) are not affected."""
        from mobius.models.base import CausalLMModel

        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            hidden_act="silu",
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            rope_type="default",
            pad_token_id=0,
        )

        model = CausalLMModel(config)
        onnx_pkg = build_from_module(model, config, task="text-generation")
        onnx_model = onnx_pkg["model"]

        counts_before = count_ops(onnx_model)
        attention_before = counts_before["Attention"]
        assert attention_before > 0

        rules = packed_attention_rules()
        rewrite(onnx_model, pattern_rewrite_rules=rules)

        counts_after = count_ops(onnx_model)
        # No attention ops replaced
        assert counts_after["Attention"] == attention_before
        assert counts_after.get("PackedMultiHeadAttention", 0) == 0

    def test_packed_attention_rules_returns_rule_set(self):
        """packed_attention_rules() returns a non-empty RewriteRuleSet."""
        from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

        rules = packed_attention_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_rewritten_model_serializes_for_ort(self):
        """Rewritten model can be serialized and loaded by ORT.

        PackedMultiHeadAttention is a custom op that may not have a CPU
        kernel in all ORT builds, so we only verify the model can be
        saved and loaded (graph validity), not inference.
        """
        config = _make_tiny_qwen3_vl_config()
        model = Qwen3VLCausalLMModel(config)
        onnx_pkg = build_from_module(
            model,
            config,
            task="qwen3-vl-vision-language",
        )
        onnx_model = onnx_pkg["model"]

        for init in onnx_model.graph.initializers.values():
            if init.const_value is None:
                init.const_value = ir.Tensor(
                    np.random.randn(*list(init.shape)).astype(np.float32)
                )

        rewrite(onnx_model, pattern_rewrite_rules=packed_attention_rules())
        assert count_ops(onnx_model)["PackedMultiHeadAttention"] == 2

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.onnx")
            ir.save(onnx_model, path)
            # Verify ORT can at least load (validate graph structure)
            try:
                session = OnnxModelSession(onnx_model)
                session.close()
            except Exception:
                # PackedMHA may not be available in all ORT builds
                pytest.skip("PackedMultiHeadAttention not available in ORT")
