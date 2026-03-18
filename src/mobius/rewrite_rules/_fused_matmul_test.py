# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius import build
from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.rewrite_rules import fused_matmul_rules
from mobius.rewrite_rules._testing_utils import (
    count_ops,
    fill_random_weights,
    make_prefill_feeds,
)


class TestFusedMatMulRules:
    def test_rules_returns_rule_set(self):
        rules = fused_matmul_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_fuses_transpose_matmul_in_llm(self):
        """Qwen3-0.6B has 197 Linear layers → 197 Transpose+MatMul pairs."""
        pkg = build("Qwen/Qwen3-0.6B", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)
        assert counts_before["Transpose"] == 197
        assert counts_before["MatMul"] == 197

        rewrite(model, pattern_rewrite_rules=fused_matmul_rules())

        counts_after = count_ops(model)
        assert counts_after.get("Transpose", 0) == 0
        assert counts_after.get("MatMul", 0) == 0
        assert counts_after["FusedMatMul"] == 197

    def test_preserves_non_linear_transpose(self):
        """Transpose nodes NOT feeding MatMul are preserved."""
        # GPT-2 has Transpose ops in attention (perm != [1,0])
        # and in Linear layers. Only Linear ones should be fused.
        pkg = build("openai-community/gpt2", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)

        rewrite(model, pattern_rewrite_rules=fused_matmul_rules())

        counts_after = count_ops(model)
        # All Linear Transpose+MatMul should become FusedMatMul
        assert counts_after["FusedMatMul"] > 0
        # Original MatMul count minus fused ones
        fused_count = counts_after["FusedMatMul"]
        remaining_matmul = counts_after.get("MatMul", 0)
        assert fused_count + remaining_matmul == counts_before["MatMul"]

    def test_rewritten_model_runs_with_ort(self):
        """FusedMatMul-rewritten model produces correct output with ORT."""
        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
        )
        model = registry.get("qwen2")(config)
        pkg = build_from_module(model, config)
        m = pkg["model"]
        fill_random_weights(m)

        counts_before = count_ops(m)
        assert counts_before["Transpose"] > 0
        assert counts_before["MatMul"] > 0

        rewrite(m, pattern_rewrite_rules=fused_matmul_rules())

        counts_after = count_ops(m)
        assert counts_after["FusedMatMul"] > 0
        assert counts_after.get("MatMul", 0) == 0

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()

    def test_combined_with_skip_norm(self):
        """FusedMatMul + SkipNorm rewrites compose correctly."""
        from mobius.rewrite_rules import skip_norm_rules

        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
        )
        model = registry.get("qwen2")(config)
        pkg = build_from_module(model, config)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=skip_norm_rules())
        rewrite(m, pattern_rewrite_rules=fused_matmul_rules())

        counts = count_ops(m)
        assert counts["SkipSimplifiedLayerNormalization"] > 0
        assert counts["FusedMatMul"] > 0
        assert counts.get("MatMul", 0) == 0

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()
