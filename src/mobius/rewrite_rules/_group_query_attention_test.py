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
from mobius.rewrite_rules import group_query_attention_rules
from mobius.rewrite_rules._testing_utils import (
    count_ops,
    fill_random_weights,
    make_prefill_feeds,
)


class TestGroupQueryAttentionRules:
    def test_rules_returns_rule_set(self):
        rules = group_query_attention_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_replaces_attention_with_gqa(self):
        pkg = build("Qwen/Qwen3-0.6B", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)
        assert counts_before["Attention"] == 28

        rewrite(model, pattern_rewrite_rules=group_query_attention_rules())

        counts_after = count_ops(model)
        assert counts_after.get("Attention", 0) == 0
        assert counts_after["GroupQueryAttention"] == 28

    def test_absorbs_rotary_embedding(self):
        """RotaryEmbedding ops are absorbed into GQA with do_rotary=1."""
        pkg = build("Qwen/Qwen3-0.6B", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)
        assert counts_before["RotaryEmbedding"] == 56

        rewrite(model, pattern_rewrite_rules=group_query_attention_rules())

        counts_after = count_ops(model)
        assert counts_after.get("RotaryEmbedding", 0) == 0
        assert counts_after["GroupQueryAttention"] == 28

    def test_absorbs_rotary_without_qk_norm(self):
        """Models without QK norm also get rotary absorbed."""
        pkg = build("HuggingFaceTB/SmolLM2-135M-Instruct", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)
        assert counts_before["RotaryEmbedding"] == 60

        rewrite(model, pattern_rewrite_rules=group_query_attention_rules())

        counts_after = count_ops(model)
        assert counts_after.get("RotaryEmbedding", 0) == 0
        assert counts_after["GroupQueryAttention"] == 30

    def test_preserves_non_matching_model(self):
        """Vision encoder attention (no KV cache) is not replaced."""
        pkg = build(
            "Qwen/Qwen3-VL-2B-Instruct",
            load_weights=False,
        )
        # Vision model: no KV cache, Attention should remain untouched
        vision = pkg["vision"]
        vision_attn_before = count_ops(vision).get("Attention", 0)
        assert vision_attn_before == 24

        rewrite(
            vision,
            pattern_rewrite_rules=group_query_attention_rules(),
        )
        vision_counts = count_ops(vision)
        assert vision_counts.get("Attention", 0) == vision_attn_before
        assert vision_counts.get("GroupQueryAttention", 0) == 0

    def test_rewritten_model_runs_with_ort(self):
        """GQA-rewritten model can be serialized and run with ORT."""
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
            attn_qk_norm=True,
        )
        model = registry.get("qwen3")(config)
        pkg = build_from_module(model, config)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())
        assert count_ops(m)["GroupQueryAttention"] == 2

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()

    def test_combined_gqa_and_skip_norm_runs_with_ort(self):
        """Applying GQA + SkipNorm together produces a valid ORT model."""
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
            attn_qk_norm=True,
        )
        model = registry.get("qwen3")(config)
        pkg = build_from_module(model, config)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())
        rewrite(m, pattern_rewrite_rules=skip_norm_rules())

        counts = count_ops(m)
        assert counts["GroupQueryAttention"] == 2
        assert counts["SkipSimplifiedLayerNormalization"] > 0
        assert counts.get("Attention", 0) == 0

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()
