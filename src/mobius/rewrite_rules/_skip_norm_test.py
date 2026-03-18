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
from mobius.rewrite_rules import skip_norm_rules
from mobius.rewrite_rules._testing_utils import (
    count_ops,
    fill_random_weights,
    make_prefill_feeds,
)


class TestSkipNormRules:
    def test_rules_returns_rule_set(self):
        rules = skip_norm_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_fuses_add_rmsnorm(self):
        pkg = build("Qwen/Qwen3-0.6B", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)
        assert counts_before["RMSNormalization"] == 113
        assert counts_before["Add"] == 56

        rewrite(model, pattern_rewrite_rules=skip_norm_rules())

        counts_after = count_ops(model)
        assert counts_after["SkipSimplifiedLayerNormalization"] == 55
        # Last layer's Add → final norm has only 1 consumer, not fused
        assert counts_after["Add"] == 1
        # Remaining RMSNorms: first input_layernorm(1) + final norm(1) + QK norms(56)
        assert counts_after["RMSNormalization"] == 58

    def test_preserves_single_consumer_add(self):
        """Add nodes with only one consumer are not fused."""
        pkg = build("Qwen/Qwen3-0.6B", load_weights=False)
        model = pkg["model"]
        count_ops(model)  # baseline

        rewrite(model, pattern_rewrite_rules=skip_norm_rules())

        counts_after = count_ops(model)
        # The 1 remaining Add is the last layer's residual → final norm
        assert counts_after["Add"] == 1

    def test_rewritten_model_runs_with_ort(self):
        """SkipNorm-rewritten model can be serialized and run with ORT."""
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
        assert count_ops(m)["SkipSimplifiedLayerNormalization"] == 3

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()
