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
from mobius.rewrite_rules import bias_gelu_rules
from mobius.rewrite_rules._testing_utils import (
    count_ops,
    fill_random_weights,
    make_prefill_feeds,
)


class TestBiasGeluRules:
    def test_rules_returns_rule_set(self):
        rules = bias_gelu_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_fuses_add_gelu(self):
        """GPT-2 has 12 Gelu ops preceded by Add → expect 12 BiasGelu."""
        pkg = build("openai-community/gpt2", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)
        assert counts_before["Gelu"] == 12

        rewrite(model, pattern_rewrite_rules=bias_gelu_rules())

        counts_after = count_ops(model)
        assert counts_after.get("Gelu", 0) == 0
        assert counts_after["BiasGelu"] == 12

    def test_preserves_silu_model(self):
        """Models using SiLU (not Gelu) are not affected."""
        pkg = build("Qwen/Qwen3-0.6B", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)
        assert counts_before.get("Gelu", 0) == 0

        rewrite(model, pattern_rewrite_rules=bias_gelu_rules())

        counts_after = count_ops(model)
        assert counts_after.get("BiasGelu", 0) == 0

    def test_skips_exact_gelu(self):
        """Gelu with approximate='none' should NOT be fused into BiasGelu."""
        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="gelu",  # exact gelu (approximate='none')
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
        model_cls = registry.get("gpt2")(config)
        pkg = build_from_module(model_cls, config)
        m = pkg["model"]

        counts_before = count_ops(m)
        assert counts_before["Gelu"] > 0

        rewrite(m, pattern_rewrite_rules=bias_gelu_rules())

        counts_after = count_ops(m)
        # Exact Gelu should NOT be fused — BiasGelu requires tanh approx
        assert counts_after.get("BiasGelu", 0) == 0
        assert counts_after["Gelu"] == counts_before["Gelu"]

    def test_rewritten_model_runs_with_ort(self):
        """BiasGelu-rewritten model runs with ORT."""
        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="gelu_new",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
        model = registry.get("gpt2")(config)
        pkg = build_from_module(model, config)
        m = pkg["model"]
        fill_random_weights(m)

        counts_before = count_ops(m)
        assert counts_before["Gelu"] > 0

        rewrite(m, pattern_rewrite_rules=bias_gelu_rules())
        counts_after = count_ops(m)
        assert counts_after["BiasGelu"] > 0
        assert counts_after.get("Gelu", 0) == 0

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()

    def test_combined_skip_layer_norm_and_bias_gelu(self):
        """Applying SkipLayerNorm + BiasGelu together produces valid model."""
        from mobius.rewrite_rules import skip_layer_norm_rules

        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="gelu_new",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
        model = registry.get("gpt2")(config)
        pkg = build_from_module(model, config)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=skip_layer_norm_rules())
        rewrite(m, pattern_rewrite_rules=bias_gelu_rules())

        counts = count_ops(m)
        assert counts["SkipLayerNormalization"] > 0
        assert counts["BiasGelu"] > 0
        assert counts.get("Gelu", 0) == 0

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()
