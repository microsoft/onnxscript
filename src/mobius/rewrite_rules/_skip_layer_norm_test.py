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
from mobius.rewrite_rules import skip_layer_norm_rules
from mobius.rewrite_rules._testing_utils import (
    count_ops,
    fill_random_weights,
    make_prefill_feeds,
)


class TestSkipLayerNormRules:
    def test_rules_returns_rule_set(self):
        rules = skip_layer_norm_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_fuses_add_layernorm(self):
        """GPT-2 uses LayerNorm → expect Add+LN fusions."""
        pkg = build("openai-community/gpt2", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)
        assert counts_before["LayerNormalization"] == 25
        assert counts_before["Add"] == 97

        rewrite(model, pattern_rewrite_rules=skip_layer_norm_rules())

        counts_after = count_ops(model)
        assert counts_after["SkipLayerNormalization"] == 24
        # 1 remaining: final LayerNorm (Add has only 1 consumer)
        assert counts_after["LayerNormalization"] == 1

    def test_preserves_rmsnorm_model(self):
        """Models using RMSNorm (not LayerNorm) are not affected."""
        pkg = build("Qwen/Qwen3-0.6B", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)
        assert counts_before.get("LayerNormalization", 0) == 0

        rewrite(model, pattern_rewrite_rules=skip_layer_norm_rules())

        counts_after = count_ops(model)
        assert counts_after.get("SkipLayerNormalization", 0) == 0

    def test_rewritten_model_runs_with_ort(self):
        """SkipLayerNorm-rewritten model runs with ORT."""
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
        assert counts_before["LayerNormalization"] > 0

        rewrite(m, pattern_rewrite_rules=skip_layer_norm_rules())
        assert count_ops(m)["SkipLayerNormalization"] > 0

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()

    def test_fuses_bias_free_layernorm(self):
        """LayerNorm with 2 inputs (no bias) should also be fused."""
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

        # Strip bias from all LayerNorm nodes to simulate bias-free LN
        for node in m.graph:
            if node.op_type == "LayerNormalization" and len(node.inputs) == 3:
                node.resize_inputs(2)

        counts_before = count_ops(m)
        assert counts_before["LayerNormalization"] > 0

        rewrite(m, pattern_rewrite_rules=skip_layer_norm_rules())

        counts_after = count_ops(m)
        # Bias-free LN nodes that have Add with 2+ consumers should fuse
        assert counts_after.get("SkipLayerNormalization", 0) > 0
