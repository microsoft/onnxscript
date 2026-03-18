# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from mobius._testing.code_paths import (
    detect_code_paths,
    detect_code_paths_from_config,
    get_all_code_path_labels,
    get_indicator_by_label,
)


class TestDetectCodePaths:
    def test_empty_config(self):
        assert detect_code_paths({}) == set()

    def test_moe_detected(self):
        result = detect_code_paths({"num_local_experts": 8, "num_experts_per_tok": 2})
        assert "moe" in result

    def test_moe_zero_not_detected(self):
        result = detect_code_paths({"num_local_experts": 0})
        assert "moe" not in result

    def test_sliding_window(self):
        result = detect_code_paths({"sliding_window": 4096})
        assert "sliding_window" in result

    def test_sliding_window_none_not_detected(self):
        result = detect_code_paths({"sliding_window": None})
        assert "sliding_window" not in result

    def test_linear_attention(self):
        result = detect_code_paths({"layer_types": ["full_attention", "linear_attention"]})
        assert "linear_attn" in result
        assert "full_attn" in result

    def test_mamba_hybrid(self):
        result = detect_code_paths({"layer_types": ["attention", "mamba"]})
        assert "mamba" in result

    def test_qk_norm(self):
        result = detect_code_paths({"attn_qk_norm": True})
        assert "qk_norm" in result

    def test_partial_rope(self):
        result = detect_code_paths({"partial_rotary_factor": 0.5})
        assert "partial_rope" in result

    def test_partial_rope_full_not_detected(self):
        result = detect_code_paths({"partial_rotary_factor": 1.0})
        assert "partial_rope" not in result

    def test_mla(self):
        result = detect_code_paths({"q_lora_rank": 64})
        assert "mla" in result

    def test_multiple_paths(self):
        result = detect_code_paths(
            {
                "num_local_experts": 4,
                "sliding_window": 2048,
                "attn_qk_norm": True,
            }
        )
        assert result == {"moe", "sliding_window", "qk_norm"}

    def test_from_config_object(self):
        """Test with a mock config object that has attributes."""

        class MockConfig:
            num_local_experts = 4
            sliding_window = None
            attn_qk_norm = False

        result = detect_code_paths_from_config(MockConfig())
        assert "moe" in result
        assert "sliding_window" not in result


class TestHelpers:
    def test_get_all_labels(self):
        labels = get_all_code_path_labels()
        assert isinstance(labels, list)
        assert len(labels) > 0
        assert labels == sorted(labels)
        assert "moe" in labels

    def test_get_indicator_by_label(self):
        ind = get_indicator_by_label("moe")
        assert ind is not None
        assert ind.field == "num_local_experts"
        assert "num_local_experts" in ind.example_config

    def test_get_indicator_unknown(self):
        assert get_indicator_by_label("nonexistent") is None
