# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for T5 model — relative position bias and weight renaming."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mobius.models.t5 import _rename_t5_weight


class TestRelativePositionBucket:
    """Test T5 log-linear bucket computation against reference NumPy impl."""

    @staticmethod
    def _np_relative_position_bucket(
        relative_position: np.ndarray,
        *,
        bidirectional: bool,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> np.ndarray:
        """Reference NumPy implementation matching HuggingFace T5."""
        relative_buckets = np.zeros_like(relative_position, dtype=np.int64)
        if bidirectional:
            half = num_buckets // 2
            relative_buckets += (relative_position > 0).astype(np.int64) * half
            relative_position = np.abs(relative_position)
            effective_buckets = half
        else:
            relative_position = -np.minimum(
                relative_position, np.zeros_like(relative_position)
            )
            effective_buckets = num_buckets

        max_exact = effective_buckets // 2
        is_small = relative_position < max_exact
        rel_clamped = np.maximum(relative_position.astype(np.float32), 1.0)
        log_ratio = np.log(rel_clamped / max_exact)
        log_scale = math.log(max_distance / max_exact)
        bucket_float = max_exact + log_ratio * (effective_buckets - max_exact) / log_scale
        large_bucket = np.minimum(bucket_float.astype(np.int64), effective_buckets - 1)
        final_offset = np.where(is_small, relative_position, large_bucket)
        relative_buckets += final_offset
        return relative_buckets

    def test_bidirectional_4x4(self):
        """Encoder (bidirectional) bucket indices for 4x4."""
        ctx = np.arange(4)[:, None]
        mem = np.arange(4)[None, :]
        rel = mem - ctx
        buckets = self._np_relative_position_bucket(rel, bidirectional=True)
        expected = np.array(
            [
                [0, 17, 18, 19],
                [1, 0, 17, 18],
                [2, 1, 0, 17],
                [3, 2, 1, 0],
            ]
        )
        np.testing.assert_array_equal(buckets, expected)

    def test_unidirectional_4x4(self):
        """Decoder (unidirectional) bucket indices for 4x4."""
        ctx = np.arange(4)[:, None]
        mem = np.arange(4)[None, :]
        rel = mem - ctx
        buckets = self._np_relative_position_bucket(rel, bidirectional=False)
        expected = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [2, 1, 0, 0],
                [3, 2, 1, 0],
            ]
        )
        np.testing.assert_array_equal(buckets, expected)

    def test_decode_step_offset(self):
        """Decoder position bias for single-token decode at offset 3."""
        # query position = [3], key positions = [0, 1, 2, 3]
        ctx = np.array([[3]])
        mem = np.arange(4)[None, :]
        rel = mem - ctx  # [[-3, -2, -1, 0]]
        buckets = self._np_relative_position_bucket(rel, bidirectional=False)
        expected = np.array([[3, 2, 1, 0]])
        np.testing.assert_array_equal(buckets, expected)


class TestT5WeightRename:
    """Test weight name mapping from HuggingFace to ONNX."""

    def test_relative_attention_bias_lifted_to_encoder(self):
        """Block 0's relative_attention_bias maps to encoder level."""
        hf = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        assert _rename_t5_weight(hf) == "encoder.relative_attention_bias.weight"

    def test_relative_attention_bias_lifted_to_decoder(self):
        """Block 0's relative_attention_bias maps to decoder level."""
        hf = "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        assert _rename_t5_weight(hf) == "decoder.relative_attention_bias.weight"

    def test_self_attention_projections(self):
        """Self-attention projections rename correctly."""
        hf = "encoder.block.2.layer.0.SelfAttention.q.weight"
        assert _rename_t5_weight(hf) == "encoder.block.2.self_attn.q_proj.weight"

    def test_cross_attention_projections(self):
        """Cross-attention projections rename correctly."""
        hf = "decoder.block.1.layer.1.EncDecAttention.k.weight"
        assert _rename_t5_weight(hf) == "decoder.block.1.cross_attn.k_proj.weight"

    def test_ffn_rename_encoder(self):
        """Encoder FFN weights rename correctly."""
        hf = "encoder.block.3.layer.1.DenseReluDense.wi.weight"
        assert _rename_t5_weight(hf) == "encoder.block.3.ffn.wi.weight"

    def test_ffn_rename_decoder(self):
        """Decoder FFN weights rename correctly."""
        hf = "decoder.block.0.layer.2.DenseReluDense.wo.weight"
        assert _rename_t5_weight(hf) == "decoder.block.0.ffn.wo.weight"

    @pytest.mark.parametrize(
        "hf_name",
        [
            "encoder.final_layer_norm.weight",
            "decoder.final_layer_norm.weight",
        ],
    )
    def test_final_layer_norm_unchanged(self, hf_name):
        """Final layer norms keep their names."""
        assert _rename_t5_weight(hf_name) == hf_name
