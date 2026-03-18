# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for rotary embeddings."""

from __future__ import annotations

import numpy as np
import pytest

from mobius._testing import create_test_builder, create_test_input, make_config
from mobius.components._rotary_embedding import (
    ChunkedMRope,
    DefaultRope,
    InterleavedMRope,
    LinearRope,
    Llama3Rope,
    LongRope,
    _get_cos_sin_cache,
    _get_default_inv_freq,
    apply_rotary_pos_emb,
    get_rotary_pos_emb,
    initialize_rope,
)


class TestInvFreq:
    def test_default_inv_freq_shape(self):
        config = make_config(head_dim=16, partial_rotary_factor=1.0)
        inv_freq = _get_default_inv_freq(config)
        assert inv_freq.shape == (8,)  # dim/2

    def test_partial_rotary_inv_freq_shape(self):
        config = make_config(head_dim=16, partial_rotary_factor=0.5)
        inv_freq = _get_default_inv_freq(config)
        assert inv_freq.shape == (4,)  # (dim * 0.5) / 2

    def test_inv_freq_values_decrease(self):
        config = make_config(head_dim=16)
        inv_freq = _get_default_inv_freq(config)
        for i in range(len(inv_freq) - 1):
            assert inv_freq[i] > inv_freq[i + 1]


class TestCosSinCache:
    def test_cache_shape(self):
        inv_freq = np.array([1.0, 0.5, 0.25, 0.125])
        cos, sin = _get_cos_sin_cache(32, inv_freq)
        assert cos.shape == (32, 4)
        assert sin.shape == (32, 4)

    def test_cache_values_bounded(self):
        inv_freq = np.array([1.0, 0.5])
        cos, sin = _get_cos_sin_cache(16, inv_freq)
        assert np.all(cos >= -1.0) and np.all(cos <= 1.0)
        assert np.all(sin >= -1.0) and np.all(sin <= 1.0)

    def test_attention_scaling(self):
        inv_freq = np.array([1.0, 0.5])
        cos1, sin1 = _get_cos_sin_cache(16, inv_freq, attention_scaling=1.0)
        cos2, sin2 = _get_cos_sin_cache(16, inv_freq, attention_scaling=2.0)
        np.testing.assert_allclose(cos2, cos1 * 2.0)
        np.testing.assert_allclose(sin2, sin1 * 2.0)


class TestRopeVariants:
    def test_default_rope_creates_caches(self):
        config = make_config()
        rope = DefaultRope(config)
        params = list(rope.parameters())
        assert len(params) == 2  # cos_cache, sin_cache

    def test_default_rope_cache_shapes(self):
        config = make_config(max_position_embeddings=64, head_dim=16)
        rope = DefaultRope(config)
        assert list(rope.cos_cache.shape) == [64, 8]
        assert list(rope.sin_cache.shape) == [64, 8]

    def test_default_rope_forward(self):
        config = make_config()
        rope = DefaultRope(config)
        builder, op, _graph = create_test_builder()
        pos_ids = create_test_input(builder, "pos_ids", [2, 4])
        result = rope(op, pos_ids)
        assert len(result) == 2  # (cos_emb, sin_emb)

    def test_linear_rope(self):
        config = make_config(rope_scaling={"factor": 2.0})
        rope = LinearRope(config)
        assert next(iter(rope.cos_cache.shape)) == config.max_position_embeddings

    def test_llama3_rope(self):
        config = make_config(
            max_position_embeddings=131072,
            original_max_position_embeddings=8192,
            rope_scaling={
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
            },
        )
        rope = Llama3Rope(config)
        assert next(iter(rope.cos_cache.shape)) == 131072

    def test_long_rope_short_only(self):
        config = make_config(
            max_position_embeddings=32,
            original_max_position_embeddings=32,
            rope_scaling={
                "short_factor": [1.0] * 8,
                "long_factor": [1.0] * 8,
            },
        )
        rope = LongRope(config)
        assert not rope.has_long_cache

    def test_long_rope_with_long_cache(self):
        config = make_config(
            max_position_embeddings=64,
            original_max_position_embeddings=32,
            rope_scaling={
                "short_factor": [1.0] * 8,
                "long_factor": [1.0] * 8,
            },
        )
        rope = LongRope(config)
        assert rope.has_long_cache
        assert next(iter(rope.cos_cache.shape)) == 96


class TestInitializeRope:
    def test_default(self):
        config = make_config(rope_type="default")
        rope = initialize_rope(config)
        assert isinstance(rope, DefaultRope)

    def test_linear(self):
        config = make_config(rope_type="linear", rope_scaling={"factor": 2.0})
        rope = initialize_rope(config)
        assert isinstance(rope, LinearRope)

    def test_llama3(self):
        config = make_config(
            rope_type="llama3",
            original_max_position_embeddings=8192,
            rope_scaling={"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0},
        )
        rope = initialize_rope(config)
        assert isinstance(rope, Llama3Rope)

    def test_longrope(self):
        config = make_config(
            rope_type="longrope",
            rope_scaling={"short_factor": [1.0] * 8, "long_factor": [1.0] * 8},
        )
        rope = initialize_rope(config)
        assert isinstance(rope, LongRope)

    def test_unsupported_raises(self):
        config = make_config(rope_type="unknown")
        with pytest.raises(ValueError, match="Unsupported rope type"):
            initialize_rope(config)

    def test_mrope_section_without_interleaved_returns_chunked(self):
        config = make_config(mrope_section=[8, 12, 12])
        rope = initialize_rope(config)
        assert isinstance(rope, ChunkedMRope)

    def test_mrope_section_with_interleaved_returns_interleaved(self):
        config = make_config(mrope_section=[11, 11, 10], mrope_interleaved=True)
        rope = initialize_rope(config)
        assert isinstance(rope, InterleavedMRope)


class TestChunkedMRope:
    def test_creates_caches_and_masks(self):
        config = make_config(head_dim=16, mrope_section=[3, 3, 2])
        rope = ChunkedMRope(config)
        param_names = [n for n, _ in rope.named_parameters()]
        assert "cos_cache" in param_names
        assert "sin_cache" in param_names
        assert "h_mask" in param_names
        assert "w_mask" in param_names

    def test_contiguous_mask_layout(self):
        # mrope_section=[3, 3, 2] with head_dim=16 → rotary_dim=8
        config = make_config(head_dim=16, mrope_section=[3, 3, 2])
        rope = ChunkedMRope(config)
        h_mask = rope.h_mask._const_value.numpy()
        w_mask = rope.w_mask._const_value.numpy()
        # H occupies indices 3,4,5 (contiguous block after T)
        assert list(h_mask) == [False, False, False, True, True, True, False, False]
        # W occupies indices 6,7 (contiguous block after H)
        assert list(w_mask) == [False, False, False, False, False, False, True, True]

    def test_forward_builds_graph(self):
        config = make_config(head_dim=16, mrope_section=[3, 3, 2])
        rope = ChunkedMRope(config)
        builder, op, graph = create_test_builder()
        pos_ids = create_test_input(builder, "pos_ids", [3, 2, 4])
        result = rope(op, pos_ids)
        assert len(result) == 2  # (cos, sin)
        assert graph.num_nodes() > 0


class TestInterleavedMRope:
    def test_creates_caches_and_masks(self):
        config = make_config(head_dim=16, mrope_section=[3, 3, 2], mrope_interleaved=True)
        rope = InterleavedMRope(config)
        param_names = [n for n, _ in rope.named_parameters()]
        assert "cos_cache" in param_names
        assert "sin_cache" in param_names
        assert "h_mask" in param_names
        assert "w_mask" in param_names

    def test_interleaved_mask_layout(self):
        # mrope_section=[3, 3, 2] with head_dim=16 → rotary_dim=8
        # H channels at stride 3 offset 1: positions 1, 4, 7  (h_length=3*3=9)
        # W channels at stride 3 offset 2: positions 2, 5     (w_length=2*3=6)
        config = make_config(head_dim=16, mrope_section=[3, 3, 2], mrope_interleaved=True)
        rope = InterleavedMRope(config)
        h_mask = rope.h_mask._const_value.numpy()
        w_mask = rope.w_mask._const_value.numpy()
        assert list(h_mask) == [False, True, False, False, True, False, False, True]
        assert list(w_mask) == [False, False, True, False, False, True, False, False]

    def test_forward_builds_graph(self):
        config = make_config(head_dim=16, mrope_section=[3, 3, 2], mrope_interleaved=True)
        rope = InterleavedMRope(config)
        builder, op, graph = create_test_builder()
        pos_ids = create_test_input(builder, "pos_ids", [3, 2, 4])
        result = rope(op, pos_ids)
        assert len(result) == 2  # (cos, sin)
        assert graph.num_nodes() > 0


class TestApplyRotaryPosEmb:
    def test_apply_rotary_pos_emb_full(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 4, 64])
        cos = create_test_input(builder, "cos", [2, 4, 8])
        sin = create_test_input(builder, "sin", [2, 4, 8])

        result = apply_rotary_pos_emb(op, x, (cos, sin), num_heads=4, rotary_embedding_dim=0)
        assert result is not None
        assert graph.num_nodes() > 0

    def test_get_rotary_pos_emb(self):
        builder, op, _graph = create_test_builder()
        pos_ids = create_test_input(builder, "pos_ids", [2, 4])
        cos_cache = create_test_input(builder, "cos_cache", [32, 8])
        sin_cache = create_test_input(builder, "sin_cache", [32, 8])

        cos, sin = get_rotary_pos_emb(op, pos_ids, cos_cache, sin_cache)
        assert cos is not None
        assert sin is not None
