# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the parity comparison utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mobius._testing.parity import (
    ParityReport,
    ParityResult,
    compare_golden,
    compare_synthetic,
)


class TestCompareSynthetic:
    """Tests for L3 synthetic parity comparison."""

    def test_identical_logits_pass(self):
        logits = np.random.default_rng(42).standard_normal((1, 100)).astype(np.float32)
        report = compare_synthetic(logits, logits.copy())
        assert report.result == ParityResult.PASS
        assert report.max_abs_diff == pytest.approx(0.0, abs=1e-12)
        assert report.cosine_similarity == pytest.approx(1.0, abs=1e-6)

    def test_small_diff_passes(self):
        rng = np.random.default_rng(42)
        hf = rng.standard_normal((1, 100)).astype(np.float32)
        onnx = hf + rng.uniform(-1e-4, 1e-4, hf.shape).astype(np.float32)
        report = compare_synthetic(onnx, hf, atol=1e-3, rtol=1e-3)
        assert report.result == ParityResult.PASS

    def test_large_diff_fails(self):
        rng = np.random.default_rng(42)
        hf = rng.standard_normal((1, 100)).astype(np.float32)
        onnx = hf + 1.0  # large shift
        report = compare_synthetic(onnx, hf, atol=1e-3, rtol=1e-3)
        assert report.result == ParityResult.FAIL
        assert report.max_abs_diff > 0.5

    def test_3d_logits_handled(self):
        """3-D (batch, seq, vocab) input extracts last-token metrics."""
        rng = np.random.default_rng(42)
        hf = rng.standard_normal((1, 5, 100)).astype(np.float32)
        report = compare_synthetic(hf, hf.copy())
        assert report.result == ParityResult.PASS

    def test_near_tie_diagnostic(self):
        """Near-tie adds diagnostic message but does not change result."""
        # Build logits where top-1 and top-2 are very close
        hf = np.zeros((1, 100), dtype=np.float32)
        hf[0, 10] = 1.0
        hf[0, 20] = 1.0 + 1e-4  # gap < 0.01 margin
        report = compare_synthetic(hf, hf.copy())
        assert report.near_tie is True
        assert "near-tie" in report.message

    def test_shape_mismatch_raises(self):
        a = np.zeros((1, 10), dtype=np.float32)
        b = np.zeros((1, 20), dtype=np.float32)
        with pytest.raises(AssertionError, match="Shape mismatch"):
            compare_synthetic(a, b)

    def test_report_is_dataclass(self):
        logits = np.zeros((1, 10), dtype=np.float32)
        report = compare_synthetic(logits, logits)
        assert isinstance(report, ParityReport)
        assert report.level == "L3"


class TestCompareGolden:
    """Tests for L4 golden comparison."""

    def test_argmax_match_passes(self):
        logits = np.zeros((1, 100), dtype=np.float32)
        logits[0, 42] = 10.0
        logits[0, 7] = 5.0
        report = compare_golden(
            logits,
            golden_top1_id=42,
            golden_top2_id=7,
            golden_top10_ids=list(range(10)),
        )
        assert report.result == ParityResult.PASS

    def test_argmax_mismatch_fails(self):
        logits = np.zeros((1, 100), dtype=np.float32)
        logits[0, 42] = 10.0
        report = compare_golden(
            logits,
            golden_top1_id=99,
            golden_top2_id=7,
            golden_top10_ids=[99, 7],
        )
        assert report.result == ParityResult.FAIL

    def test_near_tie_ambiguous(self):
        """Mismatch + near-tie + matches top2 → AMBIGUOUS."""
        logits = np.zeros((1, 100), dtype=np.float32)
        logits[0, 7] = 10.0
        logits[0, 42] = 10.0 - 1e-4  # near-tie: gap < 0.01
        # ONNX picks 7, golden top1=42, top2=7
        report = compare_golden(
            logits,
            golden_top1_id=42,
            golden_top2_id=7,
            golden_top10_ids=[42, 7],
        )
        assert report.result == ParityResult.AMBIGUOUS

    def test_3d_input(self):
        logits = np.zeros((1, 3, 50), dtype=np.float32)
        logits[0, -1, 5] = 10.0
        report = compare_golden(
            logits,
            golden_top1_id=5,
            golden_top2_id=0,
            golden_top10_ids=[5, 0],
        )
        assert report.result == ParityResult.PASS
        assert report.level == "L4"
