# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for scripts/generate_golden.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Import the generate_golden module via path manipulation since
# scripts/ is not a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_golden


class TestExtractLogitsGolden:
    """Tests for _extract_logits_golden() helper."""

    def test_basic_extraction(self):
        """Top-1/2/10 IDs and logits are correctly extracted."""
        rng = np.random.default_rng(42)
        logits = rng.standard_normal(100).astype(np.float32)

        result = generate_golden._extract_logits_golden(logits)

        assert result["top1_id"] == int(np.argmax(logits))
        assert len(result["top10_ids"]) == 10
        assert len(result["top10_logits"]) == 10
        # Top-10 IDs should be sorted by descending logit
        assert result["top10_logits"][0] >= result["top10_logits"][-1]
        # top1_id should be the first in top10
        assert result["top10_ids"][0] == result["top1_id"]

    def test_summary_statistics(self):
        """logits_summary contains [max, min, mean, std]."""
        logits = np.array([1.0, -2.0, 3.0, 0.0, -1.0], dtype=np.float32)

        result = generate_golden._extract_logits_golden(logits)
        summary = result["logits_summary"]

        assert summary[0] == pytest.approx(3.0)  # max
        assert summary[1] == pytest.approx(-2.0)  # min
        assert summary[2] == pytest.approx(0.2)  # mean
        assert summary[3] > 0  # std > 0

    def test_small_vocab(self):
        """Works with vocab smaller than 10 tokens."""
        logits = np.array([5.0, 3.0, 1.0], dtype=np.float32)

        result = generate_golden._extract_logits_golden(logits)

        assert result["top1_id"] == 0
        assert result["top2_id"] == 1
        assert len(result["top10_ids"]) == 3

    def test_dtype_promotion(self):
        """float32 input produces float64 summary."""
        logits = np.ones(50, dtype=np.float32)
        result = generate_golden._extract_logits_golden(logits)

        assert result["logits_summary"].dtype == np.float64


class TestDryRun:
    """Tests for main() with --dry-run (no HF inference needed)."""

    def test_dry_run_all_cases(self):
        """--dry-run lists cases without running inference."""
        result = subprocess.run(
            [sys.executable, "scripts/generate_golden.py", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "Found" in result.stdout
        assert "DRY-RUN" in result.stdout

    def test_dry_run_with_filter(self):
        """--dry-run --filter limits output to matching cases."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_golden.py",
                "--dry-run",
                "--filter",
                "gpt2",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "gpt2" in result.stdout
        # Should not contain other models
        assert "qwen" not in result.stdout.lower()

    def test_dry_run_with_task_type(self):
        """--dry-run --task-type filters by task."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_golden.py",
                "--dry-run",
                "--task-type",
                "encoder",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "bert" in result.stdout.lower()

    def test_dry_run_no_matches(self):
        """--filter with no matches exits cleanly."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_golden.py",
                "--dry-run",
                "--filter",
                "nonexistent_model_xyz",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "No test cases found" in result.stdout


class TestGeneratorDispatch:
    """Tests for the _GENERATORS dispatch table."""

    def test_all_task_types_have_generators(self):
        """All task types used in YAML cases have generators."""
        expected_types = {
            "text-generation",
            "feature-extraction",
            "seq2seq",
            "image-text-to-text",
            "speech-to-text",
            "audio-feature-extraction",
        }
        assert expected_types <= set(generate_golden._GENERATORS.keys())

    def test_generator_functions_are_callable(self):
        for name, func in generate_golden._GENERATORS.items():
            assert callable(func), f"Generator for {name} is not callable"
