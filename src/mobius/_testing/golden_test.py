# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the golden file I/O library."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mobius._testing.golden import (
    GoldenRef,
    GoldenTestCase,
    Tolerances,
    discover_test_cases,
    golden_path_for_case,
    has_golden,
    load_golden_ref,
    load_test_case,
    load_tolerances,
    save_golden_ref,
)

# --- Fixtures ---


MINIMAL_YAML = """\
model_id: "test-org/test-model"
revision: "abc123"
task_type: "text-generation"
dtype: "float32"
level: "L4"
inputs:
  prompts:
    - "Hello world"
"""

VL_YAML = """\
model_id: "test-org/test-vl"
revision: "def456"
task_type: "qwen-vl"
dtype: "float32"
level: "L4+L5"
inputs:
  prompts:
    - "Describe this image."
  images:
    - "pipeline-cat-chonk.jpeg"
generation:
  max_new_tokens: 20
  do_sample: false
notes: "VL model test case"
"""

SKIP_YAML = """\
model_id: "test-org/skip-model"
revision: "aaa111"
task_type: "text-generation"
dtype: "float32"
level: "L4"
skip_reason: "Model requires gated access"
"""


def _write_yaml(tmp_path: Path, name: str, content: str) -> Path:
    """Write a YAML string to a file and return the path."""
    yaml_path = tmp_path / name
    yaml_path.write_text(content)
    return yaml_path


# --- GoldenTestCase loading tests ---


class TestLoadTestCase:
    """Tests for load_test_case()."""

    def test_load_minimal(self, tmp_path: Path):
        yaml_path = _write_yaml(tmp_path, "test-model.yaml", MINIMAL_YAML)
        case = load_test_case(yaml_path)

        assert isinstance(case, GoldenTestCase)
        assert case.case_id == "test-model"
        assert case.model_id == "test-org/test-model"
        assert case.revision == "abc123"
        assert case.task_type == "text-generation"
        assert case.dtype == "float32"
        assert case.level == "L4"
        assert case.prompts == ["Hello world"]
        assert case.images == []
        assert case.audio == []
        assert case.decoder_prompt == ""
        assert case.generation_params == {}
        assert case.trust_remote_code is False
        assert case.skip_reason is None

    def test_load_vl_case(self, tmp_path: Path):
        yaml_path = _write_yaml(tmp_path, "test-vl.yaml", VL_YAML)
        case = load_test_case(yaml_path)

        assert case.task_type == "qwen-vl"
        assert case.level == "L4+L5"
        assert case.prompts == ["Describe this image."]
        assert case.images == ["pipeline-cat-chonk.jpeg"]
        assert case.generation_params == {
            "max_new_tokens": 20,
            "do_sample": False,
        }

    def test_load_skip_case(self, tmp_path: Path):
        yaml_path = _write_yaml(tmp_path, "skip.yaml", SKIP_YAML)
        case = load_test_case(yaml_path)

        assert case.skip_reason == "Model requires gated access"

    def test_missing_required_field_raises(self, tmp_path: Path):
        bad_yaml = "model_id: test\n"
        yaml_path = _write_yaml(tmp_path, "bad.yaml", bad_yaml)

        with pytest.raises(ValueError, match="Missing required fields"):
            load_test_case(yaml_path)

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_test_case(tmp_path / "nonexistent.yaml")

    def test_empty_yaml_raises(self, tmp_path: Path):
        yaml_path = _write_yaml(tmp_path, "empty.yaml", "")

        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            load_test_case(yaml_path)

    def test_case_is_frozen(self, tmp_path: Path):
        yaml_path = _write_yaml(tmp_path, "test.yaml", MINIMAL_YAML)
        case = load_test_case(yaml_path)

        with pytest.raises(AttributeError):
            case.model_id = "changed"  # type: ignore[misc]


# --- GoldenRef round-trip tests ---


class TestGoldenRefRoundTrip:
    """Tests for save_golden_ref() and load_golden_ref()."""

    def test_l4_round_trip(self, tmp_path: Path):
        """Save and reload L4 golden data; verify all fields match."""
        json_path = tmp_path / "golden" / "causal-lm" / "test.json"

        top1_id = 42
        top2_id = 7
        top10_ids = list(range(10))
        top10_logits = [float(i) for i in range(10, 0, -1)]
        logits_summary = np.array([15.0, -12.0, 0.5, 3.0])
        input_ids = np.array([[1, 450, 3127]], dtype=np.int64)

        save_golden_ref(
            json_path,
            top1_id=top1_id,
            top2_id=top2_id,
            top10_ids=top10_ids,
            top10_logits=top10_logits,
            logits_summary=logits_summary,
            input_ids=input_ids,
        )

        assert json_path.exists()

        golden = load_golden_ref(json_path)
        assert golden is not None
        assert isinstance(golden, GoldenRef)
        assert golden.top1_id == top1_id
        assert golden.top2_id == top2_id
        assert golden.top10_ids == top10_ids
        assert golden.top10_logits == pytest.approx(top10_logits)
        assert golden.logits_summary == pytest.approx(logits_summary.tolist())
        # input_ids is flattened from (1, 3) to [1, 450, 3127]
        assert golden.input_ids == [1, 450, 3127]
        assert golden.generated_ids is None
        assert golden.component_norms == {}
        assert golden.component_shapes == {}

    def test_l5_round_trip(self, tmp_path: Path):
        """Save and reload L5 golden data with generated_ids."""
        json_path = tmp_path / "test_l5.json"
        generated = np.array([42, 7, 100, 55], dtype=np.int64)

        save_golden_ref(
            json_path,
            top1_id=42,
            top2_id=7,
            top10_ids=list(range(10)),
            top10_logits=[float(i) for i in range(10)],
            logits_summary=np.array([1.0, -1.0, 0.0, 0.5]),
            input_ids=np.array([[1, 2, 3]]),
            generated_ids=generated,
        )

        golden = load_golden_ref(json_path)
        assert golden is not None
        assert golden.generated_ids == generated.tolist()

    def test_multi_model_round_trip(self, tmp_path: Path):
        """Save and reload multi-model (VL) golden data."""
        json_path = tmp_path / "test_vl.json"

        component_norms = {"vision": 42.5, "embedding": 38.2}
        component_shapes = {
            "vision": (1, 577, 1024),
            "embedding": (1, 583, 2048),
        }

        save_golden_ref(
            json_path,
            top1_id=10,
            top2_id=20,
            top10_ids=list(range(10)),
            top10_logits=[float(i) for i in range(10)],
            logits_summary=np.array([2.0, -2.0, 0.0, 1.0]),
            input_ids=np.array([[1]]),
            component_norms=component_norms,
            component_shapes=component_shapes,
        )

        golden = load_golden_ref(json_path)
        assert golden is not None
        assert golden.component_norms == pytest.approx(component_norms)
        # JSON stores shapes as lists, not tuples
        assert golden.component_shapes == {
            "vision": [1, 577, 1024],
            "embedding": [1, 583, 2048],
        }

    def test_missing_json_returns_none(self, tmp_path: Path):
        """load_golden_ref returns None for non-existent files."""
        result = load_golden_ref(tmp_path / "nonexistent.json")
        assert result is None

    def test_creates_parent_dirs(self, tmp_path: Path):
        """save_golden_ref creates parent directories."""
        json_path = tmp_path / "a" / "b" / "c" / "test.json"

        save_golden_ref(
            json_path,
            top1_id=1,
            top2_id=2,
            top10_ids=list(range(10)),
            top10_logits=[0.0] * 10,
            logits_summary=np.zeros(4),
            input_ids=np.array([[1]]),
        )

        assert json_path.exists()

    def test_golden_ref_is_frozen(self, tmp_path: Path):
        """GoldenRef instances are immutable."""
        json_path = tmp_path / "test.json"
        save_golden_ref(
            json_path,
            top1_id=1,
            top2_id=2,
            top10_ids=list(range(10)),
            top10_logits=[0.0] * 10,
            logits_summary=np.zeros(4),
            input_ids=np.array([[1]]),
        )

        golden = load_golden_ref(json_path)
        assert golden is not None
        with pytest.raises(AttributeError):
            golden.top1_id = 999  # type: ignore[misc]


# --- discover_test_cases() tests ---


class TestDiscoverTestCases:
    """Tests for discover_test_cases()."""

    def _setup_cases(self, tmp_path: Path) -> Path:
        """Create a test directory with sample YAML files."""
        cases_root = tmp_path / "cases"
        causal = cases_root / "causal-lm"
        encoder = cases_root / "encoder"
        causal.mkdir(parents=True)
        encoder.mkdir(parents=True)

        _write_yaml(
            causal,
            "model-a.yaml",
            'model_id: "a"\nrevision: "1"\ntask_type: "text-generation"'
            '\ndtype: "float32"\nlevel: "L4+L5"\n',
        )
        _write_yaml(
            causal,
            "model-b.yaml",
            'model_id: "b"\nrevision: "2"\ntask_type: "text-generation"'
            '\ndtype: "float32"\nlevel: "L4"\n',
        )
        _write_yaml(
            encoder,
            "model-c.yaml",
            'model_id: "c"\nrevision: "3"\n'
            'task_type: "feature-extraction"'
            '\ndtype: "float32"\nlevel: "L4"\n',
        )
        return cases_root

    def test_discover_all(self, tmp_path: Path):
        root = self._setup_cases(tmp_path)
        cases = discover_test_cases(root=root)

        assert len(cases) == 3
        ids = [c.case_id for c in cases]
        assert ids == sorted(ids), "Cases should be sorted by case_id"

    def test_filter_by_task_type_dir(self, tmp_path: Path):
        """Filter matches against subdirectory name."""
        root = self._setup_cases(tmp_path)
        cases = discover_test_cases(task_type="causal-lm", root=root)

        assert len(cases) == 2
        assert all(c.task_type == "text-generation" for c in cases)

    def test_filter_by_task_type_field(self, tmp_path: Path):
        """Filter matches against the task_type field."""
        root = self._setup_cases(tmp_path)
        cases = discover_test_cases(task_type="feature-extraction", root=root)

        assert len(cases) == 1
        assert cases[0].model_id == "c"

    def test_filter_by_level(self, tmp_path: Path):
        root = self._setup_cases(tmp_path)

        # "L5" matches "L4+L5" but not "L4"
        cases = discover_test_cases(level="L5", root=root)
        assert len(cases) == 1
        assert cases[0].model_id == "a"

    def test_combined_filters(self, tmp_path: Path):
        root = self._setup_cases(tmp_path)
        cases = discover_test_cases(task_type="causal-lm", level="L4", root=root)

        # Both causal-lm cases include L4 ("L4" and "L4+L5")
        assert len(cases) == 2

    def test_empty_dir_returns_empty(self, tmp_path: Path):
        cases = discover_test_cases(root=tmp_path / "nonexistent")
        assert cases == []

    def test_malformed_yaml_skipped(self, tmp_path: Path):
        """Malformed YAML files are silently skipped."""
        root = tmp_path / "cases"
        root.mkdir()
        _write_yaml(root, "bad.yaml", "not_valid: !!binary ZZZZ\n")
        _write_yaml(
            root,
            "good.yaml",
            'model_id: "ok"\nrevision: "1"\n'
            'task_type: "text-generation"'
            '\ndtype: "float32"\nlevel: "L4"\n',
        )

        cases = discover_test_cases(root=root)
        # The malformed file may either be skipped or may parse fine
        # (depends on YAML). At minimum the good file should be found.
        assert any(c.model_id == "ok" for c in cases)


# --- golden_path_for_case() and has_golden() tests ---


class TestGoldenPathAndHasGolden:
    """Tests for golden_path_for_case() and has_golden()."""

    def test_golden_path_mapping(self, tmp_path: Path):
        yaml_path = tmp_path / "cases" / "causal-lm" / "qwen2_5-0_5b.yaml"
        yaml_path.parent.mkdir(parents=True)
        _write_yaml(yaml_path.parent, yaml_path.name, MINIMAL_YAML)

        case = load_test_case(yaml_path)
        golden = golden_path_for_case(case)

        # Should map cases/ → golden/ with .json extension
        assert golden.name == "qwen2_5-0_5b.json"
        assert golden.parent.name == "causal-lm"
        assert "golden" in str(golden)

    def test_has_golden_true(self, tmp_path: Path):
        """has_golden returns True when the json exists."""
        golden_root = tmp_path / "golden"

        # Create the case YAML
        cases_dir = tmp_path / "cases" / "causal-lm"
        cases_dir.mkdir(parents=True)
        yaml_path = _write_yaml(cases_dir, "test.yaml", MINIMAL_YAML)
        case = load_test_case(yaml_path)

        # Create the golden .json at the expected location
        golden = golden_path_for_case(case, golden_dir=golden_root)
        golden.parent.mkdir(parents=True, exist_ok=True)
        save_golden_ref(
            golden,
            top1_id=1,
            top2_id=2,
            top10_ids=list(range(10)),
            top10_logits=[0.0] * 10,
            logits_summary=np.zeros(4),
            input_ids=np.array([[1]]),
        )

        assert has_golden(case, golden_dir=golden_root) is True

    def test_has_golden_false(self, tmp_path: Path):
        """has_golden returns False when the npz doesn't exist."""
        causal_dir = tmp_path / "causal-lm"
        causal_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = _write_yaml(causal_dir, "test.yaml", MINIMAL_YAML)

        case = load_test_case(yaml_path)
        assert has_golden(case, golden_dir=tmp_path / "golden") is False


# --- Tolerances loading tests ---


class TestLoadTolerances:
    """Tests for load_tolerances()."""

    SAMPLE_TOLERANCES_YAML = """\
L4:
  float32:
    near_tie_margin: 0.01
    top10_jaccard_warn: 0.7
    cosine_similarity_warn: 0.999
  float16:
    near_tie_margin: 0.1
    top10_jaccard_warn: 0.5
    cosine_similarity_warn: 0.99
L5:
  float32:
    min_token_match_ratio: 1.0
    near_tie_margin: 0.01
    top10_jaccard_warn: 0.7
    cosine_similarity_warn: 0.999
  float16:
    min_token_match_ratio: 0.9
"""

    def test_load_l4_float32(self, tmp_path: Path):
        tol_path = _write_yaml(tmp_path, "tolerances.yaml", self.SAMPLE_TOLERANCES_YAML)
        tol = load_tolerances(level="L4", dtype="float32", path=tol_path)

        assert isinstance(tol, Tolerances)
        assert tol.near_tie_margin == pytest.approx(0.01)
        assert tol.top10_jaccard_warn == pytest.approx(0.7)
        assert tol.cosine_similarity_warn == pytest.approx(0.999)
        # min_token_match_ratio falls back to default (1.0) for L4
        assert tol.min_token_match_ratio == pytest.approx(1.0)

    def test_load_l5_float16(self, tmp_path: Path):
        tol_path = _write_yaml(tmp_path, "tolerances.yaml", self.SAMPLE_TOLERANCES_YAML)
        tol = load_tolerances(level="L5", dtype="float16", path=tol_path)

        assert tol.min_token_match_ratio == pytest.approx(0.9)

    def test_missing_file_returns_defaults(self, tmp_path: Path):
        """When the tolerances YAML is missing, use hard-coded defaults."""
        tol = load_tolerances(
            path=tmp_path / "nonexistent.yaml",
        )

        assert isinstance(tol, Tolerances)
        assert tol.near_tie_margin == pytest.approx(0.01)
        assert tol.min_token_match_ratio == pytest.approx(1.0)

    def test_missing_dtype_falls_back(self, tmp_path: Path):
        """Unknown dtype gets fallback defaults."""
        tol_path = _write_yaml(tmp_path, "tolerances.yaml", self.SAMPLE_TOLERANCES_YAML)
        tol = load_tolerances(level="L4", dtype="int4", path=tol_path)

        # int4 not in YAML → falls back to built-in defaults
        assert tol.near_tie_margin == pytest.approx(0.01)

    def test_tolerances_is_frozen(self, tmp_path: Path):
        tol = load_tolerances(path=tmp_path / "nonexistent.yaml")

        with pytest.raises(AttributeError):
            tol.near_tie_margin = 999.0  # type: ignore[misc]
