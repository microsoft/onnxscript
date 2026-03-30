#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

r"""Generate a static HTML confidence dashboard for model testing coverage.

Scans the model registry, test configurations, and test files to determine
confidence levels per model type. Outputs a self-contained HTML dashboard
with no external dependencies.

Usage::

    python scripts/generate_dashboard.py --output docs/dashboard/index.html
    python scripts/generate_dashboard.py --output docs/dashboard/index.html \
        --commit $(git rev-parse --short HEAD)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jinja2


def _json_safe(obj: Any) -> Any:
    """Convert an object to a JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(v) for v in obj)
    if isinstance(obj, type):
        return obj.__name__
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _json_safe(dataclasses.asdict(obj))
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# Ensure the source package is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "tests"))


@dataclasses.dataclass
class ModelInfo:
    """Collected information about a single registered model type."""

    model_type: str
    module_class_name: str
    task: str
    category: str
    family: str
    # Confidence levels (True if covered at that level)
    l1_graph_build: bool = False
    l2_arch_validation: bool = False
    l3_synthetic_parity: bool = False
    l4_golden_files: bool = False
    l5_generation_golden: bool = False
    # Code paths exercised by test configs
    code_paths: set[str] = dataclasses.field(default_factory=set)
    # Config overrides from test configs (for drill-down)
    config_overrides: list[dict[str, Any]] = dataclasses.field(
        default_factory=list,
    )
    # Whether the model has integration tests
    has_integration_test: bool = False
    # Whether the model has a test_model_id
    test_model_id: str | None = None
    # Golden test case coverage (from testdata/cases/ YAML files)
    l4_has_test_case: bool = False
    l5_has_test_case: bool = False
    l4_test_case_skipped: bool = False
    l5_test_case_skipped: bool = False
    yaml_test_case_file: str | None = None
    yaml_test_case_skip_reason: str | None = None
    yaml_min_token_match_ratio: float | None = None
    # L3 synthetic parity status: "pass", "xfail", "skip", or None
    l3_status: str | None = None
    l3_status_reason: str | None = None

    @property
    def confidence_level(self) -> int:
        """Return the highest confidence level achieved (0-5)."""
        if self.l5_generation_golden:
            return 5
        if self.l4_golden_files:
            return 4
        if self.l3_synthetic_parity:
            return 3
        if self.l2_arch_validation:
            return 2
        if self.l1_graph_build:
            return 1
        return 0

    @property
    def confidence_label(self) -> str:
        """Human-readable confidence label."""
        return _CONFIDENCE_LABELS.get(self.confidence_level, "Not tested")


_CONFIDENCE_LABELS = {
    0: "Not tested",
    1: "L1: Graph builds",
    2: "L2: Config compatible",
    3: "L3: Synthetic parity",
    4: "L4: Golden match",
    5: "L5: Generation verified",
}

# Task strings from the registry to dashboard category mapping.
# These include model-specific task names (e.g. "qwen-vl") because the
# registry uses custom task strings for specialized pipelines.
_TASK_CATEGORY_MAP = {
    "text-generation": "Causal LM",
    "feature-extraction": "Encoder",
    "seq2seq": "Seq2Seq",
    "image-classification": "Vision",
    "object-detection": "Detection",
    "audio-feature-extraction": "Audio",
    "speech-to-text": "Speech",
    "speech-language": "Speech",
    "codec": "Speech",
    "vision-language": "Vision-Language",
    "qwen-vl": "Vision-Language",
    "qwen3-vl-vision-language": "Vision-Language",
    "hybrid-qwen-vl": "Vision-Language",
    "mllama-vision-language": "Vision-Language",
    "multimodal": "Multimodal",
}


def _derive_family(model_type: str) -> str:
    """Derive a family name from a model type using prefix heuristic.

    Groups model types that share a common prefix into families.
    E.g., qwen2, qwen2_moe, qwen2_vl → "qwen2".
    """
    for prefix in _FAMILY_PREFIXES:
        if model_type.startswith(prefix):
            return prefix
    # Default: use the whole model_type as the family
    return model_type


# Common prefixes for model families (ordered longest-first so that
# "qwen3_5" matches before "qwen3").
_FAMILY_PREFIXES = [
    "qwen3_5",
    "qwen3",
    "qwen2_5",
    "qwen2",
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_vl",
    "gemma3n",
    "gemma3",
    "gemma2",
    "internlm",
    "internvl",
    "phi4",
    "phi3",
    "olmo",
    "llava_next",
    "llava",
    "glm4v",
    "glm4",
    "wav2vec2",
    "data2vec",
    "falcon",
    "blenderbot",
    "roberta",
    "xlm",
    "gpt_neo",
    "gpt",
    "vit",
    "swin",
    "dinov2",
    "siglip",
]


def _scan_registry() -> dict[str, ModelInfo]:
    """Scan the model registry and build initial ModelInfo entries."""
    from mobius._registry import registry

    models: dict[str, ModelInfo] = {}
    for arch in registry.architectures():
        reg = registry.get_registration(arch)
        task = reg.task or "text-generation"
        category = _TASK_CATEGORY_MAP.get(task, "Other")
        family_override = getattr(reg, "family", None)
        family = family_override or _derive_family(arch)
        test_model_id = getattr(reg, "test_model_id", None)

        models[arch] = ModelInfo(
            model_type=arch,
            module_class_name=reg.module_class.__name__,
            task=task,
            category=category,
            family=family,
            test_model_id=test_model_id,
        )
    return models


def _scan_l1_configs(models: dict[str, ModelInfo]) -> None:
    """Mark L1 coverage from test config presence in _test_configs.py.

    Also marks models in ``_SPECIALIZED_TEST_MODEL_TYPES`` (VLM/audio models
    tested via dedicated test methods rather than the parametrized config loop).
    """
    from mobius._testing.code_paths import (
        detect_code_paths,
    )

    try:
        from _test_configs import ALL_CONFIGS
    except ImportError:
        print(
            "Warning: Could not import _test_configs. L1 detection skipped.",
            file=sys.stderr,
        )
        return

    for model_type, config_overrides, _is_repr in ALL_CONFIGS:
        if model_type in models:
            models[model_type].l1_graph_build = True
            models[model_type].config_overrides.append(config_overrides)
            paths = detect_code_paths(config_overrides)
            models[model_type].code_paths.update(paths)

    # Specialized VLM/audio models have dedicated test methods in
    # build_graph_test.py but are not in ALL_CONFIGS. They still build a graph.
    try:
        from build_graph_test import _SPECIALIZED_TEST_MODEL_TYPES
    except ImportError:
        return

    for model_type in _SPECIALIZED_TEST_MODEL_TYPES:
        if model_type in models:
            models[model_type].l1_graph_build = True


def _scan_l2_arch_tests(models: dict[str, ModelInfo]) -> None:
    """Mark L2 coverage from arch_validation_test.py presence."""
    arch_test = _REPO_ROOT / "tests" / "arch_validation_test.py"
    if not arch_test.exists():
        return

    content = arch_test.read_text(encoding="utf-8")
    for model_type, info in models.items():
        # L2 requires test_model_id in registry
        if info.test_model_id:
            info.l2_arch_validation = True
        # Also check if model_type appears in the test file
        elif f'"{model_type}"' in content or f"'{model_type}'" in content:
            info.l2_arch_validation = True


def _scan_l3_synthetic_parity(models: dict[str, ModelInfo]) -> None:
    """Mark L3 coverage from synthetic_parity_test.py.

    Imports the actual test config list (``ALL_CAUSAL_LM_CONFIGS``)
    used by the parametrized test to detect which model_types have
    L3 coverage.  This avoids false positives from string matching
    (e.g. skipped models mentioned in ``_SKIP_REASONS``) and false
    negatives from models parametrized via imported config lists
    rather than inline ``pytest.mark.parametrize`` strings.
    """
    parity_test = _REPO_ROOT / "tests" / "synthetic_parity_test.py"
    if not parity_test.exists():
        return

    # Import actual configs from the test support module.
    # This is the authoritative source of which model_types are tested.
    try:
        sys.path.insert(0, str(_REPO_ROOT / "tests"))
        from _test_configs import ALL_CAUSAL_LM_CONFIGS

        l3_model_types = {mt for mt, _ov, _rep in ALL_CAUSAL_LM_CONFIGS}
    except ImportError:
        # Fallback: if import fails, do nothing rather than false-positive
        return
    finally:
        sys.path.pop(0)

    for model_type in models:
        if model_type in l3_model_types:
            models[model_type].l3_synthetic_parity = True


def _scan_l4_golden_files(models: dict[str, ModelInfo]) -> None:
    """Mark L4 coverage from testdata/golden/ directory.

    Two matching strategies:
    1. Direct: ``golden/<category>/<model_type>.json`` — works when the golden
       file stem equals the registry model_type.
    2. Indirect: when a model has a YAML test case, derive the expected golden
       path from the case_id (the YAML file stem).  This handles cases like
       ``golden/vision-language/qwen2_5-vl-3b.json`` → model_type ``qwen2_5_vl``.
    """
    golden_dir = _REPO_ROOT / "testdata" / "golden"
    if not golden_dir.exists():
        return

    # Strategy 1: direct stem → model_type match
    for golden_file in golden_dir.rglob("*.json"):
        if "_generation" in golden_file.name:
            continue
        model_type = golden_file.stem
        if model_type in models:
            models[model_type].l4_golden_files = True

    # Strategy 2: YAML-derived path (case_id may differ from model_type)
    for model_type, info in models.items():
        if info.l4_golden_files or not info.yaml_test_case_file:
            continue
        case_path = _REPO_ROOT / info.yaml_test_case_file
        case_id = case_path.stem
        task_dir = case_path.parent.name
        golden_path = golden_dir / task_dir / f"{case_id}.json"
        if golden_path.exists():
            models[model_type].l4_golden_files = True


def _scan_l5_generation_golden(models: dict[str, ModelInfo]) -> None:
    """Mark L5 coverage from generation golden files.

    Uses the same two-strategy matching as :func:`_scan_l4_golden_files`.
    """
    golden_dir = _REPO_ROOT / "testdata" / "golden"
    if not golden_dir.exists():
        return

    # Strategy 1: direct stem → model_type match
    for golden_file in golden_dir.rglob("*_generation.json"):
        model_type = golden_file.stem.removesuffix("_generation")
        if model_type in models:
            models[model_type].l5_generation_golden = True

    # Strategy 2: YAML-derived path (case_id may differ from model_type)
    for model_type, info in models.items():
        if info.l5_generation_golden or not info.yaml_test_case_file:
            continue
        case_path = _REPO_ROOT / info.yaml_test_case_file
        case_id = case_path.stem
        task_dir = case_path.parent.name
        gen_path = golden_dir / task_dir / f"{case_id}_generation.json"
        if gen_path.exists():
            models[model_type].l5_generation_golden = True


def _scan_integration_tests(models: dict[str, ModelInfo]) -> None:
    """Mark models that have integration tests."""
    tests_dir = _REPO_ROOT / "tests"
    integration_files = list(tests_dir.glob("*integration*.py"))

    for test_file in integration_files:
        content = test_file.read_text(encoding="utf-8")
        for model_type in models:
            if f'"{model_type}"' in content or f"'{model_type}'" in content:
                models[model_type].has_integration_test = True


def _scan_yaml_test_cases(models: dict[str, ModelInfo]) -> None:
    """Scan testdata/cases/ for YAML test case files and mark L4/L5 coverage.

    Builds a reverse index from test_model_id → model_type to map
    HuggingFace model IDs in YAML files back to registry model types.
    """
    cases_dir = _REPO_ROOT / "testdata" / "cases"
    if not cases_dir.exists():
        return

    try:
        import yaml
    except ImportError:
        print(
            "Warning: PyYAML not installed. YAML test case scanning skipped.",
            file=sys.stderr,
        )
        return

    # Build reverse index: HF model_id → list of model_types
    model_id_to_types: dict[str, list[str]] = {}
    for model_type, info in models.items():
        if info.test_model_id:
            model_id_to_types.setdefault(info.test_model_id, []).append(model_type)

    for yaml_file in sorted(cases_dir.rglob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        model_id = data.get("model_id", "")
        level = data.get("level", "")
        rel_path = str(yaml_file.relative_to(_REPO_ROOT))

        # Skip test cases that are explicitly skipped — they don't count as coverage,
        # but we still record them so the dashboard can show "skipped" status.
        skip_reason = data.get("skip_reason")
        min_token_match_ratio = data.get("min_token_match_ratio")
        if skip_reason:
            matched_types = model_id_to_types.get(model_id, [])
            for model_type in matched_types:
                if model_type in models:
                    models[model_type].yaml_test_case_file = rel_path
                    models[model_type].yaml_test_case_skip_reason = skip_reason
                    if min_token_match_ratio is not None:
                        models[model_type].yaml_min_token_match_ratio = float(
                            min_token_match_ratio
                        )
                    if "L4" in level:
                        models[model_type].l4_test_case_skipped = True
                    if "L5" in level:
                        models[model_type].l5_test_case_skipped = True
            continue

        # Find matching model_types via test_model_id reverse index
        matched_types = model_id_to_types.get(model_id, [])
        for model_type in matched_types:
            if model_type in models:
                models[model_type].yaml_test_case_file = rel_path
                if min_token_match_ratio is not None:
                    models[model_type].yaml_min_token_match_ratio = float(
                        min_token_match_ratio
                    )
                if "L4" in level:
                    models[model_type].l4_has_test_case = True
                if "L5" in level:
                    models[model_type].l5_has_test_case = True


def _scan_l3_parity_status(models: dict[str, ModelInfo]) -> None:
    """Extract L3 synthetic parity skip/xfail status from test file.

    Parses _SKIP_REASONS and _XFAIL_REASONS dicts from
    synthetic_parity_test.py to determine per-model status without
    actually running the tests.
    """
    parity_test = _REPO_ROOT / "tests" / "synthetic_parity_test.py"
    if not parity_test.exists():
        return

    # Import the skip/xfail dicts directly
    import importlib.util

    spec = importlib.util.spec_from_file_location("_parity_test", parity_test)
    if spec is None or spec.loader is None:
        return

    try:
        _ = importlib.util.module_from_spec(spec)
        # Don't actually run the test — just load the module-level dicts
        # by extracting them from the source text
        content = parity_test.read_text(encoding="utf-8")
    except Exception:
        return

    # Parse _SKIP_REASONS dict entries via regex
    import re

    skip_reasons: dict[str, str] = {}
    xfail_reasons: dict[str, str] = {}

    # Extract _SKIP_REASONS block
    skip_match = re.search(
        r"_SKIP_REASONS.*?=\s*\{(.*?)\}",
        content,
        re.DOTALL,
    )
    if skip_match:
        for m in re.finditer(r'"(\w+)":\s*"([^"]+)"', skip_match.group(1)):
            skip_reasons[m.group(1)] = m.group(2)

    # Extract _XFAIL_REASONS block
    xfail_match = re.search(
        r"_XFAIL_REASONS.*?=\s*\{(.*?)\}",
        content,
        re.DOTALL,
    )
    if xfail_match:
        for m in re.finditer(r'"(\w+)":\s*"([^"]+)"', xfail_match.group(1)):
            xfail_reasons[m.group(1)] = m.group(2)

    for model_type, info in models.items():
        if not info.l3_synthetic_parity:
            continue
        if model_type in skip_reasons:
            info.l3_status = "skip"
            info.l3_status_reason = skip_reasons[model_type]
        elif model_type in xfail_reasons:
            info.l3_status = "xfail"
            info.l3_status_reason = xfail_reasons[model_type]
        else:
            # Has L3 test and not skipped/xfailed → expected to pass
            info.l3_status = "pass"


def collect_all_model_info() -> dict[str, ModelInfo]:
    """Collect all model information by scanning registry and tests."""
    models = _scan_registry()
    _scan_l1_configs(models)
    _scan_l2_arch_tests(models)
    _scan_l3_synthetic_parity(models)
    _scan_l3_parity_status(models)
    # YAML test cases must be scanned before golden files so that the
    # YAML-derived golden paths can be used for indirect model_type matching.
    _scan_yaml_test_cases(models)
    _scan_l4_golden_files(models)
    _scan_l5_generation_golden(models)
    _scan_integration_tests(models)
    return models


def _group_by_family(
    models: dict[str, ModelInfo],
) -> dict[str, list[ModelInfo]]:
    """Group models by family, sorted."""
    families: dict[str, list[ModelInfo]] = {}
    for info in sorted(models.values(), key=lambda m: m.model_type):
        families.setdefault(info.family, []).append(info)
    return dict(sorted(families.items()))


def _compute_summary(
    models: dict[str, ModelInfo],
) -> dict[str, Any]:
    """Compute summary statistics for the dashboard."""
    total = len(models)
    by_level = dict.fromkeys(range(6), 0)
    by_category: dict[str, int] = {}
    all_code_paths: set[str] = set()
    code_path_coverage: dict[str, int] = {}
    l3_status_counts: dict[str, int] = {
        "pass": 0,
        "xfail": 0,
        "skip": 0,
        "untested": 0,
    }
    l4_case_count = 0
    l5_case_count = 0
    l4_skipped_count = 0
    l5_skipped_count = 0

    for info in models.values():
        # Per-flag counts: how many models have each level flag set, independently.
        # These are NOT exclusive (a model counted in L3 may also be in L1/L2).
        # by_level[0] = not-tested (no flags set at all).
        if not any(
            [
                info.l1_graph_build,
                info.l2_arch_validation,
                info.l3_synthetic_parity,
                info.l4_golden_files,
                info.l5_generation_golden,
            ]
        ):
            by_level[0] += 1
        if info.l1_graph_build:
            by_level[1] += 1
        if info.l2_arch_validation:
            by_level[2] += 1
        if info.l3_synthetic_parity and info.l3_status == "pass":
            by_level[3] += 1
        if info.l4_golden_files:
            by_level[4] += 1
        if info.l5_generation_golden:
            by_level[5] += 1
        by_category[info.category] = by_category.get(info.category, 0) + 1
        all_code_paths.update(info.code_paths)
        for cp in info.code_paths:
            code_path_coverage[cp] = code_path_coverage.get(cp, 0) + 1
        # L3 status
        if info.l3_status:
            l3_status_counts[info.l3_status] = l3_status_counts.get(info.l3_status, 0) + 1
        else:
            l3_status_counts["untested"] += 1
        # Golden case coverage
        if info.l4_has_test_case:
            l4_case_count += 1
        if info.l5_has_test_case:
            l5_case_count += 1
        if info.l4_test_case_skipped:
            l4_skipped_count += 1
        if info.l5_test_case_skipped:
            l5_skipped_count += 1

    return {
        "total": total,
        "by_level": by_level,
        "by_category": dict(sorted(by_category.items())),
        "code_path_coverage": dict(sorted(code_path_coverage.items())),
        "all_code_paths": sorted(all_code_paths),
        "l3_status_counts": l3_status_counts,
        "l4_case_count": l4_case_count,
        "l5_case_count": l5_case_count,
        "l4_skipped_count": l4_skipped_count,
        "l5_skipped_count": l5_skipped_count,
    }


def _build_component_matrix(
    models: dict[str, ModelInfo],
) -> dict[str, Any]:
    """Build component x family matrix for the heatmap visualization.

    Returns a dict suitable for JSON serialization with:
    - ``families``: sorted list of family names that have at least one component.
    - ``rows``: one entry per component, with per-family max confidence levels.
    """
    from mobius._testing.code_paths import CODE_PATH_INDICATORS

    # Gather only families that exercise at least one component.
    families_with_paths: set[str] = set()
    for info in models.values():
        if info.code_paths:
            families_with_paths.add(info.family)
    sorted_families = sorted(families_with_paths)

    # matrix[feature_label][family] = max confidence level among all models
    # in that family that exercise this feature.
    matrix: dict[str, dict[str, int]] = {ind.label: {} for ind in CODE_PATH_INDICATORS}
    for info in models.values():
        if not info.code_paths:
            continue
        for path in info.code_paths:
            if path in matrix:
                cur = matrix[path].get(info.family, -1)
                matrix[path][info.family] = max(cur, info.confidence_level)

    # Build rows — one per indicator, with a cell value per family.
    rows = []
    for ind in CODE_PATH_INDICATORS:
        fam_cells = matrix[ind.label]
        cells = [fam_cells.get(fam, -1) for fam in sorted_families]
        family_count = sum(1 for c in cells if c >= 0)
        # Total individual model count from summary (recomputed here for simplicity).
        model_count = sum(1 for info in models.values() if ind.label in info.code_paths)
        best_level = max((c for c in cells if c >= 0), default=-1)
        rows.append(
            {
                "label": ind.label,
                "description": ind.description,
                "model_count": model_count,
                "family_count": family_count,
                "best_level": best_level,
                "cells": cells,
            }
        )

    return {"families": sorted_families, "rows": rows}


_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _to_js_json(obj: Any) -> str:
    """Serialize obj to JSON safe for inline <script> injection.

    The ``</`` replacement prevents the string from accidentally closing
    a ``<script>`` tag when embedded in HTML.
    """
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def _render_html(
    models: dict[str, ModelInfo],
    commit: str | None = None,
) -> str:
    """Render the self-contained HTML dashboard via Jinja2 template."""
    summary = _compute_summary(models)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build model data as JSON for JavaScript consumption.
    model_data = []
    for info in models.values():
        model_data.append(
            {
                "model_type": info.model_type,
                "module_class": info.module_class_name,
                "task": info.task,
                "category": info.category,
                "family": info.family,
                "confidence_level": info.confidence_level,
                "confidence_label": info.confidence_label,
                "l1": info.l1_graph_build,
                "l2": info.l2_arch_validation,
                "l3": info.l3_synthetic_parity and info.l3_status == "pass",
                "l4": info.l4_golden_files,
                "l5": info.l5_generation_golden,
                "l4_case": info.l4_has_test_case,
                "l5_case": info.l5_has_test_case,
                "l4_skipped": info.l4_test_case_skipped,
                "l5_skipped": info.l5_test_case_skipped,
                "l3_status": info.l3_status,
                "l3_reason": info.l3_status_reason,
                "yaml_case": info.yaml_test_case_file,
                "yaml_skip_reason": info.yaml_test_case_skip_reason,
                "min_token_match_ratio": info.yaml_min_token_match_ratio,
                "code_paths": sorted(info.code_paths),
                "config_overrides": _json_safe(info.config_overrides),
                "has_integration_test": info.has_integration_test,
                "test_model_id": info.test_model_id,
            }
        )

    model_data_json = _to_js_json(sorted(model_data, key=lambda m: m["model_type"]))

    from mobius._testing.code_paths import CODE_PATH_INDICATORS

    code_path_info = [
        {
            "label": ind.label,
            "description": ind.description,
            "example_config": ind.example_config,
        }
        for ind in CODE_PATH_INDICATORS
    ]

    # Commit string is plain text; Jinja2 autoescape handles HTML encoding.
    component_matrix = _build_component_matrix(models)
    context = {
        "timestamp": timestamp,
        "commit": commit if commit else "unknown",
        "total_models": summary["total"],
        # JSON blobs injected into <script> tags: marked |safe in the template
        # because json.dumps already produces valid JS values and the </
        # replacement prevents premature script-tag closure.
        "model_data_json": model_data_json,
        "code_path_json": _to_js_json(code_path_info),
        "component_matrix_json": _to_js_json(component_matrix),
        "summary_json": _to_js_json(summary),
        "labels_json": _to_js_json(_CONFIDENCE_LABELS),
    }

    # autoescape=True: Jinja2 HTML-escapes all {{ var }} by default.
    # Variables containing pre-serialized JSON are marked |safe in the template
    # to bypass escaping — they are already safe for <script> injection.
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
        keep_trailing_newline=True,
    )
    template = env.get_template("dashboard.html.j2")
    return template.render(**context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the testing confidence dashboard.")
    parser.add_argument(
        "--output",
        type=str,
        default="docs/dashboard/index.html",
        help="Output HTML file path (default: docs/dashboard/index.html)",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Git commit SHA to display in the dashboard",
    )
    args = parser.parse_args()

    models = collect_all_model_info()
    html_content = _render_html(models, commit=args.commit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)

    # Print summary
    summary = _compute_summary(models)
    total = summary["total"]
    by_level = summary["by_level"]
    print(f"Dashboard generated: {output_path}")
    print(f"  Total models: {total}")
    for level, count in sorted(by_level.items()):
        pct = round(count / total * 100, 1) if total > 0 else 0
        label = _CONFIDENCE_LABELS.get(level, f"L{level}")
        print(f"  {label}: {count} ({pct}%)")


if __name__ == "__main__":
    main()
