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
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    yaml_test_case_file: str | None = None
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
    """Mark L1 coverage from test config presence in _test_configs.py."""
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


def _scan_l2_arch_tests(models: dict[str, ModelInfo]) -> None:
    """Mark L2 coverage from arch_validation_test.py presence."""
    arch_test = _REPO_ROOT / "tests" / "arch_validation_test.py"
    if not arch_test.exists():
        return

    content = arch_test.read_text()
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
    """Mark L4 coverage from testdata/golden/ directory."""
    golden_dir = _REPO_ROOT / "testdata" / "golden"
    if not golden_dir.exists():
        return

    # Walk golden directories for model-type-named files
    for golden_file in golden_dir.rglob("*.json"):
        # Convention: golden/<category>/<model_type>.json
        model_type = golden_file.stem
        if model_type in models:
            models[model_type].l4_golden_files = True


def _scan_l5_generation_golden(models: dict[str, ModelInfo]) -> None:
    """Mark L5 coverage from generation golden files."""
    golden_dir = _REPO_ROOT / "testdata" / "golden"
    if not golden_dir.exists():
        return

    for golden_file in golden_dir.rglob("*_generation.json"):
        model_type = golden_file.stem.removesuffix("_generation")
        if model_type in models:
            models[model_type].l5_generation_golden = True


def _scan_integration_tests(models: dict[str, ModelInfo]) -> None:
    """Mark models that have integration tests."""
    tests_dir = _REPO_ROOT / "tests"
    integration_files = list(tests_dir.glob("*integration*.py"))

    for test_file in integration_files:
        content = test_file.read_text()
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
            data = yaml.safe_load(yaml_file.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        model_id = data.get("model_id", "")
        level = data.get("level", "")
        rel_path = str(yaml_file.relative_to(_REPO_ROOT))

        # Find matching model_types via test_model_id reverse index
        matched_types = model_id_to_types.get(model_id, [])
        for model_type in matched_types:
            if model_type in models:
                models[model_type].yaml_test_case_file = rel_path
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
        content = parity_test.read_text()
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
    _scan_l4_golden_files(models)
    _scan_l5_generation_golden(models)
    _scan_yaml_test_cases(models)
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

    for info in models.values():
        by_level[info.confidence_level] += 1
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

    return {
        "total": total,
        "by_level": by_level,
        "by_category": dict(sorted(by_category.items())),
        "code_path_coverage": dict(sorted(code_path_coverage.items())),
        "all_code_paths": sorted(all_code_paths),
        "l3_status_counts": l3_status_counts,
        "l4_case_count": l4_case_count,
        "l5_case_count": l5_case_count,
    }


def _generate_html(
    models: dict[str, ModelInfo],
    commit: str | None = None,
) -> str:
    """Generate the self-contained HTML dashboard."""
    families = _group_by_family(models)
    summary = _compute_summary(models)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build model data as JSON for JavaScript
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
                "l3_status": info.l3_status,
                "l3_reason": info.l3_status_reason,
                "yaml_case": info.yaml_test_case_file,
                "code_paths": sorted(info.code_paths),
                "config_overrides": _json_safe(info.config_overrides),
                "has_integration_test": info.has_integration_test,
                "test_model_id": info.test_model_id,
            }
        )

    model_data_json = json.dumps(
        sorted(model_data, key=lambda m: m["model_type"]),
        indent=None,
    ).replace("</", "<\\/")  # Escape </script> injection in JSON

    from mobius._testing.code_paths import (
        CODE_PATH_INDICATORS,
    )

    code_path_info = []
    for ind in CODE_PATH_INDICATORS:
        code_path_info.append(
            {
                "label": ind.label,
                "description": ind.description,
                "example_config": ind.example_config,
            }
        )
    code_path_json = json.dumps(code_path_info, indent=None).replace("</", "<\\/")

    summary_json = json.dumps(summary, indent=None).replace("</", "<\\/")

    # Family data for grouping
    family_data = {}
    for fam_name, fam_models in families.items():
        min_level = min(m.confidence_level for m in fam_models)
        # Aggregate code paths across all variants in the family
        family_code_paths: set[str] = set()
        for m in fam_models:
            family_code_paths.update(m.code_paths)
        family_data[fam_name] = {
            "count": len(fam_models),
            "min_level": min_level,
            "models": [m.model_type for m in fam_models],
            "code_paths": sorted(family_code_paths),
        }
    family_json = json.dumps(family_data, indent=None).replace("</", "<\\/")

    commit_display = html.escape(commit) if commit else "unknown"
    labels_json = json.dumps(_CONFIDENCE_LABELS, indent=None)

    # Use manual replacement instead of .format() because the HTML
    # template contains JavaScript with curly braces and unicode escapes
    result = _HTML_TEMPLATE
    result = result.replace("{{TIMESTAMP}}", timestamp)
    result = result.replace("{{COMMIT}}", commit_display)
    result = result.replace("{{TOTAL_MODELS}}", str(summary["total"]))
    result = result.replace("{{MODEL_DATA_JSON}}", model_data_json)
    result = result.replace("{{CODE_PATH_JSON}}", code_path_json)
    result = result.replace("{{SUMMARY_JSON}}", summary_json)
    result = result.replace("{{FAMILY_JSON}}", family_json)
    result = result.replace("{{LABELS_JSON}}", labels_json)
    return result


# --- HTML building blocks (raw strings to avoid brace escaping issues) ---

_CSS_BLOCK = """\
:root, [data-theme="dark"] {
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --text-muted: #8b949e;
  --accent: #58a6ff;
  --l0: #f85149; --l1: #d29922; --l2: #e3b341;
  --l3: #3fb950; --l4: #2ea043; --l5: #58a6ff;
}
[data-theme="light"] {
  --bg: #ffffff;
  --surface: #f6f8fa;
  --border: #d0d7de;
  --text: #1f2328;
  --text-muted: #656d76;
  --accent: #0969da;
  --l0: #cf222e; --l1: #9a6700; --l2: #bf8700;
  --l3: #1a7f37; --l4: #116329; --l5: #0969da;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.5;
  padding: 16px; max-width: 1600px; margin: 0 auto;
}
h1 { font-size: 1.5em; margin-bottom: 4px; }
.subtitle { color: var(--text-muted); font-size: 0.85em; margin-bottom: 16px; }
.summary { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.summary-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 16px; min-width: 120px; text-align: center;
}
.summary-card .number { font-size: 1.8em; font-weight: 700; }
.summary-card .label { color: var(--text-muted); font-size: 0.8em; }
.summary-card.level-0 .number { color: var(--l0); }
.summary-card.level-1 .number { color: var(--l1); }
.summary-card.level-2 .number { color: var(--l2); }
.summary-card.level-3 .number { color: var(--l3); }
.summary-card.level-4 .number { color: var(--l4); }
.summary-card.level-5 .number { color: var(--l5); }
.filters {
  display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  margin-bottom: 16px; background: var(--surface);
  padding: 12px; border-radius: 8px; border: 1px solid var(--border);
}
.filters label { color: var(--text-muted); font-size: 0.85em; }
.filters input, .filters select {
  background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 4px; padding: 4px 8px; font-size: 0.85em;
}
.filters input[type="text"] { width: 200px; }
.toggle-group { display: flex; gap: 4px; }
.toggle-group button {
  background: var(--bg); color: var(--text-muted); border: 1px solid var(--border);
  border-radius: 4px; padding: 2px 8px; font-size: 0.8em; cursor: pointer;
}
.toggle-group button.active {
  background: var(--accent); color: var(--bg); border-color: var(--accent);
}
.table-container { overflow-x: auto; }
table {
  width: 100%; border-collapse: collapse;
  background: var(--surface); border-radius: 8px; overflow: hidden;
}
th, td {
  padding: 8px 12px; text-align: left;
  border-bottom: 1px solid var(--border); font-size: 0.85em;
}
th {
  background: var(--bg); color: var(--text-muted); font-weight: 600;
  position: sticky; top: 0; z-index: 1; cursor: pointer; user-select: none;
}
th:hover { color: var(--text); }
tr:hover { background: rgba(88, 166, 255, 0.05); }
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 12px;
  font-size: 0.75em; font-weight: 600; white-space: nowrap;
}
.badge-0 { background: rgba(248,81,73,0.2); color: var(--l0); }
.badge-1 { background: rgba(210,153,34,0.2); color: var(--l1); }
.badge-2 { background: rgba(227,179,65,0.2); color: var(--l2); }
.badge-3 { background: rgba(63,185,80,0.2); color: var(--l3); }
.badge-4 { background: rgba(46,160,67,0.2); color: var(--l4); }
.badge-5 { background: rgba(88,166,255,0.2); color: var(--l5); }
.level-dots { display: flex; gap: 4px; align-items: center; }
.dot {
  width: 12px; height: 12px; border-radius: 50%;
  border: 1px solid var(--border);
}
.dot.active-1 { background: var(--l1); border-color: var(--l1); }
.dot.active-2 { background: var(--l2); border-color: var(--l2); }
.dot.active-3 { background: var(--l3); border-color: var(--l3); }
.dot.active-4 { background: var(--l4); border-color: var(--l4); }
.dot.active-5 { background: var(--l5); border-color: var(--l5); }
.dot.failed { background: var(--l0); border-color: var(--l0); }
.dot.pending {
  background: transparent; border-color: var(--l1);
  border-style: dashed; border-width: 2px;
}
.dot.untested { background: transparent; border-color: var(--border); }
.tag {
  display: inline-block; padding: 1px 6px; border-radius: 4px;
  font-size: 0.7em; margin: 1px; background: rgba(88,166,255,0.15);
  color: var(--accent);
}
.detail-row td { padding: 12px 20px; background: var(--bg); }
.detail-row { display: none; }
.detail-row.open { display: table-row; }
.expand-btn {
  cursor: pointer; color: var(--accent); font-size: 0.9em;
  background: none; border: none;
}
.config-block {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; padding: 8px; font-family: monospace;
  font-size: 0.8em; white-space: pre; overflow-x: auto;
  margin: 4px 0; max-height: 200px; position: relative;
}
.copy-btn {
  position: absolute; top: 4px; right: 4px;
  background: var(--border); color: var(--text); border: none;
  border-radius: 4px; padding: 2px 8px; font-size: 0.75em; cursor: pointer;
}
.copy-btn:hover { background: var(--accent); color: var(--bg); }
.family-row { background: rgba(88,166,255,0.05) !important; cursor: pointer; }
.family-row td { font-weight: 600; }
.family-toggle { margin-right: 8px; }
.family-paths { font-weight: 400; }
.l3-status { font-size: 0.75em; font-weight: 600; padding: 1px 6px; border-radius: 4px; }
.l3-pass { background: rgba(63,185,80,0.2); color: var(--l3); }
.l3-xfail { background: rgba(210,153,34,0.2); color: var(--l1); }
.l3-skip { background: rgba(139,148,158,0.2); color: var(--text-muted); }
.golden-status { margin-top: 8px; padding: 8px; background: var(--surface); border: 1px solid var(--border); border-radius: 4px; font-size: 0.85em; }
.golden-status .status-row { display: flex; gap: 16px; align-items: center; margin: 2px 0; }
.golden-status .status-icon { width: 20px; text-align: center; }
.code-path-section {
  margin-top: 24px; background: var(--surface);
  border: 1px solid var(--border); border-radius: 8px; padding: 16px;
}
.code-path-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px; margin-top: 12px;
}
.code-path-card {
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 6px; padding: 10px;
}
.code-path-card h4 { font-size: 0.9em; margin-bottom: 4px; }
.code-path-card .desc { color: var(--text-muted); font-size: 0.8em; }
.code-path-card .coverage { font-size: 0.85em; margin-top: 4px; }
.progress {
  height: 6px; background: var(--border); border-radius: 3px;
  margin-top: 4px; overflow: hidden;
}
.progress-fill { height: 100%; border-radius: 3px; }
.footer {
  margin-top: 24px; padding-top: 12px;
  border-top: 1px solid var(--border);
  color: var(--text-muted); font-size: 0.8em;
  display: flex; justify-content: space-between;
}
body, .summary-card, .filters, .filters input, .filters select,
table, th, .detail-row td, .config-block, .code-path-section,
.code-path-card, .toggle-group button, .copy-btn, .footer {
  transition: background-color 0.25s ease, color 0.25s ease,
              border-color 0.25s ease;
}
.header-row {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 4px;
}
.theme-toggle {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 10px; cursor: pointer;
  font-size: 1.2em; line-height: 1; color: var(--text);
  transition: background-color 0.25s ease, border-color 0.25s ease;
}
.theme-toggle:hover { background: var(--border); }
.dot-legend {
  display: flex; gap: 12px; align-items: center;
  margin-left: auto; font-size: 0.8em; color: var(--text-muted);
}
.dot-legend-item { display: flex; gap: 4px; align-items: center; }
"""

_FILTERS_BLOCK = """\
<div class="filters">
  <label>Search:</label>
  <input type="text" id="search" placeholder="Filter by model type...">
  <label>Category:</label>
  <select id="category-filter"><option value="">All</option></select>
  <label>Min Level:</label>
  <select id="level-filter">
    <option value="0">Any</option>
    <option value="1">L1+</option>
    <option value="2">L2+</option>
    <option value="3">L3+</option>
    <option value="4">L4+</option>
    <option value="5">L5</option>
  </select>
  <label>Group by:</label>
  <div class="toggle-group" id="group-toggle" role="group" aria-label="Grouping mode">
    <button class="active" data-group="family" aria-pressed="true">Family</button>
    <button data-group="flat" aria-pressed="false">Flat</button>
  </div>
  <div class="dot-legend">
    <span class="dot-legend-item"><span class="dot active-3" style="display:inline-block;vertical-align:middle"></span> Passed</span>
    <span class="dot-legend-item"><span class="dot failed" style="display:inline-block;vertical-align:middle"></span> Failed</span>
    <span class="dot-legend-item"><span class="dot pending" style="display:inline-block;vertical-align:middle"></span> Has test, not run</span>
    <span class="dot-legend-item"><span class="dot untested" style="display:inline-block;vertical-align:middle"></span> No test</span>
  </div>
</div>
"""

_TABLE_BLOCK = """\
<div class="table-container">
  <table id="model-table" role="grid">
    <thead>
      <tr>
        <th data-sort="model_type" style="width: 20%" aria-sort="none">Model Type</th>
        <th data-sort="category" style="width: 10%" aria-sort="none">Category</th>
        <th data-sort="module_class" style="width: 15%" aria-sort="none">Module Class</th>
        <th data-sort="confidence_level" style="width: 8%" aria-sort="none">Confidence</th>
        <th style="width: 12%" title="Each dot represents a testing level (L1\u2013L5)">Coverage (L1\u2013L5)</th>
        <th style="width: 15%">Code Paths</th>
        <th style="width: 5%"></th>
      </tr>
    </thead>
    <tbody id="model-tbody"></tbody>
  </table>
</div>
"""

_CODE_PATH_SECTION = """\
<div class="code-path-section">
  <h2>Code Path Coverage</h2>
  <p class="subtitle">Which architectural code paths are exercised by test configs</p>
  <div class="code-path-grid" id="code-path-grid"></div>
</div>
"""

_MISSING_SECTION = """\
<div class="code-path-section" style="margin-top: 16px;">
  <h2>Missing Coverage</h2>
  <p class="subtitle">Models without any test config &mdash; copy-paste these to add coverage</p>
  <div id="missing-coverage"></div>
</div>
"""

_FOOTER_BLOCK = """\
<div class="footer">
  <span>mobius confidence dashboard</span>
  <span>Data from static analysis &mdash; no test execution required</span>
</div>
"""

_JS_BLOCK = r"""
// Escape HTML entities to prevent XSS when inserting into innerHTML.
function esc(s) {
  const el = document.createElement('span');
  el.textContent = s;
  return el.innerHTML;
}

// LEVEL_LABELS is injected from Python as {{LABELS_JSON}}

// --- Summary bar ---
const LEVEL_DESCRIPTIONS = {
  0: "No test coverage at all",
  1: "ONNX graph builds from tiny config",
  2: "Full-size HuggingFace config produces valid graph",
  3: "Random-weight forward pass matches HuggingFace (atol)",
  4: "Real-weight logits match golden reference",
  5: "Full generation output matches golden reference"
};

(function renderSummary() {
  const bar = document.getElementById('summary-bar');
  const totalCard = `<div class="summary-card">
    <div class="number">${SUMMARY.total}</div>
    <div class="label">Total models</div>
  </div>`;
  bar.innerHTML = totalCard;
  for (let i = 0; i <= 5; i++) {
    const count = SUMMARY.by_level[i] || 0;
    bar.innerHTML += `<div class="summary-card level-${i}" title="${LEVEL_DESCRIPTIONS[i]}">
      <div class="number">${count}</div>
      <div class="label">${LEVEL_LABELS[i]}</div>
    </div>`;
  }
  // L3 parity breakdown
  const l3s = SUMMARY.l3_status_counts || {};
  if (l3s.pass || l3s.xfail || l3s.skip) {
    bar.innerHTML += `<div class="summary-card" title="L3 parity status breakdown">
      <div class="number" style="font-size:1em;line-height:1.4">
        <span style="color:var(--l3)">${l3s.pass || 0}\u2713</span>
        <span style="color:var(--l1)">${l3s.xfail || 0}\u26A0</span>
        <span style="color:var(--text-muted)">${l3s.skip || 0}\u23ED</span>
      </div>
      <div class="label">L3 Parity Status</div>
    </div>`;
  }
  // Golden case coverage
  bar.innerHTML += `<div class="summary-card" title="YAML test cases defined for golden testing">
    <div class="number" style="font-size:1.2em">
      <span style="color:var(--l4)">${SUMMARY.l4_case_count || 0}</span> /
      <span style="color:var(--l5)">${SUMMARY.l5_case_count || 0}</span>
    </div>
    <div class="label">L4/L5 Cases</div>
  </div>`;
})();

// --- Category filter ---
(function populateCategoryFilter() {
  const sel = document.getElementById('category-filter');
  const cats = [...new Set(MODEL_DATA.map(m => m.category))].sort();
  cats.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c; opt.textContent = c;
    sel.appendChild(opt);
  });
})();

// --- State ---
let currentSort = { key: 'model_type', asc: true };
let groupByFamily = true;
let expandedFamilies = new Set();
let expandedDetails = new Set();

// --- Rendering ---
function renderTable() {
  const tbody = document.getElementById('model-tbody');
  const search = document.getElementById('search').value.toLowerCase();
  const catFilter = document.getElementById('category-filter').value;
  const levelFilter = parseInt(document.getElementById('level-filter').value);

  let filtered = MODEL_DATA.filter(m => {
    if (search && !m.model_type.toLowerCase().includes(search)
        && !m.module_class.toLowerCase().includes(search)) return false;
    if (catFilter && m.category !== catFilter) return false;
    if (m.confidence_level < levelFilter) return false;
    return true;
  });

  // Sort
  filtered.sort((a, b) => {
    let va = a[currentSort.key], vb = b[currentSort.key];
    if (typeof va === 'string') va = va.toLowerCase();
    if (typeof vb === 'string') vb = vb.toLowerCase();
    if (va < vb) return currentSort.asc ? -1 : 1;
    if (va > vb) return currentSort.asc ? 1 : -1;
    return 0;
  });

  let html = '';
  if (groupByFamily) {
    const groups = {};
    filtered.forEach(m => {
      if (!groups[m.family]) groups[m.family] = [];
      groups[m.family].push(m);
    });
    const sortedFamilies = Object.keys(groups).sort();
    sortedFamilies.forEach(fam => {
      const models = groups[fam];
      const minLevel = Math.min(...models.map(m => m.confidence_level));
      const expanded = expandedFamilies.has(fam);
      const arrow = expanded ? '\u25BC' : '\u25B6';
      // Aggregate code paths for the family
      const famPaths = [...new Set(models.flatMap(m => m.code_paths))].sort();
      const pathTags = famPaths.map(p => `<span class="tag">${p}</span>`).join('');
      html += `<tr class="family-row" onclick="toggleFamily('${esc(fam)}')">
        <td><span class="family-toggle">${arrow}</span>${esc(fam)} <span style="color:var(--text-muted)">(x${models.length})</span></td>
        <td></td><td></td>
        <td><span class="badge badge-${minLevel}">${LEVEL_LABELS[minLevel]}</span></td>
        <td></td>
        <td class="family-paths">${pathTags}</td>
        <td></td>
      </tr>`;
      if (expanded) {
        models.forEach(m => { html += renderModelRow(m); });
      }
    });
  } else {
    filtered.forEach(m => { html += renderModelRow(m); });
  }
  tbody.innerHTML = html;
}

function renderModelRow(m) {
  const dotLabels = ['L1: Graph', 'L2: Config', 'L3: Parity', 'L4: Golden', 'L5: Generation'];
  const dots = [1,2,3,4,5].map(i => {
    const levelKey = 'l' + i;
    // For L3, only consider it "active" (passing) if l3_status is 'pass'
    const active = (i === 3) ? (m.l3 && m.l3_status === 'pass') : m[levelKey];
    // Determine whether a test exists for this level
    const hasTest = (i === 1 && m.config_overrides.length > 0)
      || (i === 2 && m.test_model_id)
      || (i === 3 && m.l3_status != null)
      || (i === 4 && m.l4_case)
      || (i === 5 && m.l5_case);
    // Determine whether the test is in a "pending" state:
    // test infrastructure exists but hasn't been run or is expected to fail
    const isPending = hasTest && !active && (
      (i === 3 && (m.l3_status === 'xfail' || m.l3_status === 'skip'))
      || (i === 4 && m.l4_case)
      || (i === 5 && m.l5_case)
    );
    let cls = 'dot';
    let label = dotLabels[i-1];
    if (active) {
      cls += ' active-' + i;
      label += ' (passed)';
    } else if (isPending) {
      cls += ' pending';
      if (i === 3) label += ` (${m.l3_status}: ${esc(m.l3_reason || 'known issue')})`;
      else label += ' (test case exists, no golden file)';
    } else if (hasTest) {
      cls += ' failed';
      label += ' (test exists, not passing)';
    } else {
      cls += ' untested';
      label += ' (no test)';
    }
    return `<div class="${cls}" title="${label}" role="img" aria-label="${label}"></div>`;
  }).join('');

  // L3 status badge
  let l3Badge = '';
  if (m.l3_status === 'pass') {
    l3Badge = ' <span class="l3-status l3-pass" title="L3 synthetic parity passes">\u2713</span>';
  } else if (m.l3_status === 'xfail') {
    l3Badge = ` <span class="l3-status l3-xfail" title="L3 xfail: ${esc(m.l3_reason || '')}">xfail</span>`;
  } else if (m.l3_status === 'skip') {
    l3Badge = ` <span class="l3-status l3-skip" title="L3 skip: ${esc(m.l3_reason || '')}">skip</span>`;
  }

  const tags = m.code_paths.map(p =>
    `<span class="tag">${esc(p)}</span>`
  ).join('');
  const expanded = expandedDetails.has(m.model_type);

  let row = `<tr>
    <td style="padding-left: ${groupByFamily ? '32px' : '12px'}">${esc(m.model_type)}${l3Badge}</td>
    <td>${esc(m.category)}</td>
    <td><code style="font-size:0.8em">${esc(m.module_class)}</code></td>
    <td><span class="badge badge-${m.confidence_level}">${LEVEL_LABELS[m.confidence_level]}</span></td>
    <td><div class="level-dots" role="img" aria-label="Coverage: ${LEVEL_LABELS[m.confidence_level]}">${dots}</div></td>
    <td>${tags || '<span style="color:var(--text-muted)">none</span>'}</td>
    <td><button class="expand-btn" aria-expanded="${expanded}" aria-label="Expand details for ${m.model_type}" onclick="toggleDetail('${m.model_type}')">${expanded ? '\u2212' : '+'}</button></td>
  </tr>`;

  if (expanded) {
    row += renderDetailRow(m);
  }
  return row;
}

function renderDetailRow(m) {
  const l3Active = m.l3 && m.l3_status === 'pass';
  const l3Pending = !l3Active && (m.l3_status === 'xfail' || m.l3_status === 'skip');
  const levels = [
    ['L1: Graph Build', m.l1, 'Model builds a valid ONNX graph from tiny config', m.config_overrides.length > 0, false],
    ['L2: Config Compatible', m.l2, 'Full-size HF config produces valid graph', !!m.test_model_id, false],
    ['L3: Synthetic Parity', l3Active, 'Random-weight forward pass matches HF (atol)', m.l3_status != null, l3Pending],
    ['L4: Golden Match', m.l4, 'Real-weight logits match golden reference', m.l4_case, m.l4_case && !m.l4],
    ['L5: Generation', m.l5, 'Full generation matches golden output', m.l5_case, m.l5_case && !m.l5],
  ];

  let levelHtml = '<table style="width:100%;margin-bottom:8px;border:none">';
  levels.forEach(([name, active, desc, hasTest, pending]) => {
    let icon;
    if (active) icon = '\u2705';
    else if (pending) icon = '\u25D4';
    else if (hasTest) icon = '\u274C';
    else icon = '\u2796';
    levelHtml += `<tr><td style="border:none;padding:2px 8px;width:30px">${icon}</td>
      <td style="border:none;padding:2px 8px;font-weight:600">${name}</td>
      <td style="border:none;padding:2px 8px;color:var(--text-muted)">${desc}</td></tr>`;
  });
  levelHtml += '</table>';

  // Golden test case status
  let goldenHtml = '<div class="golden-status">';
  goldenHtml += '<strong>Golden Test Status:</strong>';
  if (m.yaml_case) {
    goldenHtml += `<div class="status-row"><span class="status-icon">\u2705</span> Test case: <code>${esc(m.yaml_case)}</code></div>`;
  } else {
    goldenHtml += '<div class="status-row"><span class="status-icon">\u274C</span> No YAML test case defined</div>';
  }
  goldenHtml += `<div class="status-row"><span class="status-icon">${m.l4 ? '\u2705' : m.l4_case ? '\u274C' : '\u2796'}</span> L4 golden data: ${m.l4 ? 'available' : m.l4_case ? 'test case exists, run generate_golden.py' : 'none'}</div>`;
  goldenHtml += `<div class="status-row"><span class="status-icon">${m.l5 ? '\u2705' : m.l5_case ? '\u274C' : '\u2796'}</span> L5 golden data: ${m.l5 ? 'available' : m.l5_case ? 'test case exists, run generate_golden.py' : 'none'}</div>`;

  // L3 parity status
  if (m.l3_status) {
    const statusColor = m.l3_status === 'pass' ? 'var(--l3)' : m.l3_status === 'xfail' ? 'var(--l1)' : 'var(--text-muted)';
    goldenHtml += `<div class="status-row"><span class="status-icon" style="color:${statusColor}">\u25CF</span> L3 parity: <strong>${m.l3_status}</strong>`;
    if (m.l3_reason) {
      goldenHtml += ` <span style="color:var(--text-muted)">\u2014 ${esc(m.l3_reason)}</span>`;
    }
    goldenHtml += '</div>';
  }
  goldenHtml += '</div>';

  let configHtml = '';
  if (m.config_overrides.length > 0) {
    configHtml = '<strong>Test configs:</strong>';
    m.config_overrides.forEach((cfg, i) => {
      const cfgStr = JSON.stringify(cfg, null, 2);
      configHtml += `<div class="config-block" id="cfg-${m.model_type}-${i}">` +
        `<button class="copy-btn" onclick="copyConfig('cfg-${m.model_type}-${i}')">Copy</button>` +
        cfgStr + '</div>';
    });
  } else {
    configHtml = '<span style="color:var(--text-muted)">No test configs \u2014 add to tests/_test_configs.py</span>';
  }

  let metaHtml = `<div style="display:flex;gap:24px;margin-top:8px;font-size:0.85em">
    <div><strong>Task:</strong> ${esc(m.task)}</div>
    <div><strong>Module:</strong> ${esc(m.module_class)}</div>`;
  if (m.test_model_id) {
    metaHtml += `<div><strong>Test model:</strong> ${esc(m.test_model_id)}</div>`;
  }
  if (m.has_integration_test) {
    metaHtml += '<div>\u2705 Has integration test</div>';
  }
  metaHtml += '</div>';

  return `<tr class="detail-row open"><td colspan="7">
    ${levelHtml}${goldenHtml}${configHtml}${metaHtml}
  </td></tr>`;
}

// --- Interactions ---
function toggleFamily(fam) {
  if (expandedFamilies.has(fam)) expandedFamilies.delete(fam);
  else expandedFamilies.add(fam);
  renderTable();
}

function toggleDetail(mt) {
  if (expandedDetails.has(mt)) expandedDetails.delete(mt);
  else expandedDetails.add(mt);
  renderTable();
}

function copyConfig(id) {
  const el = document.getElementById(id);
  const text = el.textContent.replace('Copy', '').trim();
  navigator.clipboard.writeText(text);
  const btn = el.querySelector('.copy-btn');
  btn.textContent = 'Copied!';
  setTimeout(() => btn.textContent = 'Copy', 1500);
}

// --- Sorting ---
document.querySelectorAll('th[data-sort]').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.sort;
    if (currentSort.key === key) currentSort.asc = !currentSort.asc;
    else { currentSort.key = key; currentSort.asc = true; }
    renderTable();
  });
});

// --- Filters ---
document.getElementById('search').addEventListener('input', renderTable);
document.getElementById('category-filter').addEventListener('change', renderTable);
document.getElementById('level-filter').addEventListener('change', renderTable);

// --- Group toggle ---
document.querySelectorAll('#group-toggle button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#group-toggle button').forEach(b => {
      b.classList.remove('active');
      b.setAttribute('aria-pressed', 'false');
    });
    btn.classList.add('active');
    btn.setAttribute('aria-pressed', 'true');
    groupByFamily = btn.dataset.group === 'family';
    renderTable();
  });
});

// --- Code path coverage grid ---
(function renderCodePaths() {
  const grid = document.getElementById('code-path-grid');
  const total = MODEL_DATA.filter(m => m.l1).length;
  CODE_PATH_INFO.forEach(cp => {
    const count = SUMMARY.code_path_coverage[cp.label] || 0;
    const pct = total > 0 ? Math.round(count / total * 100) : 0;
    const color = pct > 50 ? 'var(--l3)' : pct > 20 ? 'var(--l2)' : 'var(--l0)';
    grid.innerHTML += `<div class="code-path-card">
      <h4><span class="tag">${esc(cp.label)}</span> ${esc(cp.description)}</h4>
      <div class="coverage">${count} / ${total} models (${pct}%)</div>
      <div class="progress"><div class="progress-fill" style="width:${pct}%;background:${color}"></div></div>
      <div class="config-block" style="margin-top:8px;max-height:100px;font-size:0.75em">${JSON.stringify(cp.example_config, null, 2)}</div>
    </div>`;
  });
})();

// --- Missing coverage ---
(function renderMissing() {
  const el = document.getElementById('missing-coverage');
  const missing = MODEL_DATA.filter(m => !m.l1);
  if (missing.length === 0) {
    el.innerHTML = '<p style="color:var(--l3)">\u2705 All models have at least L1 coverage!</p>';
    return;
  }
  const groups = {};
  missing.forEach(m => {
    if (!groups[m.category]) groups[m.category] = [];
    groups[m.category].push(m);
  });
  let html = `<p style="color:var(--l0)">${missing.length} models have no test coverage</p>`;
  Object.keys(groups).sort().forEach(cat => {
    html += `<h4 style="margin-top:12px">${cat}</h4><ul style="list-style:none;padding-left:8px">`;
    groups[cat].forEach(m => {
      html += `<li style="margin:2px 0"><code style="font-size:0.85em">` +
        `("${m.model_type}", {}, True),  # TODO: add to ${cat} configs</code></li>`;
    });
    html += '</ul>';
  });
  el.innerHTML = html;
})();

// Initial render
renderTable();

// --- Theme toggle ---
(function initThemeToggle() {
  const btn = document.getElementById('theme-toggle');
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('dashboard-theme', theme);
    btn.textContent = theme === 'dark' ? '\u{1F319}' : '\u{2600}\u{FE0F}';
    btn.setAttribute('aria-label',
      theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
  }
  applyTheme(localStorage.getItem('dashboard-theme') || 'dark');
  btn.addEventListener('click', function() {
    var cur = document.documentElement.getAttribute('data-theme') || 'dark';
    applyTheme(cur === 'dark' ? 'light' : 'dark');
  });
})();
"""


_HTML_TEMPLATE = (
    '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
    '<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    "<title>ONNX GenAI Models — Testing Confidence Dashboard</title>\n"
    "<style>\n" + _CSS_BLOCK + "</style>\n</head>\n<body>\n"
    "<script>\n"
    "(function(){var t=localStorage.getItem('dashboard-theme')||'dark';"
    "document.documentElement.setAttribute('data-theme',t);})()\n"
    "</script>\n"
    '<noscript><p style="color:#e6edf3;padding:2em">'
    "This dashboard requires JavaScript to render.</p></noscript>\n"
    '<div class="header-row">\n'
    '<h1><span aria-hidden="true">\U0001f9ea</span>'
    " Testing Confidence Dashboard</h1>\n"
    '<button class="theme-toggle" id="theme-toggle"'
    ' aria-label="Toggle light/dark mode"'
    ' title="Toggle light/dark mode">\U0001f319</button>\n'
    "</div>\n"
    '<p class="subtitle">\n'
    "  Generated {{TIMESTAMP}} &middot; Commit <code>{{COMMIT}}</code>"
    " &middot; {{TOTAL_MODELS}} registered model types\n</p>\n"
    '<div class="summary" role="region" aria-label="Coverage summary"'
    ' id="summary-bar"></div>\n'
    + _FILTERS_BLOCK
    + _TABLE_BLOCK
    + _CODE_PATH_SECTION
    + _MISSING_SECTION
    + _FOOTER_BLOCK
    + "<script>\n"
    + "const MODEL_DATA = {{MODEL_DATA_JSON}};\n"
    + "const CODE_PATH_INFO = {{CODE_PATH_JSON}};\n"
    + "const SUMMARY = {{SUMMARY_JSON}};\n"
    + "const FAMILY_DATA = {{FAMILY_JSON}};\n"
    + "const LEVEL_LABELS = {{LABELS_JSON}};\n"
    + _JS_BLOCK
    + "</script>\n</body>\n</html>\n"
)


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
    html_content = _generate_html(models, commit=args.commit)

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
