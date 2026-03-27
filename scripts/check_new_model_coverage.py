#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

r"""Check that newly registered model architectures have test coverage.

For each model_type registered in ``src/mobius/_registry.py``, verifies:

    a. L1+L3 — has a parametrized test config in ``tests/_test_configs.py``,
       which enables both the L1 graph-build test (build_graph_test.py) and
       the L3 synthetic-parity test (integration_test.py).
    b. L4 — has a YAML test case in ``testdata/cases/`` matched via the
       model's ``test_model_id`` (exact HuggingFace model ID match).
    c. L5 — has golden data in ``testdata/golden/`` (or a ``skip_reason``
       in the YAML case).

Usage::

    # Audit all registered models (report all gaps, exit 1 if any found)
    python scripts/check_new_model_coverage.py

    # CI mode: only fail on models that are NEW in this PR vs base branch
    python scripts/check_new_model_coverage.py --diff-base origin/main

    # Show only missing, one-line-per-model
    python scripts/check_new_model_coverage.py --quiet
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src" / "mobius"
_TESTS_DIR = _PROJECT_ROOT / "tests"
_CASES_DIR = _PROJECT_ROOT / "testdata" / "cases"
_GOLDEN_DIR = _PROJECT_ROOT / "testdata" / "golden"

# model_types that are intentionally VL sub-models registered separately.
# These are text-decoder or embedding sub-models that reuse the VL parent's
# test coverage (YAML + golden are keyed on the parent model_type, not the
# text-only variant). Excluding them avoids spurious L4/L5 failures.
_VL_TEXT_SUFFIX_ALIASES: frozenset[str] = frozenset(
    {
        "_text",
        "_multimodal",
    }
)


def _get_all_registered_types() -> list[str]:
    """Return all model_types from the live registry."""
    sys.path.insert(0, str(_SRC_ROOT.parent))
    from mobius._registry import registry

    return sorted(registry.architectures())


def _get_l3_types() -> set[str]:
    """Return model_types that have a parametrized test config in _test_configs.py.

    These configs enable both L1 (graph-build) tests in build_graph_test.py
    and L3 (synthetic parity) tests in integration_test.py.
    """
    sys.path.insert(0, str(_TESTS_DIR))
    from _test_configs import (
        ALL_CAUSAL_LM_CONFIGS,
        ENCODER_CONFIGS,
        SEQ2SEQ_CONFIGS,
        VISION_CONFIGS,
    )

    types: set[str] = set()
    for mt, _, _ in ALL_CAUSAL_LM_CONFIGS:
        types.add(mt)
    for mt, _, _ in ENCODER_CONFIGS:
        types.add(mt)
    for mt, _, _ in SEQ2SEQ_CONFIGS:
        types.add(mt)
    for mt, _, _ in VISION_CONFIGS:
        types.add(mt)
    return types


def _get_test_model_ids() -> dict[str, str]:
    """Return ``{model_type: test_model_id}`` from the live registry.

    test_model_id is the HuggingFace model ID used for L2 config validation.
    It is also used here to match YAML test cases by exact model_id value.
    """
    sys.path.insert(0, str(_SRC_ROOT.parent))
    from mobius._registry import registry

    result: dict[str, str] = {}
    for arch in registry.architectures():
        reg = registry.get_registration(arch)
        if reg and reg.test_model_id:
            result[arch] = reg.test_model_id
    return result


def _get_yaml_model_type_map() -> dict[str, dict]:
    """Return ``{hf_model_id: yaml_data}`` for all YAML test cases.

    Keyed by the ``model_id`` field inside each YAML file (the exact
    HuggingFace model ID), enabling exact matching against registry
    ``test_model_id`` values. A ``_yaml_path`` key is added to each entry
    to allow golden file path derivation.
    """
    import yaml

    cases: dict[str, dict] = {}
    for yaml_path in _CASES_DIR.rglob("*.yaml"):
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "model_id" in data:
                entry = dict(data)
                entry["_yaml_path"] = yaml_path
                cases[data["model_id"]] = entry
        except Exception:
            pass
    return cases


def _get_golden_stems() -> set[str]:
    """Return set of golden JSON stems (without _generation suffix)."""
    stems: set[str] = set()
    for p in _GOLDEN_DIR.rglob("*.json"):
        name = p.stem
        if not name.endswith("_generation"):
            stems.add(name)
    return stems


def _get_new_model_types(diff_base: str) -> set[str] | None:
    """Return model_types newly added vs diff_base, or None if git fails.

    Detects new ``reg.register(...)`` calls added to ``_registry.py``.
    Adding entries to ``_TEST_MODEL_IDS`` does NOT count as a new model
    registration — only actual ``reg.register()`` calls are tracked.
    """
    try:
        result = subprocess.run(
            ["git", "diff", f"{diff_base}...HEAD", "--", "src/mobius/_registry.py"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            return None
        diff_text = result.stdout
    except FileNotFoundError:
        return None

    if not diff_text.strip():
        return None

    # Only detect lines like:  +    reg.register("model_type", ...)
    # Ignoring _TEST_MODEL_IDS additions (those are test metadata, not new models).
    import re

    added_types: set[str] = set()
    register_pattern = re.compile(r'^\+\s+reg\.register\(\s*"([^"]+)"')

    for line in diff_text.splitlines():
        m = register_pattern.match(line)
        if m:
            added_types.add(m.group(1))

    return added_types if added_types else set()


def _is_vl_text_alias(model_type: str) -> bool:
    """Return True if model_type is a VL sub-model text alias.

    These variants (e.g. ``qwen2_vl_text``) share YAML + golden with
    their parent (``qwen2_vl``) and don't need separate test cases.
    """
    for suffix in _VL_TEXT_SUFFIX_ALIASES:
        if model_type.endswith(suffix):
            base = model_type[: -len(suffix)]
            # Only flag as alias if stripping the suffix yields a non-empty base name.
            return len(base) > 0
    return False


def _check_coverage(
    model_types: list[str],
    l3_types: set[str],
    yaml_cases: dict[str, dict],
    golden_stems: set[str],
    test_model_ids: dict[str, str],
) -> dict[str, list[str]]:
    """Return {model_type: [list of missing coverage items]} for each type.

    YAML coverage uses exact matching: a model's ``test_model_id`` (from the
    registry) must appear as the ``model_id`` value in a YAML test case.
    This prevents false positives from substring matching (e.g. 'bert' ⊂
    'roberta', 't5' ⊂ 'mt5').
    """
    gaps: dict[str, list[str]] = {}

    for mt in model_types:
        if _is_vl_text_alias(mt):
            continue

        missing: list[str] = []

        # L1+L3: test config in _test_configs.py (enables both graph-build and parity tests)
        if mt not in l3_types:
            missing.append("No test config in tests/_test_configs.py (needed for L1+L3)")

        # L4/L5: YAML test case matched via exact test_model_id
        test_model_id = test_model_ids.get(mt)
        if not test_model_id:
            # Without a test_model_id we cannot map to a YAML case.
            missing.append(
                "No test_model_id in _registry.py "
                "(required for L2 config validation and YAML test case matching)"
            )
        elif test_model_id not in yaml_cases:
            missing.append(
                f"No YAML test case in testdata/cases/ (expected model_id: {test_model_id!r})"
            )
        else:
            yaml_data = yaml_cases[test_model_id]
            has_skip = bool(yaml_data.get("skip_reason"))
            if not has_skip:
                # Has active YAML case — check for golden data.
                # Strategy 1: YAML-derived path (case_id may differ from model_type).
                yaml_path: Path = yaml_data["_yaml_path"]
                case_id = yaml_path.stem
                task_dir = yaml_path.parent.name
                has_golden = (_GOLDEN_DIR / task_dir / f"{case_id}.json").exists()
                # Strategy 2: direct model_type stem in any golden subdirectory.
                if not has_golden:
                    has_golden = any(
                        (d / f"{mt}.json").exists()
                        for d in _GOLDEN_DIR.iterdir()
                        if d.is_dir()
                    )
                if not has_golden:
                    missing.append(
                        "No golden data in testdata/golden/ "
                        "(run generate_golden.py or set skip_reason in YAML)"
                    )

        if missing:
            gaps[mt] = missing

    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check test coverage for registered model architectures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--diff-base",
        metavar="REF",
        help=(
            "Git ref to compare against (e.g. origin/main). "
            "When set, only newly registered model_types are checked. "
            "Without this flag, all registered types are audited."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only model_types with gaps, one per line.",
    )
    args = parser.parse_args()

    all_types = _get_all_registered_types()
    l3_types = _get_l3_types()
    yaml_cases = _get_yaml_model_type_map()
    golden_stems = _get_golden_stems()
    test_model_ids = _get_test_model_ids()

    if args.diff_base:
        new_types = _get_new_model_types(args.diff_base)
        if new_types is None:
            print(
                f"⚠️  Could not determine new model types from git diff against "
                f"{args.diff_base!r}. Skipping coverage check.",
                file=sys.stderr,
            )
            return 0
        types_to_check = sorted(t for t in all_types if t in new_types)
        if not types_to_check:
            print("✅ No new model_types detected in this diff. Coverage check skipped.")
            return 0
        print(f"🔍 Checking coverage for {len(types_to_check)} new model type(s):")
        for t in types_to_check:
            print(f"   {t}")
        print()
    else:
        types_to_check = all_types
        print(f"🔍 Auditing coverage for all {len(types_to_check)} registered model types.")
        print()

    gaps = _check_coverage(types_to_check, l3_types, yaml_cases, golden_stems, test_model_ids)

    if not gaps:
        if args.diff_base:
            print("✅ All new model types have required test coverage.")
        else:
            print("✅ All registered model types have test coverage.")
        return 0

    # Print gaps
    if args.quiet:
        for mt in sorted(gaps):
            print(mt)
        return 1

    if args.diff_base:
        print(f"❌ {len(gaps)} new model type(s) missing test coverage:\n")
    else:
        print(f"❌ {len(gaps)} model type(s) missing test coverage:\n")

    for mt in sorted(gaps):
        items = gaps[mt]
        print(f"  ❌ New model {mt!r} missing coverage:")
        for item in items:
            print(f"     - {item}")
        print()

    print(
        "To fix: See .github/skills/writing-tests/SKILL.md for how to add test coverage.\n"
        "Quick guide:\n"
        "  1. Add a config entry to tests/_test_configs.py (enables L1 + L3)\n"
        "  2. Add test_model_id to _TEST_MODEL_IDS in src/mobius/_registry.py\n"
        "  3. Add a YAML case to testdata/cases/<task-type>/<model>.yaml (L4/L5)\n"
        "     The YAML model_id must match the test_model_id exactly.\n"
        "  4. Run: python scripts/generate_golden.py --filter <model_id> to create golden data\n"
        "     Or add skip_reason to the YAML if golden generation is not feasible."
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())
