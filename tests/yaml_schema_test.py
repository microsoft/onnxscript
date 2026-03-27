# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Schema validation tests for YAML test case files in testdata/cases/.

Validates every ``.yaml`` file in ``testdata/cases/`` against the JSON Schema
at ``testdata/cases/schema.json``. These tests are fast (no model downloads,
no ONNX inference) and run as part of the standard unit test suite.

Run::

    pytest tests/yaml_schema_test.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CASES_DIR = _REPO_ROOT / "testdata" / "cases"
_SCHEMA_PATH = _CASES_DIR / "schema.json"


def _load_schema() -> dict[str, Any]:
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


def _all_yaml_files() -> list[Path]:
    """Return all .yaml files under testdata/cases/, sorted for stable ordering."""
    return sorted(_CASES_DIR.rglob("*.yaml"))


# ---------------------------------------------------------------------------
# Parametrized validation test
# ---------------------------------------------------------------------------

_YAML_FILES = _all_yaml_files()
_SCHEMA = _load_schema()
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)


@pytest.mark.parametrize(
    "yaml_path",
    _YAML_FILES,
    ids=[f.relative_to(_CASES_DIR).as_posix() for f in _YAML_FILES],
)
def test_yaml_validates_against_schema(yaml_path: Path) -> None:
    """Each YAML test case must conform to testdata/cases/schema.json."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    errors = sorted(_VALIDATOR.iter_errors(data), key=lambda e: e.json_path)
    if errors:
        messages = "\n".join(f"  [{e.json_path}] {e.message}" for e in errors)
        pytest.fail(f"{yaml_path.relative_to(_REPO_ROOT)}:\n{messages}")


# ---------------------------------------------------------------------------
# Schema self-consistency tests
# ---------------------------------------------------------------------------


def test_schema_file_exists() -> None:
    """schema.json must be present at the expected path."""
    assert _SCHEMA_PATH.exists(), f"Schema file not found: {_SCHEMA_PATH}"


def test_schema_is_valid_json_schema() -> None:
    """schema.json itself must be a valid JSON Schema (meta-validation)."""
    meta_validator = jsonschema.Draft202012Validator(
        jsonschema.Draft202012Validator.META_SCHEMA
    )
    errors = list(meta_validator.iter_errors(_SCHEMA))
    if errors:
        messages = "\n".join(f"  {e.message}" for e in errors)
        pytest.fail(f"schema.json is not a valid JSON Schema:\n{messages}")


def test_at_least_one_yaml_found() -> None:
    """Sanity check: the discovery function must find at least one test case."""
    assert len(_YAML_FILES) > 0, f"No .yaml files found in {_CASES_DIR}"


def test_all_yaml_task_types_are_in_schema() -> None:
    """Every task_type value used in actual YAML files must be in the schema enum.

    This catches the case where a new YAML uses a task_type that the schema
    doesn't know about yet — the schema enum needs to be updated.
    """
    schema_task_types: set[str] = set(_SCHEMA["properties"]["task_type"]["enum"])
    missing: list[str] = []
    for yaml_path in _YAML_FILES:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        task_type = data.get("task_type") if isinstance(data, dict) else None
        if task_type and task_type not in schema_task_types:
            missing.append(f"  {yaml_path.relative_to(_REPO_ROOT)}: task_type={task_type!r}")
    if missing:
        pytest.fail(
            "These YAML files use task_type values not listed in schema.json. "
            "Add them to the task_type enum in testdata/cases/schema.json:\n"
            + "\n".join(missing)
        )
