# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Golden file loading, saving, and test case discovery for L4/L5 tests.

This module provides the data layer for golden-file-based testing:
- Load/save YAML test case definitions
- Load/save .json golden reference data
- Discover test cases by task type for pytest parametrization

Golden files store pre-computed HuggingFace reference outputs (.json)
alongside YAML test case definitions.  The design is data-driven:
adding test coverage = adding a YAML file.  No code changes needed.

Multi-model tasks (e.g. vision-language) use namespaced keys in the
JSON to store per-component data (e.g. component_norms, component_shapes).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import yaml

# --- Constants ---

# Paths are relative to the repository root.  Callers can override
# the root in discover_test_cases() and load_tolerances().
CASES_DIR = Path("testdata/cases")
GOLDEN_DIR = Path("testdata/golden")
DEFAULT_TOLERANCES_PATH = Path("testdata/default_tolerances.yaml")

# Hard-coded fallback tolerances so tests work even without a YAML file.
_FALLBACK_TOLERANCES: dict[str, dict[str, float]] = {
    "L4": {
        "near_tie_margin": 0.01,
        "top10_jaccard_warn": 0.7,
        "cosine_similarity_warn": 0.999,
        "min_token_match_ratio": 1.0,
    },
    "L5": {
        "near_tie_margin": 0.01,
        "top10_jaccard_warn": 0.7,
        "cosine_similarity_warn": 0.999,
        "min_token_match_ratio": 1.0,
    },
}


# --- Data Classes ---


@dataclasses.dataclass(frozen=True)
class GoldenTestCase:
    """A parsed YAML test case definition.

    Each test case describes one model + one input scenario.
    Fields map 1:1 to the YAML schema defined in the design doc.
    """

    case_id: str
    """Stem of the YAML file (e.g. ``qwen2_5-0_5b``)."""

    task_type: str
    """One of the task strings from ``_registry.py``."""

    model_id: str
    """HuggingFace model ID (e.g. ``Qwen/Qwen2.5-0.5B``)."""

    revision: str
    """HF model revision / commit SHA."""

    dtype: str
    """Weight dtype: ``float32``, ``float16``, or ``bfloat16``."""

    level: str
    """``L4``, ``L5``, or ``L4+L5``."""

    prompts: list[str]
    """Text inputs (may be empty for encoder/audio tasks)."""

    images: list[str]
    """Image paths relative to ``testdata/`` (VL tasks)."""

    audio: list[str]
    """Audio paths relative to ``testdata/`` (speech tasks)."""

    decoder_prompt: str
    """Forced decoder prefix (seq2seq / whisper)."""

    generation_params: dict
    """E.g. ``{"max_new_tokens": 20, "do_sample": false}``."""

    trust_remote_code: bool
    """Whether the HF model requires ``trust_remote_code``."""

    skip_reason: str | None
    """If set, the test runner should skip with this message."""

    yaml_path: Path
    """Absolute path to the source YAML file."""


@dataclasses.dataclass(frozen=True)
class GoldenRef:
    """Pre-computed HuggingFace reference outputs loaded from ``.json``.

    Stores enough data for ``compare_golden()`` plus diagnostic info
    for debugging failures without re-running HuggingFace inference.

    All numeric data is stored as JSON-native types (int / float lists)
    so zero precision is lost for fp32 values via ``float.hex()``
    round-tripping, and diffs are human-readable in code review.
    """

    # L4 data — always present
    top1_id: int
    """Argmax token ID from HF last-token logits."""

    top2_id: int
    """Second-highest token ID."""

    top10_ids: list[int]
    """Top-10 token IDs sorted by descending logit value."""

    top10_logits: list[float]
    """Corresponding logit values for the top-10 tokens."""

    logits_summary: list[float]
    """``[max, min, mean, std]`` of the full logit vector."""

    input_ids: list[int]
    """Tokenized input used during golden generation."""

    # L5 data — None if level == "L4"
    generated_ids: list[int] | None
    """Full generated token sequence (greedy, L5 only)."""

    # Multi-model diagnostics — empty dicts for single-model tasks
    component_norms: dict[str, float]
    """L2 norms of component outputs, e.g. ``{"vision": 42.5}``."""

    component_shapes: dict[str, list[int]]
    """Output shapes for components, e.g. ``{"vision": [1, 577, 1024]}``."""

    # Metadata
    json_path: Path
    """Absolute path to the source ``.json`` file."""


@dataclasses.dataclass(frozen=True)
class Tolerances:
    """Tolerance thresholds for a specific level + dtype.

    L4 is argmax-gated; the numeric thresholds here are diagnostic
    (produce warnings, not failures).  L5 uses ``min_token_match_ratio``
    for the actual gate.
    """

    near_tie_margin: float
    """Logit gap below which a top-1 prediction is considered unstable."""

    top10_jaccard_warn: float
    """Warn if top-10 Jaccard drops below this value."""

    cosine_similarity_warn: float
    """Warn if cosine similarity drops below this value."""

    min_token_match_ratio: float
    """Fraction of generated tokens that must match (L5 gate)."""


# --- Public API ---


def load_test_case(yaml_path: Path) -> GoldenTestCase:
    """Load a single test case from a YAML file.

    Args:
        yaml_path: Path to the YAML test case file.

    Returns:
        Parsed ``GoldenTestCase`` dataclass.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If required fields are missing.
    """
    yaml_path = yaml_path.resolve()
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {yaml_path}, got {type(data)}")

    # Validate required top-level fields
    _required = ("model_id", "revision", "task_type", "dtype", "level")
    missing = [k for k in _required if k not in data]
    if missing:
        raise ValueError(f"Missing required fields in {yaml_path}: {missing}")

    inputs = data.get("inputs", {}) or {}
    generation = data.get("generation", {}) or {}

    # Derive case_id from the parent dir + stem to include task type
    # e.g. causal-lm/qwen2_5-0_5b → "qwen2_5-0_5b"
    case_id = yaml_path.stem

    return GoldenTestCase(
        case_id=case_id,
        task_type=data["task_type"],
        model_id=data["model_id"],
        revision=data["revision"],
        dtype=data["dtype"],
        level=data["level"],
        prompts=inputs.get("prompts", []) or [],
        images=inputs.get("images", []) or [],
        audio=inputs.get("audio", []) or [],
        decoder_prompt=inputs.get("decoder_prompt", "") or "",
        generation_params=generation,
        trust_remote_code=data.get("trust_remote_code", False),
        skip_reason=data.get("skip_reason"),
        yaml_path=yaml_path,
    )


def load_golden_ref(json_path: Path) -> GoldenRef | None:
    """Load golden reference data from a ``.json`` file.

    Returns ``None`` if the file does not exist — the caller decides
    whether to skip or fail.  This avoids hard failures when golden
    files haven't been generated yet.

    Float values are stored as hex strings (``float.hex()``) for
    lossless fp64 round-tripping.  Integer lists are stored directly.

    Args:
        json_path: Path to the ``.json`` golden file.

    Returns:
        Parsed ``GoldenRef`` dataclass, or ``None`` if file is missing.
    """
    json_path = Path(json_path).resolve()
    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    # Decode hex-encoded floats back to Python floats
    top10_logits = [float.fromhex(h) for h in data["top10_logits"]]
    logits_summary = [float.fromhex(h) for h in data["logits_summary"]]

    # Component norms use hex encoding; shapes are plain int lists
    component_norms: dict[str, float] = {
        k: float.fromhex(v) for k, v in data.get("component_norms", {}).items()
    }
    component_shapes: dict[str, list[int]] = data.get("component_shapes", {})

    # L5 generated_ids may not be present (L4-only golden)
    generated_ids = data.get("generated_ids")

    return GoldenRef(
        top1_id=int(data["top1_id"]),
        top2_id=int(data["top2_id"]),
        top10_ids=[int(x) for x in data["top10_ids"]],
        top10_logits=top10_logits,
        logits_summary=logits_summary,
        input_ids=[int(x) for x in data["input_ids"]],
        generated_ids=generated_ids,
        component_norms=component_norms,
        component_shapes=component_shapes,
        json_path=json_path,
    )


def save_golden_ref(
    json_path: Path,
    *,
    top1_id: int,
    top2_id: int,
    top10_ids: list[int],
    top10_logits: list[float],
    logits_summary: np.ndarray | list[float],
    input_ids: np.ndarray | list[int],
    generated_ids: np.ndarray | list[int] | None = None,
    component_norms: dict[str, float] | None = None,
    component_shapes: dict[str, tuple[int, ...]] | None = None,
) -> None:
    """Save golden reference data to a ``.json`` file.

    Creates parent directories if they do not exist.  All keyword
    arguments are required except ``generated_ids`` (L5 only) and
    the ``component_*`` dicts (multi-model tasks only).

    Float values are stored as hex strings (``float.hex()``) for
    lossless fp64 round-tripping.  This preserves full precision
    while keeping the file human-readable and git-diff friendly.

    Args:
        json_path: Destination path for the ``.json`` file.
        top1_id: Argmax token ID from HF last-token logits.
        top2_id: Second-highest token ID.
        top10_ids: Top-10 token IDs sorted by descending logit.
        top10_logits: Corresponding logit values for top-10 tokens.
        logits_summary: ``[max, min, mean, std]`` of the full logit
            vector.
        input_ids: Tokenized input array.
        generated_ids: Full generated token sequence (L5 only).
        component_norms: L2 norms for multi-model component outputs.
        component_shapes: Output shapes for multi-model components.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to plain Python lists for JSON
    logits_summary_list = (
        logits_summary.tolist()
        if isinstance(logits_summary, np.ndarray)
        else list(logits_summary)
    )
    input_ids_list = (
        input_ids.tolist() if isinstance(input_ids, np.ndarray) else list(input_ids)
    )
    # Flatten nested lists from 2-D arrays, e.g. (1, seq_len)
    if input_ids_list and isinstance(input_ids_list[0], list):
        input_ids_list = input_ids_list[0]

    # Encode floats as hex strings for lossless round-tripping
    data: dict = {
        "top1_id": int(top1_id),
        "top2_id": int(top2_id),
        "top10_ids": [int(x) for x in top10_ids],
        "top10_logits": [float(x).hex() for x in top10_logits],
        "logits_summary": [float(x).hex() for x in logits_summary_list],
        "input_ids": [int(x) for x in input_ids_list],
    }

    if generated_ids is not None:
        gen_list = (
            generated_ids.tolist()
            if isinstance(generated_ids, np.ndarray)
            else list(generated_ids)
        )
        data["generated_ids"] = [int(x) for x in gen_list]

    # Multi-model component data stored as nested dicts
    if component_norms:
        data["component_norms"] = {k: float(v).hex() for k, v in component_norms.items()}
    if component_shapes:
        data["component_shapes"] = {
            k: [int(x) for x in v] for k, v in component_shapes.items()
        }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")  # trailing newline for POSIX compliance


def discover_test_cases(
    task_type: str | None = None,
    level: str | None = None,
    root: Path = CASES_DIR,
) -> list[GoldenTestCase]:
    """Discover all test cases, optionally filtered by task type and level.

    Scans ``root`` (default ``testdata/cases/``) for YAML files and
    parses them.  Results are sorted by ``case_id`` for deterministic
    pytest parametrization.

    Args:
        task_type: If set, only return cases whose parent directory name
            or ``task_type`` field matches this value.
        level: If set, only return cases that include this level
            (e.g. ``"L4"`` matches both ``"L4"`` and ``"L4+L5"``).
        root: Root directory to search.

    Returns:
        Sorted list of ``GoldenTestCase`` objects.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    cases: list[GoldenTestCase] = []
    for yaml_file in sorted(root.rglob("*.yaml")):
        try:
            case = load_test_case(yaml_file)
        except (ValueError, yaml.YAMLError):
            # Skip malformed files — discovery should be tolerant.
            continue

        # Filter by task type: match against the subdirectory name
        # or the task_type field itself.
        if task_type is not None:
            parent_name = yaml_file.parent.name
            if parent_name != task_type and case.task_type != task_type:
                continue

        # Filter by level: "L4" matches "L4" and "L4+L5"
        if level is not None and level not in case.level:
            continue

        cases.append(case)

    return sorted(cases, key=lambda c: c.case_id)


def golden_path_for_case(
    case: GoldenTestCase,
    golden_dir: Path = GOLDEN_DIR,
) -> Path:
    """Return the expected golden ``.json`` path for a test case.

    Maps ``testdata/cases/<task>/<name>.yaml``
    to  ``testdata/golden/<task>/<name>.json``.

    The path is derived from the YAML path's parent directory name
    (task type) and stem (model name).

    Args:
        case: The test case to look up.
        golden_dir: Root directory for golden files. Defaults to
            ``testdata/golden/``.
    """
    # Use the task-type subdirectory from the YAML location.
    # e.g. testdata/cases/causal-lm/qwen2_5-0_5b.yaml
    #   → testdata/golden/causal-lm/qwen2_5-0_5b.json
    task_dir = case.yaml_path.parent.name
    return golden_dir / task_dir / f"{case.case_id}.json"


def has_golden(
    case: GoldenTestCase,
    golden_dir: Path = GOLDEN_DIR,
) -> bool:
    """Check whether the golden ``.json`` file exists for a test case."""
    return golden_path_for_case(case, golden_dir=golden_dir).exists()


def load_tolerances(
    level: str = "L4",
    dtype: str = "float32",
    path: Path = DEFAULT_TOLERANCES_PATH,
) -> Tolerances:
    """Load tolerance thresholds for a given level and dtype.

    If the YAML file does not exist, returns hard-coded defaults
    so tests can run without the tolerances file.

    Args:
        level: ``"L3"``, ``"L4"``, or ``"L5"``.
        dtype: ``"float32"``, ``"float16"``, ``"bfloat16"``, or ``"int4"``.
        path: Path to the YAML tolerances file.

    Returns:
        ``Tolerances`` dataclass with resolved thresholds.
    """
    path = Path(path)
    if path.exists():
        with open(path) as f:
            all_tolerances = yaml.safe_load(f) or {}
        level_data = all_tolerances.get(level, {})
        dtype_data = level_data.get(dtype, {})
    else:
        # Fall back to built-in defaults when the file is missing.
        dtype_data = {}

    # Resolve with defaults: level-specific fallback → global fallback.
    fallback = _FALLBACK_TOLERANCES.get(level, _FALLBACK_TOLERANCES["L4"])

    return Tolerances(
        near_tie_margin=dtype_data.get("near_tie_margin", fallback["near_tie_margin"]),
        top10_jaccard_warn=dtype_data.get(
            "top10_jaccard_warn", fallback["top10_jaccard_warn"]
        ),
        cosine_similarity_warn=dtype_data.get(
            "cosine_similarity_warn", fallback["cosine_similarity_warn"]
        ),
        min_token_match_ratio=dtype_data.get(
            "min_token_match_ratio", fallback["min_token_match_ratio"]
        ),
    )
