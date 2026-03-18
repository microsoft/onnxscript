# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest configuration and fixtures for mobius tests."""

from __future__ import annotations

import random

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Marker registrations
# ---------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "golden: L4/L5 golden reference comparison tests",
    )
    config.addinivalue_line(
        "markers",
        "generation: L5 end-to-end generation tests",
    )
    config.addinivalue_line(
        "markers",
        "integration: tests that require real model weights (network)",
    )
    config.addinivalue_line(
        "markers",
        "integration_fast: fast integration tests with small models",
    )
    config.addinivalue_line(
        "markers",
        "arch_validation: L2 architecture validation tests",
    )


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run only representative (is_representative=True) model configs.",
    )
    parser.addoption(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated list of model_type names to run. "
            "Only parametrized tests matching these model_types "
            "will be selected; others are skipped."
        ),
    )


# ---------------------------------------------------------------------------
# Collection hooks
# ---------------------------------------------------------------------------
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Filter parametrized tests based on ``--fast`` and ``--models`` flags.

    ``--fast``: Skip non-representative parametrized tests (model_type
    not in the fast config sets).  Non-parametrized tests always run.

    ``--models``: Only run tests whose ``model_type`` parameter matches
    one of the comma-separated names.  Non-parametrized tests always run.

    Note: Filtering is coarse-grained (model_type-level). When a
    model_type appears multiple times with different is_representative
    flags, ALL variants of that model_type run in --fast mode. This is
    safe — it over-tests, never under-tests.
    """
    # --- --models filter ---
    models_opt = config.getoption("--models")
    if models_opt:
        selected_models = {m.strip() for m in models_opt.split(",") if m.strip()}
        skip_model = pytest.mark.skip(reason="not in --models selection")
        for item in items:
            if not hasattr(item, "callspec"):
                continue
            if "model_type" not in item.callspec.params:
                continue
            model_type = item.callspec.params["model_type"]
            if model_type not in selected_models:
                item.add_marker(skip_model)

    # --- --fast filter ---
    if not config.getoption("--fast"):
        return

    from _test_configs import (
        FAST_CAUSAL_LM_CONFIGS,
        FAST_DETECTION_CONFIGS,
        FAST_ENCODER_CONFIGS,
        FAST_SEQ2SEQ_CONFIGS,
        FAST_VISION_CONFIGS,
    )

    # Build set of representative model_types
    fast_model_types: set[str] = set()
    for cfg_list in (
        FAST_CAUSAL_LM_CONFIGS,
        FAST_ENCODER_CONFIGS,
        FAST_SEQ2SEQ_CONFIGS,
        FAST_VISION_CONFIGS,
        FAST_DETECTION_CONFIGS,
    ):
        for mt, _ in cfg_list:
            fast_model_types.add(mt)

    skip_reason = pytest.mark.skip(reason="not representative (--fast mode)")
    for item in items:
        # Only filter parametrized tests with model_type
        if not hasattr(item, "callspec"):
            continue
        if "model_type" not in item.callspec.params:
            continue
        model_type = item.callspec.params["model_type"]
        if model_type not in fast_model_types:
            item.add_marker(skip_reason)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def deterministic_seed():
    """Set numpy and python random seeds for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    return seed
