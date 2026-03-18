# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""L2 Architecture Validation Tests — real HF config, no weights.

Downloads config.json from HuggingFace for each registered model type
that has a ``test_model_id``, builds the full-size ONNX graph (no weights),
and validates graph structure and shape consistency.

These tests verify that our ONNX graph construction is compatible with
real-world model configurations — catching shape mismatches, missing
config fields, and initialization errors that tiny synthetic configs
would not trigger.

Must stay under 2 GB RAM. Run sequentially (no pytest-xdist)::

    pytest tests/arch_validation_test.py -m arch_validation -v --tb=short

To run a single architecture::

    pytest tests/arch_validation_test.py -k "llama" -v

These tests are designed for nightly CI or tag-triggered runs.
They require network access to download config.json from HuggingFace.
"""

from __future__ import annotations

import gc
import logging
import os
import resource

import pytest

from mobius._config_resolver import (
    _config_from_hf,
    _default_task_for_model,
    _try_load_config_json,
)
from mobius._registry import registry
from mobius.tasks import get_task

logger = logging.getLogger(__name__)

# 1.5 GB RSS limit — leave headroom below the 2 GB target
_MAX_RSS_BYTES = 1.5 * 1024 * 1024 * 1024

# Build parametrized test cases from registry entries that have a test_model_id
_KNOWN_XFAILS: dict[str, str] = {
    "phi3small": "gegelu activation not implemented (gated GELU variant)",
}

_ARCH_PARAMS = [
    pytest.param(
        model_type,
        registration.test_model_id,
        id=model_type,
        marks=[pytest.mark.xfail(reason=_KNOWN_XFAILS[model_type], strict=False)]
        if model_type in _KNOWN_XFAILS
        else [],
    )
    for model_type in sorted(registry.architectures())
    if (registration := registry.get_registration(model_type)).test_model_id is not None
]


def _get_rss_bytes() -> int:
    """Return current RSS (resident set size) in bytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in KB on Linux
    return usage.ru_maxrss * 1024


def _load_hf_config(model_id: str):
    """Load HF config using AutoConfig first, fallback to raw config.json.

    AutoConfig handles model-specific field mappings (e.g. GPT-2's
    ``n_embd`` → ``hidden_size``).  Raw config.json is used when the
    model type isn't registered in transformers.
    """
    import transformers

    try:
        return transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    except (ValueError, OSError):
        return _try_load_config_json(model_id)


def _resolve_hf_config(hf_config):
    """Resolve nested config wrappers (thinker, talker, text, llm, decoder).

    Mirrors the resolution logic in ``build()`` — some models
    wrap the actual config inside a parent config.
    """
    parent_config = hf_config
    if hasattr(hf_config, "talker_config"):
        hf_config = hf_config.talker_config
    elif hasattr(hf_config, "thinker_config"):
        thinker = hf_config.thinker_config
        if hasattr(thinker, "text_config"):
            hf_config = thinker.text_config
        else:
            hf_config = thinker
    elif hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config
    elif hasattr(hf_config, "llm_config"):
        # InternVL2 wraps the LLM config under llm_config
        hf_config = hf_config.llm_config
    elif hasattr(hf_config, "decoder"):
        # VisionEncoderDecoder (TrOCR) wraps decoder config
        hf_config = hf_config.decoder
    return hf_config, parent_config


def _build_graph(model_type: str, model_id: str):
    """Download config, build ONNX graph, return ModelPackage.

    Uses get_task().build() directly (same pattern as build_graph_test.py)
    to bypass ArchitectureConfig.validate() which rejects non-LM configs
    (e.g. vision models with vocab_size=0).
    """
    hf_config = _load_hf_config(model_id)
    if hf_config is None:
        pytest.skip(
            f"Cannot download config for {model_id} (gated/private model or network error)"
        )

    hf_config, parent_config = _resolve_hf_config(hf_config)
    registration = registry.get_registration(model_type)
    config = _config_from_hf(
        hf_config,
        parent_config=parent_config,
        module_class=registration.module_class,
    )

    module = registration.module_class(config)
    task_name = registration.task or _default_task_for_model(model_type)
    task = get_task(task_name)
    return task.build(module, config)


@pytest.mark.arch_validation
class TestArchValidation:
    """L2 architecture validation: download real config, build full graph.

    Each test downloads only the config.json (no model weights) from
    HuggingFace, constructs the full-size ONNX graph, and validates
    that the graph has a reasonable structure.
    """

    @pytest.mark.parametrize("model_type,model_id", _ARCH_PARAMS)
    def test_config_downloads_and_parses(self, model_type: str, model_id: str):
        """Verify config.json can be downloaded and parsed."""
        hf_config = _load_hf_config(model_id)
        if hf_config is None:
            pytest.skip(
                f"Cannot download config for {model_id} (gated/private model or network error)"
            )
        # The config must have a model_type
        assert hasattr(hf_config, "model_type"), f"Config for {model_id} missing model_type"

    @pytest.mark.parametrize("model_type,model_id", _ARCH_PARAMS)
    def test_full_graph_builds(self, model_type: str, model_id: str):
        """Build full-size ONNX graph from real HF config (no weights).

        This is the core L2 validation: if the real config triggers
        shape mismatches, missing fields, or initialization errors, this
        test will catch them.
        """
        pkg = _build_graph(model_type, model_id)

        # Validate: every component has a non-empty graph
        assert len(pkg) > 0, "ModelPackage is empty"
        for component_name, model in pkg.items():
            assert model.graph is not None, f"{component_name} graph is None"
            nodes = list(model.graph)
            assert len(nodes) > 0, f"{component_name} has no nodes"
            assert len(model.graph.inputs) > 0, f"{component_name} has no inputs"
            assert len(model.graph.outputs) > 0, f"{component_name} has no outputs"

        del pkg
        gc.collect()

    @pytest.mark.parametrize("model_type,model_id", _ARCH_PARAMS)
    def test_graph_shapes_consistent(self, model_type: str, model_id: str):
        """Validate structural consistency in the full-size graph.

        Checks that the graph has initializers (parameters) and that
        inputs/outputs are defined. Note: output type info may not be
        available for all outputs when building without shape inference.
        """
        pkg = _build_graph(model_type, model_id)

        for component_name, model in pkg.items():
            # Model should have initializers (parameters)
            initializers = list(model.graph.initializers)
            assert len(initializers) > 0, (
                f"{component_name} has no initializers — graph may be missing parameters"
            )

            # All inputs should have names
            for inp in model.graph.inputs:
                assert inp.name, f"{component_name} has an unnamed input"

            # All outputs should have names
            for output in model.graph.outputs:
                assert output.name, f"{component_name} has an unnamed output"

        del pkg
        gc.collect()

    @pytest.mark.parametrize("model_type,model_id", _ARCH_PARAMS)
    def test_memory_stays_within_budget(self, model_type: str, model_id: str):
        """Guard against memory-hungry full-size graph builds.

        Logs current RSS after graph construction and fails if it
        exceeds the 1.5 GB threshold (leaving headroom below 2 GB).

        Skipped under pytest-xdist because ``ru_maxrss`` reports peak
        RSS which is cumulative within a worker process and never
        decreases — giving false positives when a worker runs many tests.
        """
        if os.environ.get("PYTEST_XDIST_WORKER"):
            pytest.skip("RSS budget check unreliable under pytest-xdist (cumulative peak RSS)")

        pkg = _build_graph(model_type, model_id)

        rss = _get_rss_bytes()
        rss_mb = rss / (1024 * 1024)
        logger.info(
            "%s (%s): RSS after build = %.0f MB",
            model_type,
            model_id,
            rss_mb,
        )

        del pkg
        gc.collect()

        assert rss < _MAX_RSS_BYTES, (
            f"{model_type} ({model_id}): RSS {rss_mb:.0f} MB "
            f"exceeds {_MAX_RSS_BYTES / 1024 / 1024:.0f} MB limit"
        )
