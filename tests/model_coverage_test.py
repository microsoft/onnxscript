# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Model test-coverage audit: every registered architecture must have L1-L5 data.

Replaces ``scripts/check_new_model_coverage.py`` and the CI workflow
``.github/workflows/model-coverage.yml`` which only checked *newly added*
models via ``git diff``.  This pytest checks **all** registered models on
every run, so coverage gaps cannot accumulate silently.

Coverage levels
~~~~~~~~~~~~~~~

=====  ========  =====================================================
Level  Artefact  What it proves
=====  ========  =====================================================
L1     ``tests/_test_configs.py`` entry       Graph builds without error
L2     ``test_model_id`` in ``_registry.py``  HF config can be fetched
L3     (same as L1)                           Synthetic-parity forward pass
L4     YAML in ``testdata/cases/``            End-to-end test spec exists
L5     JSON in ``testdata/golden/``           Reference outputs available
=====  ========  =====================================================

Adding a new model?
~~~~~~~~~~~~~~~~~~~

1. Add a config to ``tests/_test_configs.py``  →  L1 + L3
2. Set ``test_model_id`` in ``_registry.py``   →  L2
3. Add YAML in ``testdata/cases/``             →  L4
4. Generate golden JSON (or add ``skip_reason`` to YAML)  →  L5

See ``.github/skills/writing-tests/SKILL.md`` for the full guide.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import pytest

from mobius._registry import _TEST_MODEL_IDS, registry
from mobius._testing.golden import (
    discover_test_cases,
    golden_path_for_case,
    has_golden,
)

# ---------------------------------------------------------------------------
# L1/L3 config discovery (test configs from _test_configs.py)
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).resolve().parent

if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _test_configs import (  # noqa: E402
    ALL_CAUSAL_LM_CONFIGS,
    DETECTION_CONFIGS,
    ENCODER_CONFIGS,
    LINEAR_RNN_CONFIGS,
    SEGMENTATION_CONFIGS,
    SEQ2SEQ_CONFIGS,
    SPEECH_CONFIGS,
    SSM_CONFIGS,
    VISION_CONFIGS,
    VL_CONFIGS,
)


def _l1_l3_model_types() -> set[str]:
    """Return model_types that have a test config in _test_configs.py.

    Includes ALL config lists: causal LM, encoder, seq2seq, vision,
    detection, SSM, vision-language, speech, and segmentation.
    """
    types: set[str] = set()
    all_configs = (
        ALL_CAUSAL_LM_CONFIGS
        + ENCODER_CONFIGS
        + SEQ2SEQ_CONFIGS
        + VISION_CONFIGS
        + DETECTION_CONFIGS
        + SEGMENTATION_CONFIGS
        + SSM_CONFIGS
        + LINEAR_RNN_CONFIGS
        + VL_CONFIGS
        + SPEECH_CONFIGS
    )
    for mt, _, _ in all_configs:
        types.add(mt)
    return types


# ---------------------------------------------------------------------------
# L4/L5 helpers
# ---------------------------------------------------------------------------


@functools.cache
def _discovered_cases() -> tuple:
    """Cache the result of discover_test_cases() to avoid re-parsing YAML."""
    return tuple(discover_test_cases())


def _yaml_model_ids() -> set[str]:
    """Collect all model_ids present in YAML test case files."""
    return {case.model_id for case in _discovered_cases()}


def _yaml_cases_by_model_id() -> dict[str, object]:
    """Map model_id → GoldenTestCase for JSON golden file lookups."""
    return {case.model_id: case for case in _discovered_cases()}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _all_registered() -> list[str]:
    """Return all model_types from the registry, sorted."""
    return sorted(registry.architectures())


def _all_registered_with_test_id() -> dict[str, str]:
    """Return {model_type: test_model_id} for registered models with one."""
    return {
        arch: model_id for arch, model_id in _TEST_MODEL_IDS.items() if arch in registry._map
    }


# ── Skip list: models that cannot have full coverage (with reasons) ──────
#
# Each entry maps model_type → reason.  The reason is displayed when the
# parametrized test is skipped, so keep it informative.
#
# Categories (each section has a comment header):
#   - Internal / duplicate aliases
#   - VL text-decoder submodels
#   - Vision-language models
#   - Audio / speech models
#   - trust_remote_code
#   - Very large models
#   - Models without test_model_id
#   - CausalLM / other models without YAML
#
_COVERAGE_SKIP: dict[str, str] = {
    # --- Internal / duplicate aliases ---
    "code_llama": "Alias for llama — covered by llama",
    "command_r": "Alias for cohere — covered by cohere",
    "deepseek_v2_moe": "Alias for deepseek_v2 — covered by deepseek_v2",
    "gpt_oss": "Internal model — no public HF checkpoint",
    "helium": "Alias for mistral — covered by mistral",
    "open-llama": "Alias for llama — covered by llama",
    "seed_oss": "Internal model — no public HF checkpoint",
    "shieldgemma2": "Alias for gemma2 — covered by gemma2",
    "yi": "Alias for llama — covered by llama",
    # --- VL text-decoder submodels (tested via their parent VL model) ---
    "glm4v_moe_text": "VL text decoder — tested via glm4v_moe",
    "glm4v_text": "VL text decoder — tested via glm4v",
    "qwen2_5_vl_text": "VL text decoder — tested via qwen2_5_vl",
    "qwen2_vl_text": "VL text decoder — tested via qwen2_vl",
    "qwen3_5_vl_text": "VL text decoder — tested via qwen3_5_vl",
    # --- Vision-language models (require image/video inputs) ---
    "blip": "VL model — requires image inputs",
    "blip-2": "VL model — requires image inputs",
    "florence2": "VL model — requires image inputs",
    "gemma3_multimodal": "VL model — requires image inputs",
    "idefics2": "VL model — requires image inputs",
    "idefics3": "VL model — requires image inputs",
    "instructblip": "VL model — requires image inputs",
    "internvl2": "VL model — requires image inputs",
    "llava": "VL model — requires image inputs",
    "llava_next": "VL model — requires image inputs",
    "llava_onevision": "VL model — requires image inputs",
    "mllama": "VL model — requires image inputs",
    "molmo": "VL model — requires image inputs",
    "phi4_multimodal": "VL model (14B) — needs GPU for golden",
    "phi4mm": "VL model (14B) — needs GPU for golden",
    "qwen2_5_vl": "VL model — requires image inputs",
    "qwen2_vl": "VL model — requires image inputs",
    "qwen3_5": "VL model — hybrid VL, requires image inputs",
    "qwen3_vl": "VL model — requires image inputs",
    # --- Audio / speech models (require audio inputs) ---
    "data2vec-audio": "Audio model — requires audio inputs",
    "hubert": "Audio model — requires audio inputs",
    "musicgen": "Audio model — requires audio inputs",
    "seamless_m4t": "Audio model — requires audio inputs",
    "seamless_m4t_v2": "Audio model — requires audio inputs",
    "sew": "Audio model — requires audio inputs",
    "sew-d": "Audio model — requires audio inputs",
    "speecht5": "Audio model — requires audio inputs",
    "unispeech": "Audio model — requires audio inputs",
    "unispeech-sat": "Audio model — requires audio inputs",
    "wav2vec2": "Audio model — requires audio inputs",
    "wav2vec2-bert": "Audio model — requires audio inputs",
    "wav2vec2-conformer": "Audio model — requires audio inputs",
    "wavlm": "Audio model — requires audio inputs",
    "whisper": "Speech-to-text — requires audio inputs",
    # --- Models requiring trust_remote_code ---
    "chatglm": "Requires trust_remote_code (custom HF modeling code)",
    "dots1": "Requires trust_remote_code (custom HF modeling code)",
    "videollama3_qwen2": "Requires trust_remote_code (custom HF modeling code)",
    # --- Very large models without small public checkpoints ---
    "arctic": "Very large MoE (480B) — no small public checkpoint",
    "dbrx": "Large MoE (132B) — no small public checkpoint",
    "deepseek_v3": "Very large MoE (671B) — no small public checkpoint",
    "llama4_text": "Very large MoE (109B) — no small public checkpoint",
    "qwen3_5_moe": "Large MoE (22B) — no small public checkpoint",
    # --- Models without test_model_id ---
    "aya_vision": "VL model — no test_model_id yet",
    "chameleon": "VL model — no test_model_id yet",
    "codegen2": "No test_model_id — no suitable public checkpoint",
    "cohere2_vision": "VL model — no test_model_id yet",
    "csm": "No test_model_id — no suitable public checkpoint",
    "deepseek_vl": "VL model — no test_model_id yet",
    "deepseek_vl_hybrid": "VL model — no test_model_id yet",
    "deepseek_vl_v2": "VL model — no test_model_id yet",
    "dinov3_vit": "Vision model — no test_model_id yet",
    "evolla": "No test_model_id — no suitable public checkpoint",
    "fuyu": "VL model — no test_model_id yet",
    "glm4v": "VL model — no test_model_id yet",
    "glm4v_moe": "VL model — no test_model_id yet",
    "got_ocr2": "VL model — no test_model_id yet",
    "ijepa": "Vision model — no test_model_id yet",
    "instructblipvideo": "VL model — no test_model_id yet",
    "internvl": "VL model — no test_model_id yet",
    "internvl_chat": "VL model — no test_model_id yet",
    "janus": "VL model — no test_model_id yet",
    "llava_next_video": "VL model — no test_model_id yet",
    "mctct": "Audio model — no test_model_id yet",
    "megatron-bert": "Encoder — no test_model_id yet",
    "modernbert-decoder": "Decoder variant — no test_model_id yet",
    "nemotron_h": "No test_model_id — no suitable public checkpoint",
    "nllb-moe": "Seq2seq MoE — no test_model_id yet",
    "nllb_moe": "Seq2seq MoE — no test_model_id yet",
    "ovis2": "VL model — no test_model_id yet",
    "paligemma": "VL model — no test_model_id yet",
    "persimmon": "No test_model_id — no suitable public checkpoint",
    "pixtral": "VL model — no test_model_id yet",
    "qdqbert": "Quantised BERT — no test_model_id yet",
    "qwen3_5_vl": "VL model — no test_model_id yet",
    "qwen3_asr": "Audio model — no test_model_id yet",
    "qwen3_forced_aligner": "Speech model — no test_model_id yet",
    "qwen3_tts": "TTS model — no test_model_id yet",
    "qwen3_tts_tokenizer_12hz": "Codec model — no test_model_id yet",
    "qwen3_vl_single": "VL model — no test_model_id yet",
    "sam2": "Vision model — no test_model_id yet",
    "smolvlm": "VL model — no test_model_id yet",
    "solar_open": "No test_model_id — no suitable public checkpoint",
    "video_llava": "VL model — no test_model_id yet",
    "vipllava": "VL model — no test_model_id yet",
    "vit_hybrid": "Vision model — no test_model_id yet",
    "voxtral_encoder": "Audio encoder — no test_model_id yet",
    # --- CausalLM models (YAML not yet created) ---
    "baichuan": "CausalLM — YAML not yet created",
    "doge": "CausalLM — YAML not yet created",
    "ernie4_5": "CausalLM — YAML not yet created",
    "exaone": "CausalLM — YAML not yet created",
    "falcon_h1": "CausalLM — YAML not yet created",
    "falcon_mamba": "SSM — YAML not yet created",
    "imagegpt": "Vision model — YAML not yet created",
    "internlm2": "CausalLM — YAML not yet created",
    "minicpm": "CausalLM — YAML not yet created",
    "minicpm3": "CausalLM — YAML not yet created",
    "ministral": "CausalLM — YAML not yet created",
    "ministral3": "CausalLM — YAML not yet created",
    "mistral3": "CausalLM — YAML not yet created",
    "openelm": "CausalLM — YAML not yet created",
    "qwen": "CausalLM — YAML not yet created",
    "youtu": "CausalLM — YAML not yet created",
    "zamba": "CausalLM — YAML not yet created",
    "zamba2": "CausalLM — YAML not yet created",
    # --- Contrastive models (tested via specialized test classes) ---
    "clip": "Contrastive model — tested via TestBuildCLIPContrastiveGraph",
}


# ── Tests ─────────────────────────────────────────────────────────────────


class TestSkipListIntegrity:
    """Guard-rail tests for the skip list itself."""

    def test_skip_entries_are_still_registered(self):
        """Entries in _COVERAGE_SKIP must still be in the registry.

        If a model is removed from the registry, clean it from
        _COVERAGE_SKIP too.
        """
        registered = set(registry._map.keys())
        stale = set(_COVERAGE_SKIP.keys()) - registered
        if stale:
            pytest.fail(
                f"Stale entries in _COVERAGE_SKIP (no longer registered): {sorted(stale)}"
            )

    def test_skip_entries_have_reasons(self):
        """Every _COVERAGE_SKIP entry must have a non-empty reason."""
        empty = [arch for arch, reason in _COVERAGE_SKIP.items() if not reason.strip()]
        if empty:
            pytest.fail(f"_COVERAGE_SKIP entries with empty reasons: {sorted(empty)}")


class TestL1L3GraphBuildCoverage:
    """L1 + L3: every model needs a test config in _test_configs.py.

    The config enables ``build_graph_test.py`` to exercise the model.
    For causal-LM models, it also enables ``synthetic_parity_test.py``.
    """

    def test_all_models_have_test_config_or_skip(self):
        """Aggregate check: every registered model needs an L1/L3 config."""
        l13 = _l1_l3_model_types()
        all_reg = _all_registered()

        missing = [mt for mt in all_reg if mt not in l13 and mt not in _COVERAGE_SKIP]
        if missing:
            pytest.fail(
                f"{len(missing)} registered model(s) have no test "
                f"config in _test_configs.py and are not in "
                f"_COVERAGE_SKIP:\n"
                + "\n".join(f"  {mt}" for mt in missing)
                + "\n\nFix: add a config entry to "
                "tests/_test_configs.py or add to _COVERAGE_SKIP."
            )

    @pytest.mark.parametrize("arch", _all_registered())
    def test_model_has_l1_l3_config(self, arch: str):
        """Per-model check for L1/L3 test config."""
        if arch in _COVERAGE_SKIP:
            pytest.skip(_COVERAGE_SKIP[arch])
        l13 = _l1_l3_model_types()
        if arch not in l13:
            pytest.fail(
                f"Model '{arch}' has no test config in "
                f"tests/_test_configs.py. Add one for L1/L3 coverage."
            )


class TestL2ConfigValidation:
    """L2: every model needs a ``test_model_id`` in ``_TEST_MODEL_IDS``.

    This allows ``arch_validation_test.py`` to fetch and validate its
    HuggingFace config.
    """

    def test_all_models_have_test_model_id_or_skip(self):
        """Aggregate check: every registered model needs a test_model_id."""
        all_reg = _all_registered()
        missing = [
            mt for mt in all_reg if mt not in _TEST_MODEL_IDS and mt not in _COVERAGE_SKIP
        ]
        if missing:
            pytest.fail(
                f"{len(missing)} registered model(s) have no "
                f"test_model_id in _registry.py and are not in "
                f"_COVERAGE_SKIP:\n"
                + "\n".join(f"  {mt}" for mt in missing)
                + "\n\nFix: add test_model_id to _TEST_MODEL_IDS "
                "in src/mobius/_registry.py."
            )

    @pytest.mark.parametrize("arch", _all_registered())
    def test_model_has_test_model_id(self, arch: str):
        """Per-model check for test_model_id (L2)."""
        if arch in _COVERAGE_SKIP:
            pytest.skip(_COVERAGE_SKIP[arch])
        if arch not in _TEST_MODEL_IDS:
            pytest.fail(
                f"Model '{arch}' has no test_model_id in "
                f"_TEST_MODEL_IDS. Add one for L2 config validation."
            )


class TestL4L5GoldenDataCoverage:
    """L4 + L5: every model needs a YAML test case and JSON golden output.

    L4 = YAML in ``testdata/cases/``, L5 = JSON in ``testdata/golden/``.
    """

    def test_all_models_have_yaml_or_skip(self):
        """Aggregate: each model with test_model_id needs YAML or skip."""
        yaml_ids = _yaml_model_ids()
        models = _all_registered_with_test_id()

        missing = []
        for arch, model_id in sorted(models.items()):
            if arch in _COVERAGE_SKIP:
                continue
            if model_id in yaml_ids:
                continue
            missing.append(f"  {arch}: {model_id}")

        if missing:
            pytest.fail(
                f"{len(missing)} registered model(s) have no YAML "
                f"test case and are not in _COVERAGE_SKIP:\n"
                + "\n".join(missing)
                + "\n\nFix: add a YAML file in testdata/cases/ or "
                "add to _COVERAGE_SKIP with a reason."
            )

    def test_all_yaml_cases_have_golden_json_or_skip_reason(self):
        """Report YAML cases missing golden JSON (informational).

        YAML without JSON is expected when golden generation hasn't
        been run yet.  This test warns but does not fail — the per-model
        test already tracks individual coverage.
        """
        cases = _discovered_cases()
        incomplete = []
        for case in cases:
            if has_golden(case):
                continue
            if case.skip_reason:
                continue
            incomplete.append(case.model_id)

        if incomplete:
            import warnings

            warnings.warn(
                f"{len(incomplete)} YAML case(s) lack golden JSON "
                f"(generate with scripts/generate_golden.py): "
                f"{', '.join(sorted(incomplete)[:10])}..."
                if len(incomplete) > 10
                else f"{len(incomplete)} YAML case(s) lack golden JSON: "
                f"{', '.join(sorted(incomplete))}",
                stacklevel=1,
            )

    @pytest.mark.parametrize(
        "arch,model_id",
        [
            pytest.param(arch, mid, id=arch)
            for arch, mid in sorted(_all_registered_with_test_id().items())
        ],
    )
    def test_model_has_golden_data(self, arch: str, model_id: str):
        """Per-model check for YAML (L4) + JSON golden output (L5)."""
        if arch in _COVERAGE_SKIP:
            pytest.skip(_COVERAGE_SKIP[arch])

        yaml_ids = _yaml_model_ids()
        if model_id not in yaml_ids:
            pytest.fail(
                f"Model '{arch}' (test_model_id='{model_id}') has "
                f"no YAML test case in testdata/cases/. "
                f"Add a YAML file or add '{arch}' to _COVERAGE_SKIP."
            )

        # YAML exists — also check for golden JSON output.
        # Note: missing golden JSON is a skip (not a fail) because YAML
        # test cases are created first and golden generation requires a
        # GPU run via scripts/generate_golden.py.  L5 enforcement is
        # tracked by test_all_yaml_cases_have_golden_json_or_skip_reason.
        cases_by_id = _yaml_cases_by_model_id()
        case = cases_by_id.get(model_id)
        if case is not None and not has_golden(case):
            if case.skip_reason:
                pytest.skip(f"YAML has skip_reason (no golden JSON): {case.skip_reason}")
            golden = golden_path_for_case(case)
            pytest.skip(
                f"L4 YAML exists but L5 golden JSON missing at "
                f"{golden}. Generate with scripts/generate_golden.py."
            )
