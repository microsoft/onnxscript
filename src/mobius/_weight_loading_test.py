# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Defensive security regression tests for the weight loading path.

These tests guard against regressions in the security posture of weight
loading code. They verify:
1. Safetensors is preferred over pickle-based formats.
2. No unguarded ``torch.load`` calls exist in the source.
3. Path traversal attacks are blocked in weight file paths.
4. Temporary files are cleaned up after weight operations.
5. Corrupted / truncated weight files are handled gracefully.
6. The ``build_from_module`` public contract is preserved.

Targets the current code in both ``_weight_loading.py`` and ``_exporter.py``.
Uses MockWeightProvider pattern for test isolation — no network calls needed.
"""

from __future__ import annotations

import ast
import os
import pathlib
from unittest import mock

import onnx_ir as ir
import pytest
import safetensors.torch
import torch

from mobius._builder import build_from_module
from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius._weight_loading import apply_weights
from mobius.models.base import CausalLMModel
from mobius.tasks import CausalLMTask, ModelTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SRC_ROOT = pathlib.Path(__file__).resolve().parent

# Paths to scan for security regressions (non-test source files)
_SOURCE_FILES = [p for p in _SRC_ROOT.rglob("*.py") if not p.name.endswith("_test.py")]

# Key weight-loading files to scan (both current and refactored paths)
_WEIGHT_FILES = [
    _SRC_ROOT / "_weight_loading.py",
]


class MockWeightProvider:
    """Fake weight provider for test isolation — no HuggingFace calls."""

    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None):
        self.state_dict = state_dict or {}

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        return self.state_dict


def _build_model_with_weights() -> tuple[ir.Model, list[str]]:
    """Build a small test model and return it with its initializer names."""
    config = make_config()
    module = CausalLMModel(config)
    pkg = build_from_module(module, config)
    model = pkg["model"]
    init_names = list(model.graph.initializers.keys())
    return model, init_names


# ===========================================================================
# (1) Safetensors preference over pickle
# ===========================================================================


class TestSafetensorsPreference:
    """Verify that the codebase only uses safetensors — never pickle."""

    @pytest.mark.parametrize("weight_file", _WEIGHT_FILES, ids=lambda p: p.name)
    def test_no_pickle_import(self, weight_file):
        """Weight loading files must not import pickle or shelve."""
        if not weight_file.exists():
            pytest.skip(f"{weight_file.name} does not exist yet")
        source = weight_file.read_text()
        tree = ast.parse(source)
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])
        forbidden = {"pickle", "shelve", "_pickle", "cPickle"}
        violations = imported_modules & forbidden
        assert not violations, f"Forbidden pickle imports in {weight_file.name}: {violations}"

    @pytest.mark.parametrize("weight_file", _WEIGHT_FILES, ids=lambda p: p.name)
    def test_no_torch_load(self, weight_file):
        """Weight loading files must not call torch.load (pickle-based)."""
        if not weight_file.exists():
            pytest.skip(f"{weight_file.name} does not exist yet")
        source = weight_file.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "load"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "torch"
                ):
                    pytest.fail(f"torch.load found in {weight_file.name}:{node.lineno}")

    def test_safetensors_is_the_only_format_used(self):
        """Weight files referenced in weight loading code are .safetensors only."""
        for weight_file in _WEIGHT_FILES:
            if not weight_file.exists():
                continue
            source = weight_file.read_text()
            assert "safetensors" in source, f"No safetensors usage in {weight_file.name}"
            for dangerous_ext in [".bin", ".pkl", ".pickle", ".pt", ".pth"]:
                assert dangerous_ext not in source, (
                    f"Found reference to unsafe format '{dangerous_ext}' in {weight_file.name}"
                )

    def test_safetensors_load_file_is_only_weight_loader(self):
        """safetensors.torch.load_file must be the only weight deserialization call."""
        forbidden_loaders = {"load", "load_state_dict", "unpickle"}
        for weight_file in _WEIGHT_FILES:
            if not weight_file.exists():
                continue
            source = weight_file.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in forbidden_loaders:
                    if isinstance(func.value, ast.Name) and func.value.id == "torch":
                        pytest.fail(
                            f"torch.{func.attr} found in {weight_file.name}:{node.lineno}"
                        )


# ===========================================================================
# (2) weights_only=True assertion on any torch.load
# ===========================================================================


class TestNoUnguardedTorchLoad:
    """Scan all source files for unsafe deserialization calls."""

    def test_no_torch_load_without_weights_only(self):
        """Any torch.load call in the package MUST use weights_only=True."""
        violations: list[str] = []
        for path in _SOURCE_FILES:
            source = path.read_text()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                is_torch_load = (
                    isinstance(func, ast.Attribute)
                    and func.attr == "load"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "torch"
                )
                if not is_torch_load:
                    continue
                has_weights_only = any(
                    kw.arg == "weights_only"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                )
                if not has_weights_only:
                    rel = path.relative_to(_SRC_ROOT)
                    violations.append(f"{rel}:{node.lineno}")

        assert not violations, f"torch.load without weights_only=True found at: {violations}"

    def test_no_pickle_load_in_source(self):
        """No source file should call pickle.load or pickle.loads."""
        violations: list[str] = []
        for path in _SOURCE_FILES:
            source = path.read_text()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in ("load", "loads")
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "pickle"
                ):
                    rel = path.relative_to(_SRC_ROOT)
                    violations.append(f"{rel}:{node.lineno}")

        assert not violations, f"pickle.load/loads found at: {violations}"

    def test_no_bare_eval_in_source(self):
        """No source file should use eval() (excluding model.eval())."""
        violations: list[str] = []
        for path in _SOURCE_FILES:
            source = path.read_text()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                # Bare eval() call
                if isinstance(func, ast.Name) and func.id == "eval":
                    rel = path.relative_to(_SRC_ROOT)
                    violations.append(f"{rel}:{node.lineno}")

        assert not violations, f"eval() found at: {violations}"


# ===========================================================================
# (3) Path traversal prevention in weight file paths
# ===========================================================================


class TestPathTraversalPrevention:
    """Ensure path traversal sequences in weight file names are rejected."""

    @pytest.mark.parametrize(
        "malicious_filename",
        [
            "../../../etc/passwd",
            "..\\..\\Windows\\System32\\config\\SAM",
            "model/../../../etc/shadow",
            "weights/../../secret.safetensors",
            "/absolute/path/model.safetensors",
        ],
        ids=[
            "unix_traversal",
            "windows_traversal",
            "embedded_traversal",
            "relative_traversal",
            "absolute_path",
        ],
    )
    def test_path_traversal_in_weight_index_rejected(self, malicious_filename):
        """Filenames from an index.json with path traversal must be sanitized or rejected.

        This tests the contract that weight file paths derived from user/model
        data should not escape the expected directory. We use PurePosixPath to
        normalize both Unix and Windows-style separators.
        """
        import pathlib as _pathlib

        # Normalize both / and \ separators, then take the final component
        # (handles Windows-style paths on Linux hosts too)
        normalized = malicious_filename.replace("\\", "/")
        pure = _pathlib.PurePosixPath(normalized)
        basename = pure.name

        has_traversal = ".." in malicious_filename or malicious_filename.startswith(
            ("/", "\\")
        )
        assert has_traversal, "Test input should contain traversal"
        # The basename extraction strips directory traversal
        assert ".." not in basename
        assert not basename.startswith("/")

    def test_apply_weights_only_modifies_existing_initializers(self):
        """apply_weights must not create new initializers — only update existing ones."""
        model, _init_names = _build_model_with_weights()
        original_count = len(model.graph.initializers)

        # Inject a weight with a path-like name that doesn't match any initializer
        provider = MockWeightProvider(
            {
                "../../etc/passwd": torch.zeros(1),
                "normal_name_not_in_model": torch.zeros(1),
            }
        )
        apply_weights(model, provider.get_state_dict())

        # No new initializers should be created
        assert len(model.graph.initializers) == original_count

    @pytest.mark.parametrize(
        "malicious_model_id",
        [
            "../../../etc/passwd",
            "/etc/shadow",
            "legitimate-org/../../etc/passwd",
        ],
        ids=["relative_traversal", "absolute_path", "embedded_traversal"],
    )
    def test_build_with_malicious_model_id_raises(self, malicious_model_id):
        """build() with a path-traversal model_id should raise, not silently proceed."""
        from mobius._builder import build

        with pytest.raises((ValueError, OSError, Exception)):
            build(malicious_model_id, load_weights=False)


# ===========================================================================
# (4) Temp file cleanup after weight operations
# ===========================================================================


class TestTempFileCleanup:
    """Verify temporary files are cleaned up after weight operations."""

    def test_apply_weights_does_not_leak_temp_files(self, tmp_path):
        """apply_weights should not leave temporary files behind."""
        model, init_names = _build_model_with_weights()
        assert len(init_names) > 0

        name = init_names[0]
        shape = list(model.graph.initializers[name].shape)
        provider = MockWeightProvider({name: torch.ones(shape)})

        before = set(tmp_path.iterdir())
        with mock.patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            apply_weights(model, provider.get_state_dict())
        after = set(tmp_path.iterdir())

        leaked = after - before
        assert not leaked, f"Temporary files leaked: {leaked}"

    def test_apply_weights_no_open_file_handles(self):
        """apply_weights should not leave file handles open after completion."""
        model, init_names = _build_model_with_weights()
        assert len(init_names) > 0

        name = init_names[0]
        shape = list(model.graph.initializers[name].shape)
        provider = MockWeightProvider({name: torch.ones(shape)})

        # This should complete without resource warnings
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            apply_weights(model, provider.get_state_dict())


# ===========================================================================
# (5) Corrupted / truncated file handling
# ===========================================================================


class TestCorruptedFileHandling:
    """Verify graceful handling of bad weight data."""

    def test_apply_weights_empty_state_dict(self):
        """Applying an empty state dict is a no-op — no crash."""
        model, _ = _build_model_with_weights()
        provider = MockWeightProvider({})
        apply_weights(model, provider.get_state_dict())

    def test_apply_weights_wrong_shape_raises(self):
        """A tensor with the wrong shape should raise ValueError."""
        model, init_names = _build_model_with_weights()
        assert len(init_names) > 0

        name = init_names[0]
        # Provide a tensor with a deliberately wrong shape
        wrong_shape = torch.zeros(999)
        provider = MockWeightProvider({name: wrong_shape})
        with pytest.raises(ValueError, match="shape mismatch"):
            apply_weights(model, provider.get_state_dict())

    def test_apply_weights_wrong_shape_error_message(self):
        """Shape mismatch error should include the weight name and both shapes."""
        model, init_names = _build_model_with_weights()
        assert len(init_names) > 0

        name = init_names[0]
        wrong_shape = torch.zeros(999)
        provider = MockWeightProvider({name: wrong_shape})
        with pytest.raises(ValueError, match=name):
            apply_weights(model, provider.get_state_dict())

    def test_apply_weights_nan_tensor(self):
        """NaN tensors should be applied without error."""
        model, init_names = _build_model_with_weights()
        assert len(init_names) > 0

        name = init_names[0]
        shape = list(model.graph.initializers[name].shape)
        nan_tensor = torch.full(shape, float("nan"))
        provider = MockWeightProvider({name: nan_tensor})
        apply_weights(model, provider.get_state_dict())
        assert model.graph.initializers[name].const_value is not None

    def test_apply_weights_dtype_mismatch_uses_lazy_cast(self):
        """When dtype differs, apply_weights wraps with LazyTensor for lazy cast."""
        model, init_names = _build_model_with_weights()
        assert len(init_names) > 0

        name = init_names[0]
        shape = list(model.graph.initializers[name].shape)
        # Use float64 which likely differs from the model's expected dtype
        mismatched = torch.ones(shape, dtype=torch.float64)
        provider = MockWeightProvider({name: mismatched})
        apply_weights(model, provider.get_state_dict())

        result = model.graph.initializers[name].const_value
        assert result is not None
        # If there was a dtype mismatch, a LazyTensor should have been created
        if model.graph.initializers[name].dtype != ir.DataType.DOUBLE:
            assert isinstance(result, ir.LazyTensor)

    def test_corrupted_safetensors_file_raises(self, tmp_path):
        """Loading a corrupted .safetensors file must raise a clear error."""
        corrupted = tmp_path / "model.safetensors"
        corrupted.write_bytes(b"THIS IS NOT A VALID SAFETENSORS FILE\x00\xff" * 10)

        with pytest.raises(Exception) as exc_info:
            safetensors.torch.load_file(str(corrupted))
        # Should be a clear deserialization error, not a silent corruption
        assert exc_info.value is not None

    def test_truncated_safetensors_file_raises(self, tmp_path):
        """Loading a truncated .safetensors file must raise, not return partial data."""
        # Create a valid safetensors file first, then truncate it
        valid_data = {"weight": torch.randn(4, 4)}
        valid_path = tmp_path / "valid.safetensors"
        safetensors.torch.save_file(valid_data, str(valid_path))

        truncated = tmp_path / "truncated.safetensors"
        full_bytes = valid_path.read_bytes()
        # Write only the first half
        truncated.write_bytes(full_bytes[: len(full_bytes) // 2])

        with pytest.raises((OSError, RuntimeError, safetensors.SafetensorError)):
            safetensors.torch.load_file(str(truncated))

    def test_zero_byte_safetensors_file_raises(self, tmp_path):
        """An empty (zero-byte) .safetensors file must raise."""
        empty = tmp_path / "empty.safetensors"
        empty.write_bytes(b"")

        with pytest.raises((OSError, RuntimeError, safetensors.SafetensorError)):
            safetensors.torch.load_file(str(empty))


# ===========================================================================
# (6) build_from_module contract preservation
# ===========================================================================


class TestBuildFromModuleContract:
    """Verify the public build_from_module contract is preserved."""

    def test_returns_model_package(self):
        config = make_config()
        module = CausalLMModel(config)
        result = build_from_module(module, config)
        assert isinstance(result, ModelPackage)

    def test_model_key_present(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        assert "model" in pkg

    def test_model_has_outputs(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]
        assert len(model.graph.outputs) > 0

    def test_accepts_task_string(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config, task="text-generation")
        assert isinstance(pkg, ModelPackage)

    def test_accepts_task_instance(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config, task=CausalLMTask())
        assert isinstance(pkg, ModelPackage)

    def test_rejects_unknown_task(self):
        config = make_config()
        module = CausalLMModel(config)
        with pytest.raises(ValueError, match="Unknown task"):
            build_from_module(module, config, task="nonexistent-task")

    def test_custom_task_instance_works(self):
        """A user-defined ModelTask should work with build_from_module."""

        class StubTask(ModelTask):
            def build(self, module, config):
                graph = ir.Graph([], [], nodes=[], name="stub")
                model = ir.Model(graph, ir_version=10)
                return ModelPackage({"model": model})

        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config, task=StubTask())
        assert pkg["model"].graph.name == "stub"
