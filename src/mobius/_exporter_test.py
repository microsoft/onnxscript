# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the build API (build, build_from_module, registry)."""

from __future__ import annotations

import logging

import onnx_ir as ir
import pytest
import torch

from mobius._builder import (
    _cast_module_dtype,
    build,
    build_from_module,
    resolve_dtype,
)
from mobius._config_resolver import (
    _config_from_hf,
    _default_task_for_model,
)
from mobius._model_package import ModelPackage
from mobius._registry import (
    MODEL_MAP,
    ModelRegistry,
    registry,
)
from mobius._testing import make_config
from mobius._weight_loading import apply_weights
from mobius.models.base import CausalLMModel
from mobius.tasks import CausalLMTask, ModelTask


class TestBuildFromModule:
    def test_default_task(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        assert isinstance(pkg, ModelPackage)
        model = pkg["model"]
        assert "logits" in [v.name for v in model.graph.outputs]

    def test_explicit_task_string(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config, task="text-generation")
        assert isinstance(pkg, ModelPackage)

    def test_explicit_task_instance(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config, task=CausalLMTask())
        assert isinstance(pkg, ModelPackage)

    def test_invalid_task_raises(self):
        config = make_config()
        module = CausalLMModel(config)
        with pytest.raises(ValueError, match="Unknown task"):
            build_from_module(module, config, task="nonexistent")

    def test_invalid_config_raises_before_build(self):
        """build_from_module validates config and rejects invalid fields."""
        from mobius._configs import DEFAULT_INT, ArchitectureConfig

        config = ArchitectureConfig()  # All fields at DEFAULT_INT sentinel
        assert config.hidden_size == DEFAULT_INT
        module = CausalLMModel(make_config())  # valid module
        with pytest.raises(ValueError, match="Invalid ArchitectureConfig"):
            build_from_module(module, config)

    def test_custom_task(self):
        """User-defined task works with build_from_module."""

        class MinimalTask(ModelTask):
            def build(self, module, config):
                graph = ir.Graph([], [], nodes=[], name="minimal")
                model = ir.Model(graph, ir_version=10)
                return ModelPackage({"model": model})

        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config, task=MinimalTask())
        assert pkg["model"].graph.name == "minimal"


class TestApplyWeights:
    def test_apply_weights(self):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]

        # Find a weight name from the model
        init_names = list(model.graph.initializers.keys())
        assert len(init_names) > 0
        name = init_names[0]
        shape = list(model.graph.initializers[name].shape)
        weight = torch.ones(shape)

        apply_weights(model, {name: weight})
        assert model.graph.initializers[name].const_value is not None

    def test_apply_weights_unknown_key_warns(self, caplog):
        config = make_config()
        module = CausalLMModel(config)
        model = build_from_module(module, config)["model"]

        with caplog.at_level(logging.WARNING):
            apply_weights(model, {"nonexistent.weight": torch.zeros(1)})
        assert "not found in the model" in caplog.text


class TestModelRegistry:
    def test_register_and_get(self):
        reg = ModelRegistry()
        reg.register("test", CausalLMModel)
        assert reg.get("test") is CausalLMModel

    def test_contains(self):
        reg = ModelRegistry()
        reg.register("test", CausalLMModel)
        assert "test" in reg
        assert "other" not in reg

    def test_len(self):
        reg = ModelRegistry()
        assert len(reg) == 0
        reg.register("a", CausalLMModel)
        reg.register("b", CausalLMModel)
        assert len(reg) == 2

    def test_architectures_sorted(self):
        reg = ModelRegistry()
        reg.register("zulu", CausalLMModel)
        reg.register("alpha", CausalLMModel)
        assert reg.architectures() == ["alpha", "zulu"]

    def test_get_unknown_raises(self):
        reg = ModelRegistry()
        with pytest.raises(KeyError, match="Unknown model_type"):
            reg.get("unknown")

    def test_default_registry_has_all_architectures(self):
        assert len(registry) >= 50
        assert "llama" in registry
        assert "phi3" in registry
        assert "mixtral" in registry

    def test_model_map_is_registry_backed(self):
        """MODEL_MAP dict is backed by the registry."""
        assert MODEL_MAP is registry._map


class TestResolveDtype:
    @pytest.mark.parametrize(
        ("input_dtype", "expected"),
        [
            ("f32", ir.DataType.FLOAT),
            ("float32", ir.DataType.FLOAT),
            ("f16", ir.DataType.FLOAT16),
            ("float16", ir.DataType.FLOAT16),
            ("bf16", ir.DataType.BFLOAT16),
            ("bfloat16", ir.DataType.BFLOAT16),
        ],
    )
    def test_string_dtypes(self, input_dtype, expected):
        assert resolve_dtype(input_dtype) is expected

    def test_none_passthrough(self):
        assert resolve_dtype(None) is None

    def test_ir_datatype_passthrough(self):
        assert resolve_dtype(ir.DataType.FLOAT16) is ir.DataType.FLOAT16

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            resolve_dtype("int8")


class TestCastModuleDtype:
    def test_float_is_noop(self):
        """Casting to FLOAT should not change parameter dtypes."""
        config = make_config()
        module = CausalLMModel(config)
        original_dtypes = [p.dtype for p in module.parameters()]
        _cast_module_dtype(module, ir.DataType.FLOAT)
        assert [p.dtype for p in module.parameters()] == original_dtypes

    def test_cast_to_float16(self):
        """Casting to FLOAT16 converts FLOAT parameters."""
        config = make_config()
        module = CausalLMModel(config)
        _cast_module_dtype(module, ir.DataType.FLOAT16)
        for param in module.parameters():
            if param.dtype in (ir.DataType.FLOAT16, ir.DataType.INT64):
                continue  # Integer params are untouched
            assert param.dtype != ir.DataType.FLOAT, (
                "Parameter should have been cast from FLOAT"
            )


class TestConfigFromHf:
    def _make_hf_config(self, model_type="llama", **kwargs):
        """Create a minimal HF-config-like object for testing."""
        defaults = {
            "model_type": model_type,
            "vocab_size": 100,
            "max_position_embeddings": 32,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "head_dim": 16,
            "pad_token_id": 0,
        }
        defaults.update(kwargs)
        return type("FakeHFConfig", (), defaults)()

    def test_fallback_to_architecture_config(self):
        """Registered model without custom config_class uses ArchitectureConfig."""
        from mobius._configs import ArchitectureConfig

        # llama is registered but has no custom config_class, so
        # _config_from_hf should fall through to ArchitectureConfig
        hf_config = self._make_hf_config(model_type="llama")
        result = _config_from_hf(hf_config)
        assert isinstance(result, ArchitectureConfig)

    def test_module_class_config_override(self):
        """module_class.config_class takes priority when set."""
        from mobius._configs import ArchitectureConfig

        class CustomConfig(ArchitectureConfig):
            pass

        class FakeModule:
            config_class = CustomConfig

        hf_config = self._make_hf_config()
        result = _config_from_hf(hf_config, module_class=FakeModule)
        assert isinstance(result, CustomConfig)

    def test_registered_model_type(self):
        """Registered model type resolves config from registry."""
        hf_config = self._make_hf_config(model_type="llama")
        result = _config_from_hf(hf_config)
        # llama uses default ArchitectureConfig (no custom config_class)
        from mobius._configs import ArchitectureConfig

        assert isinstance(result, ArchitectureConfig)
        assert result.vocab_size == 100


class TestDefaultTaskForModel:
    def test_registered_with_explicit_task(self):
        """Registered model with explicit task returns it."""
        assert _default_task_for_model("whisper") == "speech-to-text"

    def test_registered_without_task_falls_back(self):
        """Registered model without explicit task uses default_task or text-generation."""
        result = _default_task_for_model("llama")
        assert result == "text-generation"

    def test_unregistered_returns_text_generation(self):
        assert _default_task_for_model("nonexistent_arch_xyz") == "text-generation"


class TestBuildOrchestration:
    """Tests for the build() function with mocked HuggingFace dependencies."""

    def _mock_hf_config(self, model_type="llama"):
        """Create a mock HF config for testing build() without network."""
        return type(
            "FakeHFConfig",
            (),
            {
                "model_type": model_type,
                "vocab_size": 100,
                "max_position_embeddings": 32,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_act": "silu",
                "head_dim": 16,
                "pad_token_id": 0,
                "torch_dtype": "float32",
            },
        )()

    def test_build_no_weights(self, monkeypatch):
        """build() with load_weights=False produces a valid ModelPackage."""
        import transformers

        fake_config = self._mock_hf_config()
        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            lambda *a, **kw: fake_config,
        )
        pkg = build("fake/model-id", load_weights=False)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg
        assert "logits" in [v.name for v in pkg["model"].graph.outputs]

    def test_build_explicit_module_class(self, monkeypatch):
        """build() with explicit module_class uses it instead of registry."""
        import transformers

        fake_config = self._mock_hf_config()
        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            lambda *a, **kw: fake_config,
        )
        pkg = build("fake/model-id", module_class=CausalLMModel, load_weights=False)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg

    def test_build_explicit_dtype(self, monkeypatch):
        """build() with dtype override applies it to the config."""
        import transformers

        fake_config = self._mock_hf_config()
        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            lambda *a, **kw: fake_config,
        )
        pkg = build("fake/model-id", dtype="f16", load_weights=False)
        assert isinstance(pkg, ModelPackage)
        # Verify that FLOAT16 was applied — initializer dtypes should be FLOAT16
        model = pkg["model"]
        for init in model.graph.initializers.values():
            if init.dtype == ir.DataType.FLOAT16:
                break
        else:
            pytest.fail("Expected at least one FLOAT16 initializer after dtype='f16'")

    def test_build_explicit_task(self, monkeypatch):
        """build() with explicit task string uses it."""
        import transformers

        fake_config = self._mock_hf_config()
        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            lambda *a, **kw: fake_config,
        )
        pkg = build("fake/model-id", task="text-generation", load_weights=False)
        assert isinstance(pkg, ModelPackage)
        assert "logits" in [v.name for v in pkg["model"].graph.outputs]

    def test_build_sets_graph_name(self, monkeypatch):
        """build() sets graph names to model_id/component."""
        import transformers

        fake_config = self._mock_hf_config()
        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            lambda *a, **kw: fake_config,
        )
        pkg = build("fake/model-id", load_weights=False)
        assert pkg["model"].graph.name == "fake/model-id/model"
