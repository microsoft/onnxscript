# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for _diffusers_builder.py — diffusers pipeline building."""

from __future__ import annotations

from unittest.mock import patch

import onnx_ir as ir
import pytest

from mobius._diffusers_builder import (
    _DIFFUSERS_CLASS_MAP,
    _init_diffusers_class_map,
    build_diffusers_pipeline,
)
from mobius._model_package import ModelPackage

# ── Helpers ──────────────────────────────────────────────────────────────


def _fake_pipeline_index(
    components: dict[str, list[str]] | None = None,
) -> dict:
    """Build a fake model_index.json dict.

    Args:
        components: Mapping of component name to [library, class_name].
            Defaults to a single FluxTransformer2DModel component.
    """
    index: dict = {"_class_name": "FluxPipeline"}
    if components is None:
        components = {
            "transformer": ["diffusers", "FluxTransformer2DModel"],
        }
    index.update(components)
    return index


# ── _init_diffusers_class_map ────────────────────────────────────────────


class TestInitDiffusersClassMap:
    """Tests for lazy initialization of the diffusers class map."""

    def test_populates_expected_classes(self):
        _init_diffusers_class_map()
        expected_keys = {
            "DiTTransformer2DModel",
            "HunyuanDiT2DModel",
            "PixArtTransformer2DModel",
            "FluxTransformer2DModel",
            "SD3Transformer2DModel",
            "QwenImageTransformer2DModel",
            "AutoencoderKL",
            "AutoencoderKLQwenImage",
            "AutoencoderKLCogVideoX",
            "CogVideoXTransformer3DModel",
        }
        assert expected_keys == set(_DIFFUSERS_CLASS_MAP.keys())

    def test_each_entry_is_three_tuple(self):
        _init_diffusers_class_map()
        for class_name, entry in _DIFFUSERS_CLASS_MAP.items():
            assert len(entry) == 3, (
                f"Entry for {class_name} should be (module_class, config_class, task_name)"
            )
            module_class, config_class, task_name = entry
            assert callable(module_class)
            assert callable(config_class)
            assert isinstance(task_name, str)

    def test_task_names_are_valid(self):
        _init_diffusers_class_map()
        valid_tasks = {"denoising", "vae", "qwen-image-vae", "video-denoising"}
        for class_name, (_, _, task_name) in _DIFFUSERS_CLASS_MAP.items():
            assert task_name in valid_tasks, f"Unknown task '{task_name}' for {class_name}"

    def test_idempotent(self):
        """Calling _init_diffusers_class_map twice does not duplicate entries."""
        _init_diffusers_class_map()
        count_before = len(_DIFFUSERS_CLASS_MAP)
        _init_diffusers_class_map()
        assert len(_DIFFUSERS_CLASS_MAP) == count_before


# ── build_diffusers_pipeline error handling ──────────────────────────────


class TestBuildDiffusersPipelineErrors:
    """Tests for error paths in build_diffusers_pipeline."""

    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
        return_value=None,
    )
    def test_raises_when_no_model_index(self, _mock_load):
        """ValueError when model_index.json is not found."""
        with pytest.raises(ValueError, match="does not appear to be a diffusers pipeline"):
            build_diffusers_pipeline("fake/no-index-model", load_weights=False)

    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
        return_value={"_class_name": "SomePipeline"},
    )
    def test_raises_when_no_supported_components(self, _mock_load):
        """ValueError when pipeline has no registered neural network components."""
        with pytest.raises(ValueError, match="No supported neural network components"):
            build_diffusers_pipeline("fake/empty-pipeline", load_weights=False)

    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
        return_value={
            "_class_name": "SomePipeline",
            "scheduler": ["diffusers", "EulerDiscreteScheduler"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
        },
    )
    def test_raises_when_only_non_nn_components(self, _mock_load):
        """ValueError when pipeline only has non-NN components (scheduler, tokenizer)."""
        with pytest.raises(ValueError, match="No supported neural network components"):
            build_diffusers_pipeline("fake/scheduler-only", load_weights=False)


# ── build_diffusers_pipeline component filtering ─────────────────────────


class TestBuildDiffusersPipelineFiltering:
    """Tests for how build_diffusers_pipeline filters pipeline components."""

    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_skips_underscore_prefixed_keys(self, mock_load_index, mock_load_config):
        """Keys starting with '_' (like _class_name) are skipped."""
        mock_load_index.return_value = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0",
        }
        # Should raise because no valid components remain
        with pytest.raises(ValueError, match="No supported neural network"):
            build_diffusers_pipeline("fake/model", load_weights=False)
        # _load_diffusers_component_config should never be called
        mock_load_config.assert_not_called()

    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_skips_non_list_entries(self, mock_load_index, mock_load_config):
        """Non-list entries (e.g. strings, dicts) are skipped."""
        mock_load_index.return_value = {
            "_class_name": "FluxPipeline",
            "some_string": "not a list",
            "some_dict": {"key": "value"},
            "some_int": 42,
        }
        with pytest.raises(ValueError, match="No supported neural network"):
            build_diffusers_pipeline("fake/model", load_weights=False)
        mock_load_config.assert_not_called()

    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_skips_lists_with_wrong_length(self, mock_load_index, mock_load_config):
        """Lists that don't have exactly 2 elements are skipped."""
        mock_load_index.return_value = {
            "_class_name": "FluxPipeline",
            "single": ["only_one"],
            "triple": ["a", "b", "c"],
            "empty": [],
        }
        with pytest.raises(ValueError, match="No supported neural network"):
            build_diffusers_pipeline("fake/model", load_weights=False)
        mock_load_config.assert_not_called()

    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_skips_unregistered_class_names(self, mock_load_index, mock_load_config):
        """Components with unregistered class names are skipped with a log message."""
        mock_load_index.return_value = {
            "_class_name": "FluxPipeline",
            "scheduler": ["diffusers", "EulerDiscreteScheduler"],
            "text_encoder": ["transformers", "CLIPTextModel"],
        }
        with pytest.raises(ValueError, match="No supported neural network"):
            build_diffusers_pipeline("fake/model", load_weights=False)
        mock_load_config.assert_not_called()


# ── build_diffusers_pipeline successful build ────────────────────────────


class TestBuildDiffusersPipelineSuccess:
    """Tests for successful build_diffusers_pipeline calls."""

    def _mock_build_for_vae(self, mock_load_index, mock_load_config, mock_build_from_module):
        """Set up mocks for a minimal VAE component build."""
        mock_load_index.return_value = _fake_pipeline_index(
            {"vae": ["diffusers", "AutoencoderKL"]}
        )
        # Minimal diffusers VAE config
        mock_load_config.return_value = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
        }
        # Return a fake ModelPackage with a minimal model
        graph = ir.Graph([], [], nodes=[], name="fake_vae")
        model = ir.Model(graph, ir_version=10)
        mock_build_from_module.return_value = ModelPackage({"model": model})

    @patch("mobius._diffusers_builder.build_from_module")
    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_returns_model_package(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """Successful build returns a ModelPackage."""
        self._mock_build_for_vae(mock_load_index, mock_load_config, mock_build_from_module)
        result = build_diffusers_pipeline("fake/vae-model", load_weights=False)
        assert isinstance(result, ModelPackage)

    @patch("mobius._diffusers_builder.build_from_module")
    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_single_model_subpackage_flattened(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """When sub-package has one 'model' entry, it's flattened to component name."""
        self._mock_build_for_vae(mock_load_index, mock_load_config, mock_build_from_module)
        result = build_diffusers_pipeline("fake/vae-model", load_weights=False)
        # The "model" key from sub-package becomes "vae" in the top-level package
        assert "vae" in result
        assert "model" not in result

    @patch("mobius._diffusers_builder.build_from_module")
    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_graph_name_set_to_model_id_component(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """Graph name is set to '{model_id}/{component_name}'."""
        self._mock_build_for_vae(mock_load_index, mock_load_config, mock_build_from_module)
        result = build_diffusers_pipeline("fake/vae-model", load_weights=False)
        assert result["vae"].graph.name == "fake/vae-model/vae"

    @patch("mobius._diffusers_builder.build_from_module")
    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_multi_model_subpackage_prefixed(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """When sub-package has multiple entries, they're prefixed with component name."""
        mock_load_index.return_value = _fake_pipeline_index(
            {"vae": ["diffusers", "AutoencoderKL"]}
        )
        mock_load_config.return_value = {"in_channels": 3}
        # Return a sub-package with multiple models
        graph_a = ir.Graph([], [], nodes=[], name="encoder")
        graph_b = ir.Graph([], [], nodes=[], name="decoder")
        mock_build_from_module.return_value = ModelPackage(
            {
                "encoder": ir.Model(graph_a, ir_version=10),
                "decoder": ir.Model(graph_b, ir_version=10),
            }
        )
        result = build_diffusers_pipeline("fake/multi", load_weights=False)
        assert "vae_encoder" in result
        assert "vae_decoder" in result
        assert result["vae_encoder"].graph.name == "fake/multi/vae_encoder"
        assert result["vae_decoder"].graph.name == "fake/multi/vae_decoder"

    @patch("mobius._diffusers_builder.build_from_module")
    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_multiple_components_built(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """Pipeline with multiple supported components builds all of them."""
        mock_load_index.return_value = _fake_pipeline_index(
            {
                "transformer": ["diffusers", "FluxTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                # This should be skipped (unregistered)
                "text_encoder": ["transformers", "CLIPTextModel"],
            }
        )
        mock_load_config.return_value = {}

        def fake_build(module, config, task_name):
            graph = ir.Graph([], [], nodes=[], name="g")
            return ModelPackage({"model": ir.Model(graph, ir_version=10)})

        mock_build_from_module.side_effect = fake_build

        result = build_diffusers_pipeline("fake/flux", load_weights=False)
        assert "transformer" in result
        assert "vae" in result
        assert "text_encoder" not in result

    @patch("mobius._diffusers_builder.build_from_module")
    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_dtype_string_resolved(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """String dtype is resolved to ir.DataType and passed to config."""
        self._mock_build_for_vae(mock_load_index, mock_load_config, mock_build_from_module)
        # Should not raise — dtype string "f16" is resolved by resolve_dtype()
        build_diffusers_pipeline("fake/vae-model", dtype="f16", load_weights=False)
        # Verify build_from_module was called (string dtype resolved without error)
        mock_build_from_module.assert_called_once()

    @patch("mobius._diffusers_builder.build_from_module")
    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_dtype_ir_datatype_passthrough(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
    ):
        """ir.DataType dtype is passed through without conversion."""
        self._mock_build_for_vae(mock_load_index, mock_load_config, mock_build_from_module)
        build_diffusers_pipeline(
            "fake/vae-model",
            dtype=ir.DataType.FLOAT16,
            load_weights=False,
        )
        # Verify build_from_module was called (ir.DataType accepted without error)
        mock_build_from_module.assert_called_once()


# ── build_diffusers_pipeline weight loading ──────────────────────────────


class TestBuildDiffusersPipelineWeights:
    """Tests for weight loading paths in build_diffusers_pipeline."""

    @patch("mobius._diffusers_builder.apply_weights")
    @patch(
        "mobius._diffusers_builder._download_diffusers_component_weights",
    )
    @patch("mobius._diffusers_builder.build_from_module")
    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_load_weights_true_downloads_and_applies(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
        mock_download_weights,
        mock_apply_weights,
    ):
        """When load_weights=True, weights are downloaded and applied."""
        mock_load_index.return_value = _fake_pipeline_index(
            {"vae": ["diffusers", "AutoencoderKL"]}
        )
        mock_load_config.return_value = {}
        graph = ir.Graph([], [], nodes=[], name="vae")
        model = ir.Model(graph, ir_version=10)
        mock_build_from_module.return_value = ModelPackage({"model": model})
        mock_download_weights.return_value = {}

        build_diffusers_pipeline("fake/model", load_weights=True)

        mock_download_weights.assert_called_once_with("fake/model", "vae")
        mock_apply_weights.assert_called_once()

    @patch(
        "mobius._diffusers_builder._download_diffusers_component_weights",
    )
    @patch("mobius._diffusers_builder.build_from_module")
    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_load_weights_false_skips_download(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
        mock_download_weights,
    ):
        """When load_weights=False, no weight download occurs."""
        mock_load_index.return_value = _fake_pipeline_index(
            {"vae": ["diffusers", "AutoencoderKL"]}
        )
        mock_load_config.return_value = {}
        graph = ir.Graph([], [], nodes=[], name="vae")
        mock_build_from_module.return_value = ModelPackage(
            {"model": ir.Model(graph, ir_version=10)}
        )

        build_diffusers_pipeline("fake/model", load_weights=False)
        mock_download_weights.assert_not_called()

    @patch("mobius._diffusers_builder.apply_weights")
    @patch(
        "mobius._diffusers_builder._download_diffusers_component_weights",
    )
    @patch("mobius._diffusers_builder.build_from_module")
    @patch(
        "mobius._diffusers_builder._load_diffusers_component_config",
    )
    @patch(
        "mobius._diffusers_builder._load_diffusers_pipeline_index",
    )
    def test_preprocess_weights_called_when_available(
        self,
        mock_load_index,
        mock_load_config,
        mock_build_from_module,
        mock_download_weights,
        mock_apply_weights,
    ):
        """preprocess_weights is called on the module when it has the method."""
        mock_load_index.return_value = _fake_pipeline_index(
            {"vae": ["diffusers", "AutoencoderKL"]}
        )
        mock_load_config.return_value = {}
        graph = ir.Graph([], [], nodes=[], name="vae")
        model = ir.Model(graph, ir_version=10)
        mock_build_from_module.return_value = ModelPackage({"model": model})
        raw_weights = {"weight.data": "raw"}
        processed_weights = {"weight.data": "processed"}
        mock_download_weights.return_value = raw_weights

        # The module class will have preprocess_weights set by AutoencoderKLModel
        # We patch it at the module instance level via build_from_module's first arg
        def capture_build(module, config, task_name):
            module.preprocess_weights = lambda sd: processed_weights
            return ModelPackage({"model": model})

        mock_build_from_module.side_effect = capture_build

        build_diffusers_pipeline("fake/model", load_weights=True)
        # apply_weights should receive the processed weights
        call_args = mock_apply_weights.call_args
        assert call_args[0][1] is processed_weights
