# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelPackage."""

from __future__ import annotations

import logging

import onnx_ir as ir
import torch

from mobius._builder import build_from_module
from mobius._configs import VisionConfig
from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius.models.base import CausalLMModel
from mobius.models.gemma3 import Gemma3MultiModalModel
from mobius.tasks import CausalLMTask, VisionLanguageTask


def _make_simple_model(name: str = "test") -> ir.Model:
    """Create a minimal ir.Model for testing."""
    graph = ir.Graph([], [], nodes=[], name=name)
    return ir.Model(graph, ir_version=10)


class TestModelPackageDict:
    def test_getitem(self):
        m = _make_simple_model()
        pkg = ModelPackage({"a": m})
        assert pkg["a"] is m

    def test_setitem(self):
        pkg = ModelPackage()
        m = _make_simple_model()
        pkg["new"] = m
        assert pkg["new"] is m

    def test_delitem(self):
        pkg = ModelPackage({"a": _make_simple_model()})
        del pkg["a"]
        assert "a" not in pkg

    def test_contains(self):
        pkg = ModelPackage({"a": _make_simple_model()})
        assert "a" in pkg
        assert "b" not in pkg

    def test_len(self):
        pkg = ModelPackage({"a": _make_simple_model(), "b": _make_simple_model()})
        assert len(pkg) == 2

    def test_iter(self):
        pkg = ModelPackage({"x": _make_simple_model(), "y": _make_simple_model()})
        assert sorted(pkg) == ["x", "y"]

    def test_keys_values_items(self):
        m1 = _make_simple_model("m1")
        m2 = _make_simple_model("m2")
        pkg = ModelPackage({"a": m1, "b": m2})
        assert list(pkg.keys()) == ["a", "b"]
        assert list(pkg.values()) == [m1, m2]
        assert list(pkg.items()) == [("a", m1), ("b", m2)]

    def test_repr(self):
        pkg = ModelPackage({"text_decoder": _make_simple_model()})
        assert "text_decoder" in repr(pkg)

    def test_empty(self):
        pkg = ModelPackage()
        assert len(pkg) == 0

    def test_config_stored(self):
        config = make_config()
        pkg = ModelPackage({"m": _make_simple_model()}, config=config)
        assert pkg.config is config


class TestModelPackageSaveLoad:
    def test_save_creates_files(self, tmp_path):
        pkg = ModelPackage(
            {
                "text_decoder": _make_simple_model("decoder"),
                "vision_encoder": _make_simple_model("encoder"),
            }
        )
        pkg.save(str(tmp_path))
        assert (tmp_path / "text_decoder" / "model.onnx").exists()
        assert (tmp_path / "vision_encoder" / "model.onnx").exists()

    def test_roundtrip(self, tmp_path):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)

        pkg.save(str(tmp_path), check_weights=False)
        loaded = ModelPackage.load(str(tmp_path))

        assert len(loaded) == 1
        assert "model" in loaded
        assert loaded["model"].graph.num_nodes() == pkg["model"].graph.num_nodes()

    def test_load_multiple(self, tmp_path):
        pkg = ModelPackage(
            {
                "a": _make_simple_model("a"),
                "b": _make_simple_model("b"),
            }
        )
        pkg.save(str(tmp_path))
        loaded = ModelPackage.load(str(tmp_path))
        assert sorted(loaded) == ["a", "b"]

    def test_save_creates_directory(self, tmp_path):
        outdir = tmp_path / "nested" / "dir"
        pkg = ModelPackage({"m": _make_simple_model()})
        pkg.save(str(outdir))
        assert (outdir / "model.onnx").exists()


class TestModelPackageApplyWeights:
    def test_single_component(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers.keys())
        name = init_names[0]
        shape = list(model.graph.initializers[name].shape)
        weight = torch.ones(shape)

        pkg.apply_weights({name: weight})
        assert model.graph.initializers[name].const_value is not None

    def test_multi_component_with_prefix_map(self):
        config = make_config()
        m1 = CausalLMModel(config)
        m2 = CausalLMModel(config)
        pkg1 = build_from_module(m1, config)
        pkg2 = build_from_module(m2, config)
        model1 = pkg1["model"]
        model2 = pkg2["model"]

        pkg = ModelPackage({"text": model1, "vision": model2})

        # Get a weight name from model1
        init_name = next(iter(model1.graph.initializers.keys()))
        shape = list(model1.graph.initializers[init_name].shape)
        weight = torch.ones(shape)

        # Route via prefix
        pkg.apply_weights(
            {f"text.{init_name}": weight},
            prefix_map={"text.": "text", "vision.": "vision"},
        )
        assert model1.graph.initializers[init_name].const_value is not None


class TestBuildPackageFromModule:
    def test_returns_model_package(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        assert isinstance(pkg, ModelPackage)
        assert len(pkg) == 1
        assert "model" in pkg

    def test_package_model_has_correct_outputs(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]
        output_names = [v.name for v in model.graph.outputs]
        assert "logits" in output_names

    def test_with_task_string(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config, task="text-generation")
        assert isinstance(pkg, ModelPackage)

    def test_with_task_instance(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config, task=CausalLMTask())
        assert isinstance(pkg, ModelPackage)

    def test_config_stored(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        assert pkg.config is config

    def test_package_save_load_roundtrip(self, tmp_path):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)

        pkg.save(str(tmp_path), check_weights=False)
        loaded = ModelPackage.load(str(tmp_path))

        assert len(loaded) == 1
        assert loaded["model"].graph.num_nodes() == pkg["model"].graph.num_nodes()


class TestMultiModalPackageIntegration:
    """Integration tests for multimodal model packages."""

    def _make_multimodal_config(self):
        return make_config(
            sliding_window=8,
            layer_types=["full_attention", "sliding_attention"],
            attn_qk_norm=True,
            rope_local_base_freq=10_000.0,
            vision=VisionConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                image_size=32,
                patch_size=8,
                norm_eps=1e-6,
                image_token_id=999,
            ),
            image_token_id=999,
        )

    def test_build_multimodal_package(self):
        config = self._make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert set(pkg.keys()) == {"decoder", "vision", "embedding"}
        assert pkg["decoder"].graph.num_nodes() > 0

    def test_multimodal_package_save_load(self, tmp_path):
        config = self._make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)

        pkg.save(str(tmp_path), check_weights=False)
        loaded = ModelPackage.load(str(tmp_path))
        assert len(loaded) == len(pkg)
        for name in pkg:
            assert loaded[name].graph.num_nodes() == pkg[name].graph.num_nodes()

    def test_multimodal_model_has_vision_params(self):
        config = self._make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)

        # Vision model has vision_tower and projector params
        vision_inits = list(pkg["vision"].graph.initializers.keys())
        assert any("vision_tower" in n for n in vision_inits)
        assert any("multi_modal_projector" in n for n in vision_inits)

        # Decoder has language model params
        decoder_inits = list(pkg["decoder"].graph.initializers.keys())
        assert any("lm_head" in n for n in decoder_inits)

        # Embedding has embed_tokens
        embed_inits = list(pkg["embedding"].graph.initializers.keys())
        assert any("embed_tokens" in n for n in embed_inits)


class TestApplyWeightsLogging:
    """Tests for unmapped-weight warnings and DEBUG mapping logs."""

    def test_unmapped_weights_logged_as_info(self, caplog):
        """Weights not matching any ONNX initializer produce an INFO message."""
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        init_name = next(iter(model.graph.initializers.keys()))
        shape = list(model.graph.initializers[init_name].shape)

        state_dict = {
            init_name: torch.ones(shape),
            "unmapped.weight": torch.zeros(4, 4),
        }

        with caplog.at_level(logging.INFO, logger="mobius"):
            pkg.apply_weights(state_dict)

        assert "unmapped.weight" in caplog.text
        assert "(4, 4)" in caplog.text
        assert "1 weight(s) not applied" in caplog.text

    def test_all_weights_mapped_no_info_message(self, caplog):
        """When all weights are mapped, no unmapped INFO message is emitted."""
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        init_name = next(iter(model.graph.initializers.keys()))
        shape = list(model.graph.initializers[init_name].shape)

        state_dict = {init_name: torch.ones(shape)}

        with caplog.at_level(logging.INFO, logger="mobius"):
            pkg.apply_weights(state_dict)

        assert "not applied" not in caplog.text

    def test_debug_log_includes_applied_weights(self, caplog):
        """At DEBUG level, applied weights are logged."""
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        init_name = next(iter(model.graph.initializers.keys()))
        shape = list(model.graph.initializers[init_name].shape)

        state_dict = {init_name: torch.ones(shape)}

        with caplog.at_level(logging.DEBUG, logger="mobius"):
            pkg.apply_weights(state_dict)

        assert "Applied 1 of 1 weight(s)" in caplog.text
        assert init_name in caplog.text

    def test_empty_state_dict_no_info_message(self, caplog):
        """An empty state dict produces no unmapped INFO message."""
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)

        with caplog.at_level(logging.INFO, logger="mobius"):
            pkg.apply_weights({})

        assert "not applied" not in caplog.text
