# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CLI (``__main__.py``).

These tests invoke ``main()`` directly with argv lists, so they do not
require network access. All build tests use ``--no-weights``.
"""

from __future__ import annotations

import os
import tempfile

import onnx
import pytest

from mobius.__main__ import main


class TestCLIList:
    """Test the ``list`` subcommand."""

    def test_list_models(self, capsys):
        main(["list", "models"])
        out = capsys.readouterr().out
        assert "Supported model architectures" in out
        assert "llama" in out

    def test_list_tasks(self, capsys):
        main(["list", "tasks"])
        out = capsys.readouterr().out
        assert "Available tasks" in out
        assert "text-generation" in out

    def test_list_dtypes(self, capsys):
        main(["list", "dtypes"])
        out = capsys.readouterr().out
        assert "Available dtypes" in out
        assert "f32" in out


class TestCLIBuild:
    """Test the ``build`` subcommand with ``--no-weights``."""

    def test_build_no_weights_creates_model_onnx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            main(["build", "--model", "Qwen/Qwen2.5-0.5B", tmpdir, "--no-weights"])
            assert os.path.isfile(os.path.join(tmpdir, "model.onnx"))

    def test_build_with_dtype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--dtype",
                    "f16",
                ]
            )
            assert os.path.isfile(os.path.join(tmpdir, "model.onnx"))

    def test_build_encoder_decoder_produces_separate_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            main(["build", "--model", "facebook/bart-base", tmpdir, "--no-weights"])
            assert os.path.isfile(os.path.join(tmpdir, "encoder", "model.onnx"))
            assert os.path.isfile(os.path.join(tmpdir, "decoder", "model.onnx"))

    def test_build_missing_model_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(SystemExit):
            main(["build", tmpdir])  # no --model or --config

    def test_build_static_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--static-cache",
                ]
            )
            assert os.path.isfile(os.path.join(tmpdir, "model.onnx"))

    def test_max_seq_len_without_static_cache_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(SystemExit):
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--max-seq-len",
                    "512",
                ]
            )

    def test_static_cache_with_task_errors(self):
        """--static-cache cannot be combined with any --task."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(SystemExit):
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--static-cache",
                    "--task",
                    "text-generation",
                ]
            )

    def test_non_positive_max_seq_len_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(SystemExit):
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--static-cache",
                    "--max-seq-len",
                    "0",
                ]
            )

    def test_static_cache_with_max_seq_len(self):
        """--max-seq-len is passed through and sets cache dimensions."""
        max_seq_len = 256
        with tempfile.TemporaryDirectory() as tmpdir:
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--static-cache",
                    "--max-seq-len",
                    str(max_seq_len),
                ]
            )
            model_path = os.path.join(tmpdir, "model.onnx")
            assert os.path.isfile(model_path)

            # Verify the cache input has the expected max_seq_len
            # dimension. Static cache shape: [batch, max_seq_len, kv_hidden]
            model = onnx.load(model_path)
            cache_inputs = [
                inp for inp in model.graph.input if inp.name.startswith("key_cache.")
            ]
            assert len(cache_inputs) > 0, "No key_cache inputs found"
            seq_dim = cache_inputs[0].type.tensor_type.shape.dim[1].dim_value
            assert seq_dim == max_seq_len, (
                f"key_cache.0 seq dimension is {seq_dim}, expected {max_seq_len}"
            )


class TestCLIInfo:
    """Test the ``info`` subcommand."""

    def test_info_known_model(self, capsys):
        main(["info", "Qwen/Qwen2.5-0.5B"])
        out = capsys.readouterr().out
        assert "qwen2" in out
        assert "Supported" in out
