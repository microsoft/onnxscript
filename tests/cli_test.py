# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CLI (``__main__.py``).

These tests invoke ``main()`` directly with argv lists, so they do not
require network access. All build tests use ``--no-weights``.
"""

from __future__ import annotations

import os
import tempfile

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


class TestCLIInfo:
    """Test the ``info`` subcommand."""

    def test_info_known_model(self, capsys):
        main(["info", "Qwen/Qwen2.5-0.5B"])
        out = capsys.readouterr().out
        assert "qwen2" in out
        assert "Supported" in out
