# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

import click.testing

from onnxscript.__main__ import onnx2script, translate


class TestCli(unittest.TestCase):
    def test_onnx2script(self):
        runner = click.testing.CliRunner()
        result = runner.invoke(onnx2script, ["--help"])
        self.assertIn("Usage: onnx2script [OPTIONS]", result.output)

    def test_translate(self):
        runner = click.testing.CliRunner()
        result = runner.invoke(translate, ["--help"])
        self.assertIn("Usage: translate [OPTIONS]", result.output)


if __name__ == "__main__":
    unittest.main()
