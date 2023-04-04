# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

import click.testing

from onnxscript.__main__ import translate


class TestCli(unittest.TestCase):
    def test_translate(self):
        runner = click.testing.CliRunner()
        result = runner.invoke(translate, ["--help"])
        self.assertIn("Usage: translate [OPTIONS]", result.output)


if __name__ == "__main__":
    unittest.main()
