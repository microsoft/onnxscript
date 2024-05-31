# SPDX-License-Identifier: Apache-2.0
# pylint: disable=import-outside-toplevel

import contextlib
import io
import unittest

import onnxscript.testing.benchmark.export_model


class BenchmarkTest(unittest.TestCase):

    def test_export_model_phi_cpu_eager(self):
        args = [
            "--verbose",
            "1",
            "--config",
            "medium",
            "--dtype",
            "float32",
            "--device",
            "cpu",
            "--exporter",
            "eager",
        ]
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            onnxscript.testing.benchmark.export_model.main(args)

        out = f.getvalue()
        self.assertIn(":repeat_time,", out)

    def test_export_model_phi_cpu_dynamo(self):
        args = [
            "--verbose",
            "1",
            "--config",
            "medium",
            "--dtype",
            "float32",
            "--device",
            "cpu",
            "--exporter",
            "dynamo",
        ]
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            onnxscript.testing.benchmark.export_model.main(args)

        out = f.getvalue()
        self.assertIn(":repeat_time,", out)

    def test_export_model_phi_cpu_script(self):
        args = [
            "--verbose",
            "1",
            "--config",
            "medium",
            "--dtype",
            "float32",
            "--device",
            "cpu",
            "--exporter",
            "script",
        ]
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            onnxscript.testing.benchmark.export_model.main(args)

        out = f.getvalue()
        self.assertIn(":repeat_time,", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
