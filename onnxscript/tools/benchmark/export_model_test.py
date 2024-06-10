# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import contextlib
import io
import unittest

import onnxscript.tools.benchmark.export_model
from onnxscript.tools.transformers_models import has_transformers


class BenchmarkTest(unittest.TestCase):
    @unittest.skipIf(not has_transformers(), reason="transformers missing")
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
            "--model",
            "phi",
        ]
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            onnxscript.tools.benchmark.export_model.main(args)

        out = f.getvalue()
        self.assertIn(":repeat_time,", out)

    @unittest.skipIf(not has_transformers(), reason="transformers missing")
    def test_export_model_llama_cpu_eager(self):
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
            "--model",
            "llama",
        ]
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            onnxscript.tools.benchmark.export_model.main(args)

        out = f.getvalue()
        self.assertIn(":repeat_time,", out)

    @unittest.skipIf(not has_transformers(), reason="transformers missing")
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
            "--model",
            "phi",
        ]
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            onnxscript.tools.benchmark.export_model.main(args)

        out = f.getvalue()
        self.assertIn(":repeat_time,", out)

    @unittest.skipIf(not has_transformers(), reason="transformers missing")
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
            "--model",
            "phi",
        ]
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            onnxscript.tools.benchmark.export_model.main(args)

        out = f.getvalue()
        self.assertIn(":repeat_time,", out)

    @unittest.skipIf(not has_transformers(), reason="transformers missing")
    def test_export_model_phi_cpu_dynamo_llama0(self):
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
            "--optimization",
            "rewrite,optimize,inline,llama0",
            "--model",
            "phi",
        ]
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            onnxscript.tools.benchmark.export_model.main(args)

        out = f.getvalue()
        self.assertIn(":repeat_time,", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
