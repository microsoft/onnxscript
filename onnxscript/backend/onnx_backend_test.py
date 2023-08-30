# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import unittest

import onnxruntime as ort

from onnxscript.backend import onnx_backend


def load_function(obj):
    return ort.InferenceSession(obj.SerializeToString(), providers=("CPUExecutionProvider",))


def run_function(obj, *inputs):
    names = [i.name for i in obj.get_inputs()]
    if len(names) < len(inputs):
        raise AssertionError(f"Got {len(inputs)} inputs but expecting {len(names)}.")
    feeds = {names[i]: inputs[i] for i in range(len(inputs))}
    got = obj.run(None, feeds)
    return got


class TestOnnxBackEnd(unittest.TestCase):
    folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), "onnx_backend_test_code")

    def test_enumerate_onnx_tests(self):
        name = "test_abs"
        code = list(onnx_backend.enumerate_onnx_tests("node", lambda folder: folder == name))
        self.assertEqual(len(code), 1)

    def test_enumerate_onnx_tests_run_one(self):
        done = 0
        for backend_test in onnx_backend.enumerate_onnx_tests(
            "node", lambda folder: folder == "test_abs"
        ):
            self.assertIn(backend_test.name, repr(backend_test))
            self.assertGreater(len(backend_test), 0)
            backend_test.run(load_function, run_function)
            done += 1
        self.assertEqual(done, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
