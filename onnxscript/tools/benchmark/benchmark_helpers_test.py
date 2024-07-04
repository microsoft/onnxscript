# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import onnxscript.tools.benchmark.benchmark_helpers as bh


class BenchmarkHelperTest(unittest.TestCase):
    def test_make_configs(self):
        value = {
            "warmup": 5,
            "model": "llama,phi",
            "device": "cpu,cuda",
            "config": "medium",
            "dump_folder": "",
        }
        self.assertTrue(bh.multi_run(value))
        configs = bh.make_configs(value)
        expected = [
            {
                "warmup": 5,
                "model": "llama",
                "device": "cpu",
                "config": "medium",
                "dump_folder": "",
            },
            {
                "warmup": 5,
                "model": "llama",
                "device": "cuda",
                "config": "medium",
                "dump_folder": "",
            },
            {
                "warmup": 5,
                "model": "phi",
                "device": "cpu",
                "config": "medium",
                "dump_folder": "",
            },
            {
                "warmup": 5,
                "model": "phi",
                "device": "cuda",
                "config": "medium",
                "dump_folder": "",
            },
        ]
        self.assertEqual(expected, configs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
