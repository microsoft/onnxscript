# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import onnxscript.tools.benchmark.benchmark_helpers as bh

import torch
import onnx
import unittest.mock
import numpy as np
import sys

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

    def test_run_benchmark_high_verbose(self):
        script_name = "example_script"
        configs = [{"arg1": "value1"}]
        with unittest.mock.patch('subprocess.Popen') as mock_popen:
            process_mock = unittest.mock.Mock()
            attrs = {'communicate.return_value': (b"output", b"")}
            process_mock.configure_mock(**attrs)
            mock_popen.return_value = process_mock
            result = bh.run_benchmark(script_name, configs, verbose=10, stop_if_exception=False)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertIn("ERROR", result[0])
            self.assertIn("OUTPUT", result[0])


    def test_common_export_valid_exporter_with_optimization(self):
        import torch
        import onnx
        model = torch.nn.Linear(2, 2)
        inputs = (torch.randn(1, 2),)
        onnx_model = bh.common_export(model, inputs, exporter="script", optimization="optimize")
        self.assertIsInstance(onnx_model, onnx.ModelProto)


    def test_run_benchmark_with_onnxruntime_error_no_exception(self):
        script_name = "example_script"
        configs = [{"arg1": "value1"}]
        with unittest.mock.patch('subprocess.Popen') as mock_popen:
            process_mock = unittest.mock.Mock()
            attrs = {'communicate.return_value': (b"", b"ONNXRuntimeError")}
            process_mock.configure_mock(**attrs)
            mock_popen.return_value = process_mock
            result = bh.run_benchmark(script_name, configs, stop_if_exception=False)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertIn("ERROR", result[0])
            self.assertIn("OUTPUT", result[0])


    def test_optimize_model_proto_unknown_step(self):
        import onnx
        model_proto = onnx.ModelProto()
        with self.assertRaises(AssertionError):
            bh.optimize_model_proto(model_proto, optimization="unknown_step")


    def test_apply_rule_sets_valid_rule_set(self):
        import onnx
        model_proto = onnx.ModelProto()
        rule_sets = ["llama0"]
        result = bh.apply_rule_sets(model_proto, rule_sets)
        self.assertIsInstance(result, onnx.ModelProto)


    def test_run_benchmark_no_metrics_no_exception(self):
        script_name = "example_script"
        configs = [{"arg1": "value1"}]
        result = bh.run_benchmark(script_name, configs, stop_if_exception=False)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn("ERROR", result[0])
        self.assertIn("OUTPUT", result[0])


    def test_common_export_unknown_exporter(self):
        model = None
        inputs = []
        with self.assertRaises(ValueError):
            bh.common_export(model, inputs, exporter="unknown")


    def test_measure_discrepancies_shape_mismatch(self):
        expected = [(np.array([1, 2, 3]),)]
        outputs = [(np.array([1, 2]),)]
        with self.assertRaises(AssertionError):
            bh.measure_discrepancies(expected, outputs)


    def test_run_benchmark_with_onnxruntime_error(self):
        script_name = "example_script"
        configs = [{"arg1": "value1"}]
        with self.assertRaises(RuntimeError):
            bh.run_benchmark(script_name, configs, stop_if_exception=True)


    def test_make_prefix(self):
        script_name = "example_script.py"
        index = 3
        expected = "example_script_dort_c3_"
        result = bh._make_prefix(script_name, index)
        self.assertEqual(result, expected)


    def test_extract_metrics_with_no_metrics(self):
        text = "No metrics here"
        expected = {}
        result = bh._extract_metrics(text)
        self.assertEqual(result, expected)


    def test_get_machine_with_cuda(self):
        import torch
        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_capability = lambda x: (7, 5)
        torch.cuda.get_device_name = lambda x: "Mock CUDA Device"
        
        result = bh.get_machine()
        self.assertIn("has_cuda", result)
        self.assertTrue(result["has_cuda"])
        self.assertEqual(result["capability"], (7, 5))
        self.assertEqual(result["device_name"], "Mock CUDA Device")


    def test_get_parsed_args_with_custom_args(self):
        name = "test_script"
        new_args = ["--n_trees", "20", "--learning_rate", "0.05"]
        kwargs = {"n_trees": (10, "number of trees to train"), "learning_rate": (0.01, "learning rate")}
        expected = {"n_trees": 20, "learning_rate": 0.05}
        result = bh.get_parsed_args(name, new_args=new_args, **kwargs)
        self.assertEqual(result, expected)


    def test_cmd_line_with_kwargs(self):
        script_name = "example_script"
        kwargs = {"arg1": "value1", "arg2": "value2"}
        expected = [sys.executable, "-m", "example_script", "--arg1", "value1", "--arg2", "value2"]
        result = bh._cmd_line(script_name, **kwargs)
        self.assertEqual(result, expected)



if __name__ == "__main__":
    unittest.main(verbosity=2)
