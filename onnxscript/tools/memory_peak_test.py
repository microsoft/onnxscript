# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import sys
import time
import unittest

import numpy as np
import torch

import onnxscript.tools.memory_peak

import multiprocessing

class TestMemoryPeak(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", reason="other test are failing")
    def test_memory(self):
        mem = onnxscript.tools.memory_peak.get_memory_rss(os.getpid())
        self.assertIsInstance(mem, int)

    @unittest.skipIf(sys.platform == "win32", reason="other test are failing")
    def test_spy(self):
        p = onnxscript.tools.memory_peak.start_spying_on()
        res = []
        for i in range(10):
            time.sleep(0.005)
            res.append(np.empty(i * 1000000))
        del res
        time.sleep(0.02)
        pres = p.stop()
        self.assertIsInstance(pres, dict)
        self.assertLessEqual(pres["cpu"][0].end, pres["cpu"][0].max_peak)
        self.assertLessEqual(pres["cpu"][0].begin, pres["cpu"][0].max_peak)
        self.assertIsInstance(pres["cpu"][0].to_dict(), dict)

    @unittest.skipIf(not torch.cuda.is_available(), reason="CUDA not here")
    def test_spy_cuda(self):
        p = onnxscript.tools.memory_peak.start_spying_on(cuda=True)
        res = []
        for i in range(10):
            time.sleep(0.005)
            res.append(np.empty(i * 1000000))
        del res
        time.sleep(0.02)
        pres = p.stop()
        self.assertIsInstance(pres, dict)
        self.assertIsInstance(pres["cpu"], list)
        self.assertEqual(len(pres["cpu"]), 1)
        self.assertIsInstance(pres["gpus"], list)
        self.assertLessEqual(pres["cpu"][0].end, pres["cpu"][0].max_peak)
        self.assertLessEqual(pres["cpu"][0].begin, pres["cpu"][0].max_peak)
        self.assertIn("gpus", pres)
        self.assertLessEqual(pres["gpus"][0].end, pres["gpus"][0].max_peak)
        self.assertLessEqual(pres["gpus"][0].begin, pres["gpus"][0].max_peak)


    def test_flatten_with_gpus_precise(self):
        cpu_monitor = onnxscript.tools.memory_peak.Monitor()
        gpu_monitor = onnxscript.tools.memory_peak.Monitor()
        cpu_monitor.update(1000)
        gpu_monitor.update(2000)
        stats = {'cpu': [cpu_monitor], 'gpus': [gpu_monitor]}
        flat_stats = onnxscript.tools.memory_peak.flatten(stats)
        self.assertIn('gpu0_peak', flat_stats)
        self.assertAlmostEqual(flat_stats['gpu0_peak'], 0.0019073486328125)


    def test_memory_spy_start_handshake_failure(self):
        original_pipe = multiprocessing.Pipe
        def mock_pipe():
            parent_conn, child_conn = original_pipe()
            parent_conn.recv = lambda: -1  # Simulate handshake failure
            return parent_conn, child_conn
        multiprocessing.Pipe = mock_pipe
        with self.assertRaises(RuntimeError):
            onnxscript.tools.memory_peak.MemorySpy(os.getpid())
        multiprocessing.Pipe = original_pipe


if __name__ == "__main__":
    unittest.main(verbosity=2)
