# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import time
import unittest

import numpy as np
import torch

import onnxscript.tools.memory_peak as mpeak


class TestMemoryPeak(unittest.TestCase):
    def test_memory(self):
        mem = mpeak.get_memory_rss(os.getpid())
        self.assertIsInstance(mem, int)

    def test_spy(self):
        p = mpeak.start_spying_on()
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
        p = mpeak.start_spying_on(cuda=True)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
