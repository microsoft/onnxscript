# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
import onnx
from onnxscript.test.models import signal_dft
from onnxscript.test.functions.onnx_script_test_case import (
    OnnxScriptTestCase, FunctionTestParams)


class TestOnnxSignal(OnnxScriptTestCase):

    @classmethod
    def setUpClass(cls):
        OnnxScriptTestCase.setUpClass()
        cls.rtol = 1e-4

    @staticmethod
    def _fft(x, fft_length):
        ft = np.fft.fft(x, fft_length[0])
        r = np.real(ft)
        i = np.imag(ft)
        merged = np.vstack([r[np.newaxis, ...], i[np.newaxis, ...]])
        perm = np.arange(len(merged.shape))
        perm[:-1] = perm[1:]
        perm[-1] = 0
        print(r.shape, i.shape, merged.shape, perm)
        return np.transpose(merged, list(perm))

    def test_dft_rfft(self):

        # dim 1
        x = np.arange(5).astype(np.float32).reshape((1, -1))

        cases = []
        for s in [4, 5, 6]:
            le = np.array([s], dtype=np.int64)
            with self.subTest(x_shape=x.shape, le=list(le)):
                case = FunctionTestParams(signal_dft.dft, [x, le], [self._fft(x, le)])
                self.run_eager_test(case)


if __name__ == '__main__':
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    unittest.main()
