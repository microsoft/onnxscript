# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnxscript.test.models import signal_dft
from onnxscript.test.functions.onnx_script_test_case import (
    OnnxScriptTestCase, FunctionTestParams)


class TestOnnxSignal(OnnxScriptTestCase):

    @staticmethod
    def _fft(x, fft_length, axis=-1):
        ft = np.fft.fft(x, fft_length[0], axis=axis)
        r = np.real(ft)
        i = np.imag(ft)
        merged = np.vstack([r[np.newaxis, ...], i[np.newaxis, ...]])
        perm = np.arange(len(merged.shape))
        perm[:-1] = perm[1:]
        perm[-1] = 0
        tr = np.transpose(merged, list(perm))
        if tr.shape[-1] != 2:
            raise AssertionError(f"Unexpected shape {tr.shape}, x.shape={x.shape} "
                                 f"fft_length={fft_length}.")
        return tr

    @staticmethod
    def _cifft(x, fft_length, axis=-1):
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[slices]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[slices]
        c = np.squeeze(real + 1j * imag, -1)
        return TestOnnxSignal._ifft(c, fft_length, axis=axis)

    @staticmethod
    def _ifft(x, fft_length, axis=-1):
        ft = np.fft.ifft(x, fft_length[0], axis=axis)
        r = np.real(ft)
        i = np.imag(ft)
        merged = np.vstack([r[np.newaxis, ...], i[np.newaxis, ...]])
        perm = np.arange(len(merged.shape))
        perm[:-1] = perm[1:]
        perm[-1] = 0
        tr = np.transpose(merged, list(perm))
        if tr.shape[-1] != 2:
            raise AssertionError(f"Unexpected shape {tr.shape}, x.shape={x.shape} "
                                 f"fft_length={fft_length}.")
        return tr

    @staticmethod
    def _cfft(x, fft_length, axis=-1):
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[slices]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[slices]
        c = np.squeeze(real + 1j * imag, -1)
        return TestOnnxSignal._fft(c, fft_length, axis=axis)

    @staticmethod
    def _complex2float(c):
        real = np.real(c)
        imag = np.imag(c)
        x = np.vstack([real[np.newaxis, ...], imag[np.newaxis, ...]])
        perm = list(range(len(x.shape)))
        perm[:-1] = perm[1:]
        perm[-1] = 0
        return np.transpose(x, perm)

    def test_dft_rfft_last_axis(self):

        xs = [np.arange(5).astype(np.float32),
              np.arange(5).astype(np.float32).reshape((1, -1)),
              np.arange(30).astype(np.float32).reshape((2, 3, -1)),
              np.arange(60).astype(np.float32).reshape((2, 3, 2, -1))]

        for onesided in [False, True]:
            for x_ in xs:
                x = x_[..., np.newaxis]
                for s in [4, 5, 6]:
                    le = np.array([s], dtype=np.int64)
                    expected = self._fft(x_, le)
                    if onesided:
                        slices = [slice(0, a) for a in expected.shape]
                        slices[-2] = slice(0, (expected.shape[-2] + 1) // 2)
                        expected = expected[slices]
                    with self.subTest(x_shape=x.shape, le=list(le),
                                      expected_shape=expected.shape,
                                      onesided=onesided):
                        if onesided:
                            case = FunctionTestParams(
                                signal_dft.dft_last_axis, [x, le, True], [expected])
                        else:
                            case = FunctionTestParams(
                                signal_dft.dft_last_axis, [x, le], [expected])
                        self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    def test_dft_cfft_last_axis(self):

        xs = [np.arange(5).astype(np.float32),
              np.arange(5).astype(np.float32).reshape((1, -1)),
              np.arange(30).astype(np.float32).reshape((2, 3, -1)),
              np.arange(60).astype(np.float32).reshape((2, 3, 2, -1))]
        ys = [np.arange(5).astype(np.float32) / 10,
              np.arange(5).astype(np.float32).reshape((1, -1)) / 10,
              np.arange(30).astype(np.float32).reshape((2, 3, -1)) / 10,
              np.arange(60).astype(np.float32).reshape((2, 3, 2, -1)) / 10]
        cs = [x + 1j * y for x, y in zip(xs, ys)]

        for c in cs:
            x = self._complex2float(c)
            for s in [4, 5, 6]:
                le = np.array([s], dtype=np.int64)
                expected1 = self._fft(c, le)
                expected2 = self._cfft(x, le)
                assert_almost_equal(expected1, expected2)
                with self.subTest(c_shape=c.shape, le=list(le),
                                  expected_shape=expected1.shape):
                    case = FunctionTestParams(
                        signal_dft.dft_last_axis, [x, le, False], [expected1])
                    self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    def test_dft_rfft(self):

        xs = [np.arange(5).astype(np.float32),
              np.arange(10).astype(np.float32).reshape((2, -1)),
              np.arange(30).astype(np.float32).reshape((2, 3, -1)),
              np.arange(60).astype(np.float32).reshape((2, 3, 2, -1))]

        for x_ in xs:
            x = x_[..., np.newaxis]
            for s in [4, 5, 6]:
                le = np.array([s], dtype=np.int64)
                for ax in range(len(x_.shape)):
                    expected = self._fft(x_, le, axis=ax)
                    nax = np.array([ax], dtype=np.int64)
                    with self.subTest(x_shape=x.shape, le=list(le), ax=ax,
                                      expected_shape=expected.shape):
                        case = FunctionTestParams(
                            signal_dft.dft, [x, le, nax], [expected])
                        self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    def test_dft_cfft(self):

        xs = [np.arange(5).astype(np.float32),
              np.arange(5).astype(np.float32).reshape((1, -1)),
              np.arange(30).astype(np.float32).reshape((2, 3, -1)),
              np.arange(60).astype(np.float32).reshape((2, 3, 2, -1))]
        ys = [np.arange(5).astype(np.float32) / 10,
              np.arange(5).astype(np.float32).reshape((1, -1)) / 10,
              np.arange(30).astype(np.float32).reshape((2, 3, -1)) / 10,
              np.arange(60).astype(np.float32).reshape((2, 3, 2, -1)) / 10]
        cs = [x + 1j * y for x, y in zip(xs, ys)]

        for c in cs:
            x = self._complex2float(c)
            for s in [4, 5, 6]:
                le = np.array([s], dtype=np.int64)
                for ax in range(len(c.shape)):
                    nax = np.array([ax], dtype=np.int64)
                    expected1 = self._fft(c, le, axis=ax)
                    expected2 = self._cfft(x, le, axis=ax)
                    assert_almost_equal(expected1, expected2)
                    with self.subTest(c_shape=c.shape, le=list(le), ax=ax,
                                      expected_shape=expected1.shape):
                        case = FunctionTestParams(
                            signal_dft.dft, [x, le, nax, False], [expected1])
                        self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    def test_dft_rifft(self):

        xs = [np.arange(5).astype(np.float32),
              np.arange(10).astype(np.float32).reshape((2, -1)),
              np.arange(30).astype(np.float32).reshape((2, 3, -1)),
              np.arange(60).astype(np.float32).reshape((2, 3, 2, -1))]

        for x_ in xs:
            x = x_[..., np.newaxis]
            for s in [4, 5, 6]:
                le = np.array([s], dtype=np.int64)
                for ax in range(len(x_.shape)):
                    expected = self._ifft(x_, le, axis=ax)
                    nax = np.array([ax], dtype=np.int64)
                    with self.subTest(x_shape=x.shape, le=list(le), ax=ax,
                                      expected_shape=expected.shape):
                        case = FunctionTestParams(
                            signal_dft.idft, [x, le, nax], [expected])
                        self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    def test_dft_cifft(self):

        xs = [np.arange(5).astype(np.float32),
              np.arange(5).astype(np.float32).reshape((1, -1)),
              np.arange(30).astype(np.float32).reshape((2, 3, -1)),
              np.arange(60).astype(np.float32).reshape((2, 3, 2, -1))]
        ys = [np.arange(5).astype(np.float32) / 10,
              np.arange(5).astype(np.float32).reshape((1, -1)) / 10,
              np.arange(30).astype(np.float32).reshape((2, 3, -1)) / 10,
              np.arange(60).astype(np.float32).reshape((2, 3, 2, -1)) / 10]
        cs = [x + 1j * y for x, y in zip(xs, ys)]

        for c in cs:
            x = self._complex2float(c)
            for s in [4, 5, 6]:
                le = np.array([s], dtype=np.int64)
                for ax in range(len(c.shape)):
                    nax = np.array([ax], dtype=np.int64)
                    expected1 = self._ifft(c, le, axis=ax)
                    expected2 = self._cifft(x, le, axis=ax)
                    assert_almost_equal(expected1, expected2)
                    with self.subTest(c_shape=c.shape, le=list(le), ax=ax,
                                      expected_shape=expected1.shape):
                        case = FunctionTestParams(
                            signal_dft.idft, [x, le, nax, False], [expected1])
                        self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    def test_hann_window(self):
        le = np.array([5], dtype=np.int64)
        expected = (np.sin((np.arange(5) * np.pi) / 4) ** 2).astype(np.float32)
        case = FunctionTestParams(signal_dft.hann_window, [le], [expected])
        self.run_eager_test(case, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    # TestOnnxSignal().test_hann_window()
    unittest.main()
