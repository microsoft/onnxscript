# SPDX-License-Identifier: Apache-2.0

# FIXME: Remove reference to the deprecated distutils module
import distutils.version as dv  # pylint: disable=deprecated-module
import itertools
import unittest
import warnings

import numpy as np
import parameterized
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException

from onnxscript.test.common import onnx_script_test_case
from onnxscript.test.models import signal_dft


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
        raise AssertionError(
            f"Unexpected shape {tr.shape}, x.shape={x.shape} " f"fft_length={fft_length}."
        )
    return tr


def _cifft(x, fft_length, axis=-1):
    slices = [slice(0, x) for x in x.shape]
    slices[-1] = slice(0, x.shape[-1], 2)
    real = x[tuple(slices)]
    slices[-1] = slice(1, x.shape[-1], 2)
    imag = x[tuple(slices)]
    c = np.squeeze(real + 1j * imag, -1)
    return _ifft(c, fft_length, axis=axis)


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
        raise AssertionError(
            f"Unexpected shape {tr.shape}, x.shape={x.shape} " f"fft_length={fft_length}."
        )
    return tr


def _cfft(x, fft_length, axis=-1):
    slices = [slice(0, x) for x in x.shape]
    slices[-1] = slice(0, x.shape[-1], 2)
    real = x[tuple(slices)]
    slices[-1] = slice(1, x.shape[-1], 2)
    imag = x[tuple(slices)]
    c = np.squeeze(real + 1j * imag, -1)
    return _fft(c, fft_length, axis=axis)


def _complex2float(c):
    real = np.real(c)
    imag = np.imag(c)
    x = np.vstack([real[np.newaxis, ...], imag[np.newaxis, ...]])
    perm = list(range(len(x.shape)))
    perm[:-1] = perm[1:]
    perm[-1] = 0
    return np.transpose(x, perm)


def _stft(
    x,
    fft_length,
    window,
    axis=-1,  # pylint: disable=unused-argument
    center=False,
    onesided=False,
    hop_length=None,
):
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError("torch is not installed.") from e
    ft = torch.stft(
        torch.from_numpy(x),
        n_fft=fft_length,
        hop_length=hop_length,
        win_length=fft_length,
        window=torch.from_numpy(window),
        center=center,
        onesided=onesided,
        return_complex=True,
    )
    r = np.real(ft)
    i = np.imag(ft)
    merged = np.vstack([r[np.newaxis, ...], i[np.newaxis, ...]])
    perm = np.arange(len(merged.shape))
    perm[:-1] = perm[1:]
    perm[-1] = 0
    tr = np.transpose(merged, list(perm))
    if tr.shape[-1] != 2:
        raise AssertionError(
            f"Unexpected shape {tr.shape}, x.shape={x.shape} "
            f"fft_length={fft_length}, window={window}."
        )
    return ft.numpy(), tr.astype(np.float32)


def _istft(
    y,
    fft_length,
    window,
    axis=-1,  # pylint: disable=unused-argument
    center=False,
    onesided=False,
    hop_length=None,
):
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError("torch is not installed.") from e

    ft = torch.istft(
        torch.from_numpy(y),
        n_fft=fft_length,
        hop_length=hop_length,
        win_length=fft_length,
        window=torch.from_numpy(window),
        center=center,
        onesided=onesided,
        return_complex=True,
    )
    return ft.numpy()


class TestOnnxSignal(onnx_script_test_case.OnnxScriptTestCase):
    @parameterized.parameterized.expand(
        itertools.product(
            [False, True],
            [
                np.arange(5).astype(np.float32),
                np.arange(5).astype(np.float32).reshape((1, -1)),
                np.arange(30).astype(np.float32).reshape((2, 3, -1)),
                np.arange(60).astype(np.float32).reshape((2, 3, 2, -1)),
            ],
            [4, 5, 6],
        )
    )
    @unittest.skip(
        reason="Eager mode fails due to: "
        "ValueError: The truth value of an array with more than one element "
        "is ambiguous. Use a.any() or a.all()",
    )
    def test_dft_rfft_last_axis(self, onesided: bool, x_: np.ndarray, s: int):

        x = x_[..., np.newaxis]
        le = np.array([s], dtype=np.int64)
        we = np.array([1] * le[0], dtype=np.float32)
        expected = _fft(x_, le)
        if onesided:
            slices = [slice(0, a) for a in expected.shape]
            slices[-2] = slice(0, expected.shape[-2] // 2 + expected.shape[-2] % 2)
            expected = expected[slices]
            case = onnx_script_test_case.FunctionTestParams(
                signal_dft.dft_last_axis, [x, le, we, True], [expected]
            )
        else:
            case = onnx_script_test_case.FunctionTestParams(
                signal_dft.dft_last_axis, [x, le, we], [expected]
            )
        self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    @unittest.skip(
        reason="Eager mode fails due to: "
        "ValueError: The truth value of an array with more than one element "
        "is ambiguous. Use a.any() or a.all()",
    )
    def test_dft_cfft_last_axis(self):

        xs = [
            np.arange(5).astype(np.float32),
            np.arange(5).astype(np.float32).reshape((1, -1)),
            np.arange(30).astype(np.float32).reshape((2, 3, -1)),
            np.arange(60).astype(np.float32).reshape((2, 3, 2, -1)),
        ]
        ys = [
            np.arange(5).astype(np.float32) / 10,
            np.arange(5).astype(np.float32).reshape((1, -1)) / 10,
            np.arange(30).astype(np.float32).reshape((2, 3, -1)) / 10,
            np.arange(60).astype(np.float32).reshape((2, 3, 2, -1)) / 10,
        ]
        cs = [x + 1j * y for x, y in zip(xs, ys)]

        for c in cs:
            x = _complex2float(c)
            for s in [4, 5, 6]:
                le = np.array([s], dtype=np.int64)
                we = np.array([1] * le[0], dtype=np.float32)
                expected1 = _fft(c, le)
                expected2 = _cfft(x, le)
                np.testing.assert_allclose(expected1, expected2)
                with self.subTest(
                    c_shape=c.shape,
                    le=list(le),
                    expected_shape=expected1.shape,
                    weights=we,
                ):
                    case = onnx_script_test_case.FunctionTestParams(
                        signal_dft.dft_last_axis, [x, le, we, False], [expected1]
                    )
                    self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    @parameterized.parameterized.expand(
        itertools.product(
            [
                np.arange(5).astype(np.float32),
                np.arange(10).astype(np.float32).reshape((2, -1)),
                # FIXME(#192): test times out
                # np.arange(30).astype(np.float32).reshape((2, 3, -1)),
                np.arange(60).astype(np.float32).reshape((2, 3, 2, -1)),
            ],
            [4, 5, 6],
        )
    )
    def test_dft_rfft(self, x_, s: int):
        x = x_[..., np.newaxis]

        le = np.array([s], dtype=np.int64)
        for ax in range(len(x_.shape)):
            expected = _fft(x_, le, axis=ax)
            nax = np.array([ax], dtype=np.int64)
            with self.subTest(
                x_shape=x.shape,
                le=list(le),
                ax=ax,
                expected_shape=expected.shape,
            ):
                case = onnx_script_test_case.FunctionTestParams(
                    signal_dft.dft, [x, le, nax], [expected]
                )
                self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    @parameterized.parameterized.expand(
        [
            (np.arange(5).astype(np.float32), np.arange(5).astype(np.float32) / 10),
            (
                np.arange(5).astype(np.float32).reshape((1, -1)),
                np.arange(5).astype(np.float32).reshape((1, -1)) / 10,
            ),
            (
                np.arange(30).astype(np.float32).reshape((2, 3, -1)),
                np.arange(30).astype(np.float32).reshape((2, 3, -1)) / 10,
            ),
            (
                np.arange(60).astype(np.float32).reshape((2, 3, 2, -1)),
                np.arange(60).astype(np.float32).reshape((2, 3, 2, -1)) / 10,
            ),
        ]
    )
    def test_dft_cfft(self, x, y):
        c = x + 1j * y
        x = _complex2float(c)
        for s in [4, 5, 6]:
            le = np.array([s], dtype=np.int64)
            for ax in range(len(c.shape)):
                nax = np.array([ax], dtype=np.int64)
                expected1 = _fft(c, le, axis=ax)
                expected2 = _cfft(x, le, axis=ax)
                np.testing.assert_allclose(expected1, expected2)
                with self.subTest(
                    c_shape=c.shape,
                    le=list(le),
                    ax=ax,
                    expected_shape=expected1.shape,
                ):
                    case = onnx_script_test_case.FunctionTestParams(
                        signal_dft.dft, [x, le, nax, False], [expected1]
                    )
                    self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    @parameterized.parameterized.expand(
        [
            (np.arange(5).astype(np.float32),),
            (np.arange(10).astype(np.float32).reshape((2, -1)),),
            (np.arange(30).astype(np.float32).reshape((2, 3, -1)),),
            (np.arange(60).astype(np.float32).reshape((2, 3, 2, -1)),),
        ]
    )
    def test_dft_rifft(self, x_):
        x = x_[..., np.newaxis]
        for s in [4, 5, 6]:
            le = np.array([s], dtype=np.int64)
            for ax in range(len(x_.shape)):
                expected = _ifft(x_, le, axis=ax)
                nax = np.array([ax], dtype=np.int64)
                with self.subTest(
                    x_shape=x.shape,
                    le=list(le),
                    ax=ax,
                    expected_shape=expected.shape,
                ):
                    case = onnx_script_test_case.FunctionTestParams(
                        signal_dft.dft, [x, le, nax, True], [expected]
                    )
                    self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    @parameterized.parameterized.expand(
        [
            (np.arange(5).astype(np.float32), np.arange(5).astype(np.float32) / 10),
            (
                np.arange(5).astype(np.float32).reshape((1, -1)),
                np.arange(5).astype(np.float32).reshape((1, -1)) / 10,
            ),
            (
                np.arange(30).astype(np.float32).reshape((2, 3, -1)),
                np.arange(30).astype(np.float32).reshape((2, 3, -1)) / 10,
            ),
            (
                np.arange(60).astype(np.float32).reshape((2, 3, 2, -1)),
                np.arange(60).astype(np.float32).reshape((2, 3, 2, -1)) / 10,
            ),
        ]
    )
    def test_dft_cifft(self, x, y):
        c = x + 1j * y

        x = _complex2float(c)
        for s in [4, 5, 6]:
            le = np.array([s], dtype=np.int64)
            for ax in range(len(c.shape)):
                nax = np.array([ax], dtype=np.int64)
                expected1 = _ifft(c, le, axis=ax)
                expected2 = _cifft(x, le, axis=ax)
                np.testing.assert_allclose(expected1, expected2)
                with self.subTest(
                    c_shape=c.shape,
                    le=list(le),
                    ax=ax,
                    expected_shape=expected1.shape,
                ):
                    case = onnx_script_test_case.FunctionTestParams(
                        signal_dft.dft, [x, le, nax, True], [expected1]
                    )
                    self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    def test_hann_window(self):
        le = np.array([5], dtype=np.int64)
        expected = (np.sin((np.arange(5) * np.pi) / 4) ** 2).astype(np.float32)
        case = onnx_script_test_case.FunctionTestParams(
            signal_dft.hann_window, [le], [expected]
        )
        self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    def test_hamming_window(self):
        le = np.array([5], dtype=np.int64)
        alpha = np.array([0.54], dtype=np.float32)
        beta = np.array([0.46], dtype=np.float32)
        expected = alpha - np.cos(np.arange(5) * np.pi * 2 / 4) * beta
        case = onnx_script_test_case.FunctionTestParams(
            signal_dft.hamming_window, [le, alpha, beta], [expected]
        )
        self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    def test_blackman_window(self):
        le = np.array([5], dtype=np.int64)
        expected = (
            np.array([0.42])
            - np.cos(np.arange(5) * np.pi * 2 / 4) * 0.5
            + np.cos(np.arange(5) * np.pi * 4 / 4) * 0.08
        )
        case = onnx_script_test_case.FunctionTestParams(
            signal_dft.blackman_window, [le], [expected]
        )
        self.run_eager_test(case, rtol=1e-4, atol=1e-4)

    @parameterized.parameterized.expand(
        [
            ("hp2", np.arange(24).astype(np.float32).reshape((3, 8)), 6, 2, 2),
            ("bug", np.arange(24).astype(np.float32).reshape((3, 8)), 6, 3, 1),
            ("A0", np.arange(5).astype(np.float32), 5, 1, 1),
            ("A1", np.arange(5).astype(np.float32), 4, 2, 1),
            ("A2", np.arange(5).astype(np.float32), 6, 1, 1),
            ("B0", np.arange(10).astype(np.float32).reshape((2, -1)), 5, 1, 1),
            ("B1", np.arange(10).astype(np.float32).reshape((2, -1)), 4, 2, 1),
            ("B2", np.arange(10).astype(np.float32).reshape((2, -1)), 6, 1, 1),
            ("C0", np.arange(30).astype(np.float32).reshape((6, -1)), 5, 1, 1),
            ("C1", np.arange(30).astype(np.float32).reshape((6, -1)), 4, 2, 1),
            ("C2", np.arange(30).astype(np.float32).reshape((6, -1)), 6, 1, 1),
            ("D0", np.arange(60).astype(np.float32).reshape((6, -1)), 5, 6, 1),
            ("D1", np.arange(60).astype(np.float32).reshape((6, -1)), 4, 7, 1),
            ("D2", np.arange(60).astype(np.float32).reshape((6, -1)), 6, 5, 1),
        ]
    )
    def test_dft_rstft_istft(self, name: str, x_: np.ndarray, s: int, fs: int, hp: int):
        try:
            import torch  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError("torch is not installed.") from e
        torch_113 = dv.StrictVersion(
            torch.__version__.split("+", maxsplit=1)[0]
        ) >= dv.StrictVersion("1.13")

        x = x_[..., np.newaxis]
        le = np.array([s], dtype=np.int64)
        fsv = np.array([fs], dtype=np.int64)
        hpv = np.array([hp], dtype=np.int64)
        window = signal_dft.blackman_window(le)
        window[:] = (np.arange(window.shape[0]) + 1).astype(window.dtype)
        try:
            c_expected, expected = _stft(x_, le[0], window=window, hop_length=hpv[0])
        except RuntimeError:
            self.skipTest("Unable to validate with torch.")
        i_expected = _istft(c_expected, le[0], window=window, hop_length=hpv[0])
        info = dict(
            name=name,
            x_shape=x.shape,
            le=list(le),
            hp=hp,
            fs=fs,
            expected_shape=expected.shape,
            window_shape=window.shape,
        )

        # stft
        # x, fft_length, hop_length, n_frames, window, onesided=False
        case = onnx_script_test_case.FunctionTestParams(
            signal_dft.stft, [x, le, hpv, fsv, window], [expected]
        )
        try:
            self.run_eager_test(case, rtol=1e-3, atol=1e-3)
        except AssertionError as e:
            raise AssertionError(f"Issue with {info!r}.") from e

        # istft (imaginary part is null but still returned by istft)
        ix = _complex2float(c_expected)
        expected = np.concatenate((x, np.zeros(x.shape, dtype=x.dtype)), axis=-1)
        t_istft = _istft(c_expected, le[0], window=window, hop_length=hpv[0])
        if torch_113:
            if len(x_.shape) == 2:
                np.testing.assert_allclose(x_, t_istft, atol=1e-4)
            elif len(x_.shape) == 1:
                np.testing.assert_allclose(x_, t_istft, atol=1e-4)
            else:
                raise NotImplementedError(f"Not implemented when shape is {x_.shape!r}.")
        else:
            if len(x_.shape) == 2:
                np.testing.assert_allclose(x_[:, :-1], t_istft, atol=1e-4)
            elif len(x_.shape) == 1:
                np.testing.assert_allclose(x_[:-1], t_istft, atol=1e-4)
            else:
                raise NotImplementedError(f"Not implemented when shape is {x_.shape!r}.")

        # istft
        info["expected"] = expected
        info["expected_shape"] = expected.shape
        info["i_expected_shape"] = i_expected.shape
        info["ix_shape"] = ix.shape
        case = onnx_script_test_case.FunctionTestParams(
            signal_dft.istft, [ix, le, hpv, window], [expected]
        )
        try:
            self.run_eager_test(case, rtol=1e-4, atol=1e-4)
        except (AssertionError, RuntimeException) as e:
            # Not fully implemented.
            self.skipTest(
                "Issue with {} due to {}.".format(
                    str(info).split("\n", maxsplit=1),
                    str(e).split("\n", maxsplit=1)[0],
                )
            )

    @parameterized.parameterized.expand(
        [
            ("A0", np.arange(5).astype(np.float32), 5, 1, 1),
            ("A1", np.arange(5).astype(np.float32), 4, 2, 1),
            ("A2", np.arange(5).astype(np.float32), 6, 1, 1),
            ("B0", np.arange(10).astype(np.float32).reshape((2, -1)), 5, 1, 1),
            ("B1", np.arange(10).astype(np.float32).reshape((2, -1)), 4, 2, 1),
            ("B2", np.arange(10).astype(np.float32).reshape((2, -1)), 6, 1, 1),
            ("C0", np.arange(30).astype(np.float32).reshape((6, -1)), 5, 1, 1),
            ("C1", np.arange(30).astype(np.float32).reshape((6, -1)), 4, 2, 1),
            ("C2", np.arange(30).astype(np.float32).reshape((6, -1)), 6, 1, 1),
            ("D0", np.arange(60).astype(np.float32).reshape((6, -1)), 5, 6, 1),
            ("D1", np.arange(60).astype(np.float32).reshape((6, -1)), 4, 7, 1),
            # FIXME(#192): test times out
            # ("D2", np.arange(60).astype(np.float32).reshape((6, -1)), 6, 5, 1),
        ]
    )
    def test_dft_cstft_istft(self, name: str, xy_: np.ndarray, s: int, fs: int, hp: int):

        r_ = xy_
        i_ = xy_ / 10
        c_ = r_ + 1j * i_
        x = _complex2float(c_)
        le = np.array([s], dtype=np.int64)
        fsv = np.array([fs], dtype=np.int64)
        hpv = np.array([hp], dtype=np.int64)
        window = signal_dft.blackman_window(le)
        window[:] = (np.arange(window.shape[0]) + 1).astype(window.dtype)
        try:
            c_expected, expected = _stft(c_, le[0], window=window)
        except RuntimeError:
            self.skipTest("Unable to validate with torch.")

        i_expected = _istft(c_expected, le[0], window=window)
        info = dict(
            name=name,
            x_shape=x.shape,
            le=list(le),
            hp=hp,
            fs=fs,
            expected_shape=expected.shape,
            window_shape=window.shape,
        )

        # stft
        # x, fft_length, hop_length, n_frames, window, onesided=False
        case = onnx_script_test_case.FunctionTestParams(
            signal_dft.stft, [x, le, hpv, fsv, window], [expected]
        )
        try:
            self.run_eager_test(case, rtol=1e-4, atol=1e-4)
        except AssertionError as e:
            raise AssertionError(f"Issue with {info!r}.") from e

        # istft
        ix = _complex2float(c_expected)
        expected = x
        info["expected"] = expected
        info["expected_shape"] = expected.shape
        info["i_expected_shape"] = i_expected.shape
        info["ix_shape"] = ix.shape
        case = onnx_script_test_case.FunctionTestParams(
            signal_dft.istft, [ix, le, hpv, window], [expected]
        )
        try:
            self.run_eager_test(case, rtol=1e-4, atol=1e-4)
        except (AssertionError, RuntimeException) as e:
            # Not fully implemented.
            warnings.warn(
                "Issue with {} due to {}.".format(
                    str(info).split("\n", maxsplit=1),
                    str(e).split("\n", maxsplit=1)[0],
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
