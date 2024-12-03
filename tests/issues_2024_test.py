# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import onnx
import onnxruntime as ort

from onnxscript import FLOAT, script
from onnxscript import opset20 as op
from tests.common import testutils


class TestIssues2024(testutils.TestBase):
    def test_issue_1969(self):
        # https://github.com/microsoft/onnxscript/issues/1969

        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.float32)
        y = np.array(x[len(x) - 3 : len(x)], dtype=np.float32)
        z = np.array(x[len(x) - 8 : len(x) - 3], dtype=np.float32)

        x_coef = np.array(
            [1.1, 1.5, 1.7, 1.9, 1.2, 3.1, 4.5, 5.2, 8.5, 9.0, 11.0], dtype=np.float32
        )
        y_coef = np.array([8.5, 9.0, 11.0], dtype=np.float32)
        z_coef = np.array([5, 7, 9, 9, 10, 11], dtype=np.float32)

        t1 = len(x)
        t2 = len(y)
        t3 = len(z)

        const_term = 2.37698
        h1 = 1.5689
        h2 = 1.799

        @script()
        def conv(a1: FLOAT, a2: FLOAT, a3: FLOAT, a4: FLOAT, a5: FLOAT, a6: FLOAT) -> FLOAT:
            const = op.Constant(value_float=const_term)
            # Define constants
            coeff1 = op.Constant(value_float=h1)
            coeff2 = op.Constant(value_float=h2)
            s = coeff1 * a1 + coeff2 * a2 - const

            for inx in range(t1):
                if a4 == x[inx]:
                    s = s + x_coef[inx]
            for inx2 in range(t2):
                if a5 == y[inx2]:
                    s = s + y_coef[inx2]
            for inx3 in range(t3):
                if a6 == z[inx3]:
                    s = s + z_coef[inx3]

            return op.Exp(s)

        onx = conv.to_model_proto()
        # To save to model and visualize it.
        # onnx.save(onx, "test_issue_1969.onnx")
        onnx.checker.check_model(onx)

        sess = ort.InferenceSession(onx.SerializeToString())
        feeds = {f"a{i}": np.array([i], dtype=np.float32) for i in range(1, 7)}
        got = sess.run(None, feeds)

        def conv_np(a1, a2, a3, a4, a5, a6):
            del a3
            const = const_term
            # Define constants
            coeff1 = h1
            coeff2 = h2
            s = coeff1 * a1 + coeff2 * a2 - const

            for inx in range(t1):
                if a4 == x[inx]:
                    s = s + x_coef[inx]
            for inx2 in range(t2):
                if a5 == y[inx2]:
                    s = s + y_coef[inx2]
            for inx3 in range(t3):
                if a6 == z[inx3]:
                    s = s + z_coef[inx3]

            return np.exp(s)

        expected = conv_np(**feeds)
        np.allclose(expected, got[0], rtol=1e-6)
        # Unexpected onnxscript value type '<class 'numpy.float32'>'.
        # Valid value types are 'Tensor | list[Tensor] | None | np.ndarray | list[np.ndarray]'
        expected2 = conv(**feeds)
        np.allclose(expected2, got[0], rtol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
