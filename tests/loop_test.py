# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import subprocess
import sys
import textwrap
import unittest

import numpy as np

from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64
from tests.common import testutils

# tests/models/loop_multi_state.py has 5 loop-carried variables: enough for the
# converter's output-name ordering (built via a set union) to diverge from its
# input ordering (built by iterating the same set directly) under CPython's
# hash randomization. Run as a subprocess under a fixed PYTHONHASHSEED so the
# divergence reproduces deterministically instead of depending on whatever
# seed the test runner's own process happens to pick.
_MULTI_STATE_LOOP_CHECK = textwrap.dedent(
    """
    import numpy as np
    import onnx
    from onnxruntime import InferenceSession

    from tests.models.loop_multi_state import loop_multi_state

    x = np.array([2.0, 3.0], dtype=np.float32)
    eager = loop_multi_state(x, 3)

    model = loop_multi_state.to_model_proto()
    onnx.checker.check_model(model)
    sess = InferenceSession(model.SerializeToString(), providers=("CPUExecutionProvider",))
    got = sess.run(None, {"x": x, "n": np.array(3, dtype=np.int64)})

    for eager_arr, got_arr in zip(eager, got):
        assert np.allclose(eager_arr, got_arr), (eager_arr, got_arr)
    """
)


class LoopOpTest(testutils.TestBase):
    def test_loop(self):
        """Basic loop test."""

        @script()
        def sumprod(x: FLOAT["N"], N: INT64) -> (FLOAT["N"], FLOAT["N"]):  # noqa: F821
            sum = op.Identity(x)
            prod = op.Identity(x)
            for _ in range(N):
                sum = sum + x
                prod = prod * x
            return sum, prod

        self.validate(sumprod)
        x = np.array([2])
        M = 3
        sum, prod = sumprod(x, M)
        self.assertEqual(sum, np.array([8]))
        self.assertEqual(prod, np.array([16]))

    def test_loop_bound(self):
        """Test with an expression for loop bound."""

        @script()
        def sumprod(x: FLOAT["N"], N: INT64) -> (FLOAT["N"], FLOAT["N"]):  # noqa: F821
            sum = op.Identity(x)
            prod = op.Identity(x)
            for _ in range(2 * N + 1):
                sum = sum + x
                prod = prod * x
            return sum, prod

        self.validate(sumprod)

    def test_loop_state_var_output_order_matches_eager(self):
        """A Loop with >1 loop-carried variable must bind each returned Python
        name to the ONNX value it actually computes, not to whichever value a
        hash-randomized set iteration happened to line up with it.
        """
        for seed in ("3", "4", "8"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            result = subprocess.run(
                [sys.executable, "-c", _MULTI_STATE_LOOP_CHECK],
                capture_output=True,
                text=True,
                env=env,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            with self.subTest(seed=seed):
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
