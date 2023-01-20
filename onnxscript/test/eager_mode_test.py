# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest
from typing import Sequence

import numpy as np
import onnxscript
import onnxscript.tensor

from onnxscript import script
from onnxscript import opset17 as op
from onnxscript import FLOAT


class EagerModeTest(unittest.TestCase):
    def test_sequence_input(self):
        @script()
        def Concat(seq: Sequence[FLOAT["N"]]):
            return op.ConcatFromSequence(seq, axis=0)
        
        np_array = np.array([1, 2, 3], dtype=np.float32)
        output1 = Concat([np_array, np_array])
        self.assertIsInstance(output1, np.ndarray)

        os_tensor = onnxscript.tensor.Tensor(np_array)
        output2 = Concat([os_tensor, os_tensor])
        self.assertIsInstance(output2, onnxscript.tensor.Tensor)

if __name__ == "__main__":
    unittest.main(verbosity=2)
