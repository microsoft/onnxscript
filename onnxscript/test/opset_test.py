# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

from onnx.defs import onnx_opset_version

import onnxscript.onnx_opset as mod_opset
from onnxscript.test.testutils import TestBase


class TestOpset(TestBase):

    def test_opset15(self):
        op = mod_opset.opset15
        self.assertEqual(op.domain, '')
        self.assertEqual(op.version, 15)

    def test_opset16(self):
        op = mod_opset.opset16
        self.assertEqual(op.domain, '')
        self.assertEqual(op.version, 16)

    def test_opset_last(self):
        v = onnx_opset_version()
        name = "opset%d" % v
        opset = getattr(mod_opset, name)
        self.assertEqual(opset.domain, '')
        self.assertEqual(opset.version, v)


if __name__ == '__main__':
    unittest.main()
