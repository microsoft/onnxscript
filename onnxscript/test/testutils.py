# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest
from onnx import FunctionProto
from onnxscript import OnnxFunction
from .checker import isomorphic


def function_proto(f):
    if isinstance(f, FunctionProto):
        return f
    if isinstance(f, OnnxFunction):
        return f.to_function_proto()
    raise TypeError(f"Cannot convert {type(f)} to FunctionProto")


class TestBase(unittest.TestCase):
    def validate(self, fn):
        '''validate script function translation'''
        return fn.to_function_proto()

    def assertSame(self, fn1, fn2):
        self.assertTrue(isomorphic(function_proto(fn1), function_proto(fn2)))
