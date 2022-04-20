# SPDX-License-Identifier: Apache-2.0

import unittest
from onnx import FunctionProto
from onnxscript import OnnxFunction
from .checker import isomorphic
import onnx.helper

def print_ir_function(f):
    print(str(f))
    for s in f.stmts:
        for attr in s.attrs:
            if attr.attr_proto.HasField("g"):
                print(onnx.helper.printable_graph(attr.attr_proto.g))

def function_proto(f):
    if isinstance(f, FunctionProto):
        return f
    if isinstance(f, OnnxFunction):
        print_ir_function(f.function_ir)
        return f.to_function_proto()
    raise TypeError(f"Cannot convert {type(f)} to FunctionProto")


class TestBase(unittest.TestCase):
    def assertSame(self, fn1, fn2):
        self.assertTrue(isomorphic(function_proto(fn1), function_proto(fn1)))
