# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

from onnx import FunctionProto, GraphProto, parser

from onnxscript import OnnxFunction
from onnxscript.test.checker import isomorphic


def function_proto(f):
    if isinstance(f, FunctionProto):
        return f
    if isinstance(f, OnnxFunction):
        return f.to_function_proto()
    raise TypeError(f"Cannot convert {type(f)} to FunctionProto")


def graph_proto(g):
    if isinstance(g, GraphProto):
        return g
    if isinstance(g, OnnxFunction):
        return g.to_model_proto().graph
    if isinstance(g, str):
        return parser.parse_graph(g)
    raise TypeError(f"Cannot convert {type(g)} to ModelProto")


class TestBase(unittest.TestCase):
    def validate(self, fn):
        """validate script function translation"""
        return fn.to_function_proto()

    def assertSame(self, fn1, fn2):
        self.assertTrue(isomorphic(function_proto(fn1), function_proto(fn2)))

    def assertSameGraph(self, graph1, graph2):
        self.assertTrue(isomorphic(graph_proto(graph1), graph_proto(graph2)))
