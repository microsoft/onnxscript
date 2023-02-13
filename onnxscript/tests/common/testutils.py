# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest

from onnxscript import testing


class TestBase(unittest.TestCase):
    def validate(self, fn):
        """Validate script function translation."""
        return fn.to_function_proto()

    def assertSame(self, fn1, fn2):
        testing.assert_isomorphic(fn1, fn2)

    def assertSameGraph(self, graph1, graph2):
        testing.assert_isomorphic_graph(graph1, graph2)

    def assertSameFunction(self, fn1, fn2):
        testing.assert_isomorphic_function(fn1, fn2)
