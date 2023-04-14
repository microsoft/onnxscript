# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest


class TestBase(unittest.TestCase):
    """The base class for testing ONNX Script functions for internal use."""

    def validate(self, fn):
        """Validate script function translation."""
        return fn.to_function_proto()
