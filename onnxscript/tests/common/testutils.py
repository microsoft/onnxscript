# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest


class TestBase(unittest.TestCase):
    def validate(self, fn):
        """Validate script function translation."""
        return fn.to_function_proto()
