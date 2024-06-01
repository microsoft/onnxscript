from __future__ import annotations

import unittest

from onnxscript.ir import convenience


class NamespaceTest(unittest.TestCase):

    def test_module_members_is_public(self):
        for name in convenience.__all__:
            function = getattr(convenience, name)
            self.assertEqual(function.__module__, "onnxscript.ir.convenience")
