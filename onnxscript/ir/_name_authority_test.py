# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

from onnxscript import ir
from onnxscript.ir import _name_authority


class NameAuthorityTest(unittest.TestCase):
    def test_register_or_name_value(self):
        name_authority = _name_authority.NameAuthority()
        value = ir.Value()
        name_authority.register_or_name_value(value)
        self.assertEqual(value.name, "val_0")

    def test_register_or_name_node(self):
        name_authority = _name_authority.NameAuthority()
        node = ir.Node("", "Test", [])
        name_authority.register_or_name_node(node)
        self.assertEqual(node.name, "node_Test_0")


if __name__ == "__main__":
    unittest.main()
