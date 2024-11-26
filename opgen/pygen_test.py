# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest
from io import StringIO
from textwrap import dedent

import opgen.pygen as cg


class PygenTest(unittest.TestCase):
    def assert_code(self, node: cg.Node, expected: str):
        node.accept(cg.ImportAdjuster())
        writer = StringIO()
        node.accept(cg.PythonWriter(writer))
        actual = writer.getvalue().strip()
        expected = dedent(expected).strip()
        self.assertEqual(actual, expected)

    def test_function_takes_one_or_many_body_stmts(self):
        self.assert_code(
            cg.FunctionDef("single_stmt", body=cg.Pass()),
            """
            def single_stmt():
                pass
            """,
        )
        self.assert_code(
            cg.FunctionDef(
                "many_stmts",
                return_type=cg.NoneTypeRef(),
                body=[cg.EllipsisTypeRef(), cg.Raise(cg.ThunkExpr("ValueError(...)"))],
            ),
            """
            def many_stmts() -> None:
                ...
                raise ValueError(...)
            """,
        )

    def test_raise(self):
        self.assert_code(
            cg.FunctionDef("raise_", body=cg.Raise(cg.Call(cg.Name("ValueError")))),
            """
            def raise_():
                raise ValueError()
            """,
        )

        self.assert_code(
            cg.FunctionDef("raise_", body=cg.Raise(cg.ThunkExpr("ValueError()"))),
            """
            def raise_():
                raise ValueError()
            """,
        )

    def test_node_replace_with_self(self):
        class TestNode(cg.Node):
            def accept(self, visitor: cg.Visitor):
                pass
        node = TestNode()
        try:
            node.replace(node)
        except ValueError:
            self.fail("replace() raised ValueError unexpectedly!")


    def test_node_replace_root_node(self):
        class TestNode(cg.Node):
            def accept(self, visitor: cg.Visitor):
                pass
        node = TestNode()
        with self.assertRaises(ValueError):
            node.replace(TestNode())


    def test_node_get_ancestors_and_self(self):
        class TestNode(cg.Node):
            def accept(self, visitor: cg.Visitor):
                pass
        root = TestNode()
        child = TestNode()
        root.append_child(child, cg.Role("child"))
        ancestors = list(child.get_ancestors(and_self=True))
        self.assertEqual(ancestors, [child, root])


    def test_node_replace_with_none(self):
        class TestNode(cg.Node):
            def accept(self, visitor: cg.Visitor):
                pass
        parent = TestNode()
        child = TestNode()
        parent.append_child(child, cg.Role("child"))
        child.replace(None)
        self.assertIsNone(child.parent)
        self.assertIsNone(parent.first_child)


    def test_node_predicate_type_matching(self):
        class TestNode(cg.Node):
            def accept(self, visitor: cg.Visitor):
                pass
        predicate = cg.NodePredicate(type_=TestNode)
        node = TestNode()
        self.assertTrue(predicate.matches(node))


    def test_node_remove(self):
        class TestNode(cg.Node):
            def accept(self, visitor: cg.Visitor):
                pass
        parent = TestNode()
        child = TestNode()
        parent.append_child(child, cg.Role("child"))
        removed_node = child.remove()
        self.assertIsNone(removed_node.parent)
        self.assertIsNone(parent.first_child)


    def test_node_replace_with_new_node(self):
        class TestNode(cg.Node):
            def accept(self, visitor: cg.Visitor):
                pass
        parent = TestNode()
        old_node = TestNode()
        new_node = TestNode()
        parent.append_child(old_node, cg.Role("child"))
        old_node.replace(new_node)
        self.assertIsNone(old_node.parent)
        self.assertEqual(new_node.parent, parent)


    def test_single_or_none_multiple_elements(self):
        with self.assertRaises(StopIteration):
            cg.single_or_none([1, 2])



if __name__ == "__main__":
    unittest.main()
