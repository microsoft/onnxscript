# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import io
import textwrap
import unittest

from onnxscript._internal.opgen import pygen as cg


class PygenTest(unittest.TestCase):
    def assert_code(self, node: cg.Node, expected: str):
        node.accept(cg.ImportAdjuster())
        writer = io.StringIO()
        node.accept(cg.PythonWriter(writer))
        actual = writer.getvalue().strip()
        expected = textwrap.dedent(expected).strip()
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


if __name__ == "__main__":
    unittest.main()
