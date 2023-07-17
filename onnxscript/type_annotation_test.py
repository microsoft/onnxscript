# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import unittest
from typing import Any, List, Optional, Sequence, TypeVar, Union

import parameterized

import onnxscript
import onnxscript.testing
from onnxscript import FLOAT, INT64, script, type_annotation
from onnxscript.onnx_opset import opset15 as op
from onnxscript.tests.common import testutils


class TypeAnnotationTest(testutils.TestBase):
    def test_type_annotation(self):
        """Test type annotations."""

        @script()
        def static_shape(A: FLOAT[100], B: FLOAT[100]) -> FLOAT[100]:
            C = op.Add(A, B)
            return C

        static_shape_txt = """
            static_shape (float[100] A, float[100] B) => (float[100] C) {
                C = Add (A, B)
            }
        """
        onnxscript.testing.assert_isomorphic_graph(static_shape, static_shape_txt)

        @script()
        def symbolic_shape(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:  # noqa: F821
            C = op.Add(A, B)
            return C

        symbolic_shape_txt = """
            symbolic_shape (float[N] A, float[N] B) => (float[N] C) {
                C = Add (A, B)
            }
        """
        onnxscript.testing.assert_isomorphic_graph(symbolic_shape, symbolic_shape_txt)

        @script()
        def tensor_scalar(A: FLOAT["N"], B: FLOAT) -> FLOAT["N"]:  # noqa: F821
            C = op.Add(A, B)
            return C

        tensor_scalar_txt = """
            tensor_scalar (float[N] A, float B) => (float[N] C) {
                C = Add (A, B)
            }
        """
        onnxscript.testing.assert_isomorphic_graph(tensor_scalar, tensor_scalar_txt)

        @script()
        def unknown_rank(A: FLOAT[...], B: FLOAT[...]) -> FLOAT[...]:
            C = op.Add(A, B)
            return C

        unknown_rank_txt = """
            unknown_rank (float[] A, float[] B) => (float[] C) {
                C = Add (A, B)
            }
        """
        onnxscript.testing.assert_isomorphic_graph(unknown_rank, unknown_rank_txt)

        with self.assertRaises(ValueError):
            FLOAT[10][20]  # Invalid usage. pylint: disable=pointless-statement

    def test_type_annotation_with_bool_type_for_attribute(self):
        @script()
        def bool_type_for_attribute(self: FLOAT[...], sorted: bool) -> FLOAT[...]:
            out = op.Unique(self, sorted=sorted)
            return out

        bool_type_for_attribute_txt = """
            <
                domain: "this",
                opset_import: ["": 15]
            >
            bool_type_for_attribute <sorted>(self) => (out) {
                out = Unique <sorted: int = @sorted> (self)
            }

        """
        onnxscript.testing.assert_isomorphic_function(
            bool_type_for_attribute, bool_type_for_attribute_txt
        )


_TestTypeVarConstraints = TypeVar("_TestTypeVarConstraints", INT64, FLOAT)
_TestTypeVarOneBound = TypeVar("_TestTypeVarOneBound", bound=INT64)
_TestTypeVarTwoBound = TypeVar("_TestTypeVarTwoBound", bound=Union[INT64, FLOAT])


class TypeConversionFunctionsTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (
                "tensor_type_all",
                onnxscript.onnx_types.TensorType,
                list(type_annotation.ALL_TENSOR_TYPE_STRINGS),
            ),
            ("none", None, list(type_annotation.ALL_TENSOR_TYPE_STRINGS)),
            ("tensor_type", INT64, ["tensor(int64)"]),
            ("tensor_type_union", Union[INT64, FLOAT], ["tensor(float)", "tensor(int64)"]),
            ("tensor_type_variadic_shape", INT64[...], ["tensor(int64)"]),
            ("tensor_type_shape", INT64[10], ["tensor(int64)"]),
            (
                "type_var_constraints",
                _TestTypeVarConstraints,
                ["tensor(float)", "tensor(int64)"],
            ),
            ("type_bound_one", _TestTypeVarOneBound, ["tensor(int64)"]),
            ("type_bound_two", _TestTypeVarTwoBound, ["tensor(float)", "tensor(int64)"]),
            (
                "optional_tensor_type_all",
                Optional[onnxscript.onnx_types.TensorType],
                [
                    *[
                        f"optional({tensor_type})"
                        for tensor_type in type_annotation.ALL_TENSOR_TYPE_STRINGS
                    ],
                    *type_annotation.ALL_TENSOR_TYPE_STRINGS,
                ],
            ),
            (
                "optional_tensor_type",
                Optional[INT64],
                ["optional(tensor(int64))", "tensor(int64)"],
            ),
            (
                "optional_tensor_type_union",
                Optional[Union[INT64, FLOAT]],
                [
                    "optional(tensor(float))",
                    "optional(tensor(int64))",
                    "tensor(float)",
                    "tensor(int64)",
                ],
            ),
            (
                "optional_tensor_type_variadic_shape",
                Optional[INT64[...]],
                ["optional(tensor(int64))", "tensor(int64)"],
            ),
            (
                "optional_tensor_type_shape",
                Optional[INT64[10]],
                ["optional(tensor(int64))", "tensor(int64)"],
            ),
            (
                "optional_type_var_constraints",
                Optional[_TestTypeVarConstraints],
                [
                    "optional(tensor(float))",
                    "optional(tensor(int64))",
                    "tensor(float)",
                    "tensor(int64)",
                ],
            ),
            (
                "optional_type_bound_one",
                Optional[_TestTypeVarOneBound],
                ["optional(tensor(int64))", "tensor(int64)"],
            ),
            (
                "optional_type_bound_two",
                Optional[_TestTypeVarTwoBound],
                [
                    "optional(tensor(float))",
                    "optional(tensor(int64))",
                    "tensor(float)",
                    "tensor(int64)",
                ],
            ),
            (
                "sequence_type_all",
                Sequence[onnxscript.onnx_types.TensorType],
                [
                    f"seq({tensor_type})"
                    for tensor_type in type_annotation.ALL_TENSOR_TYPE_STRINGS
                ],
            ),
            ("sequence_type", Sequence[INT64], ["seq(tensor(int64))"]),
            (
                "union_sequence_type",
                Union[Sequence[INT64], Sequence[FLOAT]],
                ["seq(tensor(float))", "seq(tensor(int64))"],
            ),
            (
                "sequence_type_variadic_shape",
                Sequence[INT64[...]],
                ["seq(tensor(int64))"],
            ),
            ("sequence_type_shape", Sequence[INT64[10]], ["seq(tensor(int64))"]),
            (
                "sequence_type_var_constraints",
                Sequence[_TestTypeVarConstraints],
                ["seq(tensor(float))", "seq(tensor(int64))"],
            ),
            (
                "sequence_type_bound_one",
                Sequence[_TestTypeVarOneBound],
                ["seq(tensor(int64))"],
            ),
            (
                "sequence_type_bound_two",
                Sequence[_TestTypeVarTwoBound],
                ["seq(tensor(float))", "seq(tensor(int64))"],
            ),
        ]
    )
    def test_pytype_to_type_strings(self, _, pytype: Any, expected: List[str]):
        self.assertEqual(type_annotation.pytype_to_type_strings(pytype), expected)

    @parameterized.parameterized.expand(
        [
            ("type_var", _TestTypeVarConstraints, "_TestTypeVarConstraints"),
            ("type_var_bound", _TestTypeVarOneBound, "_TestTypeVarOneBound"),
            (
                "optional_type_var",
                Optional[_TestTypeVarOneBound],
                "Optional__TestTypeVarOneBound",
            ),
            (
                "sequence_type_var",
                Sequence[_TestTypeVarOneBound],
                "Sequence__TestTypeVarOneBound",
            ),
            ("normal_type", INT64, None),
            ("union_type", Union[INT64, FLOAT], None),
            ("optional_type", Optional[INT64], None),
            ("sequence_type", Sequence[INT64], None),
            ("optional_sequence_type", Optional[Sequence[INT64]], None),
            ("optional_union_type", Optional[Union[INT64, FLOAT]], None),
        ]
    )
    def test_get_type_constraint_name(self, _: str, pytype: Any, expected: Optional[str]):
        self.assertEqual(type_annotation.get_type_constraint_name(pytype), expected)


if __name__ == "__main__":
    unittest.main()
