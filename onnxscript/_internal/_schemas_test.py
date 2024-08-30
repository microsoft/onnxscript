# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import unittest
from typing import Any, Optional, Sequence, TypeVar, Union

import onnxscript
import onnxscript.testing
import parameterized
from onnxscript import FLOAT, INT64, ir

from torch_onnx import _schemas

_TestTypeVarConstraints = TypeVar("_TestTypeVarConstraints", INT64, FLOAT)
_TestTypeVarOneBound = TypeVar("_TestTypeVarOneBound", bound=INT64)
_TestTypeVarTwoBound = TypeVar("_TestTypeVarTwoBound", bound=Union[INT64, FLOAT])


class TypeConversionFunctionsTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (
                "tensor_type_all",
                onnxscript.onnx_types.TensorType,
                {ir.TensorType(dtype) for dtype in ir.DataType},
            ),
            ("tensor_type", INT64, {ir.TensorType(ir.DataType.INT64)}),
            (
                "tensor_type_union",
                Union[INT64, FLOAT],
                {ir.TensorType(ir.DataType.INT64), ir.TensorType(ir.DataType.FLOAT)},
            ),
            (
                "tensor_type_variadic_shape",
                INT64[...],
                {ir.TensorType(ir.DataType.INT64)},
            ),
            ("tensor_type_shape", INT64[10], {ir.TensorType(ir.DataType.INT64)}),
            (
                "type_var_constraints",
                _TestTypeVarConstraints,
                {ir.TensorType(ir.DataType.INT64), ir.TensorType(ir.DataType.FLOAT)},
            ),
            (
                "type_bound_one",
                _TestTypeVarOneBound,
                {ir.TensorType(ir.DataType.INT64)},
            ),
            (
                "type_bound_two",
                _TestTypeVarTwoBound,
                {ir.TensorType(ir.DataType.INT64), ir.TensorType(ir.DataType.FLOAT)},
            ),
            (
                "optional_tensor_type_all",
                Optional[onnxscript.onnx_types.TensorType],
                {ir.TensorType(dtype) for dtype in ir.DataType},
            ),
            (
                "optional_tensor_type",
                Optional[INT64],
                {ir.TensorType(ir.DataType.INT64)},
            ),
            (
                "optional_tensor_type_union",
                Optional[Union[INT64, FLOAT]],
                {ir.TensorType(ir.DataType.INT64), ir.TensorType(ir.DataType.FLOAT)},
            ),
            (
                "optional_tensor_type_variadic_shape",
                Optional[INT64[...]],
                {ir.TensorType(ir.DataType.INT64)},
            ),
            (
                "optional_tensor_type_shape",
                Optional[INT64[10]],
                {ir.TensorType(ir.DataType.INT64)},
            ),
            (
                "optional_type_var_constraints",
                Optional[_TestTypeVarConstraints],
                {ir.TensorType(ir.DataType.INT64), ir.TensorType(ir.DataType.FLOAT)},
            ),
            (
                "optional_type_bound_one",
                Optional[_TestTypeVarOneBound],
                {ir.TensorType(ir.DataType.INT64)},
            ),
            (
                "optional_type_bound_two",
                Optional[_TestTypeVarTwoBound],
                {ir.TensorType(ir.DataType.INT64), ir.TensorType(ir.DataType.FLOAT)},
            ),
            (
                "sequence_type_all",
                Sequence[onnxscript.onnx_types.TensorType],
                {ir.SequenceType(ir.TensorType(dtype)) for dtype in ir.DataType},
            ),
            (
                "sequence_type",
                Sequence[INT64],
                {ir.SequenceType(ir.TensorType(ir.DataType.INT64))},
            ),
            (
                "union_sequence_type",
                Union[Sequence[INT64], Sequence[FLOAT]],
                {
                    ir.SequenceType(ir.TensorType(ir.DataType.INT64)),
                    ir.SequenceType(ir.TensorType(ir.DataType.FLOAT)),
                },
            ),
            (
                "sequence_type_variadic_shape",
                Sequence[INT64[...]],
                {ir.SequenceType(ir.TensorType(ir.DataType.INT64))},
            ),
            (
                "sequence_type_shape",
                Sequence[INT64[10]],
                {ir.SequenceType(ir.TensorType(ir.DataType.INT64))},
            ),
            (
                "sequence_type_var_constraints",
                Sequence[_TestTypeVarConstraints],
                {
                    ir.SequenceType(ir.TensorType(ir.DataType.INT64)),
                    ir.SequenceType(ir.TensorType(ir.DataType.FLOAT)),
                },
            ),
            (
                "sequence_type_bound_one",
                Sequence[_TestTypeVarOneBound],
                {ir.SequenceType(ir.TensorType(ir.DataType.INT64))},
            ),
            (
                "sequence_type_bound_two",
                Sequence[_TestTypeVarTwoBound],
                {
                    ir.SequenceType(ir.TensorType(ir.DataType.INT64)),
                    ir.SequenceType(ir.TensorType(ir.DataType.FLOAT)),
                },
            ),
        ]
    )
    def test_pytype_to_ir_type(self, _, pytype: Any, expected: set[ir.TypeProtocol]):
        self.assertEqual(
            _schemas._get_allowed_types_from_type_annotation(pytype), expected
        )

    @parameterized.parameterized.expand(
        [
            ("type_var", _TestTypeVarConstraints, "_TestTypeVarConstraints"),
            ("type_var_bound", _TestTypeVarOneBound, "_TestTypeVarOneBound"),
            (
                "optional_type_var",
                Optional[_TestTypeVarOneBound],
                "_TestTypeVarOneBound",
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
    def test_get_type_constraint_name(self, _: str, pytype: Any, expected: str | None):
        self.assertEqual(_schemas._get_type_constraint_name(pytype), expected)


if __name__ == "__main__":
    unittest.main()
