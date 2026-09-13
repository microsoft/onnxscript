# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx_ir as ir

from onnxscript._internal import autocast


def _signature(params: list) -> ir.schemas.OpSignature:
    return ir.schemas.OpSignature(
        domain="", name="TestOp", overload="", params=params, outputs=[]
    )


class CastInputsTest(unittest.TestCase):
    """Unit tests for `autocast.cast_inputs` (exercised via `dynamic_cast_inputs`)."""

    def test_it_raises_value_error_for_extra_args_when_op_has_no_inputs(self):
        # A signature with zero formal (non-attribute) input parameters, such as
        # RandomNormal, which only has attributes.
        op_signature = _signature(
            [
                ir.schemas.AttributeParameter(
                    name="seed", type=ir.AttributeType.FLOAT, required=False, default=None
                ),
            ]
        )
        self.assertEqual(op_signature.inputs, [])

        with self.assertRaisesRegex(ValueError, "exceeds number of formal parameters"):
            autocast.dynamic_cast_inputs(op_signature, (1.0,))

    def test_it_does_not_raise_when_op_has_no_inputs_and_no_args(self):
        op_signature = _signature(
            [
                ir.schemas.AttributeParameter(
                    name="seed", type=ir.AttributeType.FLOAT, required=False, default=None
                ),
            ]
        )

        self.assertEqual(autocast.dynamic_cast_inputs(op_signature, ()), ())

    def test_it_raises_value_error_for_extra_args_beyond_non_variadic_inputs(self):
        type_constraint = ir.schemas.TypeConstraintParam.any_tensor("T")
        op_signature = _signature(
            [
                ir.schemas.Parameter(
                    name="a", type_constraint=type_constraint, required=True, variadic=False
                ),
            ]
        )

        with self.assertRaisesRegex(ValueError, "exceeds number of formal parameters"):
            autocast.dynamic_cast_inputs(op_signature, (1.0, 2.0))

    def test_it_accepts_extra_args_for_homogeneous_variadic_inputs(self):
        type_constraint = ir.schemas.TypeConstraintParam.any_tensor("T")
        op_signature = _signature(
            [
                ir.schemas.Parameter(
                    name="inputs",
                    type_constraint=type_constraint,
                    required=True,
                    variadic=True,
                    homogeneous=True,
                ),
            ]
        )

        # Should not raise: a homogeneous variadic parameter can absorb any number
        # of trailing positional arguments.
        result = autocast.dynamic_cast_inputs(op_signature, (1.0, 2.0, 3.0))
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
