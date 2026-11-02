# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest
from typing import Optional, Sequence, cast

import onnx_ir as ir

from onnxscript._internal import autocast, converter


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

    def test_static_cast_inputs_raises_value_error_for_extra_args_when_op_has_no_inputs(
        self,
    ):
        # `cast_inputs` is shared by `dynamic_cast_inputs` (eager mode, tested above)
        # and `static_cast_inputs` (script conversion). This exercises the same
        # zero-formal-inputs case through the static path. `converter_` is never
        # touched on this code path (the loop raises before reaching it), so a
        # `None` stand-in is enough.
        op_signature = _signature(
            [
                ir.schemas.AttributeParameter(
                    name="seed", type=ir.AttributeType.FLOAT, required=False, default=None
                ),
            ]
        )
        dummy_converter = cast(converter.Converter, None)
        dummy_args = cast(Sequence[Optional[ir.Value]], (1.0,))

        with self.assertRaisesRegex(ValueError, "exceeds number of formal parameters"):
            autocast.static_cast_inputs(dummy_converter, op_signature, dummy_args)

    def test_static_cast_inputs_does_not_raise_when_op_has_no_inputs_and_no_args(self):
        op_signature = _signature(
            [
                ir.schemas.AttributeParameter(
                    name="seed", type=ir.AttributeType.FLOAT, required=False, default=None
                ),
            ]
        )
        dummy_converter = cast(converter.Converter, None)

        self.assertEqual(autocast.static_cast_inputs(dummy_converter, op_signature, ()), ())


if __name__ == "__main__":
    unittest.main()
