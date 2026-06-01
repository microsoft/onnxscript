# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extended tests for CastConstantOfShape rules.

Adds coverage for: additional target dtypes (int32, float64, bfloat16),
the no-value variant with int target, and a same-dtype (identity) cast.
"""

from __future__ import annotations

import unittest

import onnx.parser
import onnx_ir as ir

from onnxscript.rewriter.rules.common import _cast_constant_of_shape


class CastConstantOfShapeExtendedTest(unittest.TestCase):
    """Extended tests for CastConstantOfShape rewrite rules."""

    def _apply(self, model_text: str, expected_count: int = 1):
        """Parse ONNX text, apply rule, and return the model."""
        model_proto = onnx.parser.parse_model(model_text)
        model = ir.serde.deserialize_model(model_proto)
        count = _cast_constant_of_shape.rules.apply_to_model(model)
        self.assertEqual(count, expected_count)
        return model

    # --- Positive: additional dtypes ---

    def test_cast_to_int32(self):
        """ConstantOfShape(float) → Cast(to=INT32) fuses."""
        model = self._apply(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (int32[1, 4] output)
            {
                constant = ConstantOfShape <value: tensor = float[1] {1.}>(input_x)
                output = Cast <to = 6> (constant)
            }
            """
        )
        self.assertEqual(len(model.graph), 1)
        # dtype 6 = INT32
        self.assertEqual(model.graph[0].attributes["value"].value.dtype, 6)

    def test_cast_to_float64(self):
        """ConstantOfShape(float) → Cast(to=DOUBLE) fuses."""
        model = self._apply(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (double[1, 4] output)
            {
                constant = ConstantOfShape <value: tensor = float[1] {2.5}>(input_x)
                output = Cast <to = 11> (constant)
            }
            """
        )
        self.assertEqual(len(model.graph), 1)
        # dtype 11 = DOUBLE
        self.assertEqual(model.graph[0].attributes["value"].value.dtype, 11)

    def test_cast_to_bfloat16(self):
        """ConstantOfShape(float) → Cast(to=BFLOAT16) fuses."""
        model = self._apply(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (bfloat16[1, 4] output)
            {
                constant = ConstantOfShape <value: tensor = float[1] {1.}>(input_x)
                output = Cast <to = 16> (constant)
            }
            """
        )
        self.assertEqual(len(model.graph), 1)
        # dtype 16 = BFLOAT16
        self.assertEqual(model.graph[0].attributes["value"].value.dtype, 16)

    def test_without_value_cast_to_int32(self):
        """ConstantOfShape (no value) → Cast(to=INT32) fuses with zero default."""
        model = self._apply(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (int32[1, 4] output)
            {
                constant = ConstantOfShape (input_x)
                output = Cast <to = 6> (constant)
            }
            """
        )
        self.assertEqual(len(model.graph), 1)
        self.assertEqual(model.graph[0].attributes["value"].value.dtype, 6)

    def test_same_dtype_cast_still_fuses(self):
        """Cast to same dtype as ConstantOfShape is still a valid fusion (removes Cast)."""
        model = self._apply(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (int64[2] input_x) => (float[1, 4] output)
            {
                constant = ConstantOfShape <value: tensor = float[1] {3.}>(input_x)
                output = Cast <to = 1> (constant)
            }
            """
        )
        # Cast should be removed (fused), leaving only ConstantOfShape
        self.assertEqual(len(model.graph), 1)
        self.assertEqual(model.graph[0].op_type, "ConstantOfShape")
        self.assertEqual(model.graph[0].attributes["value"].value.dtype, 1)


if __name__ == "__main__":
    unittest.main()
