# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Test cases for type constraint deduction functionality."""

from __future__ import annotations

import inspect
import logging
import unittest
from typing import Generator

import parameterized

import onnxscript
import onnxscript.function_libs.torch_lib.ops  # Import to populate registry
from onnxscript.function_libs.tools.torch_lib import deduce_type_constraints
from onnxscript.function_libs.torch_lib import registration

logger = logging.getLogger(__name__)


def torch_lib_onnx_functions_from_registry() -> Generator[onnxscript.OnnxFunction, None, None]:
    for op in registration.default_registry.values():
        for func in (*op.overloads, *op.privates, *op.complex):
            if isinstance(func, onnxscript.OnnxFunction):
                yield func


class TestDeduceTypeConstraints(unittest.TestCase):
    _SKIP_FUNCTIONS_WITH_LOOP_OR_SCAN = (
        "_aten_as_strided_onnx",
        "_aten_unfold_onnx",
        "_aten_embedding_bag_onnx",
        "_aten_embedding_bag_1d_padding_idx_onnx",
    )

    @parameterized.parameterized.expand(
        ((op,) for op in torch_lib_onnx_functions_from_registry()),
        name_func=lambda func, _, p: f"{func.__name__}_{p.args[0].name}",
    )
    def test_deduce_type_constraints_does_not_crash_for_onnx_function(
        self, onnx_function: onnxscript.OnnxFunction
    ):
        if onnx_function.name in self._SKIP_FUNCTIONS_WITH_LOOP_OR_SCAN:
            self.skipTest("Unimplemented: function contains loop or scan node.")
        try:
            signature_type_constraint = deduce_type_constraints.deduce_type_constraints(
                onnx_function
            )
        except NotImplementedError as e:
            if "Nested function" in str(e):
                self.skipTest("Unimplemented: function contains nested function.")
        logger.info(
            "Original signature: %s%s",
            onnx_function.name,
            inspect.signature(onnx_function.function),
        )
        logger.info(signature_type_constraint)


    def test_type_constraint_repr_unordered(self):
        value1 = deduce_type_constraints.Value("value1")
        value2 = deduce_type_constraints.Value("value2")
        type_constraint = deduce_type_constraints.TypeConstraint("test_constraint", {"tensor(float)", "tensor(int64)"})
        type_constraint.bind_value(value1)
        type_constraint.bind_value(value2)
        expected_repr = "TypeConstraint(name=test_constraint, type_strs={'tensor(float)', 'tensor(int64)'}, values=['value1', 'value2'])"
        self.assertIn(repr(type_constraint), [
            "TypeConstraint(name=test_constraint, type_strs={'tensor(float)', 'tensor(int64)'}, values=['value1', 'value2'])",
            "TypeConstraint(name=test_constraint, type_strs={'tensor(int64)', 'tensor(float)'}, values=['value1', 'value2'])"
        ])


    def test_deduce_raises_not_implemented_error_for_loop_and_scan(self):
        class MockOnnxFunction:
            def to_function_proto(self):
                class MockFunctionProto:
                    opset_import = [type('MockOpset', (object,), {'version': 1})]
                    node = [type('MockNode', (object,), {'op_type': 'Loop', 'domain': 'onnx'})]
                return MockFunctionProto()
            
            def param_schemas(self):
                return []
        
        onnx_function = MockOnnxFunction()
        deducer = deduce_type_constraints.TypeConstraintDeducer(onnx_function)
        
        with self.assertRaises(NotImplementedError):
            deducer.deduce()


    def test_onnx_function_type_constraints_repr(self):
        input_constraints = {"input1": deduce_type_constraints.TypeConstraint("T0", {"tensor(float)"})}
        output_constraints = {"output1": deduce_type_constraints.TypeConstraint("T1", {"tensor(int64)"})}
        intermediate_constraints = {"intermediate1": deduce_type_constraints.TypeConstraint("T2", {"tensor(int32)"})}
        constraints = deduce_type_constraints.OnnxFunctionTypeConstraints(input_constraints, output_constraints, intermediate_constraints)
        expected_repr = (
            "Type Constraints:\n"
            "  Inputs: \n"
            "    input1: T0\n"
            "  Outputs: \n"
            "    output1: T1\n"
            "  Type Constraints: \n"
            "    T0: {'tensor(float)'}\n"
            "    T1: {'tensor(int64)'}\n"
            "  Intermediate Values: \n"
            "    intermediate1: T2\n"
            "  Intermediate Type Constraints: \n"
            "    T2: {'tensor(int32)'}"
        )
        self.assertEqual(repr(constraints), expected_repr)


    def test_value_merge_type_constraint_no_constraints(self):
        value1 = deduce_type_constraints.Value("value1")
        value2 = deduce_type_constraints.Value("value2")
        with self.assertRaises(ValueError):
            value1.merge_type_constraint(value2)


if __name__ == "__main__":
    unittest.main()
