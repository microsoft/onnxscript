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
    _SKIP_FUNCTIONS_WITH_NESTED_FUNCTION = ()

    @parameterized.parameterized.expand(
        ((op,) for op in torch_lib_onnx_functions_from_registry()),
        name_func=lambda func, _, p: f"{func.__name__}_{p.args[0].name}",
    )
    def test_deduce_type_constraints_does_not_crash_for_onnx_function(
        self, onnx_function: onnxscript.OnnxFunction
    ):
        if onnx_function.name in self._SKIP_FUNCTIONS_WITH_LOOP_OR_SCAN:
            self.skipTest("Unimplemented: function contains loop or scan node.")
        if onnx_function.name in self._SKIP_FUNCTIONS_WITH_NESTED_FUNCTION:
            self.skipTest("Unimplemented: function contains nested function.")
        signature_type_constraint = deduce_type_constraints.deduce_type_constraints(
            onnx_function
        )
        logger.info(
            "Original signature: %s%s",
            onnx_function.name,
            inspect.signature(onnx_function.function),
        )
        logger.info(signature_type_constraint)


if __name__ == "__main__":
    unittest.main()
