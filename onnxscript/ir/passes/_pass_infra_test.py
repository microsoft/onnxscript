# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import unittest

from onnxscript import ir
from onnxscript.ir.passes import _pass_infra


class PassBaseTest(unittest.TestCase):
    def test_pass_results_can_be_used_as_pass_input(self):
        class TestPass(_pass_infra.PassBase):
            @property
            def in_place(self) -> bool:
                return True

            @property
            def changes_input(self) -> bool:
                return False

            def call(self, model: ir.Model) -> _pass_infra.PassResult:
                # This is a no-op pass
                return _pass_infra.PassResult(model=model, modified=False)

        pass_ = TestPass()
        model = ir.Model(graph=ir.Graph([], [], nodes=[]), ir_version=10)
        result = pass_(model)
        self.assertIsInstance(result, _pass_infra.PassResult)
        # pass can take the result of another pass as input
        result_1 = pass_(result)
        # It can also take the model as input
        result_2 = pass_(result.model)
        self.assertIs(result_1.model, result_2.model)


if __name__ == "__main__":
    unittest.main()
