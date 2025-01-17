# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
from typing import List

import onnx
import torch

from onnxscript._internal.version_utils import torch_older_than


def _index_put_failing_function(x_len, start_idx, left_window=0, right_window=0):
    start_idx = torch.Tensor(start_idx).long()
    start_pad = torch.nn.functional.pad(start_idx, (1, 0))
    end_pad = torch.nn.functional.pad(start_idx, (0, 1), value=x_len)
    seq_range = torch.arange(0, x_len).unsqueeze(-1)
    idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]
    seq_range_expand = torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)
    idx_left = idx - left_window
    idx_left[idx_left < 0] = 0
    boundary_left = start_pad[idx_left]
    mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
    idx_right = idx + right_window
    idx_right[idx_right > len(start_idx)] = len(start_idx)
    boundary_right = end_pad[idx_right]
    mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
    return mask_left & mask_right


class TestEnd2End(unittest.TestCase):
    @unittest.skipIf(torch_older_than("2.6"), reason="fails to export")
    def test_index_put_failing_function(self):
        class MyModule(torch.nn.Module):
            def forward(self, X):
                x_len = 10  # 368
                start_idx = [4]
                left_window = 18
                result = _index_put_failing_function(
                    x_len, start_idx, left_window, right_window=0
                )
                return X + torch.unsqueeze(result, -1)

        torch_model = MyModule()
        torch_model.eval()
        inputs = (torch.randn(1, 1, 368),)
        expected = torch_model(*inputs)

        program = torch.onnx.export(torch_model, inputs, dynamo=True)
        # program.save(r"test_index_put_failing_function_not_optimized.onnx")
        program.optimize()
        program.save(r"test_index_put_failing_function.onnx")
        ref = onnx.reference.ReferenceEvaluator(program.model_proto)
        got = ref.run(None, {"x": inputs[0].numpy()})
        torch.testing.assert_close(expected, torch.tensor(got[0]))

    @unittest.skipIf(torch_older_than("2.6"), reason="no infer_schema")
    def test_register_custom_op(self):
        def index_put_failing_function(
            device: torch.device,
            x_len: int,
            start_idx: List[int],
            left_window: int,
            right_window: int,
        ) -> torch.Tensor:
            return _index_put_failing_function(
                x_len, start_idx, left_window, right_window
            )

        def index_put_failing_function_shape(device, x_len, start_idx, left_window, right_window):
            return torch.empty((x_len, x_len), dtype=torch.bool).to(device)

        def register_custom_op(fct, fct_shape, namespace, fname):
            schema_str = torch.library.infer_schema(fct, mutates_args=())
            custom_def = torch.library.CustomOpDef(namespace, fname, schema_str, fct)
            custom_def.register_kernel("cpu")(fct)
            custom_def._abstract_fn = fct_shape

        register_custom_op(
            index_put_failing_function,
            index_put_failing_function_shape,
            "test_delayed",
            "index_put_failing_function",
        )

        class MyModule(torch.nn.Module):
            def forward(self, X):
                x_len = 10  # 368
                start_idx = [4]
                left_window = 18
                result = torch.ops.test_delayed.index_put_failing_function(
                    "cpu", x_len, start_idx, left_window, 0
                )
                return X + torch.unsqueeze(result, -1)

        inputs = (torch.randn(1, 1, 368),)
        ep = torch.export.export(MyModule(), args=inputs, strict=False)
        self.assertIn("torch.ops.test_delayed.index_put_failing_function.default", str(ep))


if __name__ == "__main__":
    unittest.main(verbosity=2)
