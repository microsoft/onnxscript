# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TODO(pytorch/pytorch#129279): Migrate these tests to the PyTorch repo

import itertools
import unittest

import onnxruntime
import torch

from tests.common import testutils


class TorchLibe2eTest(testutils.TestBase):
    def test_investigate_one_particular_model(self):
        """This test can be used to investigate a particular issue."""
        red, include, stype = "amin", False, "int32"
        dtype = getattr(torch, stype)

        class Model(torch.nn.Module):
            def __init__(self, include, red):
                super().__init__()
                self.include = include
                self.red = red

            def forward(self, x, indices, updates):
                x = x.clone()
                return x.scatter_reduce(
                    0, indices, updates, self.red, include_self=self.include
                )

        model = Model(include, red)
        xs = (
            torch.tensor([[-2, 0, 2], [2, -2, 0]], dtype=dtype),
            torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int64),
            torch.tensor([[-1, -1, -1], [-1, -1, -1]], dtype=dtype),
        )
        expected = model(*xs)
        model_path = (
            f"test_aten_scatter_{red}_"
            f"{'include' if include else 'exclude'}_{stype}.onnx"
        )
        torch.onnx.export(model, xs, model_path, dynamo=True)
        feeds = dict(zip(["x", "indices", "updates"], [x.numpy() for x in xs]))

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        torch.testing.assert_close(
            expected, torch.from_numpy(got), atol=1e-5, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
