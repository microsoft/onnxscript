# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import onnxruntime
import torch


class TestOnnxExportAten(unittest.TestCase):
    def test_aten_index_put_mask_bool_fixed_broadcast_2d(self):
        class Model(torch.nn.Module):
            def forward(self, x, values):
                x = x.clone()
                mask = torch.tensor([True, False, True, True, False]).to(torch.bool)
                x[mask] = values
                return x

        model = Model()
        xs = (
            torch.arange(25).reshape((5, 5)).to(torch.float32),
            torch.tensor([700, 800, 900, 1000, 1100], dtype=torch.float32),
        )
        expected = model(*xs)
        ep = torch.onnx.export(model, xs, dynamo=True)
        sess = onnxruntime.InferenceSession(
            ep.model_proto.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        torch.testing.assert_close(expected, torch.from_numpy(got))

    def test_aten_index_put_mask_bool_fixed_broadcast_3d(self):
        class Model(torch.nn.Module):
            def forward(self, x, values):
                x = x.clone()
                mask = torch.tensor([True, False]).to(torch.bool)
                x[mask] = values
                return x
                # return torch.ops.aten.index_put(x, (mask,), values)

        model = Model()
        xs = (
            torch.arange(2 * 3 * 5).reshape((2, 3, 5)).to(torch.float32),
            torch.tensor([700, 800, 900, 1000, 1100], dtype=torch.float32),
        )
        expected = model(*xs)
        ep = torch.onnx.export(model, xs, dynamo=True)
        sess = onnxruntime.InferenceSession(
            ep.model_proto.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        torch.testing.assert_close(expected, torch.from_numpy(got))

    def test_aten_index_put_mask_bool_fixed_broadcast_3d_2(self):
        class Model(torch.nn.Module):
            def forward(self, x, values):
                x = x.clone()
                mask = torch.tensor([[True, False, False], [True, True, False]]).to(torch.bool)
                x[mask] = values
                return x
                # return torch.ops.aten.index_put(x, (mask,), values)

        model = Model()
        xs = (
            torch.arange(2 * 3 * 5).reshape((2, 3, 5)).to(torch.float32),
            torch.tensor([700, 800, 900, 1000, 1100], dtype=torch.float32),
        )
        expected = model(*xs)
        ep = torch.onnx.export(model, xs, dynamo=True)
        sess = onnxruntime.InferenceSession(
            ep.model_proto.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        torch.testing.assert_close(expected, torch.from_numpy(got))


if __name__ == "__main__":
    unittest.main(verbosity=2)
