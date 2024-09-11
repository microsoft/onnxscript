# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
import parameterized
import torch


class TestIssuesOnnxExporter(unittest.TestCase):
    def assertEqualArray(self, a, b, atol=None, rtol=None):
        if not isinstance(a, torch.Tensor):
            a = torch.Tensor(a)
        if isinstance(b, torch.Tensor):
            b = torch.Tensor(b)
        torch.testing.allclose(a, b, atol=atol, rtol=rtol)

    @parameterized.parameterized.expand(["script", "dynamo"])
    def test_updated_parameter(self, exporter):
        class UpdateModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = torch.zeros((2, 1, 10))

            def forward(self, update: torch.Tensor, index: torch.LongTensor):
                indices = torch.arange(update.shape[0])
                middle = torch.zeros((1,), dtype=torch.long)
                copy = self.params.clone()
                copy[index, middle, indices] = update.transpose(1, 0)
                return copy

        model = UpdateModel()

        n = 6
        update = torch.ones((n, 1))
        kv_index = torch.tensor([0])
        model(update, kv_index)

        model_path = f"test_updated_cache_{exporter}.onnx"

        if exporter == "script":
            torch.onnx.export(
                model,
                (update, kv_index),
                model_path,
                input_names=["update", "kv_index"],
                output_names=["updated"],
                dynamic_axes={"update": {0: "n"}},
                opset_version=13,
                verbose=False,
            )
        elif exporter == "dynamo":
            torch.onnx.export(
                model,
                (update, kv_index),
                model_path,
                input_names=["update", "kv_index"],
                output_names=["updated"],
                dynamic_axes={"update": {0: "n"}},
                verbose=False,
                dynamo=True,
            )
        else:
            raise AssertionError(f"Unknown exporter {exporter!r}")

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )

        def gen_numpy_inputs(n: int, idx: int):
            return {
                "update": 5 * np.ones((n, 1), dtype=np.float32),
                "kv_index": np.array([idx], dtype=np.int64),
            }

        input_n = gen_numpy_inputs(n, 0)
        e1 = session.run(["updated"], input_n)
        self.assertEqual(e1[0].shape, model.params.shape)

        input_2 = gen_numpy_inputs(2, 0)
        e2 = session.run(["updated"], input_2)
        self.assertEqual(e2[0].shape, model.params.shape)

    @parameterized.parameterized.expand(["script", "dynamo"])
    def test_scaled_dot_product_attention(self, exporter):
        class ScaledDotProductAttentionModel(torch.nn.Module):
            def __init__(self, d_model, scale):
                super().__init__()
                self.scale = scale  # scaling factor for attention scores
                self.d_model = d_model  # dimensionality of input embeddings
                self.query_linear = torch.nn.Linear(d_model, d_model)
                self.key_linear = torch.nn.Linear(d_model, d_model)
                self.value_linear = torch.nn.Linear(d_model, d_model)

            def forward(self, query_states, key_states, value_states):
                # Project the input states
                query = self.query_linear(query_states)
                key = self.key_linear(key_states)
                value = self.value_linear(value_states)

                # Perform scaled dot product attention
                attn_output = torch.functional.scaled_dot_product_attention(
                    query, key, value, scale=self.scale
                )
                return attn_output

        d_model = 64
        scale = 1.0 / (d_model**0.5)
        model = ScaledDotProductAttentionModel(d_model, scale)

        batch_size = 2
        seq_length_q = 10  # length of query
        seq_length_kv = 15  # length of key and value
        embedding_dim = d_model

        query_states = torch.randn(batch_size, seq_length_q, embedding_dim)
        key_states = torch.randn(batch_size, seq_length_kv, embedding_dim)
        value_states = torch.randn(batch_size, seq_length_kv, embedding_dim)

        output = model(query_states, key_states, value_states)

        onnx_file_path = f"scaled_dot_product_attention_{exporter}.onnx"

        if exporter == "script":
            torch.onnx.export(
                model,  # model being exported
                (query_states, key_states, value_states),  # example input (tuple)
                onnx_file_path,  # where to save the ONNX model
                input_names=[
                    "query_states",
                    "key_states",
                    "value_states",
                ],  # input names
                output_names=["attn_output"],  # output names
                opset=13,
            )
        elif exporter == "dynamo":
            torch.onnx.export(
                model,  # model being exported
                (query_states, key_states, value_states),  # example input (tuple)
                onnx_file_path,  # where to save the ONNX model
                input_names=[
                    "query_states",
                    "key_states",
                    "value_states",
                ],  # input names
                output_names=["attn_output"],  # output names
                dynamo=True,
            )
        else:
            raise AssertionError(f"Unknown exporter {exporter!r}")

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(
            onnx_file_path,
            sess_options=sess_options,
            providers=[("CPUExecutionProvider")],
        )
        inputs_names = [i.name for i in session.get_inputs()]
        output = session.run(
            None,
            dict(
                zip(
                    inputs_names,
                    (query_states.numpy(), key_states.numpy(), value_states.numpy()),
                )
            ),
        )
        expected_output = model(query_states, key_states, value_states)
        self.assertEqual(expected_output[0].shape, output[0].shape)
        self.assertEqualArray(expected_output[0], output[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
