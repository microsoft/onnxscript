# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import onnxruntime
import torch
import transformers.models.llama.modeling_llama as modeling_llama
from parameterized import parameterized
from transformers import LlamaConfig

import onnxscript.ir._io as io
import onnxscript.optimizer
from onnxscript.rewriter.onnxruntime.xformers import (
    _optimize_transformers as optimize_transformers,
)

# Create a LlamaConfig object with the desired parameters
_config = LlamaConfig(
    _name_or_path="HuggingFaceTB/SmolLM-1.7B",
    architectures=["LlamaForCausalLM"],
    attention_bias=False,
    attention_dropout=0.0,
    bos_token_id=0,
    eos_token_id=0,
    hidden_act="silu",
    hidden_size=2048,
    initializer_range=0.02,
    intermediate_size=8192,
    max_position_embeddings=2048,
    model_type="llama",
    num_attention_heads=32,
    num_hidden_layers=24,
    num_key_value_heads=32,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=None,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    torch_dtype="float32",
    transformers_version="4.37.2",
    use_cache=True,
    vocab_size=49152,
)

# Dimensions for inputs:
_batch_size = 1
_seq_len = 10
_hidden_size = _config.hidden_size
_num_attention_heads = _config.num_attention_heads
dim = _hidden_size // _num_attention_heads

# Generate inputs:
_hidden_states = torch.rand(_batch_size, _seq_len, _hidden_size, dtype=torch.float32)
_causal_mask = torch.tril(torch.ones(_seq_len, _seq_len, dtype=torch.float32))
_attention_mask = _causal_mask.unsqueeze(0).unsqueeze(0).expand(_batch_size, 1, _seq_len, _seq_len)
_position_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64).reshape(1, 10)

# Get model in ONNX format
# def _get_model(llama_attention_class, with_mask: bool):
#     model = llama_attention_class(_config, 0)
#     if with_mask:
#         inputs = (_hidden_states, _attention_mask, _position_ids)
#     else:
#         inputs = (_hidden_states, None, _position_ids)
#     exported = torch.onnx.export(model, inputs, dynamo=True)
#     # ORT Transformer optimizations are applied after basic optimization.
#     onnxscript.optimizer.optimize(exported.model)
#     return exported.model

class _TestData:
    def __init__(self, name: str, attention_class, with_mask: bool):
        self.name = name
        self.attention_class = attention_class
        self.with_mask = with_mask

    def get_torch_model(self):
        return self.attention_class(_config, 0)

    def get_onnx_model(self):
        model = self.get_torch_model()
        inputs = self.get_inputs()
        input_names = ["input" + str(i) for i in range(len(inputs)) if inputs[i] is not None]
        exported = torch.onnx.export(model, inputs, input_names=input_names, dynamo=True)
        # ORT Transformer optimizations are applied after basic optimization.
        onnxscript.optimizer.optimize(exported.model)
        return exported.model

    def get_inputs(self):
        if self.with_mask:
            return (_hidden_states, _attention_mask, _position_ids)
        else:
            return (_hidden_states, None, _position_ids)

    def get_torch_outputs(self):
        return self.get_torch_model()(*self.get_inputs())

    def get_ort_inputs(self):
        inputs = self.get_inputs()
        return {f"input{i}": input for i, input in enumerate(inputs) if input is not None}

_test_cases = [
    _TestData("attention", modeling_llama.LlamaAttention, False),
    _TestData("masked_attention", modeling_llama.LlamaAttention, True),
    _TestData("sdpa_attention", modeling_llama.LlamaSdpaAttention, False),
    _TestData("masked_sdpa_attention", modeling_llama.LlamaSdpaAttention, True),
]

_test_case_tuples = [ (t,) for t in _test_cases]

def _ort_check(model_name: str, model, inputs, expected_outputs, rtol=1e-2, atol=1e-2):
    providers = ["CPUExecutionProvider"]
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, f"{model_name}.onnx")
        io.save(model, model_path)
        # Run optimized model
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        ort_outputs = session.run(None, inputs)

        for i, (baseline_output, optimized_output) in enumerate(
            zip(expected_outputs, ort_outputs)
        ):
            try:
                np.testing.assert_equal(baseline_output.shape, optimized_output.shape)
                np.testing.assert_allclose(
                    baseline_output, optimized_output, rtol=rtol, atol=atol
                )
            except AssertionError as e:
                print(
                    f"Failed for model {model_name} and output {i} with rtol={rtol} and atol={atol}\n{e}"
                )
                raise

class TestOptimizeTransformers(unittest.TestCase):
    @parameterized.expand(_test_case_tuples)
    def test_attention_optimization(self, test_data: _TestData):
        model = test_data.get_onnx_model()
        # io.save(model, os.path.join(r"C:\repos\onnxscript\smy\Models", f"{test_data.name}.onnx"))
        # model.display()
        # print("======>")
        optimize_transformers.fuse_rotary_embedding(model)
        # model.display()
        op_types = [n.op_type for n in model.graph]
        self.assertIn("RotaryEmbedding", op_types)
        # _ort_check(test_data.name, model, test_data.get_ort_inputs(), test_data.get_torch_outputs())


if __name__ == "__main__":
    unittest.main()
