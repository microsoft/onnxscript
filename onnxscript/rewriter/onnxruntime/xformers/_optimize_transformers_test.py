# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import torch
import transformers.models.llama.modeling_llama as modeling_llama
from transformers import LlamaConfig

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

# Create a LlamaAttention object with the desired parameters
# model = modeling_llama.LlamaAttention(_config, 0)
model = modeling_llama.LlamaSdpaAttention(_config, 0)

# Dimensions for inputs:
_batch_size = 1
_seq_len = 10
_hidden_size = _config.hidden_size
_num_attention_heads = _config.num_attention_heads
dim = _hidden_size // _num_attention_heads

# Generate inputs:
_hidden_states = torch.rand(_batch_size, _seq_len, _hidden_size, dtype=torch.float32)
_attention_mask = torch.rand(_batch_size, 1, _seq_len, _seq_len, dtype=torch.float32)
_position_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64).reshape(1, 10)


# Get model in ONNX format
def _get_model(llama_attention_class, with_mask: bool):
    model = llama_attention_class(_config, 0)
    if with_mask:
        inputs = (_hidden_states, _attention_mask, _position_ids)
    else:
        inputs = (_hidden_states, None, _position_ids)
    exported = torch.onnx.export(model, inputs, dynamo=True)
    onnxscript.optimizer.optimize(exported.model)
    # optimize_transformers.basic_optimize(exported.model)
    return exported.model


class TestOptimizeTransformers(unittest.TestCase):
    def test_attention(self):
        model = _get_model(modeling_llama.LlamaAttention, with_mask=False)
        optimize_transformers.optimize(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("MultiHeadAttention", op_types)

    def test_masked_attention(self):
        model = _get_model(modeling_llama.LlamaAttention, with_mask=True)
        optimize_transformers.optimize(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("MultiHeadAttention", op_types)

    def test_sdpa_attention(self):
        model = _get_model(modeling_llama.LlamaSdpaAttention, with_mask=False)
        optimize_transformers.optimize(model, verbose=10)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("MultiHeadAttention", op_types)

    def test_masked_sdpa_attention(self):
        model = _get_model(modeling_llama.LlamaSdpaAttention, with_mask=True)
        optimize_transformers.optimize(model, verbose=10)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("MultiHeadAttention", op_types)


if __name__ == "__main__":
    unittest.main()
