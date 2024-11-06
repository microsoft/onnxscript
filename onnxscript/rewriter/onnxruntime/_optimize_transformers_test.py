# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnxscript.ir as ir
import onnxscript.optimizer
import onnxscript.rewriter.onnxruntime._optimize_transformers as optimize_transformers

def _get_smollm_model() -> ir.Model:
    checkpoint = "HuggingFaceTB/SmolLM-1.7B"
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
    program = torch.onnx.export(model, inputs, "tsmodel.onnx", dynamo=True)
    model = program.model
    onnxscript.optimizer.optimize_ir(model)
    return model

class TestOptimizeTransformers(unittest.TestCase):

    def test_optimize_transformers(self):
        model = _get_smollm_model()
        optimize_transformers.optimize(model)
        

if __name__ == "__main__":
    unittest.main()