# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

__all__ = [
    "TorchScriptTensor",
    "TorchScriptGraph",
    "TorchScriptTracingEvaluator",
]


class _RemovedClass:
    """A onnxscript tensor that wraps a torchscript Value."""

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "Support for dynamo_export has been dropped since onnxscript 0.4.0. "
            "Please use `torch.onnx.export(..., dynamo=True)`, or downgrade to onnxscript<0.4"
        )


TorchScriptTensor = _RemovedClass
TorchScriptGraph = _RemovedClass
TorchScriptTracingEvaluator = _RemovedClass
