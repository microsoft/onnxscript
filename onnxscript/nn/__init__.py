# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PyTorch-like module interface for building ONNX graphs."""

from onnxscript.nn._module import Module
from onnxscript.nn._parameter import Parameter

__all__ = ["Module", "Parameter"]
