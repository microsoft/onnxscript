# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import glob
import os
from typing import Any

import torch
from torch.onnx import ExportOptions
from torch.onnx import _OrtBackend as OrtBackend
from torch.onnx import _OrtBackendOptions as OrtBackendOptions


def make_aot_ort(dynamic: bool = False) -> Any:
    """Implements an autograd backend for torch.compile based on onnxrt backend."""
    export_options = ExportOptions(dynamic_shapes=dynamic)
    options = OrtBackendOptions(export_options=export_options)
    ort_backend = OrtBackend(options=options)
    return ort_backend


def train_loop(
    model: Any,
    *args,
    loss_fn: Any | None = None,
    optimizer: Any | None = None,
    dump_onnx_models: bool = False,
    dump_prefix: str = "dump_train_loop",
    dump_clean_first: bool = True,
) -> tuple[Any, tuple[Any, ...]] | tuple[Any, tuple[Any, ...], list[str]]:
    """Implements a training loop to be used in tests.
    The function returns the forward output and gradients in a tuple.

    if dump_onnx_models is True, the function returns the forward output,
    the gradients in a tuple and the generated onnx_files.
    If there is no graph break, there should be
    two graphs, one for forward, one for backward.

    Args:
        model: pytorch model
        args: inputs
        loss_fn: loss function, default is MSELoss
        optimizer: optimizer, default is SGD
        dump_onnx_models: dumps the model onnxrt backend is producing
        dump_prefix: names will be `<dump_prefix>0.onnx`, `<dump_prefix>1.onnx`, ...
        dump_clean_first: clean all files starting with the given prefix

    Returns:
        - the forward outputs
        - the backwards gradients
        - the dumped onnx models, 2 at least unless the forward, backward
          were called before this function is executed or if the model
          is not a compiled model
    """

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    if dump_onnx_models:
        if dump_clean_first:
            names = glob.glob(f"{dump_prefix}*")
            for name in names:
                os.remove(name)

        old_value = os.environ.get("ONNXRT_DUMP_PATH", None)
        os.environ["ONNXRT_DUMP_PATH"] = f"{dump_prefix}_forward"
        existing_files = glob.glob(f"{dump_prefix}*.onnx")

    # Compute prediction and loss
    pred = model(*args)
    if isinstance(pred, tuple):
        v = pred[0]
    elif hasattr(pred, "last_hidden_state"):
        v = pred.last_hidden_state
    else:
        v = pred
    loss = loss_fn(v, torch.ones_like(v))

    # Backpropagation
    if dump_onnx_models:
        os.environ["ONNXRT_DUMP_PATH"] = f"{dump_prefix}_backward"
    loss.backward()
    optimizer.step()
    # skip that part to retrieve the gradients
    # optimizer.zero_grad()

    # returns the gradients
    res = tuple(p.grad for p in model.parameters() if p.grad is not None)
    assert len(res) > 0, f"No gradient, loss is {loss}"

    if dump_onnx_models:
        if old_value is None:
            del os.environ["ONNXRT_DUMP_PATH"]
        else:
            os.environ["ONNXRT_DUMP_PATH"] = old_value
        new_files = glob.glob(f"{dump_prefix}*.onnx")
        added_files = set(new_files) - set(existing_files)
        return pred, res, [f for f in new_files if f in added_files]

    return pred, res
