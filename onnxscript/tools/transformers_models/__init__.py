# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=import-outside-toplevel
from __future__ import annotations

import random
from typing import Any, Sequence

import onnx
import onnx.inliner
import torch

import onnxscript.optimizer
import onnxscript.rewriter


def export_to_onnx(
    model: Any,
    *args: Sequence[Any],
    optimize: bool = True,
    export_api: bool = True,
    no_grad: bool = False,
) -> onnx.ModelProto:
    """
    Export a model to ONNX.
    If optimize is True, it calls *onnxscript.optimizer.optimize*,
    *onnxscript.rewriter.rewriter*, *onnx.inliner.inline_local_functions*.
    If *export_api* is True, the function uses ``torch.onnx.export``
    and not ``torch.onnx.dynamo_export``.
    """
    if no_grad:
        with torch.no_grad():
            if export_api:
                prog = torch.onnx.export(model, args, dynamo=True)  # pylint: disable=no-value-for-parameter
            else:
                prog = torch.onnx.dynamo_export(model, *args)
    else:
        if export_api:
            prog = torch.onnx.export(model, args, dynamo=True)  # pylint: disable=no-value-for-parameter
        else:
            prog = torch.onnx.dynamo_export(model, *args)
    assert prog is not None
    model_proto = prog.model_proto
    if optimize:
        model_proto = onnxscript.optimizer.optimize(
            model_proto,
            num_iterations=2,
            onnx_shape_inference=True,
        )
        model_proto = onnxscript.rewriter.rewrite(model_proto)
        model_proto = onnx.inliner.inline_local_functions(model_proto)
    return model_proto


def ids_tensor(
    shape: Sequence[int],
    vocab_size: int,
    rng: random.Random | None = None,
    name: str | None = None,
):
    """Creates a random int32 tensor of the shape within the vocab size."""
    del name  # unused

    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


def get_input_dims_for_llm(
    dynamic_shapes: bool, warmup: int, repeat: int
) -> list[tuple[int, int]]:
    """Returns input dimensions for model such as llama, phi, ..."""
    if not dynamic_shapes:
        return [(2, 1024)] * (warmup + repeat)
    w = [(2, 1024), (3, 1024), (2, 1096)] * warmup
    w = w[:warmup]
    r = [(2, 1024), (3, 1024), (4, 1024), (2, 1096), (2, 1112)] * repeat
    r = r[:repeat]
    return w + r


def get_model_and_inputs(
    model: str,
    config: str,
    dynamic_shapes: bool,
    device: str = "cpu",
    num_hidden_layers: int = 1,
    with_mask: bool = True,
    implementation: str = "eager",
    dtype: str | None = None,
    warmup: int = 5,
    repeat: int = 10,
) -> tuple[Any, list[tuple[torch.Tensor, ...]], dict | None]:
    """
    Returns a model and a couple of dummy inputs.

    Args:
        model: model name, 'phi', 'llama', 'phi3', ...
        config: 'small', 'medium', 'large', ...
        dynamic_shapes: dynamic or static shapes
        device: 'cpu' or 'cuda'
        num_hidden_layers: Number of hidden layers.
        with_mask: One input or two inputs.
        implementation: eager or sdpa
        warmup: Number of inputs to generate.
        repeat: Number of inputs to generate for repeat.
        dtype: If specified, cast the model and the inputs into this type.

    Returns:
        model and list of inputs
    """
    if model == "llama":
        import onnxscript.tools.transformers_models.llama as m_llama

        tmodel, inputs, dynamic_shapes_def = m_llama.get_llama_model_from_config(
            warmup=warmup,
            repeat=repeat,
            implementation=implementation,
            with_mask=with_mask,
            num_hidden_layers=num_hidden_layers,
            dynamic_shapes=dynamic_shapes,
            config=config,
        )

    elif model == "mistral":
        import onnxscript.tools.transformers_models.mistral as m_mistral

        tmodel, inputs, dynamic_shapes_def = m_mistral.get_mistral_model_from_config(
            warmup=warmup,
            repeat=repeat,
            implementation=implementation,
            with_mask=with_mask,
            num_hidden_layers=num_hidden_layers,
            dynamic_shapes=dynamic_shapes,
            config=config,
        )

    elif model == "phi":
        import onnxscript.tools.transformers_models.phi as m_phi

        tmodel, inputs, dynamic_shapes_def = m_phi.get_phi_model_from_config(
            warmup=warmup,
            repeat=repeat,
            implementation=implementation,
            with_mask=with_mask,
            num_hidden_layers=num_hidden_layers,
            dynamic_shapes=dynamic_shapes,
            config=config,
        )

    elif model == "phi3":
        import onnxscript.tools.transformers_models.phi3 as m_phi3

        tmodel, inputs, dynamic_shapes_def = m_phi3.get_phi3_model_from_config(
            warmup=warmup,
            repeat=repeat,
            implementation=implementation,
            with_mask=with_mask,
            num_hidden_layers=num_hidden_layers,
            dynamic_shapes=dynamic_shapes,
            config=config,
        )

    else:
        raise ValueError(f"Model {model!r} is unknown.")

    if dtype is not None:
        dt = getattr(torch, dtype)
        tmodel = tmodel.to(dt)
        inputs = [
            tuple((i if i.dtype in {torch.int64, torch.int32} else i.to(dt)) for i in inp)
            for inp in inputs
        ]

    if device == "cuda":
        tmodel = tmodel.to("cuda")
        inputs = [tuple(i.to("cuda") for i in inp) for inp in inputs]

    return tmodel, inputs, dynamic_shapes_def
