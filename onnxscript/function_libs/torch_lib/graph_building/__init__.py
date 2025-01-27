# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""APIs for building an ONNX graph from a PyTorch model.

This module exposes only three classes that will be used to build an ONNX graph
by the ONNX exporter in PyTorch:

- :class:`TorchScriptTensor`: Represents a symbolic value in the ONNX graph.
- :class:`TorchScriptGraph`: Stores the graph being built.
- :class:`TorchScriptTracingEvaluator`: An evaluator that will record all operators
    applied on the ``TorchScriptTensor``. It has a reference to the ``TorchScriptGraph``
    being built, will write to it, and will handle eager evaluations of Torch Lib
    functions when desired.

The usage is in https://github.com/pytorch/pytorch/blob/136f8378e1b5a8cb7127977b8d068fbf9c3e1247/torch/onnx/_internal/fx/fx_onnx_interpreter.py#L698-L702,
and it is very simple::

    with onnxscript.evaluator.default_as(onnxscript_tracer):  # onnxscript_tracer is a TorchScriptTracingEvaluator
        output: Union[
            onnxscript_graph_building.TorchScriptTensor,
            Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
        ] = symbolic_fn(*onnx_args, **onnx_kwargs)

Here, we set the default evaluator to be ``onnxscript_tracer`` so
that ONNX Script will dispatch all operators calls to the evaluator. The ``symbolic_fn``
can be a pure Python function (e.g. trace-only) or an ONNX Script function. Either way,
they are recorded by ``onnxscript_tracer`` and onto the graph.

The outputs, as ``TorchScriptTensor``, are then handed by to the exporter. On line
https://github.com/pytorch/pytorch/blob/136f8378e1b5a8cb7127977b8d068fbf9c3e1247/torch/onnx/_internal/fx/fx_onnx_interpreter.py#L707
the exporter fills in type and shape information from PyTorch by calling the setters
on ``TorchScriptTensor.dtype`` and ``TorchScriptTensor.shape``.
"""

from __future__ import annotations

__all__ = [
    "TorchScriptTensor",
    "TorchScriptGraph",
    "TorchScriptTracingEvaluator",
]

from onnxscript.function_libs.torch_lib import _flags

if _flags.EXPERIMENTAL_USE_IR:
    from ._graph_building_ir import (
        TorchScriptGraph,
        TorchScriptTensor,
        TorchScriptTracingEvaluator,
    )
else:
    from ._graph_building_torch import (  # type: ignore[assignment]
        TorchScriptGraph,
        TorchScriptTensor,
        TorchScriptTracingEvaluator,
    )
