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
