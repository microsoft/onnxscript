# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime inference session wrapper for ir.Model objects.

Uses ``onnxruntime-easy`` which handles bfloat16 and other non-standard
dtypes transparently.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime_easy as ort_easy

from mobius import _flags
from mobius._model_package import ModelPackage


def _has_scan_nodes(model: ir.Model) -> bool:
    """Return True if the model or any of its functions contain Scan ops.

    Uses ``graph.all_nodes()`` which recurses into Loop/If/Scan
    subgraph bodies so nested Scan nodes are not missed.
    """
    for node in model.graph.all_nodes():
        if node.op_type == "Scan":
            return True
    for func in model.functions.values():
        for node in func:
            if node.op_type == "Scan":
                return True
    return False


class OnnxModelSession:
    """Wraps an ``onnxruntime_easy.EasySession`` for an ``ir.Model``.

    Serializes the model to a temporary file and creates an ORT session.
    Provides a simple ``run()`` interface that accepts and returns numpy arrays.

    Example::

        from mobius._testing.ort_inference import OnnxModelSession

        session = OnnxModelSession(model)
        outputs = session.run({"input_ids": np.array([[1, 2, 3]])})
        logits = outputs["logits"]
    """

    def __init__(
        self,
        model: ir.Model | ModelPackage,
        **load_kwargs,
    ):
        if isinstance(model, ModelPackage):
            if len(model) != 1:
                raise ValueError(
                    f"ModelPackage has {len(model)} models; pass a "
                    f"single ir.Model or index into the package."
                )
            model = next(iter(model.values()))

        # Disable memory-pattern pre-allocation for models with Scan
        # nodes on CUDA.  ORT's memory planner cannot resolve symbolic
        # dims in Scan body outputs, so it pre-allocates undersized
        # buffers.  Disabling the pattern forces runtime allocation
        # with actual shapes.
        is_cuda = load_kwargs.get("device") == "cuda"
        if is_cuda and _flags.flags.disable_mem_pattern_for_scan and _has_scan_nodes(model):
            load_kwargs.setdefault("enable_mem_pattern", False)
            load_kwargs.setdefault("enable_mem_reuse", False)

        self._tmpdir = tempfile.TemporaryDirectory()
        self._model_path = str(Path(self._tmpdir.name) / "model.onnx")
        ir.save(model, self._model_path, external_data="model.onnx.data")

        self._session = ort_easy.load(self._model_path, **load_kwargs)
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    def get_input_shape(self, name: str) -> list[int | str] | None:
        """Return the declared shape of an input, or ``None`` if not found.

        Shape elements may be ``int`` (static) or ``str`` (symbolic).
        """
        for inp in self._session.get_inputs():
            if inp.name == name:
                return list(inp.shape)
        return None

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference and return outputs as a name→array dict.

        Args:
            feeds: Input name → numpy array mapping. Only inputs present
                in the model are used; extra keys are ignored.

        Returns:
            Dict mapping output names to numpy arrays.
        """
        # Filter to only inputs the model expects and convert to OrtValues
        ort_feeds = {}
        for k, v in feeds.items():
            if k not in self._input_names:
                continue
            # Ensure contiguous layout for ORT. Skip 0-d scalars
            # because np.ascontiguousarray promotes them to 1-d.
            if v.ndim > 0:
                v = np.ascontiguousarray(v)
            ort_feeds[k] = ort_easy.ort_value(v)
        raw_outputs = self._session(**ort_feeds)
        return dict(zip(self._output_names, (o.numpy() for o in raw_outputs)))

    def close(self) -> None:
        self._tmpdir.cleanup()

    def __del__(self) -> None:
        self.close()
