# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""End-to-end export tests for ``torch.nn.EmbeddingBag``.

These tests export real PyTorch modules (instead of calling the torchlib function
directly, as ``ops_test.py`` does) and verify that

* the exported model agrees with eager PyTorch,
* the exported model agrees with a "vendored" bag built from ``nn.Embedding`` plus a
  plain ``sum``/``mean``/``amax`` reduction, and
* the exported graph is fully vectorized, i.e. contains no ``Loop`` (a per-bag loop is
  prohibitively slow on GPU backends such as TensorRT).

See https://github.com/microsoft/onnxscript/issues/2489.
"""

from __future__ import annotations

import unittest
from typing import Optional

import numpy as np
import onnx
import onnxruntime as ort
import parameterized
import torch

NUM_EMBEDDINGS = 12
EMBEDDING_DIM = 5

# Reductions understood by both ``nn.EmbeddingBag`` and the vendored implementation.
MODES = ("sum", "mean", "max")

# (rtol, atol) per dtype. Half precision accumulates in float32 internally in the
# exporter but is compared against torch's own half-precision result, hence the slack.
TOLERANCES = {
    torch.float16: (1e-2, 1e-2),
    torch.float32: (1e-5, 1e-5),
    torch.float64: (1e-12, 1e-12),
}
FLOAT_DTYPES = tuple(TOLERANCES)
# Dtypes for which ONNX Runtime CPU has the kernels needed for an end-to-end run.
RUNNABLE_FLOAT_DTYPES = (torch.float32, torch.float64)

# Ragged bags: (offsets, number of indices). Covers empty leading, middle and
# trailing bags in addition to the regular case.
RAGGED_BAGS = {
    "regular": ([0, 2, 5], 8),
    "empty_leading": ([0, 0, 3], 6),
    "empty_middle": ([0, 3, 3, 4], 7),
    "empty_trailing": ([0, 2, 7], 7),
}


class _NativeBag(torch.nn.Module):
    """``nn.EmbeddingBag`` driven by (indices, offsets)."""

    def __init__(
        self,
        mode: str,
        include_last_offset: bool = False,
        padding_idx: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.bag = torch.nn.EmbeddingBag(
            NUM_EMBEDDINGS,
            EMBEDDING_DIM,
            mode=mode,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
            dtype=dtype,
        )

    def forward(self, indices, offsets):
        return self.bag(indices, offsets)


class _NativeFixedBag(_NativeBag):
    """``nn.EmbeddingBag`` driven by 2D (batch, bag_length) indices."""

    def forward(self, indices):  # type: ignore[override]
        return self.bag(indices)


class _VendoredFixedBag(torch.nn.Module):
    """The same fixed-length bag, hand-built from ``nn.Embedding`` and a reduction."""

    def __init__(
        self,
        mode: str,
        weight: torch.Tensor,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.mode = mode
        self.padding_idx = padding_idx
        self.embedding = torch.nn.Embedding(
            NUM_EMBEDDINGS, EMBEDDING_DIM, padding_idx=padding_idx, dtype=weight.dtype
        )
        with torch.no_grad():
            self.embedding.weight.copy_(weight)

    def forward(self, indices):
        embedded = self.embedding(indices)  # (batch, bag_length, dim)
        if self.padding_idx is None:
            valid = torch.ones_like(indices, dtype=torch.bool)
        else:
            valid = indices != self.padding_idx
        valid = valid.unsqueeze(-1)
        if self.mode == "sum":
            return torch.where(valid, embedded, torch.zeros_like(embedded)).sum(dim=1)
        if self.mode == "mean":
            summed = torch.where(valid, embedded, torch.zeros_like(embedded)).sum(dim=1)
            count = valid.sum(dim=1).clamp(min=1).to(embedded.dtype)
            return summed / count
        lowest = torch.full_like(embedded, torch.finfo(embedded.dtype).min)
        reduced = torch.where(valid, embedded, lowest).amax(dim=1)
        has_entries = valid.any(dim=1)
        return torch.where(has_entries, reduced, torch.zeros_like(reduced))


class _ConstantInputBag(torch.nn.Module):
    """``nn.EmbeddingBag`` whose indices and offsets are constants of the model."""

    def __init__(self, mode: str, indices: torch.Tensor, offsets: torch.Tensor):
        super().__init__()
        self.bag = torch.nn.EmbeddingBag(NUM_EMBEDDINGS, EMBEDDING_DIM, mode=mode)
        self.register_buffer("indices", indices)
        self.register_buffer("offsets", offsets)

    def forward(self, scale):
        return self.bag(self.indices, self.offsets) * scale


def _export(module: torch.nn.Module, args: tuple, **kwargs) -> onnx.ModelProto:
    program = torch.onnx.export(module, args, dynamo=True, optimize=True, **kwargs)
    assert program is not None
    return program.model_proto


def _run(model: onnx.ModelProto, *args: torch.Tensor) -> np.ndarray:
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feeds = {input_.name: arg.numpy() for input_, arg in zip(session.get_inputs(), args)}
    (result,) = session.run(None, feeds)
    return result


def _op_types(model: onnx.ModelProto) -> set[str]:
    """All op types in the model, including those inside subgraphs and functions."""
    found: set[str] = set()

    def visit_graph(graph: onnx.GraphProto) -> None:
        for node in graph.node:
            found.add(node.op_type)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        visit_graph(subgraph)

    visit_graph(model.graph)
    for function in model.functions:
        for node in function.node:
            found.add(node.op_type)
    return found


def _make_indices(generator: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randint(0, NUM_EMBEDDINGS, shape, generator=generator, dtype=torch.long)


class EmbeddingBagExportTest(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(0)

    def assert_close(self, actual, expected, dtype: torch.dtype) -> None:
        rtol, atol = TOLERANCES[dtype]
        np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=rtol, atol=atol)

    @parameterized.parameterized.expand(
        [
            (f"{mode}_{dtype}".replace("torch.", ""), mode, dtype)
            for mode in MODES
            for dtype in RUNNABLE_FLOAT_DTYPES
        ]
    )
    def test_native_export_matches_vendored_embedding_with_reduction(
        self, _, mode: str, dtype: torch.dtype
    ):
        native = _NativeFixedBag(mode, dtype=dtype).eval()
        vendored = _VendoredFixedBag(mode, native.bag.weight.detach()).eval()
        indices = _make_indices(self.generator, (4, 3))

        # The two torch modules must agree before any export is involved.
        expected = native(indices)
        self.assert_close(vendored(indices).detach().numpy(), expected, dtype)

        native_result = _run(_export(native, (indices,)), indices)
        vendored_result = _run(_export(vendored, (indices,)), indices)
        self.assert_close(native_result, expected, dtype)
        self.assert_close(vendored_result, expected, dtype)

    @parameterized.parameterized.expand(
        [
            (f"{mode}_{dtype}".replace("torch.", ""), mode, dtype)
            for mode in MODES
            for dtype in RUNNABLE_FLOAT_DTYPES
        ]
    )
    def test_native_export_with_padding_idx_matches_vendored(
        self, _, mode: str, dtype: torch.dtype
    ):
        padding_idx = 3
        native = _NativeFixedBag(mode, padding_idx=padding_idx, dtype=dtype).eval()
        vendored = _VendoredFixedBag(
            mode, native.bag.weight.detach(), padding_idx=padding_idx
        ).eval()
        # Rows: padding in the middle, all padding (empty bag), and no padding.
        indices = torch.tensor([[1, padding_idx, 2], [padding_idx] * 3, [4, 5, 6]])

        expected = native(indices)
        self.assert_close(vendored(indices).detach().numpy(), expected, dtype)

        self.assert_close(_run(_export(native, (indices,)), indices), expected, dtype)
        self.assert_close(_run(_export(vendored, (indices,)), indices), expected, dtype)

    @parameterized.parameterized.expand(
        [
            (
                f"{mode}_{name}_{'last' if include_last_offset else 'nolast'}",
                mode,
                name,
                include_last_offset,
            )
            for mode in MODES
            for name in RAGGED_BAGS
            for include_last_offset in (False, True)
        ]
    )
    def test_ragged_bags_match_eager(
        self, _, mode: str, bag_name: str, include_last_offset: bool
    ):
        offsets_list, num_indices = RAGGED_BAGS[bag_name]
        if include_last_offset:
            offsets_list = [*offsets_list, num_indices]
        module = _NativeBag(mode, include_last_offset=include_last_offset).eval()
        indices = _make_indices(self.generator, (num_indices,))
        offsets = torch.tensor(offsets_list, dtype=torch.long)

        result = _run(_export(module, (indices, offsets)), indices, offsets)
        self.assert_close(result, module(indices, offsets), torch.float32)

    @parameterized.parameterized.expand([(mode,) for mode in MODES])
    def test_indices_beyond_last_offset_are_ignored(self, mode: str):
        # With include_last_offset, the final offset ends the last bag; the indices
        # after it do not belong to any bag.
        module = _NativeBag(mode, include_last_offset=True).eval()
        indices = _make_indices(self.generator, (6,))
        offsets = torch.tensor([0, 2, 3], dtype=torch.long)

        result = _run(_export(module, (indices, offsets)), indices, offsets)
        self.assert_close(result, module(indices, offsets), torch.float32)

    @parameterized.parameterized.expand([(mode,) for mode in MODES])
    def test_ragged_bags_with_padding_idx_match_eager(self, mode: str):
        padding_idx = 2
        module = _NativeBag(mode, padding_idx=padding_idx).eval()
        indices = torch.tensor([1, padding_idx, 4, padding_idx, padding_idx, 5, 7])
        offsets = torch.tensor([0, 2, 3, 5], dtype=torch.long)

        result = _run(_export(module, (indices, offsets)), indices, offsets)
        self.assert_close(result, module(indices, offsets), torch.float32)

    @parameterized.parameterized.expand([(mode,) for mode in MODES])
    def test_dynamic_number_of_bags_and_indices(self, mode: str):
        module = _NativeBag(mode).eval()
        indices = _make_indices(self.generator, (9,))
        offsets = torch.tensor([0, 4, 4], dtype=torch.long)
        model = _export(
            module,
            (indices, offsets),
            dynamic_shapes=({0: "num_indices"}, {0: "num_bags"}),
        )

        other_indices = _make_indices(self.generator, (14,))
        other_offsets = torch.tensor([0, 1, 6, 6, 10], dtype=torch.long)
        for case in ((indices, offsets), (other_indices, other_offsets)):
            self.assert_close(_run(model, *case), module(*case), torch.float32)

    @parameterized.parameterized.expand([(mode,) for mode in MODES])
    def test_int32_indices_and_offsets_match_eager(self, mode: str):
        # torch accepts int32 indices and offsets; the exported graph must as well.
        module = _NativeBag(mode).eval()
        indices = _make_indices(self.generator, (8,)).int()
        offsets = torch.tensor([0, 3, 3, 6], dtype=torch.int32)

        result = _run(_export(module, (indices, offsets)), indices, offsets)
        self.assert_close(result, module(indices, offsets), torch.float32)

    @parameterized.parameterized.expand([(mode,) for mode in MODES])
    def test_graph_reduces_rows_with_scatternd(self, mode: str):
        # ScatterND copies whole rows and is much faster on ONNX Runtime CPU than
        # ScatterElements (which would need an index per element).
        module = _NativeBag(mode).eval()
        indices = _make_indices(self.generator, (8,))
        offsets = torch.tensor([0, 2, 5], dtype=torch.long)

        reduction_nodes = [
            node
            for node in _export(module, (indices, offsets)).graph.node
            if node.op_type == "ScatterND"
        ]

        self.assertEqual(len(reduction_nodes), 1)
        (attribute,) = [a for a in reduction_nodes[0].attribute if a.name == "reduction"]
        self.assertEqual(attribute.s.decode(), "max" if mode == "max" else "add")

    @parameterized.parameterized.expand(
        [
            (f"{mode}_{name}", mode, offsets_list, num_indices)
            for mode in MODES
            for name, (offsets_list, num_indices) in {
                # One index per bag: the bag ids are 0..num_bags-1, so with constant
                # inputs they cover the whole first dimension of the scatter, which
                # the optimizer's ScatterND rewrites must not mistake for an assignment.
                "one_index_per_bag": ([0, 1, 2, 3], 4),
                "regular": ([0, 3, 5], 8),
                "empty_bag": ([0, 0, 4], 6),
            }.items()
        ]
    )
    def test_constant_inputs_survive_optimization(
        self, _, mode: str, offsets_list: list[int], num_indices: int
    ):
        indices = _make_indices(self.generator, (num_indices,))
        offsets = torch.tensor(offsets_list, dtype=torch.long)
        module = _ConstantInputBag(mode, indices, offsets).eval()
        scale = torch.tensor(1.5)

        model = _export(module, (scale,))

        self.assert_close(_run(model, scale), module(scale), torch.float32)
        # Folding constants may produce the (num_bags, dim) result buffer, but must not
        # bake a per-element (num_indices, dim) index tensor into the model.
        largest_constant = max(
            (
                np.prod(initializer.dims)
                for initializer in model.graph.initializer
                if initializer.name != "bag.weight"
            ),
            default=0,
        )
        self.assertLessEqual(largest_constant, len(offsets_list) * EMBEDDING_DIM)

    def test_per_sample_weights_match_eager(self):
        class Weighted(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bag = torch.nn.EmbeddingBag(NUM_EMBEDDINGS, EMBEDDING_DIM, mode="sum")

            def forward(self, indices, offsets, per_sample_weights):
                return self.bag(indices, offsets, per_sample_weights=per_sample_weights)

        module = Weighted().eval()
        indices = _make_indices(self.generator, (8,))
        offsets = torch.tensor([0, 3, 3, 6], dtype=torch.long)
        weights = torch.randn(8, generator=self.generator)

        result = _run(_export(module, (indices, offsets, weights)), indices, offsets, weights)
        self.assert_close(result, module(indices, offsets, weights), torch.float32)

    @parameterized.parameterized.expand(
        [
            (f"{mode}_{dtype}".replace("torch.", ""), mode, dtype)
            for mode in MODES
            for dtype in (*FLOAT_DTYPES, torch.bfloat16)
        ]
    )
    def test_exported_graph_has_no_loop(self, _, mode: str, dtype: torch.dtype):
        module = _NativeBag(mode, dtype=dtype).eval()
        indices = _make_indices(self.generator, (8,))
        offsets = torch.tensor([0, 2, 5], dtype=torch.long)

        op_types = _op_types(_export(module, (indices, offsets)))

        for loop_op in ("Loop", "Scan", "SequenceInsert", "ConcatFromSequence"):
            self.assertNotIn(loop_op, op_types)

    @parameterized.parameterized.expand([(mode,) for mode in MODES])
    def test_exported_graph_with_padding_idx_has_no_loop(self, mode: str):
        module = _NativeBag(mode, padding_idx=1).eval()
        indices = _make_indices(self.generator, (8,))
        offsets = torch.tensor([0, 2, 5], dtype=torch.long)

        op_types = _op_types(_export(module, (indices, offsets)))

        for loop_op in ("Loop", "Scan", "SequenceInsert", "ConcatFromSequence"):
            self.assertNotIn(loop_op, op_types)

    def test_half_precision_export_matches_eager(self):
        # ONNX Runtime CPU lacks some float16 kernels, so only validate the exported
        # graph structurally and by accumulating in float32 on the torch side.
        module = _NativeBag("sum", dtype=torch.float16).eval()
        indices = _make_indices(self.generator, (8,))
        offsets = torch.tensor([0, 2, 5], dtype=torch.long)

        model = _export(module, (indices, offsets))

        self.assertNotIn("Loop", _op_types(model))
        result = _run(model, indices, offsets)
        self.assertEqual(result.dtype, np.float16)
        self.assert_close(result, module(indices, offsets), torch.float16)


if __name__ == "__main__":
    unittest.main()
