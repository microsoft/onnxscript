# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnxruntime as ort
import parameterized
import torch
from torch.onnx._internal.exporter import _building, _tensors

import onnxscript
from onnxscript import ir
from onnxscript.function_libs.torch_lib.ops import core


def _inputs(layout, dtype, length=24, groups=3):
    shapes = {
        "2d_3d": ((length, 8), (groups, 8, 8)),
        "3d_2d": ((groups, 8, 8), (8, length)),
        "2d_2d": ((8, length), (length, 8)),
    }
    rng = np.random.default_rng(0)
    return tuple(rng.uniform(-1, 1, shape).astype(dtype) for shape in shapes[layout])


def _reference(a, b, offsets):
    # Fill only the groups PyTorch writes; unused trailing storage is unspecified.
    if a.ndim == b.ndim == 2:
        result = np.zeros((len(offsets), a.shape[0], b.shape[1]), dtype=a.dtype)
    else:
        result = np.zeros((a.shape[-2], b.shape[-1]), dtype=a.dtype)
    start = 0
    for i, end in enumerate(offsets):
        if a.ndim == b.ndim == 2:
            result[i] = a[:, start:end] @ b[start:end]
        elif a.ndim == 2:
            result[start:end] = a[start:end] @ b[i]
        else:
            result[:, start:end] = a[i] @ b[:, start:end]
        start = end
    return result


def _capture(shapes, dtype, *, offset_shape, **kwargs):
    opset = onnxscript.opset18
    specs = [("a", shapes[0], dtype), ("b", shapes[1], dtype)]
    if offset_shape is not None:
        specs.append(("offsets", offset_shape, ir.DataType.INT32))
    tensors = [
        _tensors.SymbolicTensor(
            opset, name=name, shape=ir.Shape(shape), type=ir.TensorType(dt)
        )
        for name, shape, dt in specs
    ]

    tracer = _building.OpRecorder(opset, {})
    with onnxscript.evaluator.default_as(tracer):
        result = core.aten_grouped_mm(*tensors, **kwargs)
    rank = 3 if len(shapes[0]) == len(shapes[1]) else 2
    result.shape = ir.Shape([None] * rank)
    result.dtype = kwargs.get("out_dtype", dtype)
    graph = ir.Graph(
        tensors, [result], nodes=tracer.nodes, opset_imports={"": 18}, name="grouped_mm"
    )
    model = ir.to_proto(ir.Model(graph, ir_version=10))
    onnx.checker.check_model(model, full_check=True)
    return model


def _session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


class GroupedMmTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (f"{layout}_{dtype.__name__}_{case}", layout, dtype, offsets)
            for layout in ("2d_3d", "3d_2d", "2d_2d")
            for dtype in (np.float16, np.float32)
            for case, offsets in (
                ("uneven", [4, 12, 24]),
                ("empty_first", [0, 8, 24]),
                ("empty_middle", [8, 8, 24]),
                ("unused_tail", [0, 8, 16]),
                ("all_empty", [0, 0, 0]),
            )
        ]
    )
    def test_offsets(self, _, layout, dtype, offsets):
        a, b = _inputs(layout, dtype)
        offsets = np.asarray(offsets, dtype=np.int32)
        expected = _reference(a, b, offsets)
        model = _capture(
            (a.shape, b.shape),
            ir.DataType.FLOAT16 if dtype == np.float16 else ir.DataType.FLOAT,
            offset_shape=offsets.shape,
        )
        actual = _session(model).run(None, {"a": a, "b": b, "offsets": offsets})[0]
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        rtol, atol = (1e-3, 1e-3) if dtype == np.float16 else (1e-5, 1e-6)
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        if hasattr(torch.ops.aten, "_grouped_mm"):
            native = torch.ops.aten._grouped_mm.default(
                torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(offsets)
            ).numpy()
            self.assertEqual(actual.shape, native.shape)
            # Do not compare uninitialized trailing rows/columns from the native op.
            if layout == "2d_3d":
                actual, native = actual[: offsets[-1]], native[: offsets[-1]]
            elif layout == "3d_2d":
                actual, native = actual[:, : offsets[-1]], native[:, : offsets[-1]]
            np.testing.assert_allclose(actual, native, rtol=rtol, atol=atol)

    @parameterized.parameterized.expand([("2d_3d",), ("3d_2d",), ("2d_2d",)])
    def test_empty_group_list(self, layout):
        a, b = _inputs(layout, np.float32, groups=0)
        offsets = np.asarray([], dtype=np.int32)
        model = _capture((a.shape, b.shape), ir.DataType.FLOAT, offset_shape=(0,))
        actual = _session(model).run(None, {"a": a, "b": b, "offsets": offsets})[0]
        np.testing.assert_array_equal(actual, _reference(a, b, offsets))

    def test_existing_dense_bias_and_cast(self):
        a = np.ones((3, 8, 8), dtype=np.float32)
        b = np.ones((3, 8, 8), dtype=np.float32)
        model = _capture(
            (a.shape, b.shape),
            ir.DataType.FLOAT,
            offset_shape=None,
            bias=2.0,
            out_dtype=ir.DataType.FLOAT16,
        )
        actual = _session(model).run(None, {"a": a, "b": b})[0]
        np.testing.assert_array_equal(actual, (a @ b + 2).astype(np.float16))

    @parameterized.parameterized.expand([("2d_3d",), ("3d_2d",), ("2d_2d",)])
    def test_single_group_and_explicit_output_dtype(self, layout):
        a, b = _inputs(layout, np.float32, groups=1)
        offsets = np.asarray([24], dtype=np.int32)
        model = _capture(
            (a.shape, b.shape),
            ir.DataType.FLOAT,
            offset_shape=(1,),
            out_dtype=ir.DataType.FLOAT,
        )
        actual = _session(model).run(None, {"a": a, "b": b, "offsets": offsets})[0]
        np.testing.assert_allclose(actual, _reference(a, b, offsets), rtol=1e-5, atol=1e-6)

    def test_dynamic_group_count_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "statically known number of groups"):
            _capture(((24, 8), ("groups", 8, 8)), ir.DataType.FLOAT, offset_shape=("groups",))

    def test_operand_group_count_is_checked(self):
        with self.assertRaisesRegex(ValueError, "group counts must match"):
            _capture(((24, 8), (2, 8, 8)), ir.DataType.FLOAT, offset_shape=(3,))

    def test_offset_rank_is_checked(self):
        with self.assertRaisesRegex(ValueError, "1D"):
            _capture(((24, 8), (3, 8, 8)), ir.DataType.FLOAT, offset_shape=(1, 3))

    def test_two_dense_operands_with_offsets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "2D operand"):
            _capture(((3, 8, 8), (3, 8, 8)), ir.DataType.FLOAT, offset_shape=(3,))

    def test_offset_bias_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "bias"):
            _capture(((24, 8), (3, 8, 8)), ir.DataType.FLOAT, offset_shape=(3,), bias=object())

    def test_offset_output_dtype_change_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "output dtype"):
            _capture(
                ((24, 8), (3, 8, 8)),
                ir.DataType.FLOAT,
                offset_shape=(3,),
                out_dtype=ir.DataType.FLOAT16,
            )

    @parameterized.parameterized.expand([("2d_3d",), ("3d_2d",), ("2d_2d",)])
    def test_dynamic_offsets_and_shapes(self, layout):
        shapes = {
            "2d_3d": (("length", 8), (3, 8, 8)),
            "3d_2d": ((3, 8, 8), (8, "length")),
            "2d_2d": ((8, "length"), ("length", 8)),
        }[layout]
        model = _capture(shapes, ir.DataType.FLOAT16, offset_shape=(3,))
        session = _session(model)
        self.assertIn("offsets", [value.name for value in session.get_inputs()])
        for size, values in (
            (24, [8, 16, 24]),
            (24, [0, 8, 24]),
            (24, [8, 8, 24]),
            (24, [0, 8, 16]),
            (32, [8, 16, 32]),
        ):
            with self.subTest(size=size, offsets=values):
                a, b = _inputs(layout, np.float16, length=size)
                offsets = np.asarray(values, dtype=np.int32)
                actual = session.run(None, {"a": a, "b": b, "offsets": offsets})[0]
                expected = _reference(a, b, offsets)
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    @parameterized.parameterized.expand([("2d_3d",), ("3d_2d",), ("2d_2d",)])
    @unittest.skipUnless(hasattr(torch.ops.aten, "_grouped_mm"), "requires aten::_grouped_mm")
    def test_bfloat16_export(self, layout):
        class Model(torch.nn.Module):
            def forward(self, a, b, offsets):
                return torch.ops.aten._grouped_mm.default(a, b, offsets)

        a, b = (torch.from_numpy(x).to(torch.bfloat16) for x in _inputs(layout, np.float32))
        offsets = torch.tensor([8, 16, 24], dtype=torch.int32)
        length = 8 * torch.export.Dim("blocks", min=1)
        dynamic_shapes = {
            "2d_3d": ({0: length}, {}, {}),
            "3d_2d": ({}, {1: length}, {}),
            "2d_2d": ({1: length}, {0: length}, {}),
        }[layout]
        program = torch.onnx.export(
            Model().eval(),
            (a, b, offsets),
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            optimize=False,
        )
        onnx.checker.check_model(program.model_proto, full_check=True)
        self.assertEqual(
            program.model_proto.graph.output[0].type.tensor_type.elem_type,
            onnx.TensorProto.BFLOAT16,
        )


if __name__ == "__main__":
    unittest.main()
