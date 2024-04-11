# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import operator
import pathlib
import tempfile
import unittest
from typing import Any, Callable

import numpy as np
import onnx
import onnx.external_data_helper
import parameterized

from onnxscript.ir import _core, _enums


class ExternalTensorTest(unittest.TestCase):
    """Test the memory mapped external tensor class."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.external_data_name = "test_model.bin"
        self.base_path = self.temp_dir.name
        self.data = np.random.rand(2, 42).astype(np.float32)
        self.data_float16 = np.random.rand(2, 42).astype(np.float16)
        self.model = self._simple_model_with_external(
            self.base_path, self.external_data_name, self.data
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _simple_model_with_external(
        self, base_path: str, external_data_name: str, data: np.ndarray
    ) -> onnx.ModelProto:
        input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [None])
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [None])
        raw_data = data.tobytes()
        tensor = onnx.helper.make_tensor(
            "input", onnx.TensorProto.FLOAT, data.shape, raw_data, raw=True
        )
        raw_data2 = self.data_float16.tobytes()
        tensor2 = onnx.helper.make_tensor(
            "input2", onnx.TensorProto.FLOAT16, data.shape, raw_data2, raw=True
        )
        onnx.external_data_helper.set_external_data(
            tensor, external_data_name, offset=0, length=len(raw_data)
        )
        onnx.external_data_helper.set_external_data(
            tensor2, external_data_name, offset=len(raw_data), length=len(raw_data2)
        )

        node = onnx.helper.make_node("Identity", inputs=["input"], outputs=["output"])
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [node], "test_graph", [input], [output], initializer=[tensor, tensor2]
            )
        )
        tensor.ClearField("raw_data")
        tensor2.ClearField("raw_data")
        # Save the data to disk
        with open(pathlib.Path(base_path) / external_data_name, "wb") as f:
            f.write(raw_data)
            f.write(raw_data2)
        return model

    def test_initialize(self):
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            path=pathlib.Path(self.base_path) / external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=_enums.DataType.FLOAT,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )
        self.assertEqual(tensor.dtype, _enums.DataType.FLOAT)
        np.testing.assert_equal(tensor, self.data)
        # Ensure repeated reads are consistent
        np.testing.assert_equal(tensor, self.data)

    def test_totypes_returns_correct_data_in(self):
        external_tensor = self.model.graph.initializer[0]
        external_info = onnx.external_data_helper.ExternalDataInfo(external_tensor)
        tensor = _core.ExternalTensor(
            path=pathlib.Path(self.base_path) / external_info.location,
            offset=external_info.offset,
            length=external_info.length,
            dtype=_enums.DataType.FLOAT,
            name="input",
            shape=_core.Shape(external_tensor.dims),
        )
        external_tensor2 = self.model.graph.initializer[1]
        external_info2 = onnx.external_data_helper.ExternalDataInfo(external_tensor2)
        tensor2 = _core.ExternalTensor(
            path=pathlib.Path(self.base_path) / external_info2.location,
            offset=external_info2.offset,
            length=external_info2.length,
            dtype=_enums.DataType.FLOAT16,
            name="input",
            shape=_core.Shape(external_tensor2.dims),
        )
        self.assertEqual(tensor.tobytes(), self.data.tobytes())
        self.assertEqual(tensor2.tobytes(), self.data_float16.tobytes())
        # Ensure repeated reads are consistent
        self.assertEqual(tensor.tobytes(), self.data.tobytes())
        self.assertEqual(tensor2.tobytes(), self.data_float16.tobytes())


class DimensionTest(unittest.TestCase):
    def test_initialize(self):
        dim = _core.Dimension(42, "test")
        self.assertEqual(dim.value, 42)
        self.assertEqual(dim.denotation, "test")

    @parameterized.parameterized.expand([("int", 42), ("str", "any string"), ("None", None)])
    def test_equality_with_other_dimensions(self, _: str, value: Any):
        dim1 = _core.Dimension(value, "test")
        dim2 = _core.Dimension(value, "don't care")
        self.assertEqual(dim1, dim2)

    @parameterized.parameterized.expand([("int", 42), ("str", "any string"), ("None", None)])
    def test_equality_with_python_values(self, _: str, value: Any):
        dim = _core.Dimension(value, "test")
        self.assertEqual(dim, value)
        self.assertIn(value, [dim])
        self.assertIn(dim, [value])

    @parameterized.parameterized.expand([("int", 42), ("str", "any string"), ("None", None)])
    def test_it_is_hashable(self, _: str, value: Any):
        dim = _core.Dimension(value, "test")
        self.assertEqual(hash(dim), hash(value))
        self.assertIn(dim, {dim})
        self.assertIn(dim, {value})

    @parameterized.parameterized.expand(
        [
            ("gt", operator.gt, False),
            ("ge", operator.ge, False),
            ("lt", operator.lt, True),
            ("le", operator.le, True),
        ]
    )
    def test_it_is_comparable(self, _: str, op: Callable, expected: bool):
        dim1 = _core.Dimension(0, "test")
        dim2 = _core.Dimension(42, "test")
        self.assertEqual(op(dim1, dim2), expected)

    @parameterized.parameterized.expand(
        [
            ("gt", operator.gt, False),
            ("ge", operator.ge, False),
            ("lt", operator.lt, True),
            ("le", operator.le, True),
        ]
    )
    def test_it_is_comparable_with_int(self, _: str, op: Callable, expected: bool):
        dim1 = _core.Dimension(0, "test")
        dim2 = 42
        self.assertEqual(op(dim1, dim2), expected)

    @parameterized.parameterized.expand(
        [
            ("gt", operator.gt),
            ("ge", operator.ge),
            ("lt", operator.lt),
            ("le", operator.le),
        ]
    )
    def test_it_raises_type_error_when_compared_with_non_int(self, _: str, op: Callable):
        dim = _core.Dimension(0, "test")
        dim2 = "some string"
        with self.assertRaises(TypeError):
            op(dim, "some string")
        with self.assertRaises(TypeError):
            op(dim, None)
        with self.assertRaises(TypeError):
            op(dim, dim2)


class ShapeTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("empty", (), ()),
            ("1d", (42,), (42,)),
            ("int", (42, 42), (42, 42)),
            ("str", ("any string", "any string"), ("any string", "any string")),
            ("None", (None, None), (None, None)),
        ]
    )
    def test_eq_with_other_shapes(
        self, _: str, dims_1: tuple[Any, ...], dims_2: tuple[Any, ...]
    ):
        shape_1 = _core.Shape(dims_1)
        shape_2 = _core.Shape(dims_2)
        self.assertEqual(shape_1, shape_2)

    @parameterized.parameterized.expand(
        [
            ("empty", ()),
            ("1d", (42,)),
            ("int", (42, 42)),
            ("str", ("any string", "any string")),
            ("None", (None, None)),
        ]
    )
    def test_eq_with_tuple(self, _: str, dims: tuple[Any, ...]):
        shape = _core.Shape(dims)
        self.assertEqual(shape, dims)

    @parameterized.parameterized.expand(
        [
            ("empty", []),
            (
                "1d",
                [
                    42,
                ],
            ),
            ("int", [42, 42]),
            ("str", ["any string", "any string"]),
            ("None", [None, None]),
        ]
    )
    def test_eq_with_list(self, _: str, dims: list[Any]):
        shape = _core.Shape(dims)
        self.assertEqual(shape, dims)

    def test_eq_with_np_shape(self):
        dims = (42,)
        array = np.zeros(dims)
        shape = _core.Shape(dims)
        self.assertEqual(shape, array.shape)

    @parameterized.parameterized.expand(
        [
            ("empty", (), (1,)),
            ("d", (42,), (0,)),
            ("rank", (42, 42), (42, 42, 42)),
            ("str", ("any string",), (42,)),
            ("None", (None, None), (None, 42)),
        ]
    )
    def test_ne_with_other_shapes(
        self, _: str, dims_1: tuple[Any, ...], dims_2: tuple[Any, ...]
    ):
        shape_1 = _core.Shape(dims_1)
        shape_2 = _core.Shape(dims_2)
        self.assertNotEqual(shape_1, shape_2)

    def test_ne_with_random_object(self):
        shape = _core.Shape((42,))
        self.assertNotEqual(shape, 42)


class ValueTest(unittest.TestCase):
    def test_initialize(self):
        _ = _core.Value(None, def_index=0)


class NodeTest(unittest.TestCase):
    def test_initialize_with_values(self):
        v0 = _core.Value(None, def_index=None)
        v1 = _core.Value(None, def_index=None)
        node = _core.Node("test", "TestOp", inputs=(v0, v1), num_outputs=3)
        self.assertEqual(node.domain, "test")
        self.assertEqual(node.op_type, "TestOp")
        self.assertEqual(node.inputs, (v0, v1))
        self.assertEqual(len(node.outputs), 3)
        self.assertEqual(node.attributes, {})


class GraphTest(unittest.TestCase):
    def test_initialize(self):
        v0 = _core.Input(name="v0")
        v1 = _core.Input(name="v1")
        node = _core.Node("", "Add", inputs=(v0, v1), num_outputs=1)
        graph = _core.Graph(
            (v0, v1),
            node.outputs,
            nodes=(node,),
            opset_imports={"": 1},
        )
        self.assertEqual(graph.inputs, [v0, v1])
        self.assertEqual(graph.outputs, [*node.outputs])
        self.assertEqual(graph.opset_imports, {"": 1})
        self.assertEqual(graph.initializers, {})
        self.assertIsNone(graph.doc_string)


if __name__ == "__main__":
    unittest.main()
