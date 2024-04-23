# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import pathlib
import tempfile
import unittest
from typing import Any

import numpy as np
import onnx
import onnx.external_data_helper
import parameterized
import torch

from onnxscript.ir import _core, _enums


class TensorTest(unittest.TestCase):
    def test_initialize(self):
        tensor = _core.Tensor(
            np.random.rand(1, 2).astype(np.float32),
            dtype=_enums.DataType.FLOAT,
            shape=_core.Shape((1, 2)),
            name="test",
        )
        self.assertEqual(tensor.name, "test")
        self.assertEqual(tensor.dtype, _enums.DataType.FLOAT)
        self.assertEqual(tensor.shape, _core.Shape((1, 2)))
        np.testing.assert_array_equal(tensor, tensor)

    def test_init_raises_when_value_is_not_array(self):
        with self.assertRaises(TypeError):
            _core.Tensor(42)

    def test_init_requires_type_when_value_is_not_np_array(self):
        torch_tensor = torch.tensor(42)
        with self.assertRaises(ValueError):
            _core.Tensor(torch_tensor)

    def test_init_respects_dtype_when_it_is_provided(self):
        array = np.random.rand(1, 2).astype(np.int8)
        tensor = _core.Tensor(array, dtype=_enums.DataType.UINT4)
        self.assertEqual(tensor.dtype, _enums.DataType.UINT4)

    def test_initialize_with_just_np_array(self):
        array = np.random.rand(1, 2)
        tensor = _core.Tensor(array)
        np.testing.assert_array_equal(tensor, array)

    def test_initialize_with_torch_tensor(self):
        array = np.random.rand(1, 2).astype(np.int64)
        np_tensor = _core.Tensor(array)
        torch_tensor = _core.Tensor(torch.tensor(array), dtype=_enums.DataType.INT64)
        np.testing.assert_array_equal(torch_tensor, array)
        np.testing.assert_array_equal(torch_tensor, np_tensor)

    def test_dlpack_np_to_torch(self):
        array = np.random.rand(1, 2).astype(np.float32)
        tensor = _core.Tensor(array)
        torch_tensor = torch.from_dlpack(tensor)
        np.testing.assert_array_equal(torch_tensor, array)

    def test_dlpack_torch_to_np(self):
        torch_tensor = torch.rand(1, 2)
        tensor = _core.Tensor(torch_tensor, dtype=_enums.DataType.FLOAT)
        array = np.from_dlpack(tensor)
        np.testing.assert_array_equal(array, torch_tensor)

    def test_repr(self):
        tensor = _core.Tensor(np.random.rand(1, 2).astype(np.float32))
        self.assertIsInstance(repr(tensor), str)

    def test_dtype_returns_data_type_enum(self):
        tensor = _core.Tensor(np.random.rand(1, 2).astype(np.float32))
        self.assertEqual(tensor.dtype, _enums.DataType.FLOAT)

    def test_shape(self):
        tensor = _core.Tensor(np.random.rand(1, 2).astype(np.float32))
        self.assertEqual(tensor.shape, _core.Shape((1, 2)))

    def test_numpy_returns_np_array(self):
        array = np.random.rand(1, 2).astype(np.float32)
        tensor = _core.Tensor(array)
        np.testing.assert_equal(tensor.numpy(), array)

    def test_numpy_returns_data_when_dtype_is_not_supported(self):
        array = np.array([1], dtype=np.int8)
        tensor = _core.Tensor(array, dtype=_enums.DataType.INT4)
        np.testing.assert_equal(tensor.numpy(), array)

    def test_tobytes(self):
        array = np.random.rand(1, 2).astype(np.float32)
        torch_tensor = torch.tensor(array)
        tensor = _core.Tensor(torch_tensor, dtype=_enums.DataType.FLOAT)
        self.assertEqual(tensor.tobytes(), array.tobytes())

    def test_metadata(self):
        array = np.random.rand(1, 2).astype(np.float32)
        tensor = _core.Tensor(array)
        tensor.meta["test"] = 1
        self.assertEqual(tensor.meta["test"], 1)
        tensor.metadata_props["test"] = "any string"
        self.assertEqual(tensor.metadata_props["test"], "any string")


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


class SymbolicDimTest(unittest.TestCase):
    def test_init_raises_when_value_is_int(self):
        # Static dimensions should be python integers
        with self.assertRaises(TypeError):
            _core.SymbolicDim(42)

    @parameterized.parameterized.expand([("str", "any string"), ("None", None)])
    def test_equality_with_other_dimensions(self, _: str, value: Any):
        dim1 = _core.SymbolicDim(value)
        dim2 = _core.SymbolicDim(value)
        self.assertEqual(dim1, dim2)

    @parameterized.parameterized.expand([("str", "any string"), ("None", None)])
    def test_equality_with_python_values(self, _: str, value: Any):
        dim = _core.SymbolicDim(value)
        self.assertEqual(dim, value)
        self.assertIn(value, [dim])
        self.assertIn(dim, [value])

    @parameterized.parameterized.expand([("str", "any string"), ("None", None)])
    def test_it_is_hashable(self, _: str, value: Any):
        dim = _core.SymbolicDim(value)
        self.assertEqual(hash(dim), hash(value))
        self.assertIn(dim, {dim})
        self.assertIn(dim, {value})


class ShapeTest(unittest.TestCase):
    def test_init_raises_when_denotations_and_dims_have_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "denotations"):
            _core.Shape([42], ["DATA_CHANNEL", "BATCH"])

    def test_int_dimensions_are_python_ints(self):
        shape = _core.Shape([42])
        self.assertIsInstance(shape[0], int)

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

    def test_setitem_raises_when_shape_is_frozen(self):
        shape = _core.Shape([42], denotations=("DATA_CHANNEL",), frozen=True)
        with self.assertRaisesRegex(TypeError, "frozen"):
            shape[0] = 1

    def test_getitem(self):
        shape = _core.Shape([42], denotations=("DATA_CHANNEL",))
        self.assertEqual(shape[0], 42)

    def test_getitem_accepts_a_slice(self):
        shape = _core.Shape([1, 2, 3, 4])
        self.assertEqual(shape[1:3], (2, 3))

    @parameterized.parameterized.expand(
        [
            ("int", 42),
            ("str", "any string"),
            ("None", None),
            ("SymbolicDim", _core.SymbolicDim("any string")),
        ]
    )
    def test_setitem(self, _: str, value):
        shape = _core.Shape([0])
        shape[0] = value
        dim = shape[0]
        if isinstance(dim, _core.SymbolicDim):
            self.assertEqual(dim.value, value)
        else:
            self.assertEqual(dim, value)

    def test_get_denotation(self):
        shape = _core.Shape([42], denotations=("DATA_CHANNEL",))
        self.assertEqual(shape.get_denotation(0), "DATA_CHANNEL")

    def test_set_denotation(self):
        shape = _core.Shape([42, 0], ["DATA_CHANNEL", "BATCH"])
        shape.set_denotation(1, "UPDATED")
        self.assertEqual(shape.get_denotation(1), "UPDATED")

    def test_set_denotation_is_still_possible_when_shape_is_frozen(self):
        shape = _core.Shape([42], denotations=("DATA_CHANNEL",), frozen=True)
        shape.set_denotation(0, "UPDATED")
        self.assertEqual(shape.get_denotation(0), "UPDATED")


class ValueTest(unittest.TestCase):
    def test_initialize(self):
        _ = _core.Value(None, index=0)

    def test_meta(self):
        value = _core.Value(None, index=0)
        value.meta["test"] = 1
        self.assertEqual(value.meta["test"], 1)
        value.metadata_props["test"] = "any string"
        self.assertEqual(value.metadata_props["test"], "any string")

    # TODO(justinchuby): Test all methods


class NodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.v0 = _core.Value(None, index=None)
        self.v1 = _core.Value(None, index=None)
        self.node = _core.Node("test", "TestOp", inputs=(self.v0, self.v1), num_outputs=3)

    def test_initialize_with_values(self):
        self.assertEqual(self.node.domain, "test")
        self.assertEqual(self.node.op_type, "TestOp")
        self.assertEqual(self.node.inputs, (self.v0, self.v1))
        self.assertEqual(len(self.node.outputs), 3)
        self.assertEqual(self.node.attributes, {})

    def test_metadata(self):
        self.node.meta["test"] = 1
        self.assertEqual(self.node.meta["test"], 1)
        self.node.metadata_props["test"] = "any string"
        self.assertEqual(self.node.metadata_props["test"], "any string")

    def test_it_is_added_to_a_graph_if_specified(self):
        graph = _core.Graph(
            (self.v0, self.v1),  # type: ignore
            self.node.outputs,
            nodes=(self.node,),
            opset_imports={"": 1},
        )
        self.assertIn(self.node, graph)

    # TODO(justinchuby): Test all methods


class GraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.v0 = _core.Input(name="v0")
        self.v1 = _core.Input(name="v1")
        self.node = _core.Node("", "Add", inputs=(self.v0, self.v1), num_outputs=1)
        self.graph = _core.Graph(
            (self.v0, self.v1),
            self.node.outputs,
            nodes=(self.node,),
            opset_imports={"": 1},
        )

    def test_initialize(self):
        self.assertEqual(self.graph.inputs, [self.v0, self.v1])
        self.assertEqual(self.graph.outputs, [*self.node.outputs])
        self.assertEqual(self.graph.opset_imports, {"": 1})
        self.assertEqual(self.graph.initializers, {})
        self.assertIsNone(self.graph.doc_string)

    def test_it_is_iterable_of_nodes(self):
        self.assertEqual(list(self.graph), [self.node])

    def test_metadata(self):
        self.graph.meta["test"] = 1
        self.assertEqual(self.graph.meta["test"], 1)
        self.graph.metadata_props["test"] = "any string"
        self.assertEqual(self.graph.metadata_props["test"], "any string")

    def test_remove_removes_node_from_graph(self):
        self.graph.remove(self.node)
        self.assertEqual(list(self.graph), [])
        self.assertIsNone(self.node.graph)

    def test_remove_does_not_change_input_users(self):
        self.graph.remove(self.node)
        self.assertEqual(tuple(self.v0.consumers()), ((self.node, 0),))
        self.assertEqual(tuple(self.v1.consumers()), ((self.node, 1),))

    def test_remove_does_not_change_graph_in_out(self):
        self.graph.remove(self.node)
        self.assertEqual(self.graph.inputs, [self.v0, self.v1])
        self.assertEqual(self.graph.outputs, list(self.node.outputs))

    def test_remove_raises_when_node_does_not_belong_to_graph(self):
        node = _core.Node("", "Add", inputs=(self.v0, self.v1), num_outputs=1)
        with self.assertRaisesRegex(ValueError, "graph"):
            self.graph.remove(node)

    def test_remove_safe_raises_when_node_output_is_graph_output(self):
        with self.assertRaisesRegex(ValueError, "output"):
            self.graph.remove(self.node, safe=True)

    def test_remove_safe_raises_when_node_has_users(self):
        v0 = _core.Input(name="v0")
        v1 = _core.Input(name="v1")
        add_node = _core.Node("", "Add", inputs=(v0, v1), num_outputs=1)
        identity_node = _core.Node("", "Identity", inputs=add_node.outputs, num_outputs=1)
        graph = _core.Graph(
            (v0, v1),
            identity_node.outputs,
            nodes=(add_node, identity_node),
            opset_imports={"": 1},
        )
        with self.assertRaisesRegex(ValueError, "used by other nodes"):
            graph.remove(add_node, safe=True)

    def test_remove_safe_removes_uses_of_removed_nodes(self):
        v0 = _core.Input(name="v0")
        v1 = _core.Input(name="v1")
        add_node = _core.Node("", "Add", inputs=(v0, v1), num_outputs=1)
        identity_node = _core.Node("", "Identity", inputs=add_node.outputs, num_outputs=1)
        graph = _core.Graph(
            (v0, v1),
            identity_node.outputs,
            nodes=(add_node, identity_node),
            opset_imports={"": 1},
        )
        # Remove add_node and check that it is no longer a consumer of v0 and v1
        sub_node = _core.Node("", "Sub", inputs=(v0, v1), num_outputs=1)
        identity_node.replace_input_with(0, sub_node.outputs[0])
        graph.insert_before(identity_node, sub_node)
        graph.remove(add_node, safe=True)
        self.assertEqual(tuple(v0.consumers()), ((sub_node, 0),))
        self.assertEqual(tuple(v1.consumers()), ((sub_node, 1),))
        self.assertEqual(tuple(graph), (sub_node, identity_node))
        self.assertEqual(add_node.inputs, (None, None))

    # TODO(justinchuby): Test graph mutation methods


if __name__ == "__main__":
    unittest.main()
