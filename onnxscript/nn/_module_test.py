# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import unittest
from typing import Any

import onnx_ir as ir

from onnxscript._internal.builder import GraphBuilder, OpBuilder
from onnxscript.nn import Module, ModuleList, Parameter, Sequential


def _create_graph_and_op() -> tuple[ir.Graph, OpBuilder]:
    """Create an empty graph and its default OpBuilder."""
    graph = ir.Graph(
        name="test_model",
        inputs=[],
        outputs=[],
        nodes=[],
        opset_imports={"": 23},
    )
    builder = GraphBuilder(graph)
    return graph, builder.op


class ParameterTest(unittest.TestCase):
    def test_parameter_repr(self):
        p = Parameter([3, 4], dtype=ir.DataType.FLOAT, name="weight")
        self.assertIn("weight", repr(p))
        self.assertIn("3, 4", repr(p))

    def test_realize_creates_initializer(self):
        graph, op = _create_graph_and_op()
        p = Parameter([3, 4], dtype=ir.DataType.FLOAT, name="weight")
        value = p._realize(op.builder)  # pylint: disable=protected-access

        self.assertIs(value, p)  # _realize returns self
        self.assertIsInstance(value, ir.Value)
        self.assertEqual(value.name, "weight")
        self.assertEqual(value.type.dtype, ir.DataType.FLOAT)
        self.assertEqual(list(value.shape), [3, 4])
        # Should be registered as initializer
        self.assertIn("weight", graph.initializers)

    def test_realize_is_idempotent(self):
        _, op = _create_graph_and_op()
        p = Parameter([3, 4], name="weight")
        v1 = p._realize(op.builder)  # pylint: disable=protected-access
        v2 = p._realize(op.builder)  # pylint: disable=protected-access
        self.assertIs(v1, v2)

    def test_realize_with_data(self):
        graph, op = _create_graph_and_op()
        tensor = ir.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ir.DataType.FLOAT)
        p = Parameter([2, 2], name="weight", data=tensor)
        value = p._realize(op.builder)  # pylint: disable=protected-access

        self.assertIs(value.const_value, tensor)
        self.assertIn("weight", graph.initializers)
        # The initializer in the graph IS the parameter itself
        self.assertIs(graph.initializers["weight"], p)

    def test_realize_qualifies_name(self):
        graph, op = _create_graph_and_op()
        op.builder.push_module("layer1")
        p = Parameter([3], name="bias")
        value = p._realize(op.builder)  # pylint: disable=protected-access
        op.builder.pop_module()

        self.assertEqual(value.name, "layer1.bias")
        self.assertIn("layer1.bias", graph.initializers)


class ModuleBasicTest(unittest.TestCase):
    def test_parameter_auto_registration(self):
        """Parameters assigned as attributes are automatically registered."""

        class MyModule(Module):
            def __init__(self):
                super().__init__("my_mod")
                self.weight = Parameter([3, 4], name="weight")
                self.bias = Parameter([3], name="bias")

            def forward(self, op, x):
                pass

        m = MyModule()
        param_names = list(m._parameters.keys())  # pylint: disable=protected-access
        self.assertEqual(param_names, ["weight", "bias"])

    def test_module_auto_registration(self):
        """Child modules assigned as attributes are automatically registered."""

        class Child(Module):
            def __init__(self):
                super().__init__("child")

            def forward(self, op):
                pass

        class Parent(Module):
            def __init__(self):
                super().__init__("parent")
                self.child = Child()

            def forward(self, op):
                pass

        p = Parent()
        self.assertIn("child", p._modules)  # pylint: disable=protected-access
        self.assertIs(p._modules["child"], p.child)  # pylint: disable=protected-access

    def test_module_inherits_attribute_name(self):
        """Child module with no explicit name inherits the attribute name."""

        class Child(Module):
            def __init__(self):
                super().__init__()  # name=None

            def forward(self, op):
                pass

        class Parent(Module):
            def __init__(self):
                super().__init__("parent")
                self.my_layer = Child()

            def forward(self, op):
                pass

        p = Parent()
        self.assertEqual(p.my_layer._name, "my_layer")  # pylint: disable=protected-access

    def test_parameter_inherits_attribute_name(self):
        """Parameter with no explicit name inherits the attribute name."""

        class MyModule(Module):
            def __init__(self):
                super().__init__("mod")
                self.weight = Parameter([3, 4])  # name=None

            def forward(self, op, x):
                pass

        m = MyModule()
        self.assertEqual(m.weight.name, "weight")

    def test_name_property(self):
        """The name property returns the module name."""

        class MyModule(Module):
            def __init__(self):
                super().__init__("my_mod")

            def forward(self, op):
                pass

        m = MyModule()
        self.assertEqual(m.name, "my_mod")

    def test_setattr_plain_attribute(self):
        """Non-Parameter, non-Module attributes are stored normally."""

        class MyModule(Module):
            def __init__(self):
                super().__init__("mod")
                self.hidden_size = 128

            def forward(self, op):
                pass

        m = MyModule()
        self.assertEqual(m.hidden_size, 128)

    def test_forward_not_implemented(self):
        """Calling forward() on base Module raises NotImplementedError."""
        m = Module("base")
        _, op = _create_graph_and_op()
        with self.assertRaises(NotImplementedError):
            m(op)


class ModuleForwardTest(unittest.TestCase):
    def test_parameter_is_ir_value_in_forward(self):
        """Parameters are ir.Value instances usable directly in forward()."""
        captured: dict[str, Any] = {}

        class MyModule(Module):
            def __init__(self):
                super().__init__("mod")
                self.weight = Parameter([3, 4], name="weight")

            def forward(self, op):
                # self.weight IS a Parameter (which IS an ir.Value)
                captured["is_ir_value"] = isinstance(self.weight, ir.Value)
                captured["is_parameter"] = isinstance(self.weight, Parameter)
                captured["weight_name"] = self.weight.name

        _, op = _create_graph_and_op()
        m = MyModule()
        m(op)

        self.assertTrue(captured["is_ir_value"])
        self.assertTrue(captured["is_parameter"])
        self.assertEqual(captured["weight_name"], "mod.weight")

        # After forward, self.weight is still the same Parameter
        self.assertIsInstance(m.weight, Parameter)

    def test_parameter_naming_with_hierarchy(self):
        """Parameters get hierarchically qualified names."""
        captured_names: list[str] = []

        class Inner(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter([2, 2], name="w")

            def forward(self, op):
                captured_names.append(self.w.name)

        class Outer(Module):
            def __init__(self):
                super().__init__("outer")
                self.layer = Inner()

            def forward(self, op):
                self.layer(op)

        graph, op = _create_graph_and_op()
        m = Outer()
        m(op)

        self.assertEqual(captured_names, ["outer.layer.w"])
        self.assertIn("outer.layer.w", graph.initializers)

    def test_forward_with_ops(self):
        """Module forward can use OpBuilder to create ONNX nodes."""

        class AddBias(Module):
            def __init__(self, size):
                super().__init__("add_bias")
                self.bias = Parameter([size], name="bias")

            def forward(self, op, x):
                return op.Add(x, self.bias)

        graph, op = _create_graph_and_op()
        x = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([3]),
        )
        graph.inputs.append(x)

        m = AddBias(3)
        result = m(op, x)

        self.assertIsInstance(result, ir.Value)
        nodes = list(graph)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].op_type, "Add")

    def test_composition_multiple_submodules(self):
        """Multiple submodules compose correctly with independent parameters."""

        class Linear(Module):
            def __init__(self, in_f, out_f, name=None):
                super().__init__(name)
                self.weight = Parameter([out_f, in_f], name="weight")

            def forward(self, op, x):
                w_t = op.Transpose(self.weight, perm=[1, 0])
                return op.MatMul(x, w_t)

        class TwoLinear(Module):
            def __init__(self):
                super().__init__("model")
                self.fc1 = Linear(4, 3, name="fc1")
                self.fc2 = Linear(3, 2, name="fc2")

            def forward(self, op, x):
                h = self.fc1(op, x)
                return self.fc2(op, h)

        graph, op = _create_graph_and_op()
        x = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 4]),
        )
        graph.inputs.append(x)

        m = TwoLinear()
        result = m(op, x)

        self.assertIsInstance(result, ir.Value)
        # Check initializer names are hierarchical
        self.assertIn("model.fc1.weight", graph.initializers)
        self.assertIn("model.fc2.weight", graph.initializers)
        # Check nodes: Transpose, MatMul, Transpose, MatMul
        op_types = [node.op_type for node in graph]
        self.assertEqual(op_types, ["Transpose", "MatMul", "Transpose", "MatMul"])


class ModuleIteratorTest(unittest.TestCase):
    def test_parameters_iterator(self):
        class MyModule(Module):
            def __init__(self):
                super().__init__("mod")
                self.w1 = Parameter([3], name="w1")
                self.w2 = Parameter([4], name="w2")

            def forward(self, op):
                pass

        m = MyModule()
        params = list(m.parameters())
        self.assertEqual(len(params), 2)

    def test_parameters_recursive(self):
        class Child(Module):
            def __init__(self):
                super().__init__("child")
                self.w = Parameter([3], name="w")

            def forward(self, op):
                pass

        class Parent(Module):
            def __init__(self):
                super().__init__("parent")
                self.p = Parameter([2], name="p")
                self.child = Child()

            def forward(self, op):
                pass

        m = Parent()
        params = list(m.parameters(recurse=True))
        self.assertEqual(len(params), 2)
        params_no_recurse = list(m.parameters(recurse=False))
        self.assertEqual(len(params_no_recurse), 1)

    def test_named_parameters_recursive(self):
        class Child(Module):
            def __init__(self):
                super().__init__("child")
                self.w = Parameter([3], name="w")

            def forward(self, op):
                pass

        class Parent(Module):
            def __init__(self):
                super().__init__("parent")
                self.p = Parameter([2], name="p")
                self.child = Child()

            def forward(self, op):
                pass

        m = Parent()
        named = dict(m.named_parameters())
        self.assertIn("p", named)
        self.assertIn("child.w", named)

    def test_modules_iterator(self):
        class A(Module):
            def __init__(self):
                super().__init__("a")

            def forward(self, op):
                pass

        class B(Module):
            def __init__(self):
                super().__init__("b")
                self.a = A()

            def forward(self, op):
                pass

        m = B()
        mods = list(m.modules())
        self.assertEqual(len(mods), 2)
        self.assertIs(mods[0], m)
        self.assertIs(mods[1], m.a)

    def test_named_modules_iterator(self):
        class A(Module):
            def __init__(self):
                super().__init__("a")

            def forward(self, op):
                pass

        class B(Module):
            def __init__(self):
                super().__init__("b")
                self.a = A()

            def forward(self, op):
                pass

        m = B()
        named = dict(m.named_modules())
        self.assertIn("", named)  # self
        self.assertIn("a", named)


class ModuleReprTest(unittest.TestCase):
    def test_repr(self):
        class Child(Module):
            def __init__(self):
                super().__init__("child")
                self.w = Parameter([3], name="w")

            def forward(self, op):
                pass

        class Parent(Module):
            def __init__(self):
                super().__init__("parent")
                self.child = Child()

            def forward(self, op):
                pass

        m = Parent()
        r = repr(m)
        self.assertIn("Parent", r)
        self.assertIn("child", r)
        self.assertIn("Child", r)


class ModuleChildrenTest(unittest.TestCase):
    def test_children(self):
        class A(Module):
            def __init__(self):
                super().__init__("a")

            def forward(self, op):
                pass

        class B(Module):
            def __init__(self):
                super().__init__("b")

            def forward(self, op):
                pass

        class Parent(Module):
            def __init__(self):
                super().__init__("parent")
                self.a = A()
                self.b = B()

            def forward(self, op):
                pass

        m = Parent()
        kids = list(m.children())
        self.assertEqual(len(kids), 2)
        self.assertIs(kids[0], m.a)
        self.assertIs(kids[1], m.b)

    def test_named_children(self):
        class A(Module):
            def __init__(self):
                super().__init__("a")

            def forward(self, op):
                pass

        class Parent(Module):
            def __init__(self):
                super().__init__("parent")
                self.layer = A()

            def forward(self, op):
                pass

        m = Parent()
        named = dict(m.named_children())
        self.assertIn("layer", named)
        self.assertIs(named["layer"], m.layer)

    def test_children_does_not_recurse(self):
        """children() only yields immediate children, not grandchildren."""

        class Grandchild(Module):
            def __init__(self):
                super().__init__("gc")

            def forward(self, op):
                pass

        class Child(Module):
            def __init__(self):
                super().__init__("child")
                self.gc = Grandchild()

            def forward(self, op):
                pass

        class Parent(Module):
            def __init__(self):
                super().__init__("parent")
                self.child = Child()

            def forward(self, op):
                pass

        m = Parent()
        kids = list(m.children())
        self.assertEqual(len(kids), 1)
        self.assertIs(kids[0], m.child)


class ModuleStateDictTest(unittest.TestCase):
    def test_state_dict_flat(self):
        class MyModule(Module):
            def __init__(self):
                super().__init__("mod")
                self.w = Parameter([3], name="w")
                self.b = Parameter([3], name="b")

            def forward(self, op):
                pass

        m = MyModule()
        sd = m.state_dict()
        self.assertIn("w", sd)
        self.assertIn("b", sd)
        # Uninitialized parameters have None data
        self.assertIsNone(sd["w"])
        self.assertIsNone(sd["b"])

    def test_state_dict_hierarchical(self):
        class Child(Module):
            def __init__(self):
                super().__init__("child")
                self.w = Parameter([3], name="w")

            def forward(self, op):
                pass

        class Parent(Module):
            def __init__(self):
                super().__init__("parent")
                self.p = Parameter([2], name="p")
                self.child = Child()

            def forward(self, op):
                pass

        m = Parent()
        sd = m.state_dict()
        self.assertIn("p", sd)
        self.assertIn("child.w", sd)

    def test_state_dict_with_data(self):
        tensor = ir.tensor([1.0, 2.0, 3.0], dtype=ir.DataType.FLOAT)
        p = Parameter([3], name="w", data=tensor)

        class MyModule(Module):
            def __init__(self):
                super().__init__("mod")
                self.w = p

            def forward(self, op):
                pass

        m = MyModule()
        sd = m.state_dict()
        self.assertIs(sd["w"], tensor)

    def test_load_state_dict(self):
        class MyModule(Module):
            def __init__(self):
                super().__init__("mod")
                self.w = Parameter([3], name="w")
                self.b = Parameter([3], name="b")

            def forward(self, op):
                pass

        m = MyModule()
        w_data = ir.tensor([1.0, 2.0, 3.0], dtype=ir.DataType.FLOAT)
        b_data = ir.tensor([0.1, 0.2, 0.3], dtype=ir.DataType.FLOAT)
        m.load_state_dict({"w": w_data, "b": b_data})

        self.assertIs(m.w.const_value, w_data)
        self.assertIs(m.b.const_value, b_data)

    def test_load_state_dict_hierarchical(self):
        class Child(Module):
            def __init__(self):
                super().__init__("child")
                self.w = Parameter([3], name="w")

            def forward(self, op):
                pass

        class Parent(Module):
            def __init__(self):
                super().__init__("parent")
                self.child = Child()

            def forward(self, op):
                pass

        m = Parent()
        w_data = ir.tensor([1.0, 2.0, 3.0], dtype=ir.DataType.FLOAT)
        m.load_state_dict({"child.w": w_data})
        self.assertIs(m.child.w.const_value, w_data)

    def test_load_state_dict_strict_missing_key(self):
        class MyModule(Module):
            def __init__(self):
                super().__init__("mod")
                self.w = Parameter([3], name="w")

            def forward(self, op):
                pass

        m = MyModule()
        with self.assertRaises(KeyError):
            m.load_state_dict({})

    def test_load_state_dict_strict_unexpected_key(self):
        class MyModule(Module):
            def __init__(self):
                super().__init__("mod")
                self.w = Parameter([3], name="w")

            def forward(self, op):
                pass

        m = MyModule()
        w_data = ir.tensor([1.0, 2.0, 3.0], dtype=ir.DataType.FLOAT)
        with self.assertRaises(ValueError):
            m.load_state_dict({"w": w_data, "extra": w_data})

    def test_load_state_dict_non_strict(self):
        class MyModule(Module):
            def __init__(self):
                super().__init__("mod")
                self.w = Parameter([3], name="w")
                self.b = Parameter([3], name="b")

            def forward(self, op):
                pass

        m = MyModule()
        w_data = ir.tensor([1.0, 2.0, 3.0], dtype=ir.DataType.FLOAT)
        # Only load w, skip b — no error because strict=False
        m.load_state_dict({"w": w_data}, strict=False)
        self.assertIs(m.w.const_value, w_data)
        self.assertIsNone(m.b.const_value)


class ModuleListTest(unittest.TestCase):
    def test_basic_indexing(self):
        """ModuleList supports integer indexing."""

        class Layer(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter([3], name="w")

            def forward(self, op):
                pass

        ml = ModuleList([Layer(), Layer(), Layer()])
        self.assertEqual(len(ml), 3)
        self.assertIsInstance(ml[0], Layer)
        self.assertIsInstance(ml[2], Layer)
        self.assertIsInstance(ml[-1], Layer)
        self.assertIs(ml[-1], ml[2])

    def test_index_out_of_range(self):
        ml = ModuleList()
        with self.assertRaises(IndexError):
            _ = ml[0]

    def test_append(self):
        class Layer(Module):
            def __init__(self):
                super().__init__()

            def forward(self, op):
                pass

        ml = ModuleList()
        ml.append(Layer())
        ml.append(Layer())
        self.assertEqual(len(ml), 2)

    def test_extend(self):
        class Layer(Module):
            def __init__(self):
                super().__init__()

            def forward(self, op):
                pass

        ml = ModuleList()
        ml.extend([Layer(), Layer()])
        self.assertEqual(len(ml), 2)

    def test_iteration(self):
        class Layer(Module):
            def __init__(self):
                super().__init__()

            def forward(self, op):
                pass

        layers = [Layer(), Layer(), Layer()]
        ml = ModuleList(layers)
        for i, layer in enumerate(ml):
            self.assertIs(layer, layers[i])

    def test_slice(self):
        class Layer(Module):
            def __init__(self):
                super().__init__()

            def forward(self, op):
                pass

        ml = ModuleList([Layer(), Layer(), Layer()])
        sub = ml[1:]
        self.assertIsInstance(sub, ModuleList)
        self.assertEqual(len(sub), 2)

    def test_children_auto_named(self):
        """Children get '0', '1', ... names automatically."""

        class Layer(Module):
            def __init__(self):
                super().__init__()

            def forward(self, op):
                pass

        ml = ModuleList([Layer(), Layer()])
        self.assertEqual(ml[0]._name, "0")  # pylint: disable=protected-access
        self.assertEqual(ml[1]._name, "1")  # pylint: disable=protected-access

    def test_named_parameters_with_numeric_keys(self):
        """Parameters within ModuleList children use numeric-indexed names."""

        class Layer(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter([3], name="w")

            def forward(self, op):
                pass

        ml = ModuleList([Layer(), Layer()])
        named = dict(ml.named_parameters())
        self.assertIn("0.w", named)
        self.assertIn("1.w", named)

    def test_same_class_submodules_get_distinct_names_in_graph(self):
        """Multiple sub-modules of the same class in a ModuleList get distinct
        .0., .1. prefixed initializer and node names in the graph.
        """

        class Linear(Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.weight = Parameter([out_f, in_f], name="weight")

            def forward(self, op, x):
                w_t = op.Transpose(self.weight, perm=[1, 0])
                return op.MatMul(x, w_t)

        class Model(Module):
            def __init__(self):
                super().__init__("model")
                self.layers = ModuleList([Linear(4, 4), Linear(4, 4), Linear(4, 4)])

            def forward(self, op, x):
                for layer in self.layers:
                    x = layer(op, x)
                return x

        graph, op = _create_graph_and_op()
        x = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 4]),
        )
        graph.inputs.append(x)

        m = Model()
        result = m(op, x)

        self.assertIsInstance(result, ir.Value)
        # Each layer's weight must have a distinct hierarchical name
        self.assertIn("model.layers.0.weight", graph.initializers)
        self.assertIn("model.layers.1.weight", graph.initializers)
        self.assertIn("model.layers.2.weight", graph.initializers)
        # All three must be different ir.Value objects
        self.assertIsNot(
            graph.initializers["model.layers.0.weight"],
            graph.initializers["model.layers.1.weight"],
        )
        self.assertIsNot(
            graph.initializers["model.layers.1.weight"],
            graph.initializers["model.layers.2.weight"],
        )
        # Check that we got 6 nodes: (Transpose + MatMul) * 3 layers
        op_types = [node.op_type for node in graph]
        self.assertEqual(op_types, ["Transpose", "MatMul"] * 3)

    def test_modulelist_not_directly_callable(self):
        ml = ModuleList()
        _, op = _create_graph_and_op()
        with self.assertRaises(NotImplementedError):
            ml(op)


class SequentialTest(unittest.TestCase):
    def test_sequential_chains_forward_calls(self):
        """Sequential calls children in order, passing output to next."""

        class AddOne(Module):
            def __init__(self):
                super().__init__()

            def forward(self, op, x):
                return op.Add(x, op.Constant(value_float=1.0))

        graph, op = _create_graph_and_op()
        x = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([3]),
        )
        graph.inputs.append(x)

        seq = Sequential([AddOne(), AddOne()])
        result = seq(op, x)

        self.assertIsInstance(result, ir.Value)
        op_types = [node.op_type for node in graph]
        # Two Add ops (one Constant + Add per child)
        self.assertEqual(op_types.count("Add"), 2)

    def test_sequential_parameter_naming(self):
        """Sequential produces numeric-indexed parameter names like ModuleList."""

        class Linear(Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.weight = Parameter([out_f, in_f], name="weight")

            def forward(self, op, x):
                return op.MatMul(x, op.Transpose(self.weight, perm=[1, 0]))

        class Model(Module):
            def __init__(self):
                super().__init__("model")
                self.layers = Sequential([Linear(4, 4), Linear(4, 4)])

            def forward(self, op, x):
                return self.layers(op, x)

        graph, op = _create_graph_and_op()
        x = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 4]),
        )
        graph.inputs.append(x)

        m = Model()
        m(op, x)

        self.assertIn("model.layers.0.weight", graph.initializers)
        self.assertIn("model.layers.1.weight", graph.initializers)

    def test_sequential_with_parameterless_modules(self):
        """Sequential works with mixed param/no-param children (like SiLU + Linear)."""

        class SiLU(Module):
            def forward(self, op, x):
                return op.Mul(x, op.Sigmoid(x))

        class Linear(Module):
            def __init__(self, size):
                super().__init__()
                self.weight = Parameter([size, size], name="weight")

            def forward(self, op, x):
                return op.MatMul(x, op.Transpose(self.weight, perm=[1, 0]))

        seq = Sequential([SiLU(), Linear(4)])
        named = dict(seq.named_parameters())
        # SiLU at index 0 has no params; Linear at index 1 has weight
        self.assertIn("1.weight", named)
        self.assertEqual(len(named), 1)

    def test_sequential_empty_raises(self):
        """Empty Sequential raises RuntimeError on forward."""

        seq = Sequential()
        _, op = _create_graph_and_op()
        with self.assertRaises(RuntimeError):
            seq(op, None)

    def test_sequential_append_produces_correct_initializer_names(self):
        """Sequential with append (after parent registration) gets correct names.

        This tests the pattern where a Sequential is created empty, registered
        on a parent Module, and then children are appended. The children should
        produce initializer names like ``parent.seq.0.weight``, not
        ``parent.seq.seq.0.weight`` (double-prefixed).
        """

        class Linear(Module):
            def __init__(self, size):
                super().__init__()
                self.weight = Parameter([size, size], name="weight")

            def forward(self, op, x):
                return op.MatMul(x, op.Transpose(self.weight, perm=[1, 0]))

        class Model(Module):
            def __init__(self):
                super().__init__("model")
                self.blocks = Sequential([])
                # Append AFTER __setattr__ has set Sequential._name = "blocks"
                self.blocks.append(Linear(4))
                self.blocks.append(Linear(4))

            def forward(self, op, x):
                return self.blocks(op, x)

        graph, op = _create_graph_and_op()
        x = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 4]),
        )
        graph.inputs.append(x)

        m = Model()
        m(op, x)

        self.assertIn("model.blocks.0.weight", graph.initializers)
        self.assertIn("model.blocks.1.weight", graph.initializers)
        self.assertEqual(len(graph.initializers), 2)

    def test_modulelist_append_produces_correct_initializer_names(self):
        """ModuleList with append after parent registration gets correct names.

        Tests the interleaved ModuleList pattern (like a mid_block with separate
        resnets and attentions lists). Children appended after parent registration
        should produce names like ``parent.resnets.1.weight``.
        """

        class Linear(Module):
            def __init__(self, size):
                super().__init__()
                self.weight = Parameter([size, size], name="weight")

            def forward(self, op, x):
                return op.MatMul(x, op.Transpose(self.weight, perm=[1, 0]))

        class MidBlock(Module):
            def __init__(self):
                super().__init__()
                self.resnets = ModuleList([Linear(4)])
                self.attentions = ModuleList([])
                # Appends after parent has set _name on the ModuleLists
                self.attentions.append(Linear(4))
                self.resnets.append(Linear(4))

            def forward(self, op, x):
                x = self.resnets[0](op, x)
                for i in range(len(self.attentions)):
                    x = self.attentions[i](op, x)
                    x = self.resnets[i + 1](op, x)
                return x

        class Model(Module):
            def __init__(self):
                super().__init__("model")
                self.mid = MidBlock()

            def forward(self, op, x):
                return self.mid(op, x)

        graph, op = _create_graph_and_op()
        x = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1, 4]),
        )
        graph.inputs.append(x)

        m = Model()
        m(op, x)

        expected = {
            "model.mid.resnets.0.weight",
            "model.mid.resnets.1.weight",
            "model.mid.attentions.0.weight",
        }
        self.assertEqual(set(graph.initializers.keys()), expected)


if __name__ == "__main__":
    unittest.main()
