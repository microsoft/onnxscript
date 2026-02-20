# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import unittest
from typing import Sequence

import onnx_ir as ir

import onnxscript._internal.builder as builder
from onnxscript import opset23 as op23, script


_default_opset_version = 23


def _build(
    trace_function,
    input_types: Sequence[ir.TypeAndShape],
    output_types: Sequence[ir.TypeAndShape],
) -> ir.Model:
    graph = ir.Graph(
        name="test_model",
        inputs=[],
        outputs=[],
        nodes=[],
        opset_imports={"": _default_opset_version},
    )

    onnx_model = ir.Model(graph=graph, ir_version=10)

    for i, input_type in enumerate(input_types):
        input_name = f"input_{i}"
        graph.inputs.append(ir.Value(name=input_name, type=input_type))

    graph_builder = builder.GraphBuilder(graph)
    outputs = trace_function(graph_builder.op, *graph.inputs)
    if not isinstance(outputs, Sequence):
        outputs = [outputs]
    if len(outputs) != len(output_types):
        raise ValueError(f"Expected {len(output_types)} outputs, but got {len(outputs)}.")
    for output, output_type in zip(outputs, output_types):
        output.type = output_type.type  # TODO: need merge_type method in ir.Value
        output.merge_shapes(output_type.shape)

    graph.outputs.extend(outputs)

    return onnx_model


def _create_builder_with_inputs() -> tuple[builder.OpBuilder, ir.Value, ir.Value]:
    """Create a graph builder with two float tensor inputs (shape [2, 3, 4]).

    Returns:
        A tuple of (op_builder, input_x, input_y).
    """
    graph = ir.Graph(
        name="test_model",
        inputs=[],
        outputs=[],
        nodes=[],
        opset_imports={"": 23},
    )

    for i in range(2):
        input_name = f"input_{i}"
        graph.inputs.append(
            ir.Value(
                name=input_name,
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape([2, 3, 4]),
            )
        )

    graph_builder = builder.GraphBuilder(graph)
    x, y = graph.inputs
    return graph_builder.op, x, y


class GraphBuilderTest(unittest.TestCase):
    def test_builder_basic(self):
        def _add_mul_add(op: builder.OpBuilder, x: ir.Value, y: ir.Value) -> ir.Value:
            t1 = op.Add(x, y)
            t2 = op.Mul(x, y)
            z = op.Add(t1, t2)
            return z

        float_2d = ir.TypeAndShape(ir.TensorType(ir.DataType.FLOAT), ir.Shape([3, 4]))
        model = _build(
            _add_mul_add,
            input_types=[float_2d, float_2d],
            output_types=[float_2d],
        )
        graph = model.graph
        # Expect exactly 3 nodes: Add, Mul, Add
        op_types = [node.op_type for node in graph]
        self.assertEqual(op_types, ["Add", "Mul", "Add"])

        # Verify inputs and outputs
        self.assertEqual(len(graph.inputs), 2)
        self.assertEqual(len(graph.outputs), 1)

        # Verify the connectivity: final Add takes outputs of the first Add and Mul
        nodes = list(graph)
        add1, mul, add2 = nodes
        self.assertEqual(list(add2.inputs), [add1.outputs[0], mul.outputs[0]])

    def test_value_naming(self):
        """Test that output names can be specified via the _outputs option."""

        def _add_with_custom_names(
            op: builder.OpBuilder, x: ir.Value, y: ir.Value
        ) -> ir.Value:
            # Specify custom names for output values
            t1 = op.Add(x, y, _outputs=["add_result"])
            t2 = op.Mul(x, y, _outputs=["mul_result"])
            z = op.Add(t1, t2, _outputs=["final_add"])
            return z

        float_2d = ir.TypeAndShape(ir.TensorType(ir.DataType.FLOAT), ir.Shape([3, 4]))
        model = _build(
            _add_with_custom_names,
            input_types=[float_2d, float_2d],
            output_types=[float_2d],
        )
        graph = model.graph

        # Verify that the nodes have outputs with the specified names
        nodes = list(graph)
        self.assertEqual(len(nodes), 3)

        # Check output names
        self.assertEqual(nodes[0].outputs[0].name, "add_result")
        self.assertEqual(nodes[1].outputs[0].name, "mul_result")
        self.assertEqual(nodes[2].outputs[0].name, "final_add")

        # Verify the final output has the correct name
        self.assertEqual(len(graph.outputs), 1)
        self.assertEqual(graph.outputs[0].name, "final_add")

    def test_value_naming_with_hierarchy(self):
        """Test that hierarchical naming works with user-specified output names."""
        op, x, y = _create_builder_with_inputs()

        # Test custom names at root level
        t1 = op.Add(x, y, _outputs=["my_add"])
        self.assertEqual(t1.name, "my_add")

        # Test custom names with hierarchical context
        op.builder.push_module("layer1")
        t2 = op.Mul(t1, y, _outputs=["my_mul"])
        self.assertEqual(t2.name, "layer1.my_mul")

        # Test nested hierarchical context with custom names
        op.builder.push_module("attention")
        t3 = op.Add(t2, x, _outputs=["my_nested_add"])
        self.assertEqual(t3.name, "layer1.attention.my_nested_add")

        # Pop back and verify prefix is applied correctly
        op.builder.pop_module()
        t4 = op.Mul(t3, y, _outputs=["another_mul"])
        self.assertEqual(t4.name, "layer1.another_mul")

        op.builder.pop_module()
        t5 = op.Add(t4, x, _outputs=["final_result"])
        self.assertEqual(t5.name, "final_result")

    def test_value_naming_with_ir_value_objects(self):
        """Test that hierarchical naming works when passing ir.Value objects as _outputs."""
        op, x, y = _create_builder_with_inputs()

        # Create pre-named ir.Value objects
        out1 = ir.Value(name="my_output")
        out2 = ir.Value(name="layer_output")
        out3 = ir.Value(name="nested_output")

        # Test at root level
        t1 = op.Add(x, y, _outputs=[out1])
        self.assertEqual(t1.name, "my_output")
        self.assertIs(t1, out1)

        # Test with hierarchical context
        op.builder.push_module("layer1")
        t2 = op.Mul(t1, y, _outputs=[out2])
        self.assertEqual(t2.name, "layer1.layer_output")
        self.assertIs(t2, out2)

        # Test nested hierarchical context
        op.builder.push_module("attention")
        t3 = op.Add(t2, x, _outputs=[out3])
        self.assertEqual(t3.name, "layer1.attention.nested_output")
        self.assertIs(t3, out3)

    def test_default_output_naming_strategy(self):
        """Test the default naming strategy for generated output values using op_type_output format."""

        def _ops_with_default_names(
            op: builder.OpBuilder, x: ir.Value, y: ir.Value
        ) -> ir.Value:
            # Single output operations should be named {op_type}_output
            t1 = op.Add(x, y)
            t2 = op.Mul(x, y)
            z = op.Add(t1, t2)
            return z

        float_2d = ir.TypeAndShape(ir.TensorType(ir.DataType.FLOAT), ir.Shape([3, 4]))
        model = _build(
            _ops_with_default_names,
            input_types=[float_2d, float_2d],
            output_types=[float_2d],
        )
        graph = model.graph

        # Verify the nodes use the new naming strategy
        nodes = list(graph)
        self.assertEqual(len(nodes), 3)

        # Check output names follow the {op_type}_output pattern for single outputs
        self.assertEqual(nodes[0].outputs[0].name, "Add_output")
        self.assertEqual(nodes[1].outputs[0].name, "Mul_output")
        self.assertEqual(nodes[2].outputs[0].name, "Add_output")

        # Verify the final output has the correct name
        self.assertEqual(len(graph.outputs), 1)
        self.assertEqual(graph.outputs[0].name, "Add_output")

    def test_hierarchical_naming(self):
        """Test the hierarchical naming strategy (for value and node names)."""
        op, x, y = _create_builder_with_inputs()

        # Test node and value naming at root level
        t1 = op.Add(x, y)
        self.assertEqual(t1.name, "Add_output")
        self.assertEqual(t1.producer().name, "Add_node_0")

        t2 = op.Mul(t1, y)
        self.assertEqual(t2.name, "Mul_output")
        self.assertEqual(t2.producer().name, "Mul_node_1")

        # Test node and value naming with hierarchical context prefix
        op.builder.push_module("layer1")
        t3 = op.Add(t2, x)
        self.assertEqual(t3.name, "layer1.Add_output")
        self.assertEqual(t3.producer().name, "layer1.Add_node_2")

        # Test nested hierarchical context
        op.builder.push_module("attention")
        t4 = op.Mul(t3, y)
        self.assertEqual(t4.name, "layer1.attention.Mul_output")
        self.assertEqual(t4.producer().name, "layer1.attention.Mul_node_3")

        # Pop back to layer1 and verify naming continues correctly
        op.builder.pop_module()
        t5 = op.Add(t4, x)
        self.assertEqual(t5.name, "layer1.Add_output")
        self.assertEqual(t5.producer().name, "layer1.Add_node_4")

        # Pop back to root context
        op.builder.pop_module()
        t6 = op.Mul(t5, y)
        self.assertEqual(t6.name, "Mul_output")
        self.assertEqual(t6.producer().name, "Mul_node_5")

    def test_shape_inference_add(self):
        """Test that shape inference works correctly for Add operation."""
        op, x, y = _create_builder_with_inputs()

        # Create Add node without explicitly setting output type/shape
        result = op.Add(x, y)

        # Verify output type is inferred correctly
        self.assertIsNotNone(result.type)
        self.assertEqual(result.type.dtype, ir.DataType.FLOAT)

        # Verify output shape is inferred correctly
        self.assertIsNotNone(result.shape)
        self.assertEqual(list(result.shape), [2, 3, 4])

    def test_custom_domain_explicit(self):
        """Test using operations from custom domains with explicit _domain parameter."""
        op, x, y = _create_builder_with_inputs()

        # Create a custom domain operation with explicit _domain parameter
        # Using "com.microsoft" as an example domain
        result = op.CustomOp(x, y, _domain="com.microsoft")

        # Verify the node was created with the correct domain
        nodes = list(op.builder.graph)
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.domain, "com.microsoft")
        self.assertEqual(node.op_type, "CustomOp")

        # Verify inputs and outputs are connected correctly
        self.assertEqual(list(node.inputs), [x, y])
        self.assertEqual(node.outputs[0], result)

    def test_custom_domain_with_version(self):
        """Test using operations from custom domains with explicit _domain and _version parameters."""
        op, x, y = _create_builder_with_inputs()

        # Create a custom domain operation with explicit _domain and _version parameters
        result = op.MicrosoftOp(x, y, _domain="com.microsoft", _version=10)

        # Verify the node was created with the correct domain and version
        nodes = list(op.builder.graph)
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.domain, "com.microsoft")
        self.assertEqual(node.op_type, "MicrosoftOp")
        self.assertEqual(node.version, 10)

        # Verify output value is created
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "MicrosoftOp_output")

    def test_multiple_custom_domain_operations(self):
        """Test mixing operations from multiple domains."""
        op, x, y = _create_builder_with_inputs()

        # Create standard domain operation
        t1 = op.Add(x, y)

        # Create custom domain operation
        t2 = op.CustomOp(t1, y, _domain="com.microsoft")

        # Create another custom domain operation with different domain
        _ = op.AnotherOp(t2, x, _domain="com.custom")

        # Verify all nodes were created with correct domains
        nodes = list(op.builder.graph)
        self.assertEqual(len(nodes), 3)

        self.assertEqual(nodes[0].domain, "")
        self.assertEqual(nodes[0].op_type, "Add")

        self.assertEqual(nodes[1].domain, "com.microsoft")
        self.assertEqual(nodes[1].op_type, "CustomOp")

        self.assertEqual(nodes[2].domain, "com.custom")
        self.assertEqual(nodes[2].op_type, "AnotherOp")

    def test_opset_builder_for_custom_domain(self):
        """Test creating and using an opset builder for a custom domain."""
        op, x, y = _create_builder_with_inputs()

        # Create an OpBuilder for the "com.microsoft" domain with version 1
        ms_op = op.builder.opset("com.microsoft", 1)

        # Use operations through the custom domain opset builder
        t1 = ms_op.CustomOp(x, y)
        _ = ms_op.AnotherOp(t1, x)

        # Verify all nodes were created with the correct domain
        nodes = list(op.builder.graph)
        self.assertEqual(len(nodes), 2)

        # Verify first operation
        self.assertEqual(nodes[0].domain, "com.microsoft")
        self.assertEqual(nodes[0].op_type, "CustomOp")
        self.assertEqual(nodes[0].version, 1)
        self.assertEqual(list(nodes[0].inputs), [x, y])

        # Verify second operation
        self.assertEqual(nodes[1].domain, "com.microsoft")
        self.assertEqual(nodes[1].op_type, "AnotherOp")
        self.assertEqual(nodes[1].version, 1)
        self.assertEqual(list(nodes[1].inputs), [t1, x])

    def test_mixed_domain_opsets(self):
        """Test using both standard domain and custom domain opset builders together."""
        op, x, y = _create_builder_with_inputs()

        # Create custom domain opset builder
        ms_op = op.builder.opset("com.microsoft", 2)

        # Mix operations from different domains
        t1 = op.Add(x, y)  # Standard domain operation
        t2 = ms_op.MsAdd(t1, y)  # Custom domain operation
        _ = op.Mul(t2, x)  # Back to standard domain

        # Verify nodes were created with correct domains
        nodes = list(op.builder.graph)
        self.assertEqual(len(nodes), 3)

        self.assertEqual(nodes[0].domain, "")
        self.assertEqual(nodes[0].op_type, "Add")

        self.assertEqual(nodes[1].domain, "com.microsoft")
        self.assertEqual(nodes[1].op_type, "MsAdd")
        self.assertEqual(nodes[1].version, 2)

        self.assertEqual(nodes[2].domain, "")
        self.assertEqual(nodes[2].op_type, "Mul")

    def test_scalar_constant_initializers_are_cached(self):
        """Test that scalar constants produce named initializers and are shared across nodes."""
        graph = ir.Graph(
            name="test_model",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": _default_opset_version},
        )
        # Create one int64 input and one float32 input
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([3]))
        y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([3]))
        graph.inputs.extend([x, y])

        graph_builder = builder.GraphBuilder(graph)
        op = graph_builder.op

        # Two Adds that both use the integer constant 1
        r1 = op.Add(x, 1)
        r2 = op.Add(x, 1)
        # Two Adds that both use the float constant 1.0
        r3 = op.Add(y, 1.0)
        r4 = op.Add(y, 1.0)

        # The two int Add nodes should share the same constant initializer (same ir.Value)
        int_const_1 = r1.producer().inputs[1]
        int_const_2 = r2.producer().inputs[1]
        self.assertIs(int_const_1, int_const_2)
        self.assertEqual(int_const_1.name, "const_1_i64")

        # The two float Add nodes should share the same constant initializer
        float_const_1 = r3.producer().inputs[1]
        float_const_2 = r4.producer().inputs[1]
        self.assertIs(float_const_1, float_const_2)
        self.assertEqual(float_const_1.name, "const_1.0_f32")

        # The int and float constants should be different ir.Values
        self.assertIsNot(int_const_1, float_const_1)

    def test_int_constant_cast_to_float_via_like_type(self):
        """Test that op.Add(float_x, 1) converts the int 1 to a float tensor.

        When the schema binds the int constant to the same type variable as a float
        input, the constant should be created with dtype FLOAT and named accordingly.
        """
        graph = ir.Graph(
            name="test_model",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": _default_opset_version},
        )
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([3]))
        graph.inputs.append(x)

        graph_builder = builder.GraphBuilder(graph)
        op = graph_builder.op

        _ = op.Add(x, 1)

        nodes = list(graph)
        self.assertEqual(len(nodes), 1)

        # The int constant 1 should have been cast to float and named with f32 suffix
        const_input = nodes[0].inputs[1]
        self.assertEqual(const_input.name, "const_1_f32")
        # The constant's type should be FLOAT, not INT64
        self.assertEqual(const_input.const_value.dtype, ir.DataType.FLOAT)

    def test_int_constant_with_unknown_type_uses_cast_like(self):
        """Test that op.Add(unknown_x, 1) produces an int tensor + CastLike.

        When the input type is unknown, the constant is created with its natural
        Python type (int -> i64), and a CastLike node is inserted to dynamically
        cast it at runtime.
        """
        graph = ir.Graph(
            name="test_model",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": _default_opset_version},
        )
        # Input with no type information
        x = ir.Value(name="x", shape=ir.Shape([3]))
        graph.inputs.append(x)

        graph_builder = builder.GraphBuilder(graph)
        op = graph_builder.op

        _ = op.Add(x, 1)

        nodes = list(graph)
        # Expect 2 nodes: CastLike (to cast the int constant to x's type) + Add
        self.assertEqual(len(nodes), 2)

        cast_like_node = nodes[0]
        add_node = nodes[1]

        self.assertEqual(cast_like_node.op_type, "CastLike")
        self.assertEqual(add_node.op_type, "Add")

        # The original constant should be int64-typed
        const_initializer = cast_like_node.inputs[0]
        self.assertEqual(const_initializer.name, "const_1_i64")
        self.assertEqual(const_initializer.const_value.dtype, ir.DataType.INT64)

        # CastLike's second input should be x (the like_type reference)
        self.assertIs(cast_like_node.inputs[1], x)

        # Add should use the CastLike output, not the raw constant
        self.assertIs(add_node.inputs[1], cast_like_node.outputs[0])

    def test_pop_module_raises_on_empty_stack(self):
        """Test that pop_module raises RuntimeError when no module has been pushed."""
        op, _, _ = _create_builder_with_inputs()

        # Popping without any push should raise
        with self.assertRaises(RuntimeError):
            op.builder.pop_module()

        # Push then pop is fine; a second pop should raise
        op.builder.push_module("layer1")
        op.builder.pop_module()
        with self.assertRaises(RuntimeError):
            op.builder.pop_module()

    def test_attributes_are_created_properly(self):
        """Test that int, float, str, and list attributes are set correctly on a node."""
        op, x, y = _create_builder_with_inputs()

        result = op.DummyOp(
            x,
            y,
            _domain="test.domain",
            int_attr=42,
            float_attr=3.14,
            str_attr="hello",
            ints_attr=[1, 2, 3],
            floats_attr=[1.0, 2.0, 3.0],
            strs_attr=["a", "b", "c"],
        )

        node = result.producer()
        self.assertEqual(node.op_type, "DummyOp")
        self.assertEqual(node.domain, "test.domain")

        # Verify scalar attributes
        int_attr = node.attributes["int_attr"]
        self.assertEqual(int_attr.type, ir.AttributeType.INT)
        self.assertEqual(int_attr.value, 42)

        float_attr = node.attributes["float_attr"]
        self.assertEqual(float_attr.type, ir.AttributeType.FLOAT)
        self.assertAlmostEqual(float_attr.value, 3.14)

        str_attr = node.attributes["str_attr"]
        self.assertEqual(str_attr.type, ir.AttributeType.STRING)
        self.assertEqual(str_attr.value, "hello")

        # Verify list attributes
        ints_attr = node.attributes["ints_attr"]
        self.assertEqual(ints_attr.type, ir.AttributeType.INTS)
        self.assertEqual(list(ints_attr.value), [1, 2, 3])

        floats_attr = node.attributes["floats_attr"]
        self.assertEqual(floats_attr.type, ir.AttributeType.FLOATS)
        self.assertEqual(list(floats_attr.value), [1.0, 2.0, 3.0])

        strs_attr = node.attributes["strs_attr"]
        self.assertEqual(strs_attr.type, ir.AttributeType.STRINGS)
        self.assertEqual(list(strs_attr.value), ["a", "b", "c"])

    def test_call_inlines_onnxscript_function(self):
        """Test that GraphBuilder.call inlines an @onnxscript.script function."""

        @script(default_opset=op23)
        def mul_add_relu(X, Y):
            tmp = X * Y
            tmp = tmp + X
            return op23.Relu(tmp)

        # Create a GraphBuilder and call the function
        op, x, y = _create_builder_with_inputs()
        result = op.call(mul_add_relu, x, y)

        # The inlined function should produce 3 nodes: Mul, Add, Relu
        nodes = list(op.builder.graph)
        op_types = [n.op_type for n in nodes]
        self.assertEqual(op_types, ["Mul", "Add", "Relu"])

        # The result should be a single ir.Value (the Relu output)
        self.assertIsInstance(result, ir.Value)

        # Verify connectivity: Relu takes the Add output
        relu_node = nodes[2]
        add_node = nodes[1]
        self.assertIs(relu_node.inputs[0], add_node.outputs[0])

        # Verify the Add takes the Mul output and original input x
        mul_node = nodes[0]
        self.assertIs(add_node.inputs[0], mul_node.outputs[0])
        self.assertIs(add_node.inputs[1], x)

        # Verify the Mul takes the original inputs x and y
        self.assertIs(mul_node.inputs[0], x)
        self.assertIs(mul_node.inputs[1], y)


if __name__ == "__main__":
    unittest.main()
