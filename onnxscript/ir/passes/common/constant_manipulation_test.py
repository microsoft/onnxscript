# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import parameterized

from onnxscript import ir
from onnxscript.ir.passes.common import constant_manipulation


class TestLiftConstantsToInitializersPass(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (ir.DataType.FLOAT, True),
            (ir.DataType.FLOAT, False),
            (ir.DataType.INT64, True),
            (ir.DataType.INT64, False),
        ]
    )
    def test_pass_with_lifting_float_and_int_constants_to_initializers(
        self, ir_dtype: ir.DataType, lift_all_constants: bool
    ):
        inputs = [
            ir.Value(name="input_a", type=ir.TensorType(ir_dtype), shape=ir.Shape((2, 3))),
            ir.Value(
                name="input_b",
                type=ir.TensorType(ir_dtype),
                shape=ir.Shape((2, 3)),
            ),
        ]

        constant_tensor = ir.tensor(np.random.rand(2, 3).astype(ir_dtype.numpy()))
        const_node = ir.node(
            "Constant", inputs=[], attributes={"value": constant_tensor}, num_outputs=1
        )
        add_node = ir.node("Add", inputs=[inputs[0], const_node.outputs[0]])
        mul_node = ir.node("Mul", inputs=[add_node.outputs[0], inputs[1]])

        model = ir.Model(
            graph=ir.Graph(
                inputs=inputs,
                outputs=mul_node.outputs,
                nodes=[const_node, add_node, mul_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Check that the initializer is not in the graph yet
        self.assertEqual(len(model.graph.initializers), 0)
        # And 1 constant node
        self.assertEqual(len([node for node in model.graph if node.op_type == "Constant"]), 1)

        # Perform lift constants to initializers
        result = constant_manipulation.LiftConstantsToInitializersPass(
            lift_all_constants=lift_all_constants, size_limit=0
        )(model)
        self.assertTrue(result.modified)
        # Check that the constant node is lifted to an initializer
        self.assertEqual(len(result.model.graph.initializers), 1)
        # Check the value
        self.assertEqual(
            result.model.graph.initializers[
                "val_0"
            ].const_value,  # name created by name_authority
            constant_tensor,
        )
        # And 0 constant node
        self.assertEqual(
            len([node for node in result.model.graph if node.op_type == "Constant"]), 0
        )

    @parameterized.parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_pass_with_lifting_constants_to_initializers_within_subgraph(
        self, lift_all_constants: bool
    ):
        input_value = ir.Value(
            name="input", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))
        )

        then_constant_tensor = ir.tensor(np.random.rand(2, 3).astype(np.float32))
        then_const_node = ir.node(
            "Constant", inputs=[], attributes={"value": then_constant_tensor}, num_outputs=1
        )
        # then branch adds the constant to the input
        # else branch multiplies the input by the constant
        add_node = ir.node("Add", inputs=[input_value, then_const_node.outputs[0]])
        then_graph = ir.Graph(
            inputs=[input_value],
            outputs=[add_node.outputs[0]],
            nodes=[then_const_node, add_node],
            opset_imports={"": 20},
        )
        else_constant_tensor = ir.tensor(np.random.rand(2, 3).astype(np.float32))
        else_const_node = ir.node(
            "Constant", inputs=[], attributes={"value": else_constant_tensor}, num_outputs=1
        )
        mul_node = ir.node("Mul", inputs=[input_value, else_const_node.outputs[0]])
        else_graph = ir.Graph(
            inputs=[input_value],
            outputs=[mul_node.outputs[0]],
            nodes=[else_const_node, mul_node],
            opset_imports={"": 20},
        )
        # create a conditional node that uses the then and else graphs
        cond_node = ir.node(
            "If",
            inputs=[input_value],
            attributes={"then_branch": then_graph, "else_branch": else_graph},
            num_outputs=1,
        )
        # construnct the model
        main_graph = ir.Graph(
            inputs=[input_value],
            outputs=cond_node.outputs,
            nodes=[cond_node],
            opset_imports={"": 20},
        )
        main_graph.sort()
        model = ir.Model(
            graph=main_graph,
            ir_version=10,
        )
        result = constant_manipulation.LiftConstantsToInitializersPass(
            lift_all_constants=lift_all_constants, size_limit=0
        )(model)
        self.assertTrue(result.modified)
        # Check that the constant node is lifted to the subgraph initializers
        for node in ir.traversal.RecursiveGraphIterator(result.model.graph):
            if node.op_type == "Constant":
                raise AssertionError(
                    f"Constant node '{node.name}' was not lifted to initializers"
                )
        self.assertEqual(len(else_graph.initializers), 1)
        self.assertEqual(len(then_graph.initializers), 1)
        self.assertIs(
            else_graph.initializers["val_0"].const_value,
            else_constant_tensor,
        )
        self.assertIs(
            then_graph.initializers["val_0"].const_value,
            then_constant_tensor,
        )

    @parameterized.parameterized.expand(
        [
            (1.0, "value_float", np.float32, True),
            (1.0, "value_float", np.float32, False),
            (1, "value_int", np.int64, True),
            (1, "value_int", np.int64, False),
            ("hello world!", "value_string", np.bytes_, True),
            ("hello world!", "value_string", np.bytes_, False),
            ([1.0, 2.0, 3.0], "value_floats", np.float32, True),
            ([1.0, 2.0, 3.0], "value_floats", np.float32, False),
            ([1, 2, 3], "value_ints", np.int64, True),
            ([1, 2, 3], "value_ints", np.int64, False),
            (["hello world!", "thank you."], "value_strings", np.bytes_, True),
            (["hello world!", "thank you."], "value_strings", np.bytes_, False),
        ]
    )
    def test_pass_with_lifting_constants_to_initializers_with_floats_ints_strings(
        self,
        value: float | int | str | list[float] | list[int] | list[str],
        constant_attribute: str,
        np_dtype: type[np.dtype],
        lift_all_constants: bool,
    ):
        input_value = ir.Value(
            name="input", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))
        )

        constant_value = value
        const_node = ir.node(
            "Constant",
            inputs=[],
            attributes={constant_attribute: constant_value},
            num_outputs=1,
        )
        identity_node_constant = ir.node(
            "Identity", inputs=[const_node.outputs[0]], num_outputs=1
        )
        identity_node_input = ir.node("Identity", inputs=[input_value], num_outputs=1)

        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_value],
                outputs=[identity_node_input.outputs[0], identity_node_constant.outputs[0]],
                nodes=[identity_node_input, const_node, identity_node_constant],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Check that the initializer is not in the graph yet
        self.assertEqual(len(model.graph.initializers), 0)
        # And 1 constant node
        self.assertEqual(len([node for node in model.graph if node.op_type == "Constant"]), 1)

        # Perform lift constants to initializers
        result = constant_manipulation.LiftConstantsToInitializersPass(
            lift_all_constants=lift_all_constants, size_limit=0
        )(model)
        if lift_all_constants:
            self.assertTrue(result.modified)
            # Check that the constant node is lifted to an initializer
            self.assertEqual(len(result.model.graph.initializers), 1)
            np.testing.assert_array_equal(
                result.model.graph.initializers["val_1"].const_value.numpy(),
                np.array(constant_value, dtype=np_dtype),
            )
        else:
            self.assertFalse(result.modified)
            # Check that the constant node is not lifted to an initializer
            self.assertEqual(len(result.model.graph.initializers), 0)

    def test_not_lifting_constants_to_initializers_when_it_is_output(self):
        input_value = ir.Value(
            name="input", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))
        )
        identity_node_input = ir.node("Identity", inputs=[input_value], num_outputs=1)

        constant_value = ir.tensor(np.random.rand(2, 3).astype(np.float32))
        const_node = ir.node(
            "Constant",
            inputs=[],
            attributes={"value": constant_value},
            num_outputs=1,
        )

        model = ir.Model(
            graph=ir.Graph(
                inputs=[input_value],
                outputs=[identity_node_input.outputs[0], const_node.outputs[0]],
                nodes=[identity_node_input, const_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        result = constant_manipulation.LiftConstantsToInitializersPass()(model)
        self.assertFalse(result.modified)
        # Check that the constant node is not lifted to an initializer
        self.assertEqual(len(result.model.graph.initializers), 0)


if __name__ == "__main__":
    unittest.main()
