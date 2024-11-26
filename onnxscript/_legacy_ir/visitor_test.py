# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import onnx

from onnxscript._legacy_ir import visitor


class FunctionCallsiteProtoTransformerTest(unittest.TestCase):
    def test_function_optional_input_is_recorded_by_shape_env(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17, "custom": 1]>
            agraph (float[N] x) => (float[N] z) {
                z = custom.function(x)
            }
            <
                domain: "custom",
                opset_import: ["" : 18]
            >
            function (x, optional_y, optional_z) => (return_val)
            {
                return_val = custom.custom_op (x, optional_y, optional_z)
            }
            """
        )

        model_visitor = visitor.FunctionCallsiteProtoTransformer()
        model_visitor.visit_model(model)
        self.assertIsNotNone(
            model_visitor.function_shape_env.lookup(model.functions[0], "optional_y")
        )
        self.assertIsNotNone(
            model_visitor.function_shape_env.lookup(model.functions[0], "optional_z")
        )

    def test_proto_visitor_enter_exit_function_scope(self):
        function_proto = onnx.FunctionProto()
        visitor_instance = visitor.ProtoVisitor()
        visitor_instance.enter_function_scope(function_proto)
        self.assertIsNotNone(visitor_instance.scopes.current_scope().current_function_scope())
        visitor_instance.exit_function_scope(function_proto)
        self.assertIsNone(visitor_instance.scopes.current_scope().current_function_scope())


    def test_proto_visitor_missing_input_types(self):
        node_proto = onnx.helper.make_node(
            'Add',
            inputs=['A', 'B'],
            outputs=['C']
        )
        visitor_instance = visitor.ProtoVisitor(do_shape_inference=True)
        visitor_instance.scopes.enter_graph_scope(onnx.GraphProto())
        visitor_instance.bind('A', visitor.ir.Value(name='A', type=onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [1])))
        visitor_instance.bind('B', visitor.ir.Value(name='B', type=onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [1])))
        visitor_instance.process_node(node_proto)
        output_value = visitor_instance.lookup('C')
        self.assertIsNone(output_value.type)


    def test_save_to_value_info_with_overload(self):
        shape_env = visitor.FunctionShapeEnv()
        value_info = onnx.helper.make_tensor_value_info('custom::function::overload/x', onnx.TensorProto.FLOAT, [1])
        with self.assertRaises(NotImplementedError):
            shape_env.save_to_value_info(visitor.ir.Value(name='x', type=value_info.type), 'custom', 'function', 'overload')


    def test_save_to_model_proto_with_function_id_and_value_info(self):
        model_proto = onnx.ModelProto()
        model_proto.graph.value_info.extend([
            onnx.helper.make_tensor_value_info('custom::function/x', onnx.TensorProto.FLOAT, [1])
        ])
        shape_env = visitor.FunctionShapeEnv()
        shape_env.load_from_model_proto(model_proto)
        shape_env.save_to_model_proto(model_proto)
        self.assertEqual(len(model_proto.graph.value_info), 2)


    def test_subscope_bind_and_lookup_ref_attribute(self):
        graph_proto = onnx.GraphProto()
        subscope = visitor.SubScope(graph_proto)
        attr_proto = onnx.AttributeProto()
        attr_proto.name = "attr1"
        subscope.bind_ref_attribute("attr1", attr_proto)
        self.assertEqual(subscope.lookup_ref_attribute("attr1"), attr_proto)


    def test_scope_bind_empty_name(self):
        scope = visitor.Scope()
        scope.enter_sub_scope(onnx.GraphProto())
        with self.assertRaises(ValueError):
            scope.bind("", visitor.ir.Value(name="value"))


    def test_load_from_value_info_with_function_id(self):
        value_info = onnx.helper.make_tensor_value_info('custom::function/x', onnx.TensorProto.FLOAT, [1])
        shape_env = visitor.FunctionShapeEnv()
        shape_env.load_from_value_info(value_info)
        self.assertEqual(len(shape_env._function_values), 1)


    def test_scope_bind_none_value(self):
        scope = visitor.Scope()
        scope.enter_sub_scope(onnx.GraphProto())
        with self.assertRaises(ValueError):
            scope.bind("test_name", None)


    def test_load_from_value_info_with_none_function_id(self):
        value_info = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1])
        shape_env = visitor.FunctionShapeEnv()
        shape_env.load_from_value_info(value_info)
        self.assertEqual(len(shape_env._function_values), 0)


    def test_override_inferred_value_type_with_none_values(self):
        result = visitor._override_inferred_value_type_with_symbolic_value_type(None, None)
        self.assertIsNone(result)



if __name__ == "__main__":
    unittest.main()
