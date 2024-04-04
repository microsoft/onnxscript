from __future__ import annotations

import onnx
import onnx.helper
from onnx.helper import make_attribute

import onnxscript._legacy_ir as ir


class ModelProtoBuilder:
    def __init__(self):
        self.opset_imports: dict[str, onnx.OperatorSetIdProto] = {}

    def visit_ir_model(self, ir_model: ir.Model) -> onnx.ModelProto:
        model_proto = onnx.ModelProto()
        model_proto.ir_version = ir_model.original_model_proto.ir_version
        # TODO (sbhokare) : Find a way of copying model properties without
        #                   each property individually
        # Copy over model properties
        model_proto.doc_string = ir_model.original_model_proto.doc_string
        model_proto.domain = ir_model.original_model_proto.domain
        model_proto.metadata_props.extend(ir_model.original_model_proto.metadata_props)
        model_proto.model_version = ir_model.original_model_proto.model_version
        model_proto.producer_name = ir_model.original_model_proto.producer_name
        model_proto.producer_version = ir_model.original_model_proto.producer_version
        model_proto.training_info.extend(ir_model.original_model_proto.training_info)

        for domain, version in ir_model.version_map.items():
            operator_setid_proto = model_proto.opset_import.add()
            operator_setid_proto.domain, operator_setid_proto.version = domain, version
            self.opset_imports[domain] = operator_setid_proto
        for function in ir_model.functions:
            function_proto = model_proto.functions.add()
            self.visit_ir_function(function, function_proto)
        graph_proto = model_proto.graph
        self.visit_ir_graph(ir_model.graph, graph_proto)
        return model_proto

    def visit_ir_graph(
        self, ir_graph: ir.Graph, graph_proto: onnx.GraphProto
    ) -> onnx.GraphProto:
        graph_proto.name = ir_graph.name
        # Copy over graph properties
        graph_proto.doc_string = ir_graph.original_graph_proto.doc_string
        # graph_proto.metadata_props = ir_graph.original_graph_proto.metadata_props)
        graph_proto.quantization_annotation.extend(
            ir_graph.original_graph_proto.quantization_annotation
        )

        for node in ir_graph.nodes:
            node_proto = graph_proto.node.add()
            self.process_ir_node(node, node_proto)
        for i in ir_graph.original_graph_proto.input:
            graph_proto.input.append(i)
        for o in ir_graph.original_graph_proto.output:
            graph_proto.output.append(o)
        for val in ir_graph.original_graph_proto.value_info:
            graph_proto.value_info.append(val)
        for i in ir_graph.original_graph_proto.initializer:  # type: ignore[assignment]
            graph_proto.initializer.append(i)  # type: ignore[arg-type]
        return graph_proto

    def visit_ir_function(
        self, ir_function: ir.Function, function_proto: onnx.FunctionProto
    ) -> onnx.FunctionProto:
        function_proto.name = ir_function.name
        function_proto.domain = ir_function.domain
        # Copy over function properties
        function_proto.doc_string = ir_function.original_function_proto.doc_string
        # function_proto.metadata_props = ir_function.original_function_proto.metadata_props)

        for node in ir_function.nodes:
            # TODO: deduplicate the opset import of function?
            operator_setid_proto = function_proto.opset_import.add()
            if node.domain in self.opset_imports:
                operator_setid_proto.domain = self.opset_imports[node.domain].domain
                operator_setid_proto.version = self.opset_imports[node.domain].version
            else:
                raise ValueError(f"Unknown domain {node.domain}")
            node_proto = function_proto.node.add()
            self.process_ir_node(node, node_proto)
        # TODO (shubham) : Propagate shape-type info
        for i in ir_function.original_function_proto.input:
            function_proto.input.append(i)
        for o in ir_function.original_function_proto.output:
            function_proto.output.append(o)
        for attr in ir_function.original_function_proto.attribute:
            function_proto.attribute.append(attr)
        for attr_proto in ir_function.original_function_proto.attribute_proto:
            function_proto.attribute_proto.append(attr_proto)
        for val in getattr(ir_function.original_function_proto, "value_info", []):
            function_proto.value_info.append(val)
        return function_proto

    def process_ir_node(self, ir_node: ir.Node, node_proto: onnx.NodeProto) -> onnx.NodeProto:
        node_proto.op_type = ir_node.op_type
        node_proto.domain = ir_node.domain
        # Copy over node properties
        node_proto.name = ir_node.original_node_proto.name
        node_proto.doc_string = ir_node.original_node_proto.doc_string
        # node_proto.metadata_props = ir_node.original_node_proto.metadata_props)

        for i in ir_node.inputs:
            node_proto.input.append(i.name if i is not None else "")
        for o in ir_node.outputs:
            assert o is not None
            node_proto.output.append(o.name)
        for attr in ir_node.attributes.items():
            attr_proto = self.process_attribute(attr)
            node_proto.attribute.append(attr_proto)
        return node_proto

    def process_attribute(self, attr):
        attr_name, attr_val = attr
        if isinstance(attr_val, ir.RefAttr):
            return attr_val.to_proto()
        if isinstance(attr_val, ir.Graph):
            graph_proto = onnx.GraphProto()
            attr_val = self.visit_ir_graph(attr_val, graph_proto)
        attr_proto = make_attribute(attr_name, attr_val)
        return attr_proto


def build_model_proto(model: ir.Model) -> onnx.ModelProto:
    """Builds an ONNX model proto from an IR."""
    return ModelProtoBuilder().visit_ir_model(model)
