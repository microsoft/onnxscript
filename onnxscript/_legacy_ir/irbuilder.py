from __future__ import annotations

import warnings
from typing import Any

import onnx

import onnxscript._legacy_ir as ir
from onnxscript._legacy_ir import visitor
from onnxscript.utils import utils

""" NOTE: IRBuilder and function visiting

Current IRBuilder is designed to visit function by definition, instead of function by callsite.
This has the following implications during visiting:
- Prior to IR 10 / ONNX 1.16, value_info is not defined in function. They are experimentally defined under
  main graph for models produced by PyTorch 2.2+ dynamo onnx exporter. Hence a workaround is required in `process_node`
  to load function value info from a pre-processed `FunctionShapeEnv` object.
  Post IR 10, using `process_value_info` method is enough to retrieve and process both function and graph
  value_info.
- ref_attr_name is not resolved during visiting, because it requires the function callsite information.

"""


class IRBuilder:
    def __init__(self):
        self._current_graphs: list[ir.Graph] = []
        # See NOTE: IRBuilder and function visiting
        self._current_function: ir.Function | None = None
        self._function_subgraphs: list[ir.Graph] = []
        self.functions: dict[ir.FuntionId, ir.Function] = {}

    def visit_model(self, model_proto: onnx.ModelProto) -> ir.Model:
        self._function_shape_env = visitor.FunctionShapeEnv()
        self._function_shape_env.load_from_model_proto(model_proto)
        self._ir_version = model_proto.ir_version
        self.version_map = {x.domain: x.version for x in model_proto.opset_import}
        functions = [self.visit_function(function) for function in model_proto.functions]
        self.functions = {function.id: function for function in functions}
        graph = self.visit_graph(model_proto.graph)
        model = ir.Model()
        model.set(model_proto, graph, functions, self.version_map)
        return model

    def visit_graph(self, graph: onnx.GraphProto) -> ir.Graph:
        self.enter_graph(ir.Graph(graph))
        for input in graph.input:
            self.process_graph_input(input)
        for init in graph.initializer:
            self.process_initializer(init)
        for node in graph.node:
            self.process_node(node)
        for output in graph.output:
            self.process_graph_output(output)
        for value_info in graph.value_info:
            self.process_value_info(value_info)
        return self.exit_graph()

    def visit_function(self, function: onnx.FunctionProto) -> ir.Function:
        self._current_function = ir.Function(function)
        for input in function.input:
            self.process_function_input(input)
        for node in function.node:
            self.process_node(node)
        for output in function.output:
            self.process_function_output(output)
        for value_info in getattr(function, "value_info", []):
            self.process_value_info(value_info)
        function_ir = self._current_function
        self._current_function = None
        return function_ir

    @property
    def current_graph_or_function(self) -> ir.Graph | ir.Function:
        if self._function_subgraphs:
            assert self._current_function is not None
            return self._function_subgraphs[-1]
        if self._current_function is not None:
            return self._current_function
        return self._current_graphs[-1]

    def enter_graph(self, graph: ir.Graph):
        if self._current_function is not None:
            self._function_subgraphs.append(graph)
        else:
            self._current_graphs.append(graph)

    def exit_graph(self) -> ir.Graph:
        if self._current_function is not None:
            return self._function_subgraphs.pop()
        else:
            return self._current_graphs.pop()

    def _lookup_from_graphs(self, name: str, graphs: list[ir.Graph]) -> ir.Value | None:
        for graph in reversed(graphs):
            value = graph.values.get(name, None)
            if value is not None:
                return value
        return None

    def lookup(self, name: str) -> ir.Value | None:
        if self._current_function is not None:
            value = self._lookup_from_graphs(name, self._function_subgraphs)
            if value is not None:
                return value
            return self._current_function.values.get(name, None)
        return self._lookup_from_graphs(name, self._current_graphs)

    def bind(self, name: str, value: ir.Value):
        self.current_graph_or_function.values[name] = value

    def process_graph_input(self, input: onnx.ValueInfoProto):
        newvalue = ir.Value(name=input.name, type=input.type)
        self.bind(input.name, newvalue)

    def process_initializer(self, init: onnx.TensorProto):
        # TODO(titaiwang): Take care of the case where the initializer is already defined?
        if init.name not in self.current_graph_or_function.values:
            newvalue = ir.Value(name=init.name, value=init)
            self.bind(init.name, newvalue)

    def process_node(self, node):
        node_ir = ir.Node(node)
        node_ir.set_version_if_custom_op(self.version_map)
        self.current_graph_or_function.nodes.append(node_ir)
        for name in node.input:
            value = self.lookup(name)
            node_ir.inputs.append(value)
            if value is not None:
                value.uses.append(node_ir)
            else:
                # TODO(titaiwang): Do something more than warnings?
                warnings.warn(f"Use of undefined variable {name!r}.", stacklevel=1)
        for index, output in enumerate(node.output):
            newvalue = ir.Value(name=output, node=node_ir, output_index=index)
            if self._current_function is not None:
                ir_value = self._function_shape_env.lookup(
                    self._current_function.original_function_proto, output
                )
                if ir_value is not None:
                    newvalue.identity_merge_from(ir_value)
            node_ir.outputs.append(newvalue)
            self.bind(output, newvalue)
        for attr in node.attribute:
            attr_val = self.process_attribute(attr)
            node_ir.attributes[attr.name] = attr_val
        # Set constant-value for Constant node:
        if node.op_type == "Constant" and node.domain in {"", "ai.onnx"}:
            node_ir.outputs[0].value = utils.get_constant_node_value(node, node.output[0])

    def process_attribute(self, attr: onnx.AttributeProto) -> ir.Graph | list[ir.Graph] | Any:
        if attr.HasField("g"):
            return self.visit_graph(attr.g)
        elif len(attr.graphs) > 0:
            return [self.visit_graph(graph) for graph in attr.graphs]
        elif attr.ref_attr_name:
            return ir.RefAttr(attr.name, attr.ref_attr_name, attr.type)
        else:
            # This returns Any based on onnx.helper.get_attribute_value's return type.
            return onnx.helper.get_attribute_value(attr)

    def process_graph_output(self, output: onnx.ValueInfoProto):
        value = self.lookup(output.name)
        if value is None:
            # TODO(titaiwang): Should we remove the non-output value from the graph.values?
            warnings.warn(
                f"Graph contains no definition for output '{output.name}'.",
                stacklevel=1,
            )
        else:
            value.type = output.type
            value.is_output = True

    def process_function_input(self, input: str):
        ir_value = self._function_shape_env.lookup(
            self._current_function.original_function_proto, input
        )
        if ir_value is None:
            ir_value = ir.Value(name=input)
        self.bind(input, ir_value)

    def process_function_output(self, output: str):
        value = self.lookup(output)
        if value is None:
            print(f"WARNING: Function contains no definition for output '{output.name}'.")
        else:
            value.is_output = True

    def process_value_info(self, value_info: onnx.ValueInfoProto):
        function_id, ir_value = self._function_shape_env.process_value_info(value_info)
        existing_value = self.lookup(value_info.name)
        if existing_value is not None:
            existing_value.identity_merge_from(ir_value)
            ir_value = existing_value

        if self._ir_version >= 10:
            # ONNX >= 1.16 where value_info can be defined in function
            self.bind(ir_value.name, ir_value)
        elif function_id is not None:
            # All value_infos are defined in main graph
            # This needs to be handled while visiting function, so do nothing here.
            pass
        else:
            self.bind(ir_value.name, ir_value)


def build_ir(model: onnx.ModelProto):
    """Builds an IR from an ONNX model proto."""
    return IRBuilder().visit_model(model)
