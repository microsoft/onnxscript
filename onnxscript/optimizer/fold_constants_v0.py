from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import onnx
import onnx.reference.ops

# Excluded ops include
# * Random ops, which are not deterministic
# * Control flow ops

excluded_ops = frozenset(
    {
        "RandomUniform",
        "RandomNormal",
        "RandomUniformLike",
        "RandomNormalLike",
        "Multinomial",
        "If",
        "Loop",
        "Scan",
        "SequenceMap",
    }
)

onnx_domain = frozenset({"", "onnx.ai"})


def get_evaluator(domain: str, op: str, version: int) -> callable | None:
    if op in excluded_ops and domain in onnx_domain:
        return None
    try:
        op_impl_class = onnx.reference.ops.load_op(domain, op, version)
    except Exception:
        return None
    else:
        return op_impl_class.eval


def convert_attributes(attributes: Sequence[onnx.AttributeProto]) -> dict[str, Any]:
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in attributes}


def is_control_flow_op(node: onnx.NodeProto) -> bool:
    return any(attr.HasField("g") or len(attr.graphs) > 0 for attr in node.attribute)


def is_constant_op(node: onnx.NodeProto) -> bool:
    return node.op_type == "Constant" and node.domain == ""


def get_bool_value(val) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray) and val.size == 1 and val.dtype == bool:
        return val.item(0)
    return None


def get_shape_info(type: onnx.TypeProto) -> tuple[int, ...] | None:
    if type.HasField("tensor_type") and type.tensor_type.HasField("shape"):
        if all(d.HasField("dim_value") for d in type.tensor_type.shape.dim):
            return np.array([d.dim_value for d in type.tensor_type.shape.dim], dtype=np.int64)
    return None


def get_element_type(type: onnx.TypeProto) -> int | None:
    if type.HasField("tensor_type"):
        return type.tensor_type.elem_type
    return None


class State:
    def __init__(self, default_value) -> None:
        self.scopes = [{}]
        self.default_value = default_value

    def lookup(self, name: str) -> Any:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return self.default_value

    def bind(self, name: str, value: Any) -> None:
        self.scopes[-1][name] = value

    def enter_scope(self) -> None:
        self.scopes.append({})

    def exit_scope(self) -> None:
        self.scopes.pop()


def is_onnx_op(node: onnx.NodeProto, op: str) -> bool:
    return (node.op_type == op) and (node.domain in onnx_domain)


def matches(node: onnx.NodeProto, op: str, *arg_predicates) -> bool:
    if node.op_type != op or node.domain != "":
        return False
    if len(node.input) < len(arg_predicates):
        return False
    return all(pred(input) for pred, input in zip(arg_predicates, node.input))


def get_initializer_type(initializer: onnx.TensorProto) -> onnx.TypeProto:
    type = onnx.TypeProto()
    type.tensor_type.elem_type = initializer.data_type
    dims = type.tensor_type.shape.dim
    for dim in initializer.dims:
        dims.add().dim_value = dim
    return type


def fold_constants(model: onnx.ModelProto):
    not_constant = object()
    var_info = State(default_value=not_constant)
    type_info = State(default_value=None)
    counts = {}
    sizes = {}

    def add_count(op: str, size: int = 1):
        counts[op] = counts.get(op, 0) + 1
        sizes[op] = sizes.get(op, 0) + size

    def new_constant(name, value):
        var_info.bind(name, value)
        tensor = onnx.numpy_helper.from_array(value, name=name)
        node = onnx.helper.make_node("Constant", inputs=[], outputs=[name], value=tensor)
        return node

    def lookup_version(domain: str, op: str) -> int:
        for opset in model.opset_import:
            if opset.domain == domain:
                return opset.version
        return 1  # TODO

    def transform_node(node: onnx.NodeProto):
        if is_onnx_op(node, "Transpose"):
            return [node]
        if is_onnx_op(node, "CastLike"):
            value = var_info.lookup(node.input[0]) if len(node.input) > 0 else not_constant
            if value is not_constant:
                return [node]
            type = type_info.lookup(node.input[1]) if len(node.input) > 1 else None
            element_type = get_element_type(type) if type is not None else None
            if element_type is None:
                return [node]
            evaluator = get_evaluator("", "Cast", lookup_version("", "Cast"))
            if evaluator is None:
                return [node]
            cast_value = evaluator(value, to=element_type)
            add_count("CastLike", cast_value.size)
            return [new_constant(node.output[0], cast_value)]
        if is_onnx_op(node, "Shape"):
            type = type_info.lookup(node.input[0]) if len(node.input) > 0 else None
            shape = get_shape_info(type) if type is not None else None
            if shape is not None:
                add_count("Shape", shape.size)
                return [new_constant(node.output[0], shape)]

        if is_onnx_op(node, "If"):
            cond = var_info.lookup(node.input[0]) if len(node.input) > 0 else None
            cond = get_bool_value(cond)
            if cond is not None:
                # cond is a constant-value: inline the branch
                branch = "then_branch" if cond else "else_branch"
                graph = onnx.helper.get_node_attr_value(node, branch)
                formal_outs = list(graph.output)
                actual_outs = node.output
                renamings = {
                    formal.name: actual
                    for formal, actual in zip(formal_outs, actual_outs)
                    if actual != ""
                }

                def rename(name):
                    return renamings.get(name, name)

                for node in graph.node:
                    node.input[:] = [rename(name) for name in node.input]
                    node.output[:] = [rename(name) for name in node.output]
                transform_graph(graph)
                add_count("If")
                return list(graph.node)

        if is_control_flow_op(node):
            for attr in node.attribute:
                if attr.HasField("g"):
                    transform_graph(attr.g)
                elif len(attr.graphs) > 0:
                    for graph in attr.graphs:
                        transform_graph(graph)
            return [node]

        domain = node.domain
        op = node.op_type
        version = lookup_version(domain, op)
        inputs = []
        for x in node.input:
            if x == "":
                inputs.append(None)
            else:
                v = var_info.lookup(x)
                if v is not_constant:
                    return [node]
                inputs.append(v)
        evaluator = get_evaluator(domain, op, version)
        if evaluator is None:
            return [node]
        attrs = convert_attributes(node.attribute)
        outputs = evaluator(*inputs, **attrs)
        if len(node.output) == 1 and not isinstance(outputs, tuple):
            replacement = new_constant(node.output[0], outputs)
            if is_constant_op(node):
                return [node]
            add_count(op, outputs.size)
            return [replacement]
        else:
            add_count(op)
            return [new_constant(output, outputs[i]) for i, output in enumerate(node.output)]

    def transform_graph(graph: onnx.GraphProto):
        var_info.enter_scope()
        type_info.enter_scope()
        for initializer in graph.initializer:
            array = onnx.numpy_helper.to_array(initializer)
            var_info.bind(initializer.name, array)
            type_info.bind(initializer.name, get_initializer_type(initializer))
        for input in graph.input:
            var_info.bind(input.name, not_constant)
            type_info.bind(input.name, input.type)
        for valueinfo in graph.value_info:
            type_info.bind(valueinfo.name, valueinfo.type)

        replacement = [transform_node(node) for node in graph.node]
        flattened = [node for nodes in replacement for node in nodes]
        del graph.node[:]
        graph.node.extend(flattened)
        var_info.exit_scope()
        type_info.exit_scope()

    transform_graph(model.graph)
    for op in counts:
        print(f"Constant-folded '{op}' {counts[op]} times, with {sizes[op]} size.")
