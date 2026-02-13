# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
 
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import onnx
import onnxscript.values
import onnx_ir as ir


def _make_node(
    op_type: str,
    inputs: Sequence[ir.Value | None],
    attributes: Mapping[str, ir._convenience.SupportedAttrTypes] | None = None,
    *,
    num_outputs: int | None = None,
    outputs: Sequence[ir.Value] | None = None,
    domain: str = "",
    overload: str = "",
    version: int | None = None,
    graph: ir.Graph | None = None,
    name: str | None = None,
    doc_string: str | None = None,
    metadata_props: dict[str, str] | None = None,
) -> ir.Node:
    if num_outputs is None and outputs is None:
        raise ValueError("Either num_outputs or outputs must be provided.")
    if num_outputs is not None and outputs is not None:
        raise ValueError("Both num_outputs and outputs cannot be provided simultaneously.")
    output_kwargs: dict[str, Any]
    if outputs is None:
        output_kwargs = dict(num_outputs=num_outputs)
    else:
        output_kwargs = dict(outputs=outputs)
    if attributes is None:
        attrs: Sequence[ir.Attr] = ()
    else:
        attrs = ir._convenience.convert_attributes(attributes)
    node = ir.Node(
        domain,
        op_type,
        inputs,
        attributes=attrs,
        **output_kwargs,
        overload=overload,
        version=version,
        graph=graph,
        name=name,
        doc_string=doc_string,
        metadata_props=metadata_props,
    )
    return node

  
class GraphBuilder:
    def __init__(self, graph: ir.Graph, is_function: bool) -> None:
        self._graph = graph
        self._op_builder = self.opset("", None)

        # Context stack to manage hierarchical naming. Each module/layer can push a new context, and pop it when done.
        # The current context is used as a prefix for naming values and nodes.
        # This allows us to generate names like "layer1.attention.query"
        self._context_stack: list[str] = [""]


    def opset(self, domain: str = "", version: int | None = None) -> OpBuilder:
        return OpBuilder(self, domain, version)

    @property
    def op(self) -> OpBuilder:
        return self._op_builder

    @property
    def graph(self) -> ir.Graph:
        return self._graph

    def initializer(self, tensor: ir.TensorProtocol, name: str | None = None) -> ir.Value:
        if name is None:
            name = tensor.name
        prefix = self.context_name()
        if prefix:
            name = f"{prefix}.{name}"
            # TODO: set tensor name as well
        shape = ir.Shape((d if isinstance(d, int) else d.value) for d in tensor.shape.dims)
        value = ir.Value(
            name=name, shape=shape, type=ir.TensorType(tensor.dtype), const_value=tensor
        )
        self._graph.register_initializer(value)
        return value

    def _input_to_ir_value(self, value: ir.Value | ir.TensorProtocol, like_type: ir.Value | None = None) -> ir.Value:
        if isinstance(value, ir.Value):
            return value
        elif isinstance(value, (int, float, bool, str)):
            # Scalar constant
            import numpy as np

            if like_type is not None and like_type.type is not None:
                dtype = like_type.type.dtype
            else:
                # Infer type from Python type
                if isinstance(value, bool):
                    dtype = ir.DataType.BOOL
                elif isinstance(value, int):
                    dtype = ir.DataType.INT64
                elif isinstance(value, float):
                    dtype = ir.DataType.FLOAT32
                elif isinstance(value, str):
                    dtype = ir.DataType.STRING
                else:
                    raise TypeError(f"Unsupported scalar type: {type(value)}")
            tensor = ir.Tensor(
                data=np.array(value, dtype=dtype.numpy()),
                name="const_scalar",
            )
            return self.initializer(tensor)
        else:
            # assert isinstance(value, ir.TensorProtocol):
            # TODO: We could using caching to avoid duplicate initializers. However, it seems unlikely
            # to be useful in practice, as shared use of a stateful module is rare.
            return self.initializer(value)


    def _adapt_outputs(self, outputs: int | Sequence[str | ir.Value]) -> Sequence[ir.Value]:
        prefix = self.context_name()
        if isinstance(outputs, int):
            count = self.graph.num_nodes()
            name = f"{prefix}.val_{count}" if prefix else f"val_{count}"
            if outputs == 1:
                return [ir.Value(name=name)]
            else:
                return [ir.Value(name=f"{name}.{i}") for i in range(outputs)]
        adapted_outputs = []
        for output in outputs:
            if isinstance(output, ir.Value):
                adapted_outputs.append(output)
            elif isinstance(output, str):
                adapted_outputs.append(ir.Value(name=output))
            else:
                raise TypeError(f"Output type not supported.")
        return adapted_outputs

    def _get_schema(self, op_type: str, domain: str, version: int | None) -> onnx.defs.OpSchema | None:
        if version is not None:
            try:
                return onnx.defs.get_schema(op_type, version, domain)
            except onnx.defs.SchemaError:
                pass
        return None

    def _partition_inputs_attributes(
        self,
        schema: onnx.defs.OpSchema | None,
        inputs: Sequence[ir.Value | ir.TensorProtocol],
        kwargs: dict[str, Any],
    ) -> tuple[Sequence[ir.Value | ir.TensorProtocol], dict[str, Any]]:
        # Not implemented yet
        return inputs, kwargs

    def _cast_inputs(
        self,
        schema: onnx.defs.OpSchema | None,
        inputs: Sequence[ir.Value | ir.TensorProtocol], 
    ) -> Sequence[ir.Value]:
        """Uses schema specification to support a limited form of auto-casting.

        * Scalars are promoted to tensors.
        * Further. they are cast to the required type when used in ops with other
        tensor inputs that are required to be of same type.
        Thus, in "A+1" or "Add(A, 1)", the value 1 will be converted to the same
        type as A.

        This is used by the converter in a static-mode, as well as by the eager-mode
        execution in a dynamic-mode.
        """
        if schema is None:
            return [self._input_to_ir_value(i) for i in inputs]

        expected_inputs = schema.inputs
        # We make two passes. In the first pass, we identify known type-bindings for
        # type-variables: eg., {'T1' : np.float32, 'T2' : np.int32}.
        # In the second pass, we use these bindings to cast scalar-values to
        # tensors of appropriate types. The two passes are needed to handle cases
        # like "Add(1, X)" where 1 must be cast to the same type as X.
        type_bindings: dict[str, ir.Value] = {}
        args_typevars: list[tuple[ir.Value | None, str | None]] = []
        for i, x in enumerate(inputs):
            if i < len(expected_inputs):
                expected = expected_inputs[i]
            elif expected_inputs[-1].option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
                expected = expected_inputs[-1]
                if not expected.is_homogeneous:
                    args_typevars.append((x, None))
                    continue
            else:
                raise ValueError(
                    f"Number of actual parameters {len(inputs)} "
                    f"exceeds number of formal parameters {len(expected_inputs)}."
                )
            typevar = expected.type_str
            if ("(" not in typevar) and (typevar not in type_bindings):
                # typevar is an identifier, like "T"
                if isinstance(x, ir.Value):
                    type_bindings[typevar] = x
            args_typevars.append((x, typevar))
        def adapt (x, typevar: str | None) -> ir.Value | None:
            if x is None: return None
            if typevar is None:
                return self._input_to_ir_value(x)
            type_like = type_bindings.get(typevar) if typevar is not None else None
            return self._input_to_ir_value(x, type_like)
        return [adapt(x, typevar) for x, typevar in args_typevars]

    def _cast_attributes(
        self,
        schema: onnx.defs.OpSchema | None,
        attributes: dict[str, Any],
    ) -> dict[str, Any]:
        return attributes

    def add_node(self, node: ir.Node) -> None:
        self.graph.append(node)
        onnxscript.optimizer.basic_constant_propagation([node])
        # TODO: inference.infer_outputs(node, 23)
                
    def call_op(self, op_type: str, inputs: Sequence[ir.Value | ir.TensorProtocol], kwargs: dict[str, Any]):
        domain = kwargs.pop("_domain", "")
        version = kwargs.pop("_version", None)
        outputs = kwargs.pop("_outputs", 1)

        count = self.graph.num_nodes
        node_name = f"node_{count}"

        output_values = self._adapt_outputs(outputs)

        schema = self._get_schema(op_type, domain, version)
        inputs, attributes = self._partition_inputs_attributes(schema, inputs, kwargs)
        inputs = self._cast_inputs(schema, inputs)
        attributes = self._cast_attributes(schema, attributes)

        node = _make_node(
                op_type,
                inputs=inputs,
                attributes=attributes,
                domain=domain,
                version=version,
                outputs=output_values,
                graph=self.graph,
                name=node_name
            )
        self.add_node(node)

        return node.outputs if len(node.outputs) > 1 else node.outputs[0]

    def call(self, function, *args, **kwargs):
        if isinstance(function, ir.Function):
            function_ir = function
        elif isinstance(function, onnxscript.values.OnnxFunction):
            function_proto = function.to_function_proto()
            function_ir = ir.serde.deserialize_function(function_proto)
        else:
            raise TypeError("Function must be an ir.Function or onnxscript.ONNXFunction")
        nodes, outputs = inliner.instantiate(function_ir, args, kwargs)
        for node in nodes:
            self.add_node(node)
        return outputs if len(outputs) > 1 else outputs[0]
    
    def push_module(self, module: str) -> None:
        current = self.context_name()
        if module.name:
            new_context = f"{current}.{module.name}" if current else module.name
        else:
            new_context = current
        self._context_stack.append(new_context)

    def pop_module(self) -> None:
        self._context_stack.pop()

    def context_name(self) -> str:
        return self._context_stack[-1] if self._context_stack else ""

class OpBuilder:
    def __init__(self, builder: GraphBuilder, domain: str = "", version: int | None = None) -> None:
        self._builder = builder
        self._domain = domain
        self._version = version

    @property
    def builder(self) -> GraphBuilder:
        return self._builder

    def _call_op(self, op_type: str, inputs: Sequence[Any], kwargs: dict[str, Any]):
        if "_domain" not in kwargs:
            kwargs["_domain"] = self._domain
        if self._version is not None and "_version" not in kwargs:
            kwargs["_version"] = self._version
        return self._builder.call_op(op_type, inputs, kwargs)

    def __getattr__(self, op_type: str) -> Callable:
        return lambda *args, **kwargs: self._call_op(op_type, args, kwargs)

    def initializer(self, tensor: ir.TensorProtocol, name: str | None = None) -> ir.Value:
        return self._builder.initializer(tensor, name)

    def call(self, function, *args, **kwargs):
        return self._builder.call(function, *args, **kwargs)


