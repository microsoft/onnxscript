# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, Callable, Sequence

import onnx
import onnx_ir as ir

import onnxscript._internal._inference as inference
import onnxscript.optimizer

_DTYPE_SUFFIX_MAP: dict[ir.DataType, str] = {
    ir.DataType.FLOAT: "f32",
    ir.DataType.DOUBLE: "f64",
    ir.DataType.FLOAT16: "f16",
    ir.DataType.BFLOAT16: "bf16",
    ir.DataType.INT8: "i8",
    ir.DataType.INT16: "i16",
    ir.DataType.INT32: "i32",
    ir.DataType.INT64: "i64",
    ir.DataType.UINT8: "u8",
    ir.DataType.UINT16: "u16",
    ir.DataType.UINT32: "u32",
    ir.DataType.UINT64: "u64",
    ir.DataType.BOOL: "bool",
    ir.DataType.STRING: "str",
}


class GraphBuilder:
    def __init__(self, graph: ir.Graph) -> None:
        self._graph = graph

        # Get the opset version for "" (default domain) from the graph
        if "" not in graph.opset_imports:
            # Force this for now. Default opset version for "" is problematic.
            raise ValueError('Input graph does not have an import for domain ""')
        opset_version = graph.opset_imports[""]

        self._op_builder = self.opset("", opset_version)

        # Context stack to manage hierarchical naming. Each module/layer can push a new context, and pop it when done.
        # The current context is used as a prefix for naming values and nodes.
        # This allows us to generate names like "layer1.attention.query"
        self._context_stack: list[str] = [""]

        # Cache for constant initializers (scalars and sequences), keyed by (value, dtype).
        # This avoids creating duplicate initializers for the same constant
        # and allows sharing them across different layers/contexts.
        self._constant_cache: dict[tuple[Any, ir.DataType | None], ir.Value] = {}

    def opset(self, domain: str, version: int = 1) -> OpBuilder:
        return OpBuilder(self, domain, version)

    @property
    def op(self) -> OpBuilder:
        return self._op_builder

    @property
    def graph(self) -> ir.Graph:
        return self._graph

    def initializer(
        self, tensor: ir.TensorProtocol, name: str | None = None, *, qualify: bool = True
    ) -> ir.Value:
        if name is None:
            name = tensor.name
        if qualify:
            name = self.qualify_name(name)
        tensor.name = name
        shape = ir.Shape((d if isinstance(d, int) else d.value) for d in tensor.shape.dims)
        value = ir.Value(
            name=name, shape=shape, type=ir.TensorType(tensor.dtype), const_value=tensor
        )
        self._graph.register_initializer(value)
        return value

    @staticmethod
    def _type_suffix(element_type: type) -> str:
        """Return a short type suffix for naming constants based on Python type."""
        if element_type is int:
            return "i64"
        if element_type is float:
            return "f32"
        return ""

    @staticmethod
    def _dtype_suffix(dtype: ir.DataType) -> str:
        """Return a short type suffix for naming constants based on ir.DataType."""
        return _DTYPE_SUFFIX_MAP.get(dtype, dtype.name.lower())

    @staticmethod
    def _constant_name(value: int | float | bool | str | Sequence, type_suffix: str) -> str:
        """Generate a descriptive name for a constant value."""
        if isinstance(value, (int, float, bool, str)):
            return f"const_{value}_{type_suffix}" if type_suffix else f"const_{value}"
        # Sequence: use up to 2 elements in the name
        if len(value) <= 2:
            vals = ",".join(str(v) for v in value)
        else:
            vals = f"{value[0]},{value[1]},..."
        return f"const_[{vals}]_{type_suffix}" if type_suffix else f"const_[{vals}]"

    def _input_to_ir_value(self, value, like_type: ir.Value | None = None) -> ir.Value:
        if isinstance(value, ir.Value):
            return value
        dtype = (
            like_type.type.dtype
            if like_type is not None and like_type.type is not None
            else None
        )
        needs_dynamic_cast = like_type is not None and dtype is None
        # For simple scalar/sequence constants, use a cache to avoid duplicate initializers.
        # These are shared across layers, so we don't qualify the name with context prefix.
        if isinstance(value, (int, float, bool, str)):
            cache_key = (value, dtype)
            if cache_key in self._constant_cache:
                ir_value = self._constant_cache[cache_key]
            else:
                type_suffix = (
                    self._dtype_suffix(dtype)
                    if dtype is not None
                    else self._type_suffix(type(value))
                )
                name = self._constant_name(value, type_suffix)
                tensor = ir.tensor(value, dtype=dtype, name=name)
                ir_value = self.initializer(tensor, name=name, qualify=False)
                self._constant_cache[cache_key] = ir_value
        elif (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(v, type(value[0])) for v in value)
            and isinstance(value[0], (int, float, bool, str))
        ):
            cache_key = (tuple(value), dtype)
            if cache_key in self._constant_cache:
                ir_value = self._constant_cache[cache_key]
            else:
                type_suffix = (
                    self._dtype_suffix(dtype)
                    if dtype is not None
                    else self._type_suffix(type(value[0]))
                )
                name = self._constant_name(value, type_suffix)
                tensor = ir.tensor(list(value), dtype=dtype, name=name)
                ir_value = self.initializer(tensor, name=name, qualify=False)
                self._constant_cache[cache_key] = ir_value
        else:
            # For other types (TensorProtocol, numpy arrays, torch tensors, etc.),
            # ir.tensor() handles the conversion.
            # TODO(rama): Consider caching for other tensor values.
            ir_value = self.initializer(ir.tensor(value, dtype=dtype))
        # If like_type is provided but its type is unknown, insert a dynamic CastLike
        # so the constant is cast to match like_type's type at runtime.
        if needs_dynamic_cast:
            ir_value = self.op.CastLike(ir_value, like_type)
        return ir_value

    def _adapt_outputs(
        self, outputs: int | Sequence[str | ir.Value], op_type: str = ""
    ) -> Sequence[ir.Value]:
        if isinstance(outputs, int):
            if outputs == 1:
                name = f"{op_type}_output" if op_type else "output"
                return [ir.Value(name=self.qualify_name(name))]
            else:
                names = [
                    f"{op_type}_output{i}" if op_type else f"output{i}" for i in range(outputs)
                ]
                return [ir.Value(name=self.qualify_name(n)) for n in names]
        adapted_outputs = []
        for output in outputs:
            if isinstance(output, ir.Value):
                if output.name:
                    output.name = self.qualify_name(output.name)
                adapted_outputs.append(output)
            elif isinstance(output, str):
                adapted_outputs.append(ir.Value(name=self.qualify_name(output)))
            else:
                raise TypeError("Output type not supported.")
        return adapted_outputs

    def _get_schema(
        self, op_type: str, domain: str, version: int | None
    ) -> onnx.defs.OpSchema | None:
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
        del schema
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
            elif (
                expected_inputs[-1].option == onnx.defs.OpSchema.FormalParameterOption.Variadic
            ):
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

        def adapt(x, typevar: str | None) -> ir.Value | None:
            if x is None:
                return None
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
        del schema  # Not implemented yet
        return attributes if attributes is not None else {}

    def add_node(self, node: ir.Node) -> None:
        self.graph.append(node)
        onnxscript.optimizer.basic_constant_propagation([node])
        inference.infer_outputs(node)

    def call_op(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | ir.TensorProtocol],
        kwargs: dict[str, Any],
    ):
        domain = kwargs.pop("_domain", "")
        version = kwargs.pop("_version", None)
        outputs = kwargs.pop("_outputs", 1)

        count = self.graph.num_nodes()
        node_name = self.qualify_name(f"{op_type}_node_{count}")

        output_values = self._adapt_outputs(outputs, op_type)

        schema = self._get_schema(op_type, domain, version)
        inputs, attributes = self._partition_inputs_attributes(schema, inputs, kwargs)
        inputs = self._cast_inputs(schema, inputs)
        attributes = self._cast_attributes(schema, attributes)

        node = ir.node(
            op_type,
            inputs,
            attributes=attributes or None,
            domain=domain,
            outputs=output_values,
            version=version,
            name=node_name,
        )
        self.add_node(node)

        return node.outputs if len(node.outputs) > 1 else node.outputs[0]

    def push_module(self, module: str) -> None:
        current = self.context_name()
        if module:
            new_context = f"{current}.{module}" if current else module
        else:
            new_context = current
        self._context_stack.append(new_context)

    def pop_module(self) -> None:
        if len(self._context_stack) <= 1:
            raise RuntimeError("Cannot pop_module: no module context has been pushed.")
        self._context_stack.pop()

    def context_name(self) -> str:
        return self._context_stack[-1] if self._context_stack else ""

    def qualify_name(self, name: str) -> str:
        """Prepend the current hierarchical context prefix to the given name."""
        prefix = self.context_name()
        return f"{prefix}.{name}" if prefix else name


class OpBuilder:
    def __init__(
        self, builder: GraphBuilder, domain: str = "", version: int | None = None
    ) -> None:
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
