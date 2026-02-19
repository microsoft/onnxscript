# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, Callable, Sequence, Union

import onnx
import onnx_ir as ir

import onnxscript._internal._inference as inference
import onnxscript.optimizer

# A permissible value for an op input, which can be converted to an ir.Value.
VALUE_LIKE = Union[
    ir.Value,
    ir.TensorProtocol,
    int,
    float,
    bool,
    str,
    Sequence[int],
    Sequence[float],
    Sequence[bool],
    Sequence[str],
]

# Mapping from Python scalar types to their default ONNX DataType,
# used when no schema-based type binding is available.
_PYTHON_TYPE_TO_DTYPE: dict[type, ir.DataType] = {
    int: ir.DataType.INT64,
    float: ir.DataType.FLOAT,
}


def _type_suffix(element_type: type) -> str:
    """Return a short type suffix for naming constants based on Python type."""
    dtype = _PYTHON_TYPE_TO_DTYPE.get(element_type)
    return dtype.short_name() if dtype is not None else ""


def _dtype_suffix(dtype: ir.DataType) -> str:
    """Return a short type suffix for naming constants based on ir.DataType."""
    return dtype.short_name()


def _constant_name(
    value: int | float | bool | str | Sequence, type_suffix: str, num: int = 0
) -> str:
    """Generate a descriptive name for a constant value.

    Args:
        value: The constant value
        type_suffix: Type suffix (e.g., 'F', 'I64')
        num: A number used for generating unique names for str/sequences

    Returns:
        A name string for the constant
    """
    if isinstance(value, str):
        # For strings, use a generic name with cache size as unique identifier
        return f"const_str_{num}"
    if isinstance(value, (int, float, bool)):
        return f"const_{value}_{type_suffix}" if type_suffix else f"const_{value}"
    # Sequence: use generic name with cache size as unique identifier
    return f"const_1d_{num}"


class GraphBuilder:
    """Imperative builder for constructing ONNX IR graphs with automatic constant promotion, type casting, and shape inference."""

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
        """Create an OpBuilder bound to the given domain and version."""
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
        """Register a tensor as a graph initializer, returning the corresponding ir.Value."""
        if name is None:
            name = tensor.name
        if qualify:
            name = self.qualify_name(name)
        shape = ir.Shape(tensor.shape)
        value = ir.Value(
            name=name, shape=shape, type=ir.TensorType(tensor.dtype), const_value=tensor
        )
        self._graph.register_initializer(value)
        return value

    def _input_to_ir_value(
        self, value: VALUE_LIKE, like_type: ir.Value | None = None
    ) -> ir.Value:
        """Convert a permissible input (for a call to an op) into an ir.Value.

        Permissible values include ir.Value as well as python constants that can be converted
        into ONNX constant tensors. For constant values, the like_type is used to determine the
        target onnx type.
        """
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
                    _dtype_suffix(dtype) if dtype is not None else _type_suffix(type(value))
                )
                name = _constant_name(value, type_suffix, len(self._constant_cache))
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
                    _dtype_suffix(dtype) if dtype is not None else _type_suffix(type(value[0]))
                )
                name = _constant_name(value, type_suffix, len(self._constant_cache))
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
            if outputs < 0:
                raise ValueError(f"Number of outputs must be non-negative, got {outputs}")
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
        inputs: Sequence[VALUE_LIKE],
    ) -> Sequence[ir.Value | None]:
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
            type_like = type_bindings.get(typevar)
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
        """Append a node to the graph, run constant propagation and shape inference."""
        self.graph.append(node)
        onnxscript.optimizer.basic_constant_propagation([node])
        inference.infer_outputs(node)

    def call_op(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | ir.TensorProtocol],
        kwargs: dict[str, Any],
    ):
        """Create an ONNX node and add it to the graph, returning its output value(s)."""
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
        """Push a new naming context onto the stack (e.g. a layer or module name)."""
        current = self.context_name()
        if module:
            new_context = f"{current}.{module}" if current else module
        else:
            new_context = current
        self._context_stack.append(new_context)

    def pop_module(self) -> None:
        """Pop the most recent naming context off the stack."""
        if len(self._context_stack) <= 1:
            raise RuntimeError("Cannot pop_module: no module context has been pushed.")
        self._context_stack.pop()

    def context_name(self) -> str:
        """Return the current dot-separated naming context prefix."""
        return self._context_stack[-1] if self._context_stack else ""

    def qualify_name(self, name: str) -> str:
        """Prepend the current hierarchical context prefix to the given name."""
        prefix = self.context_name()
        return f"{prefix}.{name}" if prefix else name


class OpBuilder:
    """Dynamic op dispatcher that translates attribute access into ONNX node creation via a GraphBuilder."""

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
