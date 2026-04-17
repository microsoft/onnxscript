# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Graph builder for constructing ONNX IR graphs imperatively.

This module provides imperative builders for constructing ONNX IR graphs with automatic
constant promotion, type casting, and shape inference. The GraphBuilder class enables
programmatic construction of graphs with proper scoping, constant management, and node
creation. The OpBuilder class provides dynamic op dispatching via attribute access.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Sequence, Union

import onnx
import onnx_ir as ir

import onnxscript._internal._inference as inference
import onnxscript.optimizer
from onnxscript._internal import _inliner, param_manipulation

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
    None,
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


def lift_initializers_to_constants(graph: ir.Graph) -> None:
    """Replace every initializer in *graph* with a ``Constant`` node.

    ONNX ``ir.Function`` bodies do not support initializers — all values
    must be produced by nodes.  Call this on the function-body graph
    **before** wrapping it in :class:`ir.Function` so that any constant
    initializers (e.g. from Python literals promoted by
    :class:`GraphBuilder`) become valid ``Constant`` nodes.

    The function preserves ``ir.Value`` identity: it reuses each existing
    ``ir.Value`` object as the output of the new ``Constant`` node so that
    all downstream references remain valid.

    Graph inputs that are *also* registered as initializers (the standard
    ONNX pattern for optional inputs with default values) are skipped
    because they are explicit function parameters, not embedded constants.
    """
    graph_input_set = {id(v) for v in graph.inputs}
    to_lift: list[ir.Value] = [
        v for v in list(graph.initializers.values()) if id(v) not in graph_input_set
    ]
    opset_version = graph.opset_imports.get("", 1)
    new_nodes: list[ir.Node] = []
    for value in to_lift:
        tensor = value.const_value
        if tensor is None:
            raise ValueError(f"Initializer {value.name!r} has no const_value")
        # Build a Constant node whose output is the *same* ir.Value so
        # that every existing reference keeps working.
        node = ir.Node(
            "",
            "Constant",
            inputs=[],
            attributes=[ir.Attr("value", ir.AttributeType.TENSOR, tensor)],
            outputs=[value],
            version=opset_version,
            name=f"initializer_{value.name}",
        )
        graph.initializers.pop(value.name)
        new_nodes.append(node)
    # Insert all Constant nodes at the beginning of the graph.
    if new_nodes:
        first_existing = graph.node(0) if graph.num_nodes() > 0 else None
        if first_existing is not None:
            graph.insert_before(first_existing, new_nodes)
        else:
            for n in new_nodes:
                graph.append(n)


# Type accepted as an element of *inputs* / *outputs* by
# :meth:`GraphBuilder.subgraph`.  Can be an already-resolved
# :class:`ir.TypeAndShape`, or a
# :class:`~onnxscript.onnx_types.TensorType` subclass such as ``FLOAT[1024]``.
#
# .. deprecated::
#     Use ``ir.Value`` with name/type/shape directly instead.
TypeSpec = Union[ir.TypeAndShape, Any]


def _resolve_type_spec(spec: TypeSpec) -> ir.TypeAndShape:
    """Convert a *TypeSpec* to an :class:`ir.TypeAndShape`.

    Accepts an :class:`ir.TypeAndShape` directly, or any object with a
    ``to_ir_type_and_shape()`` method (e.g. a
    :class:`~onnxscript.onnx_types.TensorType` subclass such as
    ``FLOAT[1024]`` or ``FLOAT['M', 'N']``).

    .. deprecated::
        Use :func:`make_value` or construct ``ir.Value`` directly instead.
    """
    if isinstance(spec, ir.TypeAndShape):
        return spec
    if hasattr(spec, "to_ir_type_and_shape"):
        result = spec.to_ir_type_and_shape()
        if not isinstance(result, ir.TypeAndShape):
            raise TypeError(
                f"{type(spec)!r}.to_ir_type_and_shape() returned {type(result)!r}, "
                f"expected ir.TypeAndShape."
            )
        return result
    raise TypeError(
        f"Expected ir.TypeAndShape or an object with a to_ir_type_and_shape() method, "
        f"got {type(spec)!r}."
    )


def make_value(name: str, type_spec: TypeSpec | None = None) -> ir.Value:
    """Create an :class:`ir.Value` from a name and optional :data:`TypeSpec`.

    Similar to :func:`onnx_ir.val` but accepts a :data:`TypeSpec` (e.g.
    ``FLOAT[3, 4]``) instead of separate *dtype* and *shape* arguments.

    Example::

        x = make_value("x", FLOAT[3, 4])
        y = make_value("y")  # untyped

    Args:
        name: The value name.
        type_spec: Optional type specification.  Accepts an
            :class:`ir.TypeAndShape`, or a
            :class:`~onnxscript.onnx_types.TensorType` subclass
            (e.g. ``FLOAT[3, 4]``).

    Returns:
        A fresh :class:`ir.Value` with the given name and optional type/shape.
    """
    if type_spec is not None:
        ts = _resolve_type_spec(type_spec)
        return ir.Value(name=name, type=ts.type, shape=ts.shape)
    return ir.Value(name=name)


def _split_optional_inputs(
    inputs: Sequence[ir.Value | None],
) -> tuple[list[ir.Value | None], list[ir.Value]]:
    """Split an input list into trace args and graph inputs.

    For each ``None`` entry, a placeholder :class:`ir.Value` with a generated
    name (``input_0``, ``input_1``, …) is created and added to
    *graph_inputs* so that the function/graph signature declares the formal
    parameter.  The corresponding *trace_args* entry remains ``None`` so that
    the trace function can branch with ``if x is None:``.

    Returns:
        A tuple of (trace_args, graph_inputs) where trace_args preserves
        ``None`` holes and graph_inputs includes placeholders for absent
        optional inputs.

    Raises:
        ValueError: If any non-None input already has a producer or is
            already attached to a graph.
    """
    trace_args: list[ir.Value | None] = list(inputs)
    graph_inputs: list[ir.Value] = []
    for i, v in enumerate(trace_args):
        if v is None:
            # Placeholder: declared in function signature but unused in body.
            graph_inputs.append(ir.Value(name=f"input_{i}"))
        else:
            if v.producer() is not None:
                raise ValueError(
                    f"Input {v.name!r} already has a producer node. "
                    f"Pass freshly created ir.Value objects."
                )
            if v.graph is not None:
                raise ValueError(
                    f"Input {v.name!r} is already attached to a graph. "
                    f"Pass freshly created ir.Value objects."
                )
            graph_inputs.append(v)
    return trace_args, graph_inputs


def build_graph(
    trace_function: Callable,
    inputs: Sequence[ir.Value | None],
    outputs: Sequence[ir.Value],
    *,
    opset_imports: dict[str, int],
    name: str = "subgraph",
    parent: GraphBuilder | None = None,
) -> ir.Graph:
    """Build an :class:`ir.Graph` suitable for use as a graph-valued attribute.

    This is a module-level utility that constructs a subgraph by tracing
    *trace_function*.  It is useful for building body graphs of control-flow ops
    such as ``Scan``, ``Loop``, and ``If``.

    Example - building a Scan body that adds two sequences element-wise::

        body = build_graph(
            lambda op, x, y: op.Add(x, y),
            inputs=[make_value("x", FLOAT[3, 4]), make_value("y", FLOAT[3, 4])],
            outputs=[make_value("sum", FLOAT[3, 4])],
            opset_imports={"": 23},
        )

    Args:
        trace_function: A callable with signature
            ``(op: OpBuilder, *inputs: ir.Value | None) -> ir.Value | Sequence[ir.Value]``.
            It is called once with freshly created placeholder inputs to record the
            graph topology.  ``None`` entries in *inputs* are passed through as ``None``
            to support optional inputs.
        inputs: A :class:`Sequence` of :class:`ir.Value` (or ``None`` for
            absent optional inputs).  Each ``ir.Value`` should be freshly
            created with a name and optional type/shape.  For ``None``
            entries, placeholder values are declared as formal graph inputs,
            while ``None`` is passed to *trace_function* for the
            corresponding argument position.
        outputs: A :class:`Sequence` of :class:`ir.Value` objects declaring
            the expected outputs.  After tracing, the name and type of each
            declared output are applied to the corresponding returned value.
        opset_imports: Opset version map for the subgraph (e.g.
            ``{"": 23}``).
        name: Name of the resulting :class:`ir.Graph`.
        parent: Optional parent :class:`GraphBuilder`.  When provided, the
            sub-builder's ``_root`` points to the root builder of the parent,
            so that :meth:`Parameter._realize` registers initializers in the
            root (main) graph rather than the subgraph.

    Returns:
        An :class:`ir.Graph` whose inputs and outputs are populated and whose
        nodes record the operations traced by *trace_function*.  This graph can be
        passed directly as a graph-valued attribute (e.g. the ``body`` attribute of
        a ``Scan`` or ``Loop`` node).
    """
    trace_args, graph_inputs = _split_optional_inputs(inputs)

    subgraph = ir.Graph(
        name=name,
        inputs=graph_inputs,
        outputs=[],
        nodes=[],
        opset_imports=opset_imports,
    )

    sub_builder = GraphBuilder(subgraph, parent=parent)
    if parent is not None:
        sub_builder._scope_stack = list(parent._scope_stack)
    trace_outputs = trace_function(sub_builder.op, *trace_args)
    if not isinstance(trace_outputs, Sequence):
        trace_outputs = [trace_outputs]
    if len(trace_outputs) != len(outputs):
        raise ValueError(
            f"trace_function returned {len(trace_outputs)} output(s), "
            f"but {len(outputs)} were declared in outputs."
        )
    for returned_val, declared_val in zip(trace_outputs, outputs):
        if declared_val.name:
            returned_val.name = declared_val.name
        if declared_val.type is not None:
            if returned_val.type is not None and returned_val.type != declared_val.type:
                raise ValueError(
                    f"Output {declared_val.name!r}: traced type "
                    f"{returned_val.type} conflicts with declared type "
                    f"{declared_val.type}."
                )
            returned_val.type = declared_val.type
        if declared_val.shape is not None:
            returned_val.merge_shapes(declared_val.shape)

    subgraph.outputs.extend(trace_outputs)
    return subgraph


def build_function(
    trace_function: Callable,
    inputs: Sequence[ir.Value | None],
    *,
    domain: str,
    name: str,
    attributes: Mapping[str, ir.Attr] | Sequence[ir.Attr] | None = None,
    opset_imports: dict[str, int],
) -> ir.Function:
    """Build an :class:`ir.Function` by tracing *trace_function*.

    This utility handles all boilerplate for constructing an ``ir.Function``:
    graph creation, input/output wiring, initializer lifting (so that Python
    literals work correctly inside function bodies), and attribute packaging.

    Example::

        fn = build_function(
            lambda op, x, y: op.Add(x, y),
            [make_value("x"), make_value("y")],
            domain="com.example",
            name="MyAdd",
            opset_imports={"": 23},
        )

    Args:
        trace_function: A callable with signature
            ``(op: OpBuilder, *inputs: ir.Value | None) -> ir.Value | Sequence[ir.Value] | None``.
            It is called once to trace the function body.  Return value(s)
            become function outputs.  If ``None`` is returned, the function
            uses whatever outputs were appended to ``graph.outputs`` by the
            trace function directly.
        inputs: A :class:`Sequence` of :class:`ir.Value` (or ``None`` for
            absent optional inputs).  ``None`` entries are represented by
            placeholder formal inputs in the generated function signature,
            while ``None`` is passed through to *trace_function* in the
            corresponding positions so the body can branch with
            ``if x is None``.
        domain: Function domain (e.g. ``"com.microsoft"``).
        name: Function name (e.g. ``"LinearAttention"``).
        attributes: Function-level attributes.  Accepts a
            :class:`Mapping` from name to :class:`ir.Attr`, a
            :class:`Sequence` of :class:`ir.Attr`, or ``None``.
        opset_imports: Opset version map (e.g. ``{"": 23}``).

    Returns:
        An :class:`ir.Function` with initializers automatically lifted to
        ``Constant`` nodes.
    """
    trace_args, graph_inputs = _split_optional_inputs(inputs)

    graph = ir.Graph(
        inputs=graph_inputs,
        outputs=[],
        nodes=[],
        name=f"{name}_body",
        opset_imports=opset_imports,
    )

    gb = GraphBuilder(graph)  # No parent — function is self-contained
    trace_outputs = trace_function(gb.op, *trace_args)

    # Normalize outputs: either returned or appended directly, not both.
    if trace_outputs is not None:
        if not isinstance(trace_outputs, Sequence):
            trace_outputs = [trace_outputs]
        if graph.outputs:
            raise ValueError(
                "trace_function both returned output values and appended "
                "to graph.outputs. Use one approach, not both."
            )
        graph.outputs.extend(trace_outputs)
    elif not graph.outputs:
        raise ValueError(
            "trace_function returned None and did not append any outputs to graph.outputs."
        )

    # Lift initializers → Constant nodes (required for ir.Function bodies).
    lift_initializers_to_constants(graph)

    # Build attributes dict.
    if attributes is None:
        attr_dict: dict[str, ir.Attr] = {}
    elif isinstance(attributes, Mapping):
        attr_dict = dict(attributes)
    else:
        attr_dict = {a.name: a for a in attributes}

    return ir.Function(
        domain=domain,
        name=name,
        graph=graph,
        attributes=attr_dict,
    )


@dataclasses.dataclass(frozen=True)
class GraphBuilderOptions:
    """Controls optional behaviours in :class:`GraphBuilder`.

    All options default to ``True`` for backwards compatibility with the
    existing ``GraphBuilder`` (used in tracing / export).  For a
    tape-equivalent builder (matching ``_tape.Builder`` behaviour), use
    :data:`TAPE_COMPATIBLE_OPTIONS` which sets every option to ``False``.
    """

    constant_propagation: bool = True
    """Run ``basic_constant_propagation`` on each node after creation."""

    shape_inference: bool = True
    """Run ``infer_outputs`` (shape and type inference) on each node after creation."""

    auto_cast_inputs: bool = True
    """Use ONNX schema to auto-cast Python literals and match sibling input types."""

    scope_metadata: bool = True
    """Attach namespace / class_hierarchy / name_scopes metadata_props to nodes."""

    auto_name_nodes: bool = True
    """Auto-generate qualified node names when no explicit ``_name`` kwarg is given.
    When ``False``, the node name is ``None`` unless ``_name`` is provided."""


EXPORT_OPTIONS = GraphBuilderOptions()
"""Default options for tracing / export — all features enabled."""

TAPE_COMPATIBLE_OPTIONS = GraphBuilderOptions(
    constant_propagation=False,
    shape_inference=False,
    auto_cast_inputs=False,
    scope_metadata=False,
    auto_name_nodes=False,
)
"""Options that replicate ``_tape.Builder`` behaviour — all enhanced features disabled."""


class GraphBuilder:
    """Imperative builder for constructing ONNX IR graphs with automatic constant promotion, type casting, and shape inference."""

    def __init__(
        self,
        graph: ir.Graph,
        *,
        parent: GraphBuilder | None = None,
        options: GraphBuilderOptions | None = None,
    ) -> None:
        self._graph = graph
        self._parent = parent
        self._root: GraphBuilder = parent._root if parent is not None else self

        # Resolve options: explicit > inherited from parent > default
        if options is not None:
            self._options = options
        elif parent is not None:
            self._options = parent._options
        else:
            self._options = EXPORT_OPTIONS

        # Get the opset version for "" (default domain) from the graph
        if "" not in graph.opset_imports:
            # Force this for now. Default opset version for "" is problematic.
            raise ValueError('Input graph does not have an import for domain ""')
        opset_version = graph.opset_imports[""]

        self._op_builder = self.opset("", opset_version)

        # Module scope stack. Each entry is (name, class_name) where name is
        # the module attribute name (e.g. "layers.0", "self_attn") and
        # class_name is the qualified class name (e.g. "Gemma3DecoderLayer").
        self._scope_stack: list[tuple[str, str]] = []

        # Cache for constant initializers (scalars and sequences), keyed by
        # (value, dtype).  Only the **root** builder owns a cache; child
        # builders delegate to ``self._root`` so that all constant
        # initializers live in the root graph (outer-scope initializers are
        # visible to subgraphs per the ONNX spec).
        if parent is None:
            self._constant_cache: dict[tuple[Any, ir.DataType | None], ir.Value] = {}

    def opset(self, domain: str, version: int = 1) -> OpBuilder:
        """Create an OpBuilder bound to the given domain and version."""
        return OpBuilder(self, domain, version)

    @property
    def op(self) -> OpBuilder:
        return self._op_builder

    @property
    def options(self) -> GraphBuilderOptions:
        """The options controlling this builder's behaviour."""
        return self._options

    @property
    def parent(self) -> GraphBuilder | None:
        """The parent builder, or None for a top-level builder."""
        return self._parent

    @property
    def root(self) -> GraphBuilder:
        """The root (top-level) builder in the parent chain."""
        return self._root

    @property
    def graph(self) -> ir.Graph:
        return self._graph

    def initializer(
        self, tensor: ir.TensorProtocol, name: str | None = None, *, qualify: bool = True
    ) -> ir.Value:
        """Register a tensor as a graph initializer in the **root** graph.

        Initializers created through this method are stored in the root graph
        so that inner scopes (subgraphs) can reference them via ONNX's
        outer-scope visibility rules.  This does not apply to the ONNX
        default-input pattern created via :meth:`input` with ``const_value``,
        which registers an initializer on the owning graph.  For function
        bodies (which cannot have initializers), apply
        :func:`lift_initializers_to_constants` before wrapping in
        :class:`ir.Function`.
        """
        if name is None:
            name = tensor.name
        if qualify:
            name = self._qualify_initializer_name(name)
        shape = ir.Shape(tensor.shape)
        value = ir.Value(
            name=name, shape=shape, type=ir.TensorType(tensor.dtype), const_value=tensor
        )
        self._root._graph.register_initializer(value)
        return value

    def input(
        self,
        name: str,
        dtype: ir.DataType | None = None,
        shape: ir.Shape | Sequence[int | str | None] | None = None,
        *,
        type: ir.TypeProtocol | None = None,
        const_value: ir.TensorProtocol | None = None,
        metadata_props: dict[str, str] | None = None,
    ) -> ir.Value:
        """Create an input to the graph and return the corresponding ir.Value.

        Args:
            name: The name of the value.
            dtype: The data type of the TensorType of the value. This is used only when type is None.
            shape: The shape of the value.
            type: The type of the value. Only one of dtype and type can be specified.
            const_value: The constant tensor that initializes the value. Supply this argument
                when you want to create an initializer. The type and shape can be obtained from the tensor.
            metadata_props: The metadata properties that will be serialized to the ONNX proto.

        Returns:
            A Value object.
        """
        value = ir.val(
            name=name,
            dtype=dtype,
            shape=shape,
            type=type,
            const_value=const_value,
            metadata_props=metadata_props,
        )
        self._graph.inputs.append(value)
        if const_value is not None:
            self._graph.register_initializer(value)
        return value

    def add_output(self, value: ir.Value, name: str | None) -> None:
        """Add an output to the graph.

        Args:
            value: The ir.Value to add as an output.
            name: The name to assign to the output value. If None, no renaming is done.
        """
        if name:
            value.name = name
        self._graph.outputs.append(value)

    def _get_or_create_constant(
        self, value: VALUE_LIKE, dtype: ir.DataType | None
    ) -> ir.Value:
        """Materialise a constant as an initializer in the **root** graph.

        Child builders delegate to the root so that all constant initializers
        live in the root graph.  For subgraphs this is correct because ONNX
        allows inner scopes to reference outer-scope initializers.  For
        function bodies (which cannot reference outer initializers) callers
        should apply :func:`lift_initializers_to_constants` before wrapping
        the graph in :class:`ir.Function`.
        """
        root = self._root
        if isinstance(value, (int, float, bool, str)):
            if dtype is None:
                dtype = _PYTHON_TYPE_TO_DTYPE.get(type(value))
            cache_key = (value, dtype)
            if cache_key in root._constant_cache:
                return root._constant_cache[cache_key]
            type_suffix = _dtype_suffix(dtype) if dtype is not None else ""
            name = _constant_name(value, type_suffix, len(root._constant_cache))
            tensor = ir.tensor(value, dtype=dtype, name=name)
            ir_value = root.initializer(tensor, name=name, qualify=False)
            root._constant_cache[cache_key] = ir_value
            return ir_value
        if (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(v, type(value[0])) for v in value)
            and isinstance(value[0], (int, float, bool, str))
        ):
            if dtype is None:
                dtype = _PYTHON_TYPE_TO_DTYPE.get(type(value[0]))
            cache_key = (tuple(value), dtype)
            if cache_key in root._constant_cache:
                return root._constant_cache[cache_key]
            type_suffix = _dtype_suffix(dtype) if dtype is not None else ""
            name = _constant_name(value, type_suffix, len(root._constant_cache))
            tensor = ir.tensor(list(value), dtype=dtype, name=name)
            ir_value = root.initializer(tensor, name=name, qualify=False)
            root._constant_cache[cache_key] = ir_value
            return ir_value
        # For other types (TensorProtocol, numpy arrays, torch tensors, etc.),
        # ir.tensor() handles the conversion.
        # TODO(rama): Consider caching for other tensor values.
        return self.initializer(ir.tensor(value, dtype=dtype))

    def _input_to_ir_value(
        self, value: VALUE_LIKE, like_type: ir.Value | None = None
    ) -> ir.Value | None:
        """Convert a permissible input (for a call to an op) into an ir.Value.

        Permissible values include ir.Value as well as python constants that can be converted
        into ONNX constant tensors. For constant values, the like_type is used to determine the
        target onnx type.
        """
        if isinstance(value, ir.Value):
            return value
        if value is None:
            return value
        dtype = (
            like_type.type.dtype
            if like_type is not None and like_type.type is not None
            else None
        )
        needs_dynamic_cast = like_type is not None and dtype is None
        ir_value = self._get_or_create_constant(value, dtype)
        # If like_type is provided but its type is unknown, insert a dynamic CastLike
        # so the constant is cast to match like_type's type at runtime.
        # The CastLike node is created in THIS builder's graph (not root),
        # so that it lives in the correct scope (subgraph or function body).
        if needs_dynamic_cast:
            ir_value = self.op.CastLike(ir_value, like_type)
        return ir_value

    def _adapt_outputs(
        self, outputs: int | Sequence[str | ir.Value], op_type: str = ""
    ) -> Sequence[ir.Value]:
        if isinstance(outputs, int):
            count = self.graph.num_nodes()
            if outputs < 0:
                raise ValueError(f"Number of outputs must be non-negative, got {outputs}")
            if outputs == 1:
                name = f"{op_type}_{count}" if op_type else f"{count}"
                return [ir.Value(name=self._qualify_value_name(name))]
            else:
                names = [
                    (f"{op_type}_{count}_{i}" if op_type else f"{count}_{i}")
                    for i in range(outputs)
                ]
                return [ir.Value(name=self._qualify_value_name(n)) for n in names]
        adapted_outputs = []
        for output in outputs:
            if isinstance(output, ir.Value):
                if output.name:
                    output.name = self._qualify_value_name(output.name)
                adapted_outputs.append(output)
            elif isinstance(output, str):
                adapted_outputs.append(ir.Value(name=self._qualify_value_name(output)))
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
        inputs: Sequence[ir.Value | ir.TensorProtocol | None],
        kwargs: dict[str, Any],
    ) -> tuple[Sequence[ir.Value | ir.TensorProtocol], dict[str, Any]]:
        if schema is None:
            return inputs, kwargs
        op_signature = ir.schemas.OpSignature.from_op_schema(schema)
        return param_manipulation.separate_input_attributes_from_arguments(
            op_signature,
            list(inputs),
            kwargs,
            fill_defaults=False,
            allow_extra_args=False,
        )

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
            elif expected_inputs and (
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
        """Append a node to the graph and optionally run constant propagation and shape inference."""
        self.graph.append(node)
        if self._options.constant_propagation:
            onnxscript.optimizer.basic_constant_propagation([node])
        if self._options.shape_inference:
            inference.infer_outputs(node)

    def subgraph(
        self,
        trace_function: Callable,
        inputs: Sequence[ir.Value | None],
        outputs: Sequence[ir.Value],
        *,
        name: str = "subgraph",
    ) -> ir.Graph:
        """Build an :class:`ir.Graph` suitable for use as a graph-valued attribute.

        The subgraph inherits the opset version from this :class:`GraphBuilder`.
        It is particularly useful for constructing the body graphs of control-flow ops
        such as ``Scan``, ``Loop``, and ``If``.

        Example - building a Scan body that adds two sequences element-wise::

            body = graph_builder.subgraph(
                lambda op, x, y: op.Add(x, y),
                inputs=[make_value("x", FLOAT[...]), make_value("y", FLOAT[...])],
                outputs=[make_value("sum", FLOAT[...])],
            )

        Args:
            trace_function: A callable with signature
                ``(op: OpBuilder, *inputs: ir.Value | None) -> ir.Value | Sequence[ir.Value]``.
                It is called once with freshly created placeholder inputs to record the
                graph topology.
            inputs: A :class:`Sequence` of :class:`ir.Value` (or ``None``
                for absent optional inputs).  Each ``ir.Value`` should be
                freshly created with a name and optional type/shape.
            outputs: A :class:`Sequence` of :class:`ir.Value` objects
                declaring the expected outputs.
            name: Name of the resulting :class:`ir.Graph`.

        Returns:
            An :class:`ir.Graph` whose inputs and outputs are populated and whose
            nodes record the operations traced by *trace_function*.  This graph can be
            passed directly as a graph-valued attribute (e.g. the ``body`` attribute of
            a ``Scan`` or ``Loop`` node).
        """
        return build_graph(
            trace_function,
            inputs,
            outputs,
            opset_imports=dict(self._graph.opset_imports),
            name=name,
            parent=self,
        )

    def call_op(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | ir.TensorProtocol | None],
        kwargs: dict[str, Any],
    ):
        """Create an ONNX node and add it to the graph, returning its output value(s)."""
        domain = kwargs.pop("_domain", "")
        version = kwargs.pop("_version", None)
        outputs = kwargs.pop("_outputs", 1)
        name = kwargs.pop("_name", None)

        # Node naming: auto-generate a local name if none provided, then
        # qualify with scope context when auto_name_nodes is enabled.
        if name is None and self._options.auto_name_nodes:
            count = self.graph.num_nodes()
            name = self._qualify_node_name(f"{op_type}_node_{count}")
        elif name is not None and self._options.auto_name_nodes:
            name = self._qualify_node_name(name)

        output_values = self._adapt_outputs(outputs, op_type)

        if self._options.auto_cast_inputs:
            schema = self._get_schema(op_type, domain, version)
            inputs, attributes = self._partition_inputs_attributes(schema, inputs, kwargs)
            inputs = self._cast_inputs(schema, inputs)
            attributes = self._cast_attributes(schema, attributes)
        else:
            # Tape-compatible: convert inputs to ir.Value without schema-driven
            # casting, and treat remaining kwargs as attributes directly.
            inputs = [self._input_to_ir_value(i) for i in inputs]
            attributes = kwargs

        node = ir.node(
            op_type,
            inputs,
            attributes=attributes or None,
            domain=domain,
            outputs=output_values,
            version=version,
            name=name,
        )

        if self._options.scope_metadata:
            node.metadata_props["namespace"] = self._build_namespace()
            node.metadata_props["pkg.onnxscript.class_hierarchy"] = repr(self._scope_classes())
            node.metadata_props["pkg.onnxscript.name_scopes"] = repr(self._scope_names())

        self.add_node(node)

        return node.outputs if len(node.outputs) > 1 else node.outputs[0]

    def call(
        self,
        function,
        *args,
        _outputs: Sequence[str] | None = None,
        _prefix: str = "",
        **kwargs,
    ):
        if isinstance(function, ir.Function):
            graph = function.graph
        elif isinstance(function, onnxscript.OnnxFunction):
            graph = function.graph()
        else:
            raise TypeError("Function must be an ir.Function or onnxscript.OnnxFunction")
        output_renaming: dict[str, str] = {}
        if _outputs is not None:
            if len(_outputs) != len(graph.outputs):
                raise ValueError(
                    f"Number of provided output names {_outputs} does not match "
                    f"number of function outputs {len(graph.outputs)}."
                )
            for output, name in zip(graph.outputs, _outputs):
                output_renaming[output.name] = self._qualify_value_name(name)
        else:
            for output in graph.outputs:
                output_renaming[output.name] = self._qualify_value_name(output.name)
        nodes, outputs = _inliner.instantiate(graph, args, kwargs)
        if _prefix:
            self.push_module(_prefix)
        for node in nodes:
            node.name = self._qualify_node_name(node.name)
            for output in node.outputs:
                if output.name:
                    if output.name in output_renaming:
                        output.name = output_renaming[output.name]
                    else:
                        output.name = self._qualify_value_name(output.name)
            self.add_node(node)
        if _prefix:
            self.pop_module()
        return outputs if len(outputs) > 1 else outputs[0]

    def push_module(self, module: str, class_name: str = "") -> None:
        """Push a new module scope onto the stack.

        Args:
            module: The attribute name of the module (e.g. ``"layers.0"``).
            class_name: The qualified class name (e.g. ``"Gemma3DecoderLayer"``).
        """
        self._scope_stack.append((module, class_name))

    def pop_module(self) -> None:
        """Pop the most recent module scope off the stack."""
        if not self._scope_stack:
            raise RuntimeError("Cannot pop_module: no module context has been pushed.")
        self._scope_stack.pop()

    def _scope_names(self) -> list[str]:
        """Return the list of module attribute names in the current scope."""
        return [name for name, _ in self._scope_stack]

    def _scope_classes(self) -> list[str]:
        """Return the list of class names in the current scope."""
        return [cls for _, cls in self._scope_stack]

    def _scope_name_parts(self) -> list[str]:
        """Return non-empty module names for qualifying names."""
        return [name for name, _ in self._scope_stack if name]

    def _qualify_initializer_name(self, name: str) -> str:
        """Prepend the current hierarchical context prefix to the given name.

        Uses ``.`` as separator, appropriate for parameter and initializer names.
        """
        parts = self._scope_name_parts()
        if parts:
            return ".".join(parts) + "." + name
        return name

    def _qualify_value_name(self, name: str) -> str:
        """Qualify a value name with the current scope using ``.`` separator.

        The name is prefixed with ``v_`` to distinguish values from parameters.
        """
        parts = self._scope_name_parts()
        if parts:
            return "v_" + ".".join(parts) + "." + name
        return f"v_{name}"

    def _qualify_node_name(self, name: str) -> str:
        """Qualify a node name with the current scope using ``/`` separator."""
        parts = self._scope_name_parts()
        if parts:
            return "/".join(parts) + "/" + name
        return name

    def _build_namespace(self) -> str:
        """Build the namespace string for a node.

        Each scope entry is formatted as ``name: class_name`` joined by ``/``.
        """
        parts = []
        for name, cls in self._scope_stack:
            if name or cls:
                parts.append(f"{name}: {cls}" if cls else name)
        return "/".join(parts)


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

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def version(self) -> int | None:
        return self._version

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

    def call(
        self,
        function,
        *args,
        _outputs: Sequence[str] | None = None,
        _prefix: str = "",
        **kwargs,
    ):
        """Call a function and inline it into the graph.

        Args:
            function: The function to call (ir.Function or onnxscript.OnnxFunction).
            *args: Positional arguments to pass to the function.
            _outputs: Optional sequence of output names. If provided, must match the
                number of function outputs.
            _prefix: Optional prefix for module scoping (e.g., "layers.0").
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The output value(s) from the function call.
        """
        return self._builder.call(
            function, *args, _outputs=_outputs, _prefix=_prefix, **kwargs
        )
