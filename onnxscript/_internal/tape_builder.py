# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Builder base class and tape-backed implementation.

This module defines:

- ``BuilderBase``: Abstract base class for building ONNX IR nodes via a
  dynamic dispatch interface (``op.Relu(x)``, ``op.op("Relu", x)``,
  ``op.initializer(...)``).
  Subclasses implement the storage strategy by overriding ``_add_node``,
  ``_add_initializer``, and ``_record_opset``.

- ``TapeBuilder``: Concrete subclass backed by simple lists.  Engines
  (rewriter, optimizer, version converter) create an instance, pass it to a
  rule or evaluator, and harvest the accumulated nodes / initializers / opsets
  after it returns.

- ``BuilderFeature``: Flag enum controlling optional processing steps
  (schema partitioning, input casting, shape inference, etc.).
"""

from __future__ import annotations

import abc
import enum
from typing import Any, Optional, Sequence

import onnx
import onnx_ir as ir
from onnx_ir import _convenience

from onnxscript._internal import param_manipulation

UsedOpsets = set[tuple[str, Optional[int]]]

# Mapping from Python scalar types to their default ONNX DataType,
# used when no schema-based type binding is available.
_PYTHON_TYPE_TO_DTYPE: dict[type, ir.DataType] = {
    int: ir.DataType.INT64,
    float: ir.DataType.FLOAT,
}


def _dtype_suffix(dtype: ir.DataType) -> str:
    """Return a short type suffix for naming constants based on ir.DataType."""
    return dtype.short_name()


def _constant_name(
    value: int | float | bool | str | Sequence, type_suffix: str, num: int = 0
) -> str:
    """Generate a descriptive name for a constant value."""
    if isinstance(value, str):
        return f"const_str_{num}"
    if isinstance(value, (int, float, bool)):
        return f"const_{value}_{type_suffix}" if type_suffix else f"const_{value}"
    return f"const_1d_{num}"


class BuilderFeature(enum.Flag):
    """Features that can be enabled on BuilderBase."""

    NONE = 0
    SCHEMA_PARTITION = enum.auto()
    CAST_INPUTS = enum.auto()
    CAST_ATTRIBUTES = enum.auto()
    INFER_SHAPES = enum.auto()
    CONSTANT_PROPAGATION = enum.auto()

    # Convenience combos
    SCHEMA_AWARE = SCHEMA_PARTITION | CAST_INPUTS | CAST_ATTRIBUTES
    FULL = SCHEMA_AWARE | INFER_SHAPES | CONSTANT_PROPAGATION

    @property
    def any_schema_feature(self) -> bool:
        """True if any schema-dependent feature is enabled."""
        return bool(
            self
            & (
                BuilderFeature.SCHEMA_PARTITION
                | BuilderFeature.CAST_INPUTS
                | BuilderFeature.CAST_ATTRIBUTES
            )
        )


class BuilderBase(abc.ABC):
    """Abstract base class for building ONNX IR nodes.

    Supports two creation operations:

    1. **Op creation** — ``op.op("Relu", x)`` or ``op.Relu(x)`` (syntactic sugar).
    2. **Initializer creation** — ``op.initializer(tensor, name=...)``.

    Subclasses must implement the three protected methods that define where
    created nodes and initializers are stored:

    - :meth:`_add_node`
    - :meth:`_add_initializer`
    - :meth:`_record_opset`
    """

    def __init__(self, *, features: BuilderFeature = BuilderFeature.NONE) -> None:
        self._features = features

    @property
    def features(self) -> BuilderFeature:
        return self._features

    # ------------------------------------------------------------------
    # Abstract storage interface (to be implemented by subclasses)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _add_node(self, node: ir.Node) -> None:
        """Record a newly created node."""
        raise NotImplementedError

    @abc.abstractmethod
    def _add_initializer(self, value: ir.Value) -> None:
        """Record a newly created initializer."""
        raise NotImplementedError

    @abc.abstractmethod
    def _record_opset(self, domain: str, version: int | None) -> None:
        """Record that an opset domain/version was referenced."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Overridable hook methods
    # ------------------------------------------------------------------

    def _get_schema(
        self, op_type: str, domain: str, version: int | None
    ) -> onnx.defs.OpSchema | None:
        """Look up the op schema.

        Returns None if version is not provided or schema is not found.
        """
        if version is not None:
            try:
                return onnx.defs.get_schema(op_type, version, domain)
            except onnx.defs.SchemaError:
                pass
        return None

    def _partition_inputs_attributes(
        self,
        schema: onnx.defs.OpSchema | None,
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> tuple[Sequence[Any], dict[str, Any]]:
        """Separate positional args into inputs and attributes using the schema."""
        if schema is None:
            return args, kwargs
        op_signature = ir.schemas.OpSignature.from_op_schema(schema)
        return param_manipulation.separate_input_attributes_from_arguments(
            op_signature,
            list(args),
            kwargs,
            fill_defaults=False,
            allow_extra_args=False,
        )

    def _cast_inputs(
        self,
        schema: onnx.defs.OpSchema | None,
        inputs: Sequence[Any],
    ) -> Sequence[ir.Value | None]:
        """Cast/promote inputs (e.g., scalars → tensors) using schema type info.

        Uses schema specification to support a limited form of auto-casting:
        * Scalars are promoted to tensors via _input_to_ir_value.
        * They are cast to the required type when used in ops with other
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
        """Cast attributes using schema info.

        Default: pass through unchanged.
        """
        del schema  # Not implemented yet
        return attributes if attributes is not None else {}

    def _input_to_ir_value(
        self, value: Any, like_type: ir.Value | None = None
    ) -> ir.Value | None:
        """Convert a permissible input into an ir.Value.

        Handles ir.Value (pass-through), None (pass-through), and Python
        constants/sequences/tensors (promoted to initializers via
        ``_promote_constant``).  When *like_type* is provided but its dtype
        is unknown at graph-construction time, a dynamic ``CastLike`` node
        is inserted so the constant matches *like_type* at runtime.
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
        ir_value = self._promote_constant(value, dtype)
        if needs_dynamic_cast:
            ir_value = self.call_op("CastLike", [ir_value, like_type], {})
        return ir_value

    def _promote_constant(self, value: Any, dtype: ir.DataType | None) -> ir.Value:
        """Convert a Python constant into an ir.Value via a Constant node.

        Creates a ``Constant`` op node whose output carries the tensor value.
        This avoids initializer-name collisions when the builder is used
        inside the rewriter/optimizer.

        GraphBuilder overrides this with a cache-based initializer strategy.
        """
        tensor = ir.tensor(value, dtype=dtype)
        return self.call_op("Constant", [], {"value": tensor})

    def _qualify_value_name(self, name: str) -> str:
        """Qualify a value name with scope prefix.

        Default: identity (no qualification). Override in GraphBuilder
        to add module scope prefixes.
        """
        return name

    def _generate_node_name(self, op_type: str) -> str | None:
        """Generate a node name. Default: None (no auto-naming)."""
        return None

    def _adapt_outputs(
        self, outputs: int | Sequence[str | ir.Value], op_type: str
    ) -> Sequence[ir.Value] | None:
        """Pre-create output ir.Value objects.

        Default returns ``None`` for int outputs (letting ir.Node create
        anonymous outputs), and converts string/ir.Value sequences.
        Override in GraphBuilder to always pre-create named outputs.
        """
        if isinstance(outputs, int):
            return None
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

    def _constant_propagation(self, node: ir.Node) -> None:
        """Run basic constant propagation on a newly created node.

        Called when CONSTANT_PROPAGATION feature is enabled.
        """
        # Lazy import to avoid circular dependency at module level.
        import onnxscript.optimizer  # pylint: disable=import-outside-toplevel

        onnxscript.optimizer.basic_constant_propagation([node])

    def _infer_shapes(self, node: ir.Node) -> None:
        """Run shape/type inference on a newly created node.

        Called when INFER_SHAPES feature is enabled.
        """
        from onnxscript._internal import _inference  # pylint: disable=import-outside-toplevel

        _inference.infer_outputs(node)

    def _annotate_node(self, node: ir.Node) -> None:
        """Attach metadata to a node after creation.

        Default: no-op. Override to add scope/namespace annotations.
        """

    # ------------------------------------------------------------------
    # Public API (concrete)
    # ------------------------------------------------------------------

    def __getattr__(self, op_type: str) -> Any:
        """Dynamic op dispatch: ``op.Relu(x)``, ``op.MatMul(a, b)``, etc.

        Syntactic sugar for ``op.op(op_type, ...)``.

        Returns a callable that creates a node of the given ``op_type``
        and records it via the subclass storage implementation.
        """
        return lambda *args, **kwargs: self.op(op_type, *args, **kwargs)

    def op(
        self,
        op_type: str,
        /,
        *args: ir.Value | None,
        _domain: str = "",
        _version: int | None = None,
        _outputs: int | Sequence[str] = 1,
        _name: str | None = None,
        **kwargs: Any,
    ) -> ir.Value | Sequence[ir.Value]:
        """Create an ONNX node.

        This is the single entry point for all node creation.
        ``op.Relu(x)`` is equivalent to ``op.op("Relu", x)``.

        Args:
            op_type: The operator type (e.g., ``"Relu"``, ``"Conv"``).
            *args: Positional arguments — the node's input values.
            _domain: Op domain (default ``""``).
            _version: Opset version.
            _outputs: Number of outputs or list of explicit output names.
            _name: Optional node name (must be unique).
            **kwargs: Keyword arguments — node attributes.
                Values can be Python scalars/lists (auto-converted) or
                ``ir.Attr`` instances (passed through).

        Returns:
            A single ``ir.Value`` if the node has one output, otherwise
            a sequence of ``ir.Value``.
        """
        return self.call_op(
            op_type,
            args,
            kwargs,
            domain=_domain,
            version=_version,
            outputs=_outputs,
            name=_name,
        )

    def call_op(
        self,
        op_type: str,
        args: Sequence[Any],
        kwargs: dict[str, Any],
        /,
        domain: str = "",
        version: int | None = None,
        outputs: int | Sequence[str | ir.Value] = 1,
        name: str | None = None,
    ) -> ir.Value | Sequence[ir.Value]:
        """Create an ONNX node and add it to the graph, returning its output value(s).

        This is the core node-creation method. Both ``BuilderBase.op()`` and
        ``OpBuilder.__getattr__`` delegate here. The processing steps are
        controlled by :attr:`features` flags and overridable hook methods.
        """
        features = self._features

        # 1. Schema lookup (if any schema-dependent feature is enabled)
        schema = None
        if features.any_schema_feature:
            schema = self._get_schema(op_type, domain, version)

        # 2. Partition args into inputs and attributes using schema
        if features & BuilderFeature.SCHEMA_PARTITION:
            args, kwargs = self._partition_inputs_attributes(schema, args, kwargs)

        # 3. Cast inputs (scalar→tensor promotion, type-variable matching)
        if features & BuilderFeature.CAST_INPUTS:
            args = self._cast_inputs(schema, args)

        # 4. Cast attributes using schema info
        if features & BuilderFeature.CAST_ATTRIBUTES:
            kwargs = self._cast_attributes(schema, kwargs)

        # 5. Convert remaining kwargs to ir.Attr list
        attrs: Sequence[ir.Attr] = _convenience.convert_attributes(kwargs) if kwargs else ()

        # 6. Determine outputs
        output_values = self._adapt_outputs(outputs, op_type)

        # 7. Build the node
        if name is None:
            name = self._generate_node_name(op_type)

        if output_values is not None:
            node = ir.Node(
                domain,
                op_type,
                args,
                attributes=attrs,
                outputs=output_values,
                version=version,
                name=name,
            )
        else:
            num_outputs = len(outputs) if isinstance(outputs, Sequence) else outputs
            node = ir.Node(
                domain,
                op_type,
                args,
                attributes=attrs,
                num_outputs=num_outputs,
                version=version,
                name=name,
            )

        # 8. Annotate (metadata, scope)
        self._annotate_node(node)

        # 9. Store
        self._add_node(node)
        self._record_opset(domain, version)

        # 10. Post-creation hooks (inference, const-prop)
        if features & BuilderFeature.CONSTANT_PROPAGATION:
            self._constant_propagation(node)
        if features & BuilderFeature.INFER_SHAPES:
            self._infer_shapes(node)

        # 11. Return
        if len(node.outputs) == 1:
            return node.outputs[0]
        return node.outputs

    def initializer(
        self,
        tensor: ir.TensorProtocol,
        name: str | None = None,
    ) -> ir.Value:
        """Create a new constant initializer and return its ``ir.Value``."""
        name = name or tensor.name
        if name is None:
            raise ValueError("Name must be provided for initializer.")
        shape = ir.Shape((d if isinstance(d, int) else d.value) for d in tensor.shape.dims)
        value = ir.Value(
            name=name, shape=shape, type=ir.TensorType(tensor.dtype), const_value=tensor
        )
        self._add_initializer(value)
        return value


class TapeBuilder(BuilderBase):
    """Concrete builder backed by simple lists (tape-like storage).

    Engines (rewriter, optimizer, version converter) create an instance,
    pass it to a rule or evaluator, and after it returns, harvest the
    accumulated results via the ``nodes``, ``initializers``, and
    ``used_opsets`` properties.
    """

    def __init__(self, *, features: BuilderFeature = BuilderFeature.NONE) -> None:
        super().__init__(features=features)
        self._nodes: list[ir.Node] = []
        self._initializers: list[ir.Value] = []
        self._used_opsets: UsedOpsets = set()

    def _add_node(self, node: ir.Node) -> None:
        self._nodes.append(node)

    def _add_initializer(self, value: ir.Value) -> None:
        self._initializers.append(value)

    def _record_opset(self, domain: str, version: int | None) -> None:
        self._used_opsets.add((domain, version))

    # --- Harvesting properties ---

    @property
    def nodes(self) -> Sequence[ir.Node]:
        """All nodes created during this context's lifetime."""
        return tuple(self._nodes)

    @property
    def initializers(self) -> Sequence[ir.Value]:
        """All initializers created during this context's lifetime."""
        return tuple(self._initializers)

    @property
    def used_opsets(self) -> UsedOpsets:
        """Opset domains/versions referenced by created nodes."""
        return self._used_opsets
