"""Graph building functions using the ONNX IR, compatible with the original TorchScriptGraph usage."""

from __future__ import annotations

import ctypes
import typing
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.checker
import onnx.defs
import onnx.helper
import onnx.shape_inference
import torch
from typing_extensions import TypeAlias

import onnxscript
from onnxscript import evaluator, ir
from onnxscript import tensor as onnxscript_tensor
from onnxscript._internal import param_manipulation
from onnxscript.function_libs.torch_lib import _flags
from onnxscript.function_libs.torch_lib.ops import common as common_ops

__all__ = [
    "TorchScriptTensor",
    "TorchScriptGraph",
    "TorchScriptTracingEvaluator",
]


ValidArgumentType: TypeAlias = Union[
    "TorchScriptTensor",
    Sequence["TorchScriptTensor"],
    Sequence[float],
    Sequence[int],
    complex,
    str,
    int,
    float,
    bool,
    None,
]
ValidInputType: TypeAlias = Union[
    "TorchScriptTensor",
    Sequence["TorchScriptTensor"],
    Sequence[float],
    Sequence[int],
    complex,
    str,
    int,
    float,
    bool,
    None,
]

_TORCH_DTYPE_TO_ONNX: dict[torch.dtype, ir.DataType] = {
    torch.bfloat16: ir.DataType.BFLOAT16,
    torch.bool: ir.DataType.BOOL,
    torch.complex128: ir.DataType.COMPLEX128,
    torch.complex64: ir.DataType.COMPLEX64,
    torch.float16: ir.DataType.FLOAT16,
    torch.float32: ir.DataType.FLOAT,
    torch.float64: ir.DataType.DOUBLE,
    torch.float8_e4m3fn: ir.DataType.FLOAT8E4M3FN,
    torch.float8_e4m3fnuz: ir.DataType.FLOAT8E4M3FNUZ,
    torch.float8_e5m2: ir.DataType.FLOAT8E5M2,
    torch.float8_e5m2fnuz: ir.DataType.FLOAT8E5M2FNUZ,
    torch.int16: ir.DataType.INT16,
    torch.int32: ir.DataType.INT32,
    torch.int64: ir.DataType.INT64,
    torch.int8: ir.DataType.INT8,
    torch.uint8: ir.DataType.UINT8,
}


def _torch_dtype_to_onnx_dtype(dtype: torch.dtype) -> ir.DataType:
    return _TORCH_DTYPE_TO_ONNX[dtype]


class _TorchTensor(ir.Tensor):  # pylint: disable=too-many-ancestors
    def __init__(self, tensor: torch.Tensor):
        super().__init__(tensor, dtype=_torch_dtype_to_onnx_dtype(tensor.dtype))

    def tobytes(self) -> bytes:
        # Support native PyTorch types so we can use types like bloat16
        assert isinstance(self.raw, torch.Tensor)
        tensor = self.raw.detach().cpu().contiguous()
        return bytes(
            (ctypes.c_ubyte * tensor.element_size() * tensor.numel()).from_address(
                tensor.data_ptr()
            )
        )


class TorchScriptTensor(ir.Value, onnxscript_tensor.Tensor):
    """A onnxscript tensor that wraps a torchscript Value."""

    def __init__(
        self,
        _=None,  # Unused argument for backward compatibility
        producer=None,
        index=None,
        name: str | None = None,
    ):
        onnxscript_tensor.Tensor.__init__(self, None)
        ir.Value.__init__(self, producer, index=index, name=name)
        self._is_complex: bool = False
        self._concrete_value: np.ndarray | None = None
        self._device: torch.device | None = None

    @property
    def value(self) -> Optional[np.ndarray]:
        return self._concrete_value

    @value.setter
    def value(self, value: np.ndarray) -> None:
        self._concrete_value = value

    @property  # type: ignore[override]
    def rank(self) -> int | None:
        if self.shape is None:
            return None
        return len(self.shape)

    @property  # type: ignore[override]
    def shape(self) -> ir.Shape | None:
        return super().shape

    @shape.setter
    def shape(self, shape: Union[torch.Size, Tuple[int | str | None, ...]]):
        # Normalize torch symbolic dimension size to str.
        torch_sym_types = (torch.SymInt, torch.SymFloat, torch.SymBool)
        self._shape = ir.Shape(
            tuple(str(dim.node) if isinstance(dim, torch_sym_types) else dim for dim in shape)  # type: ignore[union-attr]
        )

    @property
    def dtype(self) -> ir.DataType | None:
        return super().dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype | ir.DataType | None):
        if dtype is None:
            onnx_dtype = ir.DataType.UNDEFINED
        elif isinstance(dtype, ir.DataType):
            onnx_dtype = dtype
        else:
            onnx_dtype = _torch_dtype_to_onnx_dtype(dtype)
        if self._type is None:
            self._type = ir.TensorType(onnx_dtype)
        else:
            self._type.dtype = onnx_dtype

    # TODO: Remove this when there is no mismatch output shapes between device:
    # https://github.com/pytorch/pytorch/blob/a44f8894fa6d973693aab44a3dda079a168b05c1/torch/_decomp/decompositions.py#L1451-L1457
    @property
    def device(self) -> torch.device | None:
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device

    @property
    def is_complex(self) -> bool:
        return self._is_complex

    @is_complex.setter
    def is_complex(self, is_complex: bool):
        self._is_complex = is_complex

    @property
    def onnx_dtype(self) -> int:
        raise NotImplementedError("onnx_dtype is not supported for TorchScriptTensor.")

    def value_info(self) -> Optional[onnx.ValueInfoProto]:
        raise NotImplementedError("value_info is not supported for TorchScriptTensor.")


class TorchScriptTracingEvaluator(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph."""

    def __init__(self, graph: TorchScriptGraph):
        self._graph: TorchScriptGraph = graph

    @property
    def graph(self) -> TorchScriptGraph:
        return self._graph

    def eval(self, schema, inputs: Sequence[ValidInputType], attributes):
        return self._graph.add_op_call(schema, inputs, attributes)

    def eval_function(  # type: ignore[override]
        self,
        function: onnxscript.OnnxFunction,
        args: Sequence[ValidArgumentType],
        kwargs: Mapping[str, ValidArgumentType],
    ):
        if _flags.EXPERIMENTAL_PREFER_TRACING:
            # Special cases for handling IsScalar and Rank
            if function.name == "IsScalar":
                if len(args) != 1:
                    raise TypeError(
                        f"Expected 1 positional argument for function '{function}', got {len(args)}."
                    )
                if isinstance(args[0], TorchScriptTensor):
                    if args[0].rank is not None:
                        return args[0].rank == 0
                    else:
                        # Fall to call add_function_call
                        pass
                elif isinstance(args[0], Sequence):  # noqa: SIM103
                    return False
                else:
                    # Python constants are scalars
                    return True
            if function.name == "Rank":
                if len(args) != 1:
                    raise TypeError(
                        f"Expected 1 positional argument for function '{function}', got {len(args)}."
                    )
                if isinstance(args[0], TorchScriptTensor):
                    if args[0].rank is not None:
                        return args[0].rank
                    else:
                        # Fall to call add_function_call
                        pass
                elif isinstance(args[0], Sequence):
                    if all(isinstance(arg, (int, float)) for arg in args[0]):
                        return 1
                    else:
                        # Fall to call add_function_call
                        pass
                else:
                    # Python constants are scalars
                    return 0
            elif function.experimental_traceable:
                # Trace the function call instead of adding the function as a node
                return function.function(*args, **kwargs)

        # args/kwargs are TorchScriptTensor/python built-in based
        param_schemas = function.param_schemas()
        (
            inputs,
            attributes,
        ) = param_manipulation.separate_input_attributes_from_arguments(
            param_schemas, args, kwargs, fill_defaults=True, allow_extra_kwargs=True
        )

        # Cast attributes to the correct type based on function signature
        op_schema = function.op_schema
        assert op_schema is not None
        for name, value in attributes.items():
            attribute = op_schema.attributes[name]
            if attribute.type == onnx.defs.OpSchema.AttrType.FLOAT:
                # Cast int to float if the attribute is FLOAT
                attributes[name] = float(value)

            # In PyTorch, an attribute annotated as `int[1]?` accepts an integer
            # or a sequence. When the attribute is an integer, it is treated as
            # a single element sequence. ONNX requires an attribute to either be
            # an integer or a sequence. So we promote the value to a sequence here.
            if attribute.type == onnx.defs.OpSchema.AttrType.INTS and isinstance(value, int):
                attributes[name] = (value,)
            if attribute.type == onnx.defs.OpSchema.AttrType.FLOATS and isinstance(
                value, float
            ):
                attributes[name] = (value,)
        return self._graph.add_function_call(function, inputs, attributes)


def _build_attribute(
    key: str,
    value: Union[
        float,
        int,
        str,
        Sequence[float],
        Sequence[int],
        torch.Tensor,
        _TorchTensor,
        ir.TensorProtocol,
    ],
):
    """Initializes the right attribute based on type of value."""
    if isinstance(value, float):
        return ir.AttrFloat32(key, value)
    if isinstance(value, int):
        return ir.AttrInt64(key, value)
    if isinstance(value, str):
        return ir.AttrString(key, value)
    if isinstance(value, torch.Tensor):
        return ir.AttrTensor(key, _TorchTensor(value))
    if isinstance(value, (_TorchTensor, ir.TensorProtocol)):
        return ir.AttrTensor(key, value)
    if isinstance(value, Sequence):
        if not value:
            # Treat empty sequences as empty list tensors
            # TODO(justinchuby): Revisit ways to determine the type of the empty list
            return ir.AttrInt64s(key, [])
        if isinstance(value[0], float):
            return ir.AttrFloat32s(key, list(value))  # type: ignore[arg-type]
        if isinstance(value[0], int):
            return ir.AttrInt64s(key, list(value))  # type: ignore
        raise TypeError(f"Unsupported sequence type '{type(value)}' for attribute '{key}'")
    raise TypeError(f"Unsupported attribute type '{type(value)}' for attribute '{key}'")


def _create_op_call_in_graph(
    graph: ir.Graph,
    domain: str,
    op_type: str,
    *,
    inputs: Sequence[TorchScriptTensor],
    attributes: Mapping[str, Any],
    num_outputs: int = 1,
) -> Sequence[TorchScriptTensor]:
    """Creates a node representing an onnx op in `graph`.

    Args:
        graph: The torch graph to add the node to.
        domain: The domain of the op.
        op_type: The name of the op. E.g. "Add".
        inputs: The onnx inputs to the op.
        attributes: The onnx attributes to the op.
        num_outputs: The number of outputs the op has.

    Returns:
        The outputs of the created node.
    """
    # Filter out None attributes, this can be convenient client side because
    # now they can pass through None attributes, and have them not show up
    attributes = {k: v for k, v in attributes.items() if v is not None}

    node = ir.Node(
        domain,
        op_type,
        inputs=inputs,
        attributes=[_build_attribute(key, value) for key, value in attributes.items()],
        outputs=[TorchScriptTensor() for _ in range(num_outputs)],
    )
    graph.append(node)

    return typing.cast(Sequence[TorchScriptTensor], node.outputs)


def _shared_functions() -> list[ir.Function]:
    """Hack to always include the share ops."""

    # TODO: Remove after https://github.com/microsoft/onnxscript/issues/834 is fixed
    return [
        ir.serde.deserialize_function(common_ops.Rank.to_function_proto()),
        ir.serde.deserialize_function(common_ops.IsScalar.to_function_proto()),
    ]


class TorchScriptGraph:
    def __init__(
        self,
        parent_torch_script_graph: Optional[TorchScriptGraph] = None,
        domain_name: Optional[str] = None,
    ):
        self._graph = ir.Graph((), (), nodes=(), name="main_graph")
        # All the functions used, deduplicated by name
        # key: (name, domain)
        self._function_store: Dict[ir.OperatorIdentifier, ir.Function] = {}
        self._initializers: Dict[str, torch.Tensor] = {}
        # Mapping from initializer name to input(TorchScriptTensor).
        self._initializers_inputs: Dict[str, TorchScriptTensor] = {}
        # Mapping from initializer name to input(TorchScriptTensor) from parent graph.
        self._initializers_inputs_from_parent: Dict[str, TorchScriptTensor] = {}
        # Mapping from model local function type name to function graph.
        # Local function type name is expected to be unique. Converter creates
        # a unique name and a unique function graph for every module call.
        self._sub_torch_script_graphs: Dict[str, TorchScriptGraph] = {}
        # Parent graph. None if this is the top level graph.
        self._parent_torch_script_graph = parent_torch_script_graph
        # Domain name of the graph. None if this is the top level graph.
        self._domain_name: Optional[str] = domain_name

        if self._domain_name is None and self._parent_torch_script_graph is not None:
            raise RuntimeError(
                "Domain name is not set. It is required because this 'TorchScriptGraph' instance "
                "is a subgraph that represents an ONNX local function."
            )

    @property
    def initializers(self) -> Mapping[str, torch.Tensor]:
        return self._initializers

    # NOTE: This setter is used in torch converter when we activate fake mode,
    #       we need to filter out the initializers that has fake tensor. This
    #       is because we don't want to introduce fake tensor in onnxscript.
    @initializers.setter
    def initializers(self, initializers: Dict[str, torch.Tensor]):
        self._initializers = initializers

    @property
    def initializers_inputs(self) -> Mapping[str, TorchScriptTensor]:
        return self._initializers_inputs

    @property
    def initializers_inputs_from_parent(self) -> Mapping[str, TorchScriptTensor]:
        return self._initializers_inputs_from_parent

    @property
    def num_outputs(self) -> int:
        return len(self._graph.outputs)

    @property
    def domain_name(self) -> Optional[str]:
        return self._domain_name

    def add_input(
        self,
        input_name: Optional[str],
        shape: Optional[Union[torch.Size, Tuple[Union[int, str, None], ...]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> TorchScriptTensor | None:
        if input_name is None:
            # This input argument is None, which is mapped
            # to a NULL value in TorchScript type system.
            value = None
        else:
            value = TorchScriptTensor(name=input_name)
            value.shape = shape  # type: ignore[arg-type,assignment]
            value.device = device
            if dtype is not None:
                value.dtype = dtype  # type: ignore[assignment]
            # TODO(titaiwang): This approach loses the information that "same SymInts
            # indicates same shape", for example, [symint0, symint0, symint1]
            # would all be [None, None, None]
            # torch_value.setType(
            #     torch_value.type().with_sizes(
            #         [dim if isinstance(dim, int) else None for dim in shape]  # type: ignore[union-attr]
            #     )
            # )
            self._graph.inputs.append(value)  # type: ignore[arg-type]
        return value

    def add_initializer(self, name: str, value: torch.Tensor) -> TorchScriptTensor:
        if name in self._initializers_inputs:
            # NOTE: Previously it raises when `name` is already set. This is relaxed
            # because this will be invoked multiple times when submodule is called
            # multiple times.
            if name in self._initializers and self._initializers[name] is not value:
                raise ValueError(
                    f"Initializer '{name}' exists already with a different value."
                )
            return self._initializers_inputs[name]  # type: ignore[return-value]

        if (
            self != self._parent_torch_script_graph
            and self._parent_torch_script_graph is not None
        ):
            # Only the root graph can have initializers. Add as initializer
            # to root graph, and add as input to current graph.
            self._initializers_inputs_from_parent[name] = (
                self._parent_torch_script_graph.add_initializer(name, value)
            )
        else:
            input = TorchScriptTensor(name=name)
            input.const_value = _TorchTensor(value)
            self._initializers_inputs[name] = input
            self._initializers[name] = value
        return input

    def register_outputs(
        self, outputs: Union[TorchScriptTensor, Tuple[TorchScriptTensor, ...]]
    ):
        if isinstance(outputs, TorchScriptTensor):
            outputs = (outputs,)
        for output in outputs:
            assert isinstance(
                output, TorchScriptTensor
            ), f"output must be a TorchScriptTensor, not {type(output)}"
            self._graph.outputs.append(output)

    def _add_constant_to_graph(self, constant) -> Sequence[ir.Value | None]:
        """Add a constant to the graph.

        Returns:
            A single element of sequence of the constant value.
        """
        if constant is None:
            return (None,)

        if isinstance(constant, bool):
            # Be sure to put bool before int, because bool is a subclass of int
            constant_tensor = torch.tensor(constant, dtype=torch.bool)
        elif isinstance(constant, float):
            constant_tensor = torch.tensor(constant, dtype=torch.float)
        elif isinstance(constant, int):
            constant_tensor = torch.tensor(constant, dtype=torch.int64)
        elif isinstance(constant, (tuple, list)) and all(
            isinstance(val, int) for val in constant
        ):
            constant_tensor = torch.tensor(constant, dtype=torch.int64)
        elif isinstance(constant, (tuple, list)) and all(
            isinstance(val, float) for val in constant
        ):
            constant_tensor = torch.tensor(constant, dtype=torch.float)
        elif isinstance(constant, complex):
            # NOTE: ONNX doesn't support tensor of complex64/complex128, so we
            # convert them to float32/float64 with real representation.
            constant_tensor = torch.view_as_real(torch.tensor(constant).resolve_conj())
        else:
            raise TypeError(
                f"Constant input '{constant}' of type '{type(constant)}' is not supported"
            )
        onnx_tensor = _TorchTensor(constant_tensor)
        value = _create_op_call_in_graph(
            self._graph,
            "",
            "Constant",
            inputs=(),
            attributes=dict(value=onnx_tensor),
        )
        return value

    def _add_ir_graph_op_call(
        self,
        *,
        domain: str,
        op_type: str,
        onnx_inputs: Sequence[ValidInputType],
        onnx_attributes: Mapping[str, ValidArgumentType],
        num_outputs: int,
    ) -> Sequence[TorchScriptTensor]:
        graph_inputs: list[TorchScriptTensor] = []
        assert isinstance(onnx_inputs, Sequence)
        for input in onnx_inputs:
            # NOTE(titaiwang): input could be empty list
            if (
                isinstance(input, Sequence)
                and input
                and all(isinstance(elem, TorchScriptTensor) for elem in input)
            ):
                # If all elements in the Sequence are TorchScriptTensor we know it
                # should be a Sequence input in ONNX.
                input_sequence = _create_op_call_in_graph(
                    self._graph,
                    "",
                    "SequenceConstruct",
                    inputs=input,  # type: ignore
                    attributes={},
                )
                graph_inputs.extend(input_sequence)
            elif not isinstance(input, TorchScriptTensor):
                graph_inputs.extend(self._add_constant_to_graph(input))  # type: ignore
            else:
                # TODO(justinchuby): What is this case?
                graph_inputs.append(input)
        for key, value in onnx_attributes.items():
            assert not isinstance(
                value, TorchScriptTensor
            ), f"ONNX attribute must not be a TorchScriptTensor, got {key}: {value}."
        tensors = _create_op_call_in_graph(
            self._graph,
            domain,
            op_type,
            inputs=graph_inputs,
            attributes=onnx_attributes,
            num_outputs=num_outputs,
        )
        assert tensors, "Expected at least one output from ONNX op call."
        # NOTE: TorchScriptTensor is created here, however neither dtype nor shape is
        # set. It is expected that exporter will modify the tensor being returned and
        # set these info.
        return tensors

    def _fetch_function_dict(
        self, opset_version: int
    ) -> Mapping[ir.OperatorIdentifier, ir.Function]:
        function_dict: Dict[ir.OperatorIdentifier, ir.Function] = {}
        # Fetch local function protos. E.g., local functions representing module calls.
        for (
            sub_graph_name,
            sub_torch_script_graph,
        ) in self._sub_torch_script_graphs.items():
            function_dict.update(sub_torch_script_graph._fetch_function_dict(opset_version))  # pylint: disable=protected-access
            domain = sub_torch_script_graph.domain_name
            assert domain is not None
            name_domain = (sub_graph_name, domain, "")
            assert (
                name_domain not in function_dict
            ), f"Sub graph name already exists. {name_domain}"
            function_dict[name_domain] = sub_torch_script_graph._to_function(  # pylint: disable=protected-access
                opset_version, sub_graph_name
            )
        # Fetch torchlib function protos.
        for identifier, function in self._function_store.items():
            function_dict[identifier] = function
        return function_dict

    def add_op_call(
        self,
        onnx_op_schema: onnx.defs.OpSchema,
        onnx_inputs: Sequence[ValidInputType],
        onnx_attributes: Mapping[str, ValidArgumentType],
    ) -> Union[TorchScriptTensor, Sequence[TorchScriptTensor]]:
        # Compute outputs from the onnx_op op schema
        num_outputs = evaluator.compute_num_outputs(
            onnx_op_schema, onnx_inputs, onnx_attributes
        )
        result = self._add_ir_graph_op_call(
            domain="",
            op_type=onnx_op_schema.name,
            onnx_inputs=onnx_inputs,
            onnx_attributes=onnx_attributes,
            num_outputs=num_outputs,
        )

        if num_outputs == 1:
            return result[0]

        return result

    def add_function_call(
        self,
        onnx_function: onnxscript.OnnxFunction,
        onnx_inputs: Sequence[ValidInputType],
        onnx_attributes: Mapping[str, ValidArgumentType],
    ) -> Union[TorchScriptTensor, Sequence[TorchScriptTensor]]:
        ir_function = ir.serde.deserialize_function(onnx_function.to_function_proto())
        self._function_store[ir_function.identifier()] = ir_function
        num_outputs = len(onnx_function.function_ir.outputs)
        # Compute outputs from the function schema
        result = self._add_ir_graph_op_call(
            domain=ir_function.domain,
            op_type=ir_function.name,
            onnx_inputs=onnx_inputs,
            onnx_attributes=onnx_attributes,
            num_outputs=num_outputs,
        )

        if num_outputs == 1:
            return result[0]

        return result

    def add_module_call(
        self,
        name: str,
        sub_torch_script_graph: TorchScriptGraph,
        onnx_inputs: Sequence[ValidInputType],
    ) -> Union[TorchScriptTensor, Sequence[TorchScriptTensor]]:
        self._sub_torch_script_graphs[name] = sub_torch_script_graph
        domain_name = sub_torch_script_graph.domain_name
        assert domain_name is not None

        num_outputs = sub_torch_script_graph.num_outputs
        result = self._add_ir_graph_op_call(
            domain=domain_name,
            op_type=name,
            onnx_inputs=(
                *onnx_inputs,
                *sub_torch_script_graph.initializers_inputs_from_parent.values(),
            ),
            onnx_attributes={},
            num_outputs=num_outputs,
        )

        if num_outputs == 1:
            return result[0]

        return result

    def _to_function(self, opset_version: int, function_name: str) -> ir.Function:
        assert len(self.initializers) == 0, "Model local functions cannot have initializers."

        # Dissect the model proto and transform to function proto.
        domain = self.domain_name
        if domain is None:
            raise RuntimeError("Domain name is not set.")
        onnx_function = ir.Function(
            domain=domain,
            name=function_name,
            graph=self._graph,
            attributes=(),
        )
        onnx_function.opset_imports[""] = opset_version

        return onnx_function

    def to_model_proto(
        self, opset_version: int, include_initializers: bool = True
    ) -> onnx.ModelProto:
        function_dict: Mapping[ir.OperatorIdentifier, ir.Function] = self._fetch_function_dict(
            opset_version
        )
        unique_custom_domains: Dict[str, int] = {"": opset_version}

        for function in function_dict.values():
            # TODO(BowenBao): All local function domain versions are hardcoded as 1.
            unique_custom_domains[function.domain] = 1

        if include_initializers:
            self._graph.initializers.update(self._initializers_inputs)
        else:
            # TODO(justinchuby): Potentially set to const_value to None instead so we
            # don't lose handle on the values.
            self._graph.initializers.clear()

        onnx_model = ir.Model(
            self._graph,
            ir_version=8,
            producer_name=f"pytorch {torch.__version__}",
            functions=[*function_dict.values(), *_shared_functions()],
        )

        onnx_model.opset_imports.update(unique_custom_domains)
        # Include the library shared opset domain
        # TODO: Remove after https://github.com/microsoft/onnxscript/issues/834 is fixed
        onnx_model.opset_imports[common_ops.common_opset.domain] = (
            common_ops.common_opset.version
        )
        model_proto = ir.serde.serialize_model(onnx_model)
        return model_proto
