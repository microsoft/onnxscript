# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.checker
import onnx.defs
import onnx.helper
import onnx.shape_inference
import torch
from onnx import TensorProto
from typing_extensions import TypeAlias

import onnxscript
from onnxscript import evaluator, onnx_opset
from onnxscript._internal import autocast, param_manipulation, runtime_typing

__all__ = [
    "Graph",
    "GraphEvaluator",
]


ValidArgumentType: TypeAlias = Union[
    "Tensor",
    Sequence["Tensor"],
    Sequence[float],
    Sequence[int],
    str,
    int,
    float,
    bool,
    None,
]
ValidInputType: TypeAlias = Union[
    "Tensor",
    Sequence["Tensor"],
    Sequence[float],
    Sequence[int],
    str,
    int,
    float,
    bool,
    None,
]
ValidTorchValueType: TypeAlias = Union[
    torch.Value,
    Sequence[torch.Value],
    Sequence[float],
    Sequence[int],
    str,
    int,
    float,
    bool,
    None,
]


# TODO(titaiwang): Should we make Tensor datacalss?
class Tensor:
    """An implementation of ONNX Tensors, based on a wrapper around numpy arrays.
    Serves to define overloaded ops with an ONNX/ONNXScript semantics.
    """

    def __init__(
        self,
        name: str = "",
        nparray: Optional[np.ndarray] = None,
        shape: Optional[tuple[int, ...]] = None,
        dtype: Optional[np.dtype] = None,
        opset=None,
        onnx_type: str = "",
    ):
        if nparray is not None and not isinstance(nparray, np.ndarray):
            raise TypeError(
                f"Unexpected type {type(nparray)}. It must be a numpy array or None."
            )
        self._name = name
        self._nparray = nparray
        self._shape = nparray.shape if nparray is not None else shape
        self._dtype = nparray.dtype if nparray is not None else dtype
        # FIXME(justinhuby): Create a better way to determine the opset version
        self._opset: Any = opset or onnx_opset.opset18
        self._onnx_type = onnx_type

    # NOTE: Optional tensor
    @staticmethod
    def empty() -> Tensor:
        return Tensor(name="")

    def to_value_info(self) -> onnx.ValueInfoProto:
        # TODO: support more types?
        if self.onnx_type == "tensot_type":
            return onnx.helper.make_tensor_value_info(self.name, self.onnx_dtype, self.shape)
        if self.onnx_type == "sequence_type":
            return onnx.helper.make_tensor_sequence_value_info(
                self.name, self.onnx_dtype, self.shape
            )
        if self.onnx_type == "optional_type":
            # NOTE: no `make_optional_value_info` API
            optional_type_proto = onnx.helper.make_optional_type_proto(self.onnx_dtype)
            return onnx.helper.make_value_info(self.name, optional_type_proto)
        return onnx.helper.make_empty_value_info(self.name)

    @property
    def onnx_type(self):
        return self._onnx_type

    @property
    def name(self) -> str:
        return self._name

    # TODO(titaiwang): better name?
    @property
    def is_fake(self) -> bool:
        return self._nparray is None

    @property
    def value(self) -> np.ndarray:
        if self._nparray is None:
            raise ValueError("Tensor does not have a value.")
        return self._nparray

    @property
    def is_scalar(self) -> bool:
        return self.rank == 0

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def onnx_dtype(self) -> int:
        return onnx.helper.np_dtype_to_tensor_dtype(self.dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value!r})"

    def __bool__(self) -> bool:
        return bool(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def __len__(self) -> int:
        return self.shape[0]

    def __index__(self) -> int:
        return self.value.__index__()

    def __getitem__(self, index):
        op = self._opset
        if op.version < 13:
            raise RuntimeError("Indexing requires opset 13 or later.")
        if not isinstance(index, tuple):
            # Normalize representation to a tuple.
            # A single index-value is equivalent to a tuple with a single element.
            index = (index,)
        if len(index) > self.rank:
            raise ValueError(
                f"Number of indices {len(index)} is greater than rank {self.rank}"
            )

        # Promote integer indices to tensors of rank 0
        index = [autocast.cast_pyvalue_to_os_tensor(x) for x in index]
        # Process all elements in index
        shape = self.shape
        sliced_indices = []
        scalar_indices = []
        to_squeeze = []
        non_scalar_indices = []
        for axis_, s in enumerate(index):
            if isinstance(s, slice):
                if s.start is None and s.stop is None and s.step is None:
                    continue
                if s.step is None or s.step > 0:
                    sliced_indices.append(
                        [
                            s.start or 0,
                            s.stop if s.stop is not None else shape[axis_],
                            axis_,
                            s.step or 1,
                        ]
                    )
                else:
                    sliced_indices.append(
                        [
                            s.start if s.start is not None else (shape[axis_] - 1),
                            s.stop if s.stop is not None else -(shape[axis_] + 1),
                            axis_,
                            s.step,
                        ]
                    )
            elif isinstance(s, Tensor):
                if s.is_scalar:
                    scalar_indices.append([s, s + 1, axis_, 1])
                    to_squeeze.append(axis_)
                else:
                    non_scalar_indices.append((axis_, s))
            else:
                raise TypeError(f"Unexpected type {type(s)}: slice or int expected.")

        # Non-scalar-indexing requires the use of ONNX Gather operation.
        # Slicing can be implemented efficiently using ONNX's Slice operation.
        # Scalar-indexing can be implemented using either Gather or with the Slice operation.
        # We map scalar-indexing into the Slice operation, except in the special case
        # of a single scalar-index (with no other sliced_index), which we map directly
        # to a Gather.

        if not (sliced_indices or scalar_indices or non_scalar_indices):
            # Edge case: no index specified. Eg. A[:, :]
            return op.Identity(self)
        if not sliced_indices and len(scalar_indices) == 1:
            # Special case of indexing along a single axis: A[i], A[:, i], A[:, :, i] etc.
            # promote integer input to tensor
            axis = to_squeeze[0]
            index_value = index[axis]
            # use Gather to perform indexing
            result = op.Gather(self, index_value, axis=axis)
        elif sliced_indices or scalar_indices:
            sliced_indices = sliced_indices + scalar_indices
            indices = np.array(sliced_indices, dtype=np.int64).T
            starts = Tensor(indices[0])
            ends = Tensor(indices[1])
            axes = Tensor(indices[2])
            steps = Tensor(indices[3])
            result = op.Slice(self, starts, ends, axes, steps)
            if to_squeeze:
                result = Tensor(np.squeeze(result.value, axis=tuple(to_squeeze)))
        else:
            result = self
        for axis, value in non_scalar_indices:
            result = op.Gather(result, value, axis=axis)

        return result

    def __mod__(self, other):
        if self.onnx_dtype in {
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
        }:
            return self._opset.Mod(self, other, fmod=1)
        return self._opset.Mod(self, other)

    def __ne__(self, other):
        temp = self._opset.Equal(self, other)
        return self._opset.Not(temp)

    def __neg__(self):
        return self._opset.Neg(self)

    def __add__(self, other):
        return self._opset.Add(self, other)

    def __radd__(self, other):
        return self._opset.Add(other, self)

    def __and__(self, other):
        return self._opset.And(self, other)

    def __rand__(self, other):
        return self._opset.And(other, self)

    def __mul__(self, other):
        return self._opset.Mul(self, other)

    def __rmul__(self, other):
        return self._opset.Mul(other, self)

    def __matmul__(self, other):
        return self._opset.MatMul(self, other)

    def __or__(self, other):
        return self._opset.Or(self, other)

    def __pow__(self, other):
        return self._opset.Pow(self, other)

    def __sub__(self, other):
        return self._opset.Sub(self, other)

    def __rsub__(self, other):
        return self._opset.Sub(other, self)

    def __truediv__(self, other):
        return self._opset.Div(self, other)

    def __lt__(self, other):
        return self._opset.Less(self, other)

    def __le__(self, other):
        return self._opset.LessOrEqual(self, other)

    def __eq__(self, other):
        return self._opset.Equal(self, other)

    def __ge__(self, other):
        return self._opset.GreaterOrEqual(self, other)

    def __gt__(self, other):
        return self._opset.Greater(self, other)


###################################################################################################


class Node:
    def __init__(
        self,
        namespace: Optional[str],
        op_name: Optional[str],
        inputs: Sequence[ValidInputType],
        attributes: Mapping[str, ValidTorchValueType],
        n_outputs: int,
        function: Optional[onnxscript.OnnxFunction] = None,
    ):
        if not (namespace and op_name) or function is not None:
            raise ValueError(
                "Either provide namespace and op_name, or provide a function, but not both."
            )
        if function is None and not (namespace and op_name):
            raise ValueError("Either provide namespace and op_name, or provide a function.")

        self._namespace = namespace
        self._op_name = op_name
        self._inputs = inputs
        self._attributes = attributes
        self._n_outputs = n_outputs
        self._function = function
        self._is_function = function is not None

    def to_node_proto(self):
        onnx_node = onnx.helper.make_node(
            self.op_name,
            inputs=[t.name for t in self.inputs],
            outputs=[t.name for t in self.outputs],
            domain=self.domain,
            **self.attributes,  # TODO: check if this works
        )
        return onnx_node

    @property
    def is_function(self) -> bool:
        return self._is_function

    @property
    def namespace(self) -> Optional[str]:
        return self._namespace

    @property
    def op_name(self) -> Optional[str]:
        return self._op_name

    @property
    def function(self) -> Optional[onnxscript.OnnxFunction]:
        return self._function

    @classmethod
    def from_op_schema(
        cls, op_schema: onnx.defs.OpSchema, inputs, attributes, n_outputs
    ) -> Node:
        namespace = op_schema.domain
        op_name = op_schema.name
        return cls(
            function=None,
            namespace=namespace,
            op_name=op_name,
            inputs=inputs,
            attributes=attributes,
            n_outputs=n_outputs,
        )

    @classmethod
    def from_function(
        cls, onnx_function: onnxscript.OnnxFunction, inputs, attributes, n_outputs
    ) -> Node:
        namespace = onnx_function.function_ir.domain
        op_name = onnx_function.name
        return cls(
            function=onnx_function,
            namespace=namespace,
            op_name=op_name,
            inputs=inputs,
            attributes=attributes,
            n_outputs=n_outputs,
        )


# TODO(titaiwang): How we deal with subgraph?
class Graph:
    def __init__(self, producer_name="pytorch") -> None:
        # All the functions used, deduplicated by name
        # key: (name, domain)
        self._function_store: Dict[Tuple[str, str], onnxscript.OnnxFunction] = {}
        self._nodes: List[Node] = []
        self._inputs: List[Tensor] = []
        self._outputs: List[Tensor] = []
        self._initializer: List[Tensor] = []
        self._producer_name: str = producer_name
        # TODO: what are these two string for?
        self._doc_string: str = ""
        self._name: str = ""

    # TODO(titaiwang): Should we not expose Node?
    # if function and op_schema can be alinged, we can just use op_schema
    @runtime_typing.checked
    def add_node(self, node: Node):
        if node.is_function:
            identifier = (node.op_name, node.namespace)
            self._function_store[identifier] = node.function
        self._nodes.append(node)

    # TODO(titaiwang): Should we not expose Tensor?
    # NOTE: We don't need `add_initializer` because we can just use `add_input`
    # for constant tensors
    @runtime_typing.checked
    def add_input(self, tensor: Tensor):
        self._inputs.append(tensor)

    @runtime_typing.checked
    def add_output(self, tensor: Tensor):
        self._outputs.append(tensor)

    # TODO(titaiwang): this function is not friendly to users when
    # they want to build the graph from scratch, as they need to
    # manually set the value to initializers, so we need another
    # method to get all initializers from the graph `self.initializers`
    @runtime_typing.checked
    def add_initializer(self, tensor: Tensor, name: str):
        pass

    def _get_constant_tensors(self) -> Dict[str, Tensor]:
        input_const = [tensor for tensor in self._inputs if not tensor.is_fake]
        output_const = [tensor for tensor in self._outputs if not tensor.is_fake]
        node_related_const = [
            tensor
            for node in self._nodes
            for tensor in (node.inputs + node.outputs)
            if not tensor.is_fake
        ]
        return {
            tensor.name: tensor for tensor in input_const + output_const + node_related_const
        }

    @property
    def initializers(self) -> List[Tensor]:
        return self._initializer or self._get_constant_tensors()

    def _to_onnx_graph(self) -> onnx.GraphProto:
        # convert tensor to tensor value info according to the onnx_type
        input_value_infos = [tensor.to_value_info() for tensor in self._inputs]
        output_value_infos = [tensor.to_value_info() for tensor in self._outputs]
        # convert initializer
        initializer = [
            constant_tensor.to_value_int() for constant_tensor in self.initializers.values()
        ]
        # convert node to node proto
        node_value_infos = [node.to_node_proto() for node in self._nodes]
        # convert graph to graph proto
        graph_proto = onnx.helper.make_graph(
            nodes=node_value_infos,
            name=self._name,
            inputs=input_value_infos,
            outputs=output_value_infos,
            initializer=initializer,
            doc_string=self._doc_string,
        )
        return graph_proto

    def to_onnx_model(self, opset_version: int) -> onnx.ModelProto:
        graph_proto = self._to_onnx_graph()
        model_proto = onnx.helper.make_model(
            graph_proto,
            producer_name=self._producer_name,
            doc_string="",  # TODO
            functions=list(self._function_store.values()),
            opset_imports=opset_version,
            ir_version=8,  # TODO
        )
        return model_proto

    # TODO(titaiwang)
    @runtime_typing.checked
    def from_onnx(self, model: onnx.ModelProto):
        pass


class GraphEvaluator(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph into torchscript."""

    def __init__(self, graph: Graph):
        self._graph: Graph = graph

    @property
    def graph(self) -> Graph:
        return self._graph

    def eval(self, schema, inputs, attributes):
        n_outputs = evaluator.compute_num_outputs(schema, inputs, attributes)
        node = Node.from_op_schema(schema, inputs, attributes, n_outputs)
        return self._graph.add_node(node)

    @runtime_typing.checked
    def eval_function(  # type: ignore[override]
        self,
        function: onnxscript.OnnxFunction,
        args: Sequence[ValidArgumentType],
        kwargs: Mapping[str, ValidArgumentType],
    ):
        # args/kwargs are TorchScriptTensor/python built-in based
        param_schemas = function.param_schemas()
        (
            inputs,
            attributes,
        ) = param_manipulation.separate_input_attributes_from_arguments(
            param_schemas, args, kwargs, fill_defaults=True, allow_extra_kwargs=True
        )
        name_to_schema = {param.name: param for param in param_schemas}
        # TODO(titaiwang): DO we still need this?
        for name, value in attributes.items():
            param = name_to_schema[name]
            # Cast int to float if needed
            if param.type in {float, "float"}:
                # FIXME(justinchuby): Create invariant on the type of param.type to simplify this
                attributes[name] = float(value)
        # NOTE: Initialize node
        node = Node.from_function(
            function, inputs, attributes, n_outputs=len(function.function_ir.outputs)
        )
        return self._graph.add_node(node)
