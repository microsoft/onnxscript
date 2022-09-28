# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from enum import IntFlag
from typing import Any, List, _GenericAlias

import numpy as np
import onnx

from onnxscript import autocast, debuginfo, eager_mode_evaluator, tensor


class Opset:
    """
    Represents an ONNX Opset, which consists of a domain name, a version.
    It also contains a set of operations. This represents an Opset defined
    in the ONNX schema registry and the operations are retrieved from the
    ONNX schema registry. It also stores function definitions created for
    ops in the corresponding Opset.

    Only a single instance of Opset is created for a given (domain, version) pair.
    """

    cache = {}

    def __new__(cls, domain, version):
        key = (domain, version)
        existing = cls.cache.get(key)
        if existing:
            return existing
        instance = super().__new__(cls)
        instance.domain = domain
        instance.version = version
        instance.function_defs = {}
        cls.cache[key] = instance
        return instance

    def __repr__(self):
        return f"{self.__class__.__name__}({self.domain!r}, {self.version!r})"

    def __init__(self, domain, version) -> None:
        # Nothing to do. Object is initialized by __new__
        pass

    def __getitem__(self, opname):
        try:
            return onnx.defs.get_schema(opname, self.version, self.domain)
        except Exception:  # pylint: disable=broad-except # TODO: more specific exception
            return None

    def __contains__(self, opname):
        try:
            onnx.defs.get_schema(opname, self.version, self.domain)
            return True
        except Exception:  # pylint: disable=broad-except # TODO: more specific exception
            return False

    def __str__(self) -> str:
        return self.domain

    def __getattr__(self, attr: str):
        try:
            schema = onnx.defs.get_schema(attr, self.version, self.domain)
            return Op(self, attr, schema)
        except Exception as exc:
            raise AttributeError(f"Attribute {attr} not found.") from exc

    def add_function_def(self, fun):
        if fun.name in self.function_defs:

            logger = logging.getLogger("onnx-script")
            logger.warning(  # pylint: disable=logging-fstring-interpolation
                f"{fun.name}: Already defined."
            )
        self.function_defs[fun.name] = fun


# ONNX ops


class Op:
    """
    Represents an ONNX op instance (for example, the MatMul op from ONNX opset version 13).
    It belongs to a particular Opset and has a name.
    """

    def __init__(self, opset, opname, opschema=None) -> None:

        self.opset = opset
        self.opname = opname
        self.opschema = opschema
        self.evaluator = eager_mode_evaluator.call_ort

    def is_single_op(self):
        return isinstance(self.opname, str)

    def get_schema(self):
        if self.opschema:
            return self.opschema
        return self.opset[self.opname]

    def has_schema(self):
        return self.opschema is not None

    def __call__(self, *args, **kwargs):
        args = autocast.dynamic_cast_inputs(self.opschema, *args)
        return self.evaluator(self.opschema, *args, **kwargs)


class OnnxFunction(Op):
    """
    Represents an ONNX op for which a function-body has been defined in onnxscript.

    :param opset: opset the function belongs to
    :param pyfun: python function
    :param irfun: python code parsed by class :class:`onnxscript.converter.Converter`
    :param source: source code used to generate the function
    :param kwargs: additional properties used to construct a ModelProto
    """

    def __init__(self, opset, pyfun, irfun, source, kwargs):
        opset = opset or Opset(irfun.domain, 1)
        super().__init__(opset, irfun.name)
        self.function = pyfun
        self.function_ir = irfun
        self.source = source
        self.kwargs = kwargs

    @property
    def name(self):
        "Returns the function name."
        return self.opname

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            # Operator Constant, it is usually called within a function.
            return self._libcall(**kwargs)
        if isinstance(args[0], tensor.Tensor):
            return self._libcall(*args, **kwargs)
        return self._usercall(*args, **kwargs)

    def _usercall(self, *args, **kwargs):
        "Eager mode"
        new_args = []
        for i, a in enumerate(args):
            if isinstance(a, np.ndarray):
                new_args.append(tensor.Tensor(a))
            elif isinstance(a, bool):
                new_args.append(tensor.Tensor(np.array(a)))
            else:
                raise TypeError(f"Unexpected input type {type(a)} for an input {i}.")
        res = self.function(*new_args, **kwargs)
        if isinstance(res, np.ndarray):
            return res
        if isinstance(res, tensor.Tensor):
            return res.value
        if isinstance(res, (list, tuple)):
            unwrapped = []
            for i, r in enumerate(res):
                if isinstance(r, tensor.Tensor):
                    unwrapped.append(r.value)
                else:
                    raise TypeError(
                        f"Unexpected output type {type(r)} for an output {i} "
                        f"in function {self.function!r}."
                    )
            if isinstance(res, tuple):
                return tuple(unwrapped)
            return unwrapped
        raise TypeError(f"Unexpected output type {type(res)} in function {self.function!r}.")

    def _libcall(self, *args, **kwargs):
        """
        This method must be called when a function decoracted with `script`
        calls another one decorated with `script`.
        """
        new_args = []
        for i, a in enumerate(args):
            if isinstance(a, tensor.Tensor):
                new_args.append(a)
            elif isinstance(a, bool):
                # TODO: default values for function parameters
                # are not properly handled yet. This section
                # should disappear.
                new_args.append(tensor.Tensor(np.array(a)))
            else:
                raise TypeError(f"Unexpected input type {type(a)} for an input {i}.")
        res = self.function(*new_args, **kwargs)
        if isinstance(res, tensor.Tensor):
            return res
        if isinstance(res, tuple):
            unwrapped = []
            for i, r in enumerate(res):
                if isinstance(r, tensor.Tensor):
                    unwrapped.append(r)
                else:
                    raise TypeError(
                        f"Unexpected output type {type(r)} for an output {i} "
                        f"in function {self.function!r}."
                    )
            return tuple(unwrapped)
        raise TypeError(f"Unexpected output type {type(res)} in function {self.function!r}.")

    def to_function_proto(self, domain=None):
        "Converts the function into :class:`onnx.FunctionProto`."
        return self.function_ir.to_function_proto(domain or self.opset)

    def to_model_proto(self, **kwargs):
        "Converts the function into :class:`onnx.ModelProto`."
        if self.function_ir.attrs:
            raise ValueError("A function with attributes cannot be exported as a model.")
        # Note: The function must also have monomorphic type annotation for inputs/outputs
        # to be converted into a valid model. Otherwise, we can still produce an ONNX
        # model, but it will not pass the ONNX model checker. We do not report an error
        # at this stage.

        # Merge kwargs specified in script-decorator with those specified in this call.
        merged_kw_args = {**self.kwargs, **kwargs}
        return self.function_ir.to_model_proto(**merged_kw_args)


class Value:
    """
    A Value is used to represent information about named variables used in a script.
    At translation-time, the (local) variables of a script, including its parameters,
    are bound to a Value.

    Values fall into the following categories:

    AttrRef: Function parameters of attribute-kind, also mapped to ONNX attributes

    Dynamic: values computed at runtime (of tensor type, for now) mapped to NodeArgs.
    Dynamic values include input-parameters of the script, as well intermediate
    values computed in the script.

    For example, consider the following script definition:
    ::

        @script()
        def ThresholdedRelu(X, alpha: float):
            zero = op.CastLike(0, X)
            return op.Where(X > alpha, X, zero)

    Here, `X` has a Dynamic value, `alpha` has an AttrRef value, and `zero`
    has a Dynamic value.

    Scripts may also contain references to global variables, but the translator
    does not associate a Value with them. The python value of global variables
    is used directly in the translation, and such global variables are intended
    to be used for limited purposes, namely:
    * To identify an opset
    * To represent constant-values, translated into ONNX constants.
    """

    def __init__(self, val: Any, info: debuginfo.DebugInfo) -> None:
        if not isinstance(info, debuginfo.DebugInfo):
            raise TypeError(f"info must be of debuginfo.DebugInfo not {type(info)!r}.")
        if val is None:
            raise ValueError(info.msg("val cannot be None."))
        self.value = val
        self.info = info

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value!r})"


class AttrRef(Value):
    def __init__(self, name: str, typeinfo: type or List, info: debuginfo.DebugInfo) -> None:
        """
        Arguments:
            name: name of the attribute-parameter
            typeinfo: type annotation of the attribute.
                op's attributes in ONNX are usually single type or list of single type.
            info: for debugging use.
        """
        super().__init__(name, info)
        self.typeinfo = typeinfo
        if not isinstance(typeinfo, (type, _GenericAlias)):
            # typing._GenericAlias for List[int] and List[str], etc.
            raise TypeError(f"Expecting a type not f{type(typeinfo)} for typeinfo.")
        self.typeinfo = typeinfo

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value!r}, {self.typeinfo!r})"


class DynamicKind(IntFlag):
    Unknown = 0
    Input = 1
    Output = 2
    Intermediate = 4
    Loop = 8


class Dynamic(Value):
    def __init__(
        self, val: str, kind: DynamicKind, info: debuginfo.DebugInfo, typeinfo=None
    ) -> None:
        """
        Arguments:
            val: the name of the ONNX variable used to represent this value
            kind: the DynamicKind of this variable
            info: source-location information for error-messages/debugging
            typeinfo: type-information for the value
        """
        super().__init__(val, info)
        assert isinstance(kind, DynamicKind)
        self.kind = kind
        self.typeinfo = typeinfo

    def __repr__(self):
        if self.typeinfo is None:
            return f"{self.__class__.__name__}({self.value!r}, {self.kind!r})"
        return (
            f"{self.__class__.__name__}({self.value}, {self.kind}, typeinfo={self.typeinfo})"
        )
