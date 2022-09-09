# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
import pprint
import typing
from typing import Any, List
from enum import IntFlag
import numpy as np
import onnx
from onnx.defs import OpSchema
from .eager_array import EagerArray


class DebugInfo:

    def __init__(self, lineno, source="string", code=None):
        if hasattr(source, 'source'):
            code = source.source
            current_fn = getattr(source, 'current_fn', None)
            if current_fn is not None:
                source = getattr(source.current_fn, 'name', None)
            else:
                source = None
        if hasattr(lineno, 'lineno'):
            self.ast_obj = lineno
            self.lineno = lineno.lineno
        elif isinstance(lineno, int):
            self.ast_obj = None
            self.lineno = lineno
        elif sys.version_info[:2] < (3, 9):
            # python 3.8 and below
            self.ast_obj = None
            self.lineno = 1
        else:
            raise NotImplementedError(
                f"Unable to extract debug information from type {type(lineno)!r}, "
                f"attributes={pprint.pformat(lineno.__dict__)}.")
        self.source = source
        self.code = None if code is None else code.split('\n')

    def msg(self, text):
        return "ERROR\n%s\n    %s" % (str(self), text)

    def __str__(self):
        if self.code is None:
            line = ''
        else:
            line = "    -- line: " + self.code[self.lineno - 1]
        return "%s:%d%s" % (self.source, self.lineno, line)


class Opset:
    '''
    Represents an ONNX Opset, which consists of a domain name, a version.
    It also contains a set of operations. This represents an Opset defined
    in the ONNX schema registry and the operations are retrieved from the
    ONNX schema registry. It also stores function definitions created for
    ops in the corresponding Opset.

    Only a single instance of Opset is created for a given (domain, version) pair.
    '''
    cache = {}

    def __new__(cls, domain, version):
        key = (domain, version)
        existing = cls.cache.get(key)
        if existing:
            return existing
        instance = super(Opset, cls).__new__(cls)
        instance.domain = domain
        instance.version = version
        instance.function_defs = {}
        cls.cache[key] = instance
        return instance

    def __repr__(self):
        return "%s(%r, %r)" % (self.__class__.__name__, self.domain, self.version)

    def __init__(self, domain, version) -> None:
        # Nothing to do. Object is initialized by __new__
        pass

    def __getitem__(self, opname):
        return onnx.defs.get_schema(opname, self.version, self.domain)

    def __contains__(self, opname):
        try:
            onnx.defs.get_schema(opname, self.version, self.domain)
            return True
        except BaseException:
            return False

    def __str__(self) -> str:
        return self.domain

    def __getattr__(self, attr: str):
        try:
            schema = onnx.defs.get_schema(attr, self.version, self.domain)
            return Op(self, attr, schema)
        except BaseException:
            raise AttributeError(f"Attribute {attr} not found.")

    def add_function_def(self, fun):
        if fun.name in self.function_defs:
            import logging
            logger = logging.getLogger("onnx-script")
            logger.warning(f"{fun.name}: Already defined.")
        self.function_defs[fun.name] = fun


# ONNX ops


class Op:
    '''
    Represents an ONNX op instance (for example, the MatMul op from ONNX opset version 13).
    It belongs to a particular Opset and has a name.
    '''

    def __init__(self, opset, opname, opschema=None) -> None:
        from . import eager_mode_evaluator
        self.opset = opset
        self.opname = opname
        self.opschema = opschema
        self.evaluator = eager_mode_evaluator.call_ort

    def is_single_op(self):
        return isinstance(self.opname, str)

    def get_schema(self):
        return self.opschema

    def has_schema(self):
        return (self.opschema is not None)

    def cast_inputs(self, *args):
        '''
        Uses schema specification to support a limited form of casting.
        * Scalars are promoted to tensors.
        * Further. they are cast to the required type when used in ops with other
        tensor inputs that are required to be of same type.
        Thus, in "A+1" or "Add(A, 1)", the value 1 will be converted to the same
        type as A.

        The supported cases must be in sync with the cases supported by the converter
        to ensure that the eager-mode semantics is same as onnx-conversion semantics.
        '''
        if self.opschema is not None:
            expected_inputs = self.opschema.inputs
            # We make two passes. In the first pass, we identify known type-bindings for
            # type-variables: eg., {'T1' : np.float32, 'T2' : np.int32}.
            # In the second pass, we use these bindings to cast scalar-values to
            # tensors of appropriate types. The two passes are needed to handle cases
            # like "Add(1, X)" where 1 must be cast to the same type as X.
            type_bindings = {}
            args_typevars = []
            for i, x in enumerate(args):
                if i < len(expected_inputs):
                    expected = expected_inputs[i]
                elif expected_inputs[-1].option == OpSchema.FormalParameterOption.Variadic:
                    expected = expected_inputs[-1]
                    if not expected.isHomogeneous:
                        args_typevars.append((x, None))
                        continue
                else:
                    raise ValueError(
                        f"Number of actual parameters {len(args)} "
                        f"exceeds number of formal parameters {len(expected_inputs)}.")
                typevar = expected.typeStr
                if '(' not in typevar:
                    # typevar is an identifier, like "T"
                    if isinstance(x, EagerArray):
                        type_bindings[typevar] = x.dtype
                args_typevars.append((x, typevar))
            newargs = []
            for x, typevar in args_typevars:
                cast_x = x
                if isinstance(x, (int, float)):
                    # Scalar values are promoted to tensors of a type chosen as below:
                    if typevar in type_bindings:
                        dtype = type_bindings[typevar]
                    elif isinstance(x, int):
                        dtype = np.int32
                    elif isinstance(x, float):
                        dtype = np.float32
                    cast_x = EagerArray(np.array(x, dtype=dtype))
                newargs.append(cast_x)
            return tuple(newargs)
        else:
            # Either an error or a custom op.
            # No checks/casts in this case.
            return args

    def __call__(self, *args, **kwargs):
        args = self.cast_inputs(*args)
        return self.evaluator(self.opschema, *args, **kwargs)


class OnnxFunction(Op):
    '''
    Represents an ONNX op for which a function-body has been defined in onnxscript.

    :param opset: opset the function belongs to
    :param pyfun: python function
    :param irfun: python code parsed by class :class:`onnxscript.converter.Converter`
    :param source: source code used to generate the function
    :param kwargs: additional properties used to construct a ModelProto
    '''

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
        if isinstance(args[0], EagerArray):
            return self._libcall(*args, **kwargs)
        return self._usercall(*args, **kwargs)

    def _usercall(self, *args, **kwargs):
        "Eager mode"
        new_args = []
        for i, a in enumerate(args):
            if isinstance(a, np.ndarray):
                new_args.append(EagerArray(a))
            elif isinstance(a, bool):
                new_args.append(EagerArray(np.array(a)))
            else:
                raise TypeError(
                    f"Unexpected input type {type(a)} for an input {i}.")
        res = self.function(*new_args, **kwargs)
        if isinstance(res, np.ndarray):
            return res
        if isinstance(res, EagerArray):
            return res.value
        if isinstance(res, (list, tuple)):
            unwrapped = []
            for i, r in enumerate(res):
                if isinstance(r, EagerArray):
                    unwrapped.append(r.value)
                else:
                    raise TypeError(
                        f"Unexpected output type {type(r)} for an output {i} "
                        f"in function {self.function!r}.")
            if isinstance(res, tuple):
                return tuple(unwrapped)
            return unwrapped
        raise TypeError(
            f"Unexpected output type {type(res)} in function {self.function!r}.")

    def _libcall(self, *args, **kwargs):
        """
        This method must be called when a function decoracted with `script`
        calls another one decorated with `script`.
        """
        new_args = []
        for i, a in enumerate(args):
            if isinstance(a, EagerArray):
                new_args.append(a)
            elif isinstance(a, bool):
                # TODO: default values for function parameters
                # are not properly handled yet. This section
                # should disappear.
                new_args.append(EagerArray(np.array(a)))
            else:
                raise TypeError(
                    f"Unexpected input type {type(a)} for an input {i}.")
        res = self.function(*new_args, **kwargs)
        if isinstance(res, EagerArray):
            return res
        if isinstance(res, tuple):
            unwrapped = []
            for i, r in enumerate(res):
                if isinstance(r, EagerArray):
                    unwrapped.append(r)
                else:
                    raise TypeError(
                        f"Unexpected output type {type(r)} for an output {i} "
                        f"in function {self.function!r}.")
            return tuple(unwrapped)
        raise TypeError(
            f"Unexpected output type {type(res)} in function {self.function!r}.")

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

    def __init__(self, val: Any, info: DebugInfo) -> None:
        if not isinstance(info, DebugInfo):
            raise TypeError("info must be of DebugInfo not %r." % type(info))
        if val is None:
            raise ValueError(info.msg('val cannot be None.'))
        self.value = val
        self.info = info

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.value)


class AttrRef(Value):
    def __init__(
            self,
            name: str,
            typeinfo: type or List,
            info: DebugInfo) -> None:
        '''
        Arguments:
            name: name of the attribute-parameter
            typeinfo: type annotation of the attribute.
                op's attributes in ONNX are usually single type or list of single type.
            info: for debugging use.
        '''
        super().__init__(name, info)
        self.typeinfo = typeinfo
        if not isinstance(typeinfo, (type, typing._GenericAlias)):
            # typing._GenericAlias for List[int] and List[str], etc.
            raise TypeError(f"Expecting a type not f{type(typeinfo)} for typeinfo.")
        self.typeinfo = typeinfo

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.value, self.typeinfo)


class DynamicKind(IntFlag):
    Unknown = 0
    Input = 1
    Output = 2
    Intermediate = 4
    Loop = 8


class Dynamic(Value):
    def __init__(self, val: str, kind: DynamicKind, info: DebugInfo, typeinfo=None) -> None:
        '''
        Arguments:
            val: the name of the ONNX variable used to represent this value
            kind: the DynamicKind of this variable
            info: source-location information for error-messages/debugging
            typeinfo: type-information for the value
        '''
        super().__init__(val, info)
        assert isinstance(kind, DynamicKind)
        self.kind = kind
        self.typeinfo = typeinfo

    def __repr__(self):
        if self.typeinfo is None:
            return '%s(%r, %r)' % (self.__class__.__name__, self.value, self.kind)
        return '%s(%r, %r, typeinfo=%r)' % (
            self.__class__.__name__, self.value, self.kind, self.typeinfo)
