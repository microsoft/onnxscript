# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import typing
from typing import Any, List
from enum import IntFlag
import onnx


class DebugInfo:

    def __init__(self, lineno, source="string"):
        if hasattr(lineno, 'lineno'):
            self.ast_obj = lineno
            self.lineno = lineno.lineno
        elif isinstance(lineno, int):
            self.ast_obj = None
            self.lineno = lineno
        else:
            raise NotImplementedError(
                "Unable to extract debug information from type %r." % type(lineno))
        self.source = source

    def msg(self, text):
        return "ERROR\n%s\n    %s" % (str(self), text)

    def __str__(self):
        return "%s:%d" % (self.source, self.lineno)


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

    def __call__(self, *args, **kwargs):
        return self.evaluator(self.opschema, *args, **kwargs)


class OnnxFunction(Op):
    '''
    Represents an ONNX op for which a function-body has been defined in onnxscript.
    '''

    def __init__(self, opset, pyfun, irfun):
        opset = opset or Opset(irfun.domain, 1)
        super().__init__(opset, irfun.name)
        self.function = pyfun
        self.function_ir = irfun

    @property
    def name(self):
        return self.opname

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def to_function_proto(self, domain=None):
        return self.function_ir.to_function_proto(domain or self.opset)

    def to_model_proto(self, **kwargs):
        if self.function_ir.attrs:
            raise ValueError("A function with attributes cannot be exported as a model.")
        # Note: The function must also have monomorphic type annotation for inputs/outputs
        # to be converted into a valid model. Otherwise, we can still produce an ONNX
        # model, but it will not pass the ONNX model checker. We do not report an error
        # at this stage.
        return self.function_ir.to_model_proto(**kwargs)


# Values fall into the following categories:
# ConstValue: values known at translation-time, mapped to ONNX attributes
# AttrRef: Function parameters of attribute-kind, also mapped to ONNX attributes
# Dynamic: values computed at runtime (of tensor type, for now) mapped to NodeArgs.


class Value:
    """
    A Value is a named variable from the function script.
    Values fall into the following categories:
    ConstValue: values known at translation-time, mapped to ONNX attributes
    AttrRef: Function parameters of attribute-kind, also mapped to ONNX attributes
    Dynamic: values computed at runtime (of tensor type, for now) mapped to NodeArgs.
    """

    def __init__(self, val: Any, info: DebugInfo) -> None:
        if not isinstance(info, DebugInfo):
            raise TypeError("info must be of DebugInfo not %r." % type(info))
        self.value = val
        self.info = info

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.value)


class ConstValue(Value):
    def __init__(self, val: [float, int], info: DebugInfo) -> None:
        if not isinstance(val, (float, int)):
            raise TypeError(
                "val must be numeric not %r." % type(val))
        super().__init__(val, info)


class AttrRef(Value):
    def __init__(
            self,
            name: str,
            typeinfo: type or List,
            info: DebugInfo) -> None:
        '''
        Arguments:
            name: name of the attribute
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
    def __init__(self, val: str, kind: DynamicKind, info: DebugInfo) -> None:
        super().__init__(val, info)
        assert isinstance(kind, DynamicKind)
        self.kind = kind

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.value, self.kind)
