# SPDX-License-Identifier: Apache-2.0

from typing import Any
from enum import IntFlag
import onnx

# ONNX opsets (correspond to python modules in reference-mode)
# Have a domain-name, version, and a list of ops


class Opset:
    def __init__(self, domain, version) -> None:
        self.domain = domain
        self.version = version

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


opset15 = Opset("", 15)

msdomain1 = Opset("com.microsoft", 1)

# ONNX ops


class Op:
    def __init__(self, opset, opname) -> None:
        self.opset = opset
        self.opname = opname

    def get_schema(self):
        return self.opset[self.opname]

    def has_schema(self):
        return (self.opname in self.opset)


class Value:
    """
    A Value is a named variable from the function script.
    Values fall into the following categories:
    ConstValue: values known at translation-time, mapped to ONNX attributes
    AttrRef: Function parameters of attribute-kind, also mapped to ONNX attributes
    Dynamic: values computed at runtime (of tensor type, for now) mapped to NodeArgs.
    """
    def __init__(self, val: Any) -> None:
        self.value = val

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.value)


class ConstValue(Value):
    def __init__(self, val: [float, int]) -> None:
        if not isinstance(val, (float, int)):
            raise TypeError(
                "val must be numeric not %r." % type(val))
        super().__init__(val)


class AttrRef(Value):
    def __init__(self, name: str, typeinfo: type) -> None:
        super().__init__(name)
        if not isinstance(typeinfo, type):
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
    def __init__(self, val: str, kind: DynamicKind) -> None:
        super().__init__(val)
        assert isinstance(kind, DynamicKind)
        self.kind = kind

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.value, self.kind)
