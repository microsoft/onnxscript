import onnx

# ONNX opsets (correspond to python modules in reference-mode)
# Have a domain-name, version, and a list of ops

class Opset:
    def __init__(self, domain, version) -> None:
        self.domain = domain
        self.version = version
    
    def  __getitem__ (self, opname):
        return onnx.defs.get_schema (opname, self.version, self.domain)
    
    def __contains__ (self, opname):
        try:
            onnx.defs.get_schema (opname, self.version, self.domain)
            return True
        except:
            return False

    def __str__(self) -> str:
        return self.domain

opset15 = Opset("", 15)

# ONNX ops

class Op:
    def __init__(self, opset, opname) -> None:
        self.opset = opset
        self.opname = opname
    
    def get_schema(self):
        return self.opset[self.opname]
    
    def has_schema(self):
        return (self.opname in self.opset)

# Values fall into the following categories:
# ConstValue: values known at translation-time, mapped to ONNX attributes
# AttrRef: Function parameters of attribute-kind, also mapped to ONNX attributes
# Dynamic: values computed at runtime (of tensor type, for now) mapped to NodeArgs

class Value:
    def __init__(self, val) -> None:
        self.value = val

class ConstValue(Value):   
    def __init__(self, val) -> None:
        super().__init__(val)

class AttrRef(Value):
    def __init__(self, name: str, type) -> None:
        super().__init__(name)
        self.type = type

class Dynamic(Value):
    def __init__(self, val) -> None:
        super().__init__(val)
