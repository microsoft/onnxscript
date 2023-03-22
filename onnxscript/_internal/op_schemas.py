"""Python implementation of OpSchema in onnx/defs/schema.cc."""


class OpSchema:
    def __init__(self):
        self.kUninitializedSinceVersion = -1
        self.Single = 0
        self.Optional = 1
        self.Variadic = 2
        self.Unknown = 0
        self.Differentiable = 1
        self.NonDifferentiable = 2

class FormalParameter:
    def __init__(self):
        self.name = None
        self.type_set = None
        self.type_str = None
        self.description = None
        self.param_option = None
        self.is_homogeneous = None
        self.min_arity = None
        self.differentiation_category = None
