from typing import Any
import onnxruntime as ort

# TODO: enable invocation of ORT kernels

class Model:
    def __init__(self, onnxfile) -> None:
        self.session = ort.InferenceSession(onnxfile)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        inputs = self.session.get_inputs()
        for i, arg in enumerate(args):
            kwds[inputs[i].name] = arg
        return self.session.run(None, kwds)

def make_ort_value(v):
    pass

def make_attrs(**kwds):
    pass

def call_ort (opname, *args, **kwds):
    model = Model("todo")
    ort_args = [make_ort_value(x) for x in args]
    return model(ort_args)
