from typing import Optional

from sympy import fu
from onnxscript import evaluator, eager_mode_evaluator, values, onnx_opset

def id(schema):
    return schema.name, schema.domain, schema.since_version

class ORTEvaluator(evaluator.Evaluator):
    """Evaluates ONNX ops using ONNX Runtime."""

    def __init__(self) -> None:
        super().__init__()
        self._python_ops = {}

    def _eval(self, schema, inputs, attributes, closure):
        return eager_mode_evaluator.call_ort(schema, inputs, attributes, closure)
    
    def register(self, opset: Optional[values.Opset] = None):
        opset = opset or onnx_opset.default_opset
        def decorator(function):
            schema = opset[function.__name__]
            self._python_ops[id(schema)] = function
            return function
        return decorator

    def adapt_attributes(self, schema, attributes):
        """Transform attributes (in-place) to the expected format for the evaluator.
        
        Returns a closure that can be used to evaluate graph-valued attributes."""
        pymode = id(schema) in self._python_ops
        closure = {}
        for k, v in attributes.items():
            if isinstance(v, values.OnnxClosure):
                if pymode:
                    attributes[k] = v.function
                else:
                    attributes[k] = v.function_ir.to_graph_proto()
                    for pyvar, onnxvar in v.function_ir.outer_scope_variables:
                        closure[onnxvar.value] = v.frame.f_locals[pyvar]
            elif callable(v):
                raise ValueError(
                    f"Error: function-valued attribute {v.__name__} has no graph_proto"
                    "attribute. Did you forget to decorate it with @graph?"
                )
        return closure

    def eval(self, schema, inputs, attributes):
        inputs = self.adapt_inputs(schema, inputs)
        closure = self.adapt_attributes(schema, attributes)
        if id(schema) in self._python_ops:
            return self._python_ops[id(schema)](inputs, attributes)
        return self._eval(schema, inputs, attributes, closure)

ort_evaluator = ORTEvaluator()

evaluator.set_instance(ort_evaluator) # TODO: move this to a better place

@ort_evaluator.register()
def SequenceMap(inputs, attributes):
    """Evaluates a SequenceMap op."""
    fun = attributes["body"]
    def get_input_of(input_index, iter_num):
        input = inputs[input_index]
        if isinstance(input, list):
            return input[iter_num]
        return input
    def get_input(iter_num):
        return [get_input_of(input_index, iter_num) for input_index in range(len(inputs))]
    return [fun(*(get_input(i))) for i in range(len(inputs[0]))]