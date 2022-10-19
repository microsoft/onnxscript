from typing import Optional

from onnxscript import evaluator, eager_mode_evaluator, values, onnx_opset

def id(schema):
    return schema.name, schema.domain, schema.since_version

class ORTEvaluator(evaluator.Evaluator):
    """Evaluates ONNX ops using ONNX Runtime."""

    def _eval(self, schema, inputs, attributes, closure):
        return eager_mode_evaluator.call_ort(schema, inputs, attributes, closure)
    
class ORTMixedEvaluator(ORTEvaluator):
    """Evaluates ONNX ops using ONNX Runtime, unless an overriding python implementation
    is registered. This is useful for higher-order ops such as Scan and SequenceMap,
    allowing for python-based debugging."""

    def __init__(self) -> None:
        super().__init__()
        self._python_ops = {}
    
    def use_graph_attribute(self, schema):
        return id(schema) not in self._python_ops

    def _eval(self, schema, inputs, attributes, closure):
        if id(schema) in self._python_ops:
            return self._python_ops[id(schema)](inputs, attributes)
        else:
            return super()._eval(schema, inputs, attributes, closure)
    
    def register(self, opset: Optional[values.Opset] = None):
        opset = opset or onnx_opset.default_opset
        def decorator(function):
            schema = opset[function.__name__]
            self._python_ops[id(schema)] = function
            return function
        return decorator

ort_evaluator = ORTMixedEvaluator()

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