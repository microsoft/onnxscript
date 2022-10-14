from abc import ABC, abstractmethod

from onnxscript import autocast, values

class Evaluator(ABC):
    """Base class for evaluation of ONNX ops."""

    def eval(self, schema, inputs, attributes):
        closure = self.adapt_attributes(schema, attributes)
        inputs = self.adapt_inputs(schema, inputs)
        return self._eval(schema, inputs, attributes, closure)
    
    def adapt_inputs(self, schema, inputs):
        '''Transform inputs to the expected format for the evaluator.
        
        Enables some syntactic sugar, such as the use of Python scalars, 
        in a manner consistent with the translator. See autocast.py for details.'''
        return autocast.dynamic_cast_inputs(schema, *inputs)
    
    def adapt_attributes(self, schema, attributes):
        """Transform attributes (in-place) to the expected format for the evaluator.
        
        Returns a closure that can be used to evaluate graph-valued attributes."""
        closure = {}
        for k, v in attributes.items():
            if isinstance(v, values.OnnxClosure):
                attributes[k] = v.function_ir.to_graph_proto()
                for pyvar, onnxvar in v.function_ir.outer_scope_variables:
                    closure[onnxvar.value] = v.frame.f_locals[pyvar]
            elif callable(v):
                raise ValueError(
                    f"Error: function-valued attribute {v.__name__} has no graph_proto"
                    "attribute. Did you forget to decorate it with @graph?"
                )
        return closure
    
    @abstractmethod
    def _eval(self, schema, inputs, attributes, closure):
        pass

instance_ = None

def instance():
    """Returns the current Evaluator instance."""
    return instance_

def set_instance(instance):
    """Sets the current Evaluator instance."""
    global instance_
    instance_ = instance