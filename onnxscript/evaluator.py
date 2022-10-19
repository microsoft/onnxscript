from abc import ABC, abstractmethod
from contextlib import contextmanager

from onnxscript import autocast, values

class Evaluator(ABC):
    """Base class for evaluation of ONNX ops.
    
    The execution of onnxscript functions in eager-mode is dispatched to an Evaluator
    instance (or, more precisely, to the eval method of the Evaluator instance).
    The evaluator is expected to transform the input/output/attribute representation
    supported by onnxscript to those expected by a particular backend.
    """

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
        use_graph_attribute = self.use_graph_attribute(schema)
        closure = {}
        for k, v in attributes.items():
            if isinstance(v, values.OnnxClosure):
                if use_graph_attribute:
                    attributes[k] = v.function_ir.to_graph_proto()
                    for pyvar, onnxvar in v.function_ir.outer_scope_variables:
                        closure[onnxvar.value] = v.frame.f_locals[pyvar]
                else:
                    attributes[k] = v.function
            elif callable(v):
                raise ValueError(
                    f"Error: function-valued attribute {v.__name__} has no graph_proto"
                    "attribute. Did you forget to decorate it with @graph?"
                )
        return closure
    
    def use_graph_attribute(self, schema):
        return True

    @abstractmethod
    def _eval(self, schema, inputs, attributes, closure):
        pass

# Used to control the default evaluator instance. A simple approach for now.

instance_ = None

def instance():
    """Returns the current Evaluator instance."""
    return instance_

def set_instance(instance):
    """Sets the current Evaluator instance."""
    global instance_
    instance_ = instance

@contextmanager
def using_instance(instance):
    """Context manager that temporarily sets the current Evaluator instance."""
    old_instance = instance_
    set_instance(instance)
    try:
        yield
    finally:
        set_instance(old_instance)
