from __future__ import annotations

from onnxscript import ir
import onnx


class ShapeInferencePass(ir.passes.PassBase):
    """This pass performs shape inference on the graph."""
    in_place = False

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Store the original initializer values so they can be restored
        initializer_values = tuple(model.graph.initializers.values())
        tensors = {v.name: v.const_value for v in initializer_values}
        original_inputs_len = len(model.graph.inputs)
        initializer_names = {v.name for v in initializer_values}

        # Turn the initializers into inputs and clear the initializers
        # to limit the model size
        for initializer in initializer_values:
            if initializer not in model.graph.inputs:
                model.graph.inputs.append(initializer)
            initializer.const_value = None
        model.graph.initializers.clear()

        # Perform shape inference
        try:
            proto = ir.serde.serialize_model(model)
            proto = onnx.shape_inference.infer_shapes(proto, data_prop=True)
            model = ir.serde.deserialize_model(proto)
        finally:
            # Restore the original initializer values so the model is unchanged
            for new_input in model.graph.inputs:
                # Assign the tensors back to the initializers
                if new_input.name in initializer_names:
                    model.graph.register_initializer(new_input)
                    new_input.const_value = tensors[new_input.name]
            # Remove the inputs that were added
            inputs = model.graph.inputs[:original_inputs_len]
            model.graph.inputs.clear()
            model.graph.inputs.extend(inputs)

         # Even though modified, we know the pass will not change the model if we ran it again.
         # So set modified to False
        return ir.passes.PassResult(model, modified=False)
