import onnxscript as ir


def get_value_name(value: ir.Value) -> str:
    """Get the name of a value."""
    if value.name:
        return value.name

def serialize_model(model: ir.Model) -> str:
    """Convert the model to Python code."""
    py_lines = []
    py_lines.append("import onnxscript as ir")
    py_lines.append("def create_model():")
    py_lines.append(
f"""
    model = ir.Model(
        graph=ir.Graph(
            name={model.graph.name!r},
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={model.graph.opset_imports!r},
        ),
        name={model.name!r},
        ir_version={model.ir_version},
        producer_name={model.producer_name!r},
    )
"""
    )
    for input in model.graph.inputs:
        py_lines.append(f"    model.graph.add_input({input.name!r}, {input.type!r})")
