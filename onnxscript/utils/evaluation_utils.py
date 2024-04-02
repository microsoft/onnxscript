from __future__ import annotations

import pathlib

import numpy as np
import onnx
from onnx import helper as onnx_helper


def load_test_data(
    qual_model_dir: str, input_names: list[str]
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    test_data_dir = pathlib.Path(qual_model_dir) / "test_data_set_0"
    inputs = {}
    expected_outputs = []
    for test_data in test_data_dir.glob("input_*.pb"):
        idx = int(test_data.stem[len("input_") :])
        input_name = input_names[idx]
        input_data = onnx.TensorProto()
        with open(test_data, "rb") as f:
            input_data.ParseFromString(f.read())
        inputs[input_name] = onnx.numpy_helper.to_array(input_data)

    output_file_paths = list(test_data_dir.glob("output_*.pb"))
    expected_outputs = [None] * len(output_file_paths)
    for test_data in test_data_dir.glob("output_*.pb"):
        idx = int(test_data.stem[len("output_") :])
        output_data = onnx.TensorProto()
        with open(test_data, "rb") as f:
            output_data.ParseFromString(f.read())
        expected_outputs[idx] = onnx.numpy_helper.to_array(output_data)  # type: ignore[call-overload]

    assert all(name in inputs for name in input_names), "Some inputs are missing."
    assert not any(output is None for output in expected_outputs), "Some outputs are missing."

    return inputs, expected_outputs  # type: ignore[return-value]


def generate_random_input(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Generate random input for the model.

    NOTE: This is unused. There is parity issue with randomly generated data. Need investigation.
    """
    inputs = {}
    for _, input in enumerate(model.graph.input):
        shape = [d.dim_value for d in input.type.tensor_type.shape.dim]
        np_dtype = onnx_helper.tensor_dtype_to_np_dtype(input.type.tensor_type.elem_type)
        if np_dtype is None:
            raise ValueError(f"Unsupported dtype: {input.type.tensor_type.elem_type}")
        if np_dtype in (np.float16, np.float32, np.float64):
            inputs[input.name] = np.random.rand(*shape).astype(np_dtype) - 0.5
        else:
            inputs[input.name] = np.random.randint(3, 100, size=shape, dtype=np_dtype)
    return inputs
