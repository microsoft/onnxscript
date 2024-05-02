"""Test IR roundtrip with ONNX model zoo."""

from __future__ import annotations

import gc
import sys
import tempfile
import time

import onnx
from onnx import hub

import onnxscript.testing
from onnxscript import ir


def test_model(model_info: hub.ModelInfo) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        hub.set_dir(temp_dir)
        model_name = model_info.model
        model = hub.load(model_name)
    assert model is not None
    onnx.checker.check_model(model)
    # Fix the missing graph name of some test models
    model.graph.name = "main_graph"

    # Profile the serialization and deserialization process
    ir_model = ir.serde.deserialize_model(model)
    serialized = ir.serde.serialize_model(ir_model)
    onnxscript.testing.assert_onnx_proto_equal(serialized, model)
    onnx.checker.check_model(serialized)


def main():
    model_list = hub.list_models()
    print(f"=== Testing IR on {len(model_list)} models ===")

    # run checker on each model
    failed_models = []
    failed_messages = []
    for model_info in model_list:
        start = time.time()
        model_name = model_info.model
        model_path = model_info.model_path
        print(f"----Testing: {model_name} @ {model_path}----")
        try:
            test_model(model_info)
            print(f"[PASS]: {model_name} roundtrip test passed.")
        except Exception as e:
            print(f"[FAIL]: {e}")
            failed_models.append(model_name)
            failed_messages.append((model_name, e))
        end = time.time()
        print(f"--------------Time used: {end - start} secs-------------")
        # enable gc collection to prevent MemoryError by loading too many large models
        gc.collect()

    if len(failed_models) == 0:
        print(f"{len(model_list)} models have been checked.")
    else:
        print(f"In all {len(model_list)} models, {len(failed_models)} models failed")
        for model_name, error in failed_messages:
            print(f"{model_name} failed because: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
