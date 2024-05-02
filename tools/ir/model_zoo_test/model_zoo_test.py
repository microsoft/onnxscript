"""Test IR roundtrip with ONNX model zoo."""

from __future__ import annotations

import argparse
import contextlib
import gc
import multiprocessing.pool
import sys
import tempfile
import time
import traceback

import onnx
from onnx import hub

import onnxscript.testing
from onnxscript import ir


def test_model(model_info: hub.ModelInfo) -> float:
    model_name = model_info.model
    with contextlib.redirect_stdout(None):
        model = hub.load(model_name)
    assert model is not None
    onnx.checker.check_model(model)
    # Fix the missing graph name of some test models
    model.graph.name = "main_graph"

    # Profile the serialization and deserialization process
    start = time.time()
    ir_model = ir.serde.deserialize_model(model)
    serialized = ir.serde.serialize_model(ir_model)
    end = time.time()
    onnxscript.testing.assert_onnx_proto_equal(serialized, model)
    onnx.checker.check_model(serialized)
    return end - start


def run_one_test(model_info: hub.ModelInfo) -> tuple[str, Exception | None]:
    start = time.time()
    model_name = model_info.model
    model_path = model_info.model_path
    message = f"----Testing: {model_name} @ {model_path}----"
    try:
        time_passed = test_model(model_info)
        message += green(f"\n[PASS]: {model_name} roundtrip test passed.")
    except Exception as e:  # noqa: F841
        time_passed = -1
        error = traceback.format_exc()
        message += red(f"\n[FAIL]: {error}")
    else:
        e = None
        error = None
    end = time.time()
    message += f"\n-------Time used: {end - start} secs, roundtrip: {time_passed} secs -------"
    print(message, flush=True)
    # enable gc collection to prevent MemoryError by loading too many large models
    gc.collect()
    return model_name, e


def green(text: str) -> str:
    return f"\033[32m{text}\033[0m"


def red(text: str) -> str:
    return f"\033[31m{text}\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Test IR roundtrip with ONNX model zoo.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run. Default is 1.",
    )
    args = parser.parse_args()

    model_list = hub.list_models()
    print(f"=== Testing IR on {len(model_list)} models ===")

    # run checker on each model
    failed_models = []
    failed_messages = []
    # Use multi-processing to speed up the testing process
    with tempfile.TemporaryDirectory() as temp_dir:
        hub.set_dir(temp_dir)
        results = multiprocessing.pool.Pool(args.jobs).map(run_one_test, model_list)
    for model_name, error in results:
        if error is not None:
            failed_models.append(model_name)
            failed_messages.append((model_name, error))
    if len(failed_models) == 0:
        print(f"{len(model_list)} models have been checked.")
    else:
        print(f"In all {len(model_list)} models, {len(failed_models)} models failed")
        for model_name, error in failed_messages:
            print(f"{model_name} failed because: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
