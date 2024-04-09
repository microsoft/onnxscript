"""Lite benchmark script comparing perf between different onnx model of the same torch model.

Folders are expected to be in the following format:

<model-dir>
├── dynamo
│   ├── <model_name>_dynamo.onnx
│   ├── test_data_set_0
├── torchscript
│   ├── <model_name>_torchscript.onnx
│   ├── test_data_set_0
├── dynamo_aot_optimize
│   ├── <model_name>_dynamo_aot_optimize.onnx
│   ├── test_data_set_0
├── dynamo_aot_inline_optimize
│   ├── <model_name>_dynamo_aot_inline_optimize.onnx
│   ├── test_data_set_0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pathlib
import re
import subprocess
import time
from typing import Callable

import numpy as np
import onnx
import onnxruntime
import torch

from onnxscript.utils import evaluation_utils

np.random.seed(0)
logger = logging.getLogger(__name__)

_TORCH_TO_NUMPY_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.longlong,
    torch.bool: np.bool_,
}


def create_iobinding(
    sess: onnxruntime.InferenceSession,
    inputs: dict[str, np.ndarray],
    expected_outputs: list[np.ndarray],
) -> tuple[onnxruntime.IOBinding, list[torch.Tensor], list[torch.Tensor]]:
    iobindings = sess.io_binding()
    bound_inputs = []
    bound_outputs = []
    for input_name, input in inputs.items():
        cuda_tensor = torch.from_numpy(np.array(input)).cuda().contiguous()
        bound_inputs.append(cuda_tensor)
        device = cuda_tensor.device
        logger.debug(
            "binding input name %s to device %s %s, shape %s, dtype %s",
            input_name,
            device.type,
            device.index,
            cuda_tensor.size(),
            _TORCH_TO_NUMPY_DTYPE[cuda_tensor.dtype],
        )
        iobindings.bind_input(
            input_name,
            device.type,
            device.index,
            _TORCH_TO_NUMPY_DTYPE[cuda_tensor.dtype],
            cuda_tensor.size(),
            cuda_tensor.data_ptr(),
        )
    for ort_output_meta, expected_output in zip(sess.get_outputs(), expected_outputs):
        cuda_tensor = torch.empty_like(
            torch.from_numpy(np.array(expected_output)).cuda()
        ).contiguous()
        bound_outputs.append(cuda_tensor)
        device = cuda_tensor.device
        logger.debug(
            "binding output name %s to device %s %s, shape %s, dtype %s",
            ort_output_meta.name,
            device.type,
            device.index,
            cuda_tensor.size(),
            _TORCH_TO_NUMPY_DTYPE[cuda_tensor.dtype],
        )
        iobindings.bind_output(
            ort_output_meta.name,
            device.type,
            device.index,
            _TORCH_TO_NUMPY_DTYPE[cuda_tensor.dtype],
            cuda_tensor.size(),
            cuda_tensor.data_ptr(),
        )
    return iobindings, bound_inputs, bound_outputs


def create_timed_ort_run_callable(
    sess: onnxruntime.InferenceSession,
    inputs: dict[str, np.ndarray],
    expected_outputs: list[np.ndarray],
    provider: str,
) -> Callable[[], tuple[list[np.ndarray], float]]:
    if provider == "CUDAExecutionProvider":
        iobindings, bound_inputs, bound_outputs = create_iobinding(
            sess, inputs, expected_outputs
        )
        run_options = onnxruntime.RunOptions()
        # run_options.only_execute_path_to_fetches = True

        def run_ort_with_iobindings() -> tuple[list[np.ndarray], float]:
            iobindings.synchronize_inputs()
            run_start_time = time.perf_counter()
            sess.run_with_iobinding(iobindings, run_options=run_options)
            iobindings.synchronize_outputs()
            run_end_time = time.perf_counter()
            np_outputs = [output.cpu().numpy() for output in bound_outputs]
            return np_outputs, run_end_time - run_start_time

        return run_ort_with_iobindings
    else:

        def run_ort_directly() -> tuple[list[np.ndarray], float]:
            run_start_time = time.perf_counter()
            outputs = sess.run(None, inputs)
            run_end_time = time.perf_counter()
            return outputs, run_end_time - run_start_time

        return run_ort_directly


def check_and_run_model(
    model_path: str,
    qual_model_name: str,
    inputs: dict[str, np.ndarray],
    expected_outputs: list[np.ndarray],
    iterations: int,
    device: str,
):
    # onnx.shape_inference.infer_shapes(
    #     onnx_model, check_type=True, strict_mode=True, data_prop=True
    # )
    # onnx.checker.check_model(onnx_model, full_check=True)
    provider = "CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider"
    logger.info("Running %s with %s", qual_model_name, provider)

    try:
        session_options = onnxruntime.SessionOptions()
        # Debug mode: more logs, slower
        # session_options.log_severity_level = 0  # everything
        # session_options.log_verbosity_level = 4  # verbose
        # session_options.enable_profiling = True

        # Bench mode: no logs
        session_options.log_verbosity_level = 0  # suppress

        # NOTE: uncomment to save ort optimized model.
        # TODO: make it an arg.
        # session_options.optimized_model_filepath = (
        #     f"ort_{qual_model_name}.onnx"
        # )
        load_start_time = time.perf_counter()
        sess = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=[provider],
        )
        print(
            f"Loading {qual_model_name} model took {time.perf_counter() - load_start_time} seconds."
        )
        input_names = [i.name for i in sess.get_inputs()]
        assert set(input_names) == set(inputs.keys())

        timed_ort_callable = create_timed_ort_run_callable(
            sess, inputs, expected_outputs, provider
        )

        # warm-up
        # outputs = sess.run(None, inputs)
        timed_ort_callable()

        # quick bench
        total_time = 0
        for _ in range(iterations):
            outputs, run_time = timed_ort_callable()
            total_time += run_time

        print(f"Running {qual_model_name} model took {total_time / iterations} seconds.")

        for output, expected_output in zip(outputs, expected_outputs):
            np.testing.assert_allclose(output, expected_output, rtol=5e-1, atol=5e-1)

    except Exception as e:
        print(f"========== {qual_model_name} failed: {e}")
    else:
        print(f"========== {qual_model_name} passed")


@dataclasses.dataclass
class ORTAnalysis:
    graph_transform_hits: dict[str, int] = dataclasses.field(default_factory=dict)
    operator_ep_distribution: dict[str, int] = dataclasses.field(default_factory=dict)


def analyze_ort_logs(stderr_output: str) -> ORTAnalysis:
    analysis = ORTAnalysis()

    for line in stderr_output.split("\n"):
        # Example:
        # [I:onnxruntime:, graph_transformer.cc:15 Apply] GraphTransformer GeluFusion modified: 0 with status: OK
        match = re.search(r"GraphTransformer (\w+) modified: 1 with status: OK", line)
        if match:
            graph_transformer = match.group(1)
            analysis.graph_transform_hits[graph_transformer] = (
                analysis.graph_transform_hits.get(graph_transformer, 0) + 1
            )

        # Example:
        # [V:onnxruntime:, session_state.cc:1149 VerifyEachNodeIsAssignedToAnEp]  All nodes placed on [CPUExecutionProvider]. Number of nodes: 25
        match = re.search(r"All nodes placed on \[(\w+)\]. Number of nodes: (\d+)", line)
        if match:
            ep = match.group(1)
            num_nodes = int(match.group(2))
            analysis.operator_ep_distribution[ep] = (
                analysis.operator_ep_distribution.get(ep, 0) + num_nodes
            )

    return analysis


def run_model(compiler_name: str, model_dir: str, iterations: int, device: str) -> None:
    model_name = pathlib.Path(model_dir).stem
    qual_model_name = f"{model_name}_{compiler_name}"
    qual_model_dir = f"{model_dir}/{compiler_name}"
    model_path = f"{qual_model_dir}/{model_name}_{compiler_name}.onnx"
    model = onnx.load(model_path)
    inputs, expected_outputs = evaluation_utils.load_test_data(
        qual_model_dir, [i.name for i in model.graph.input]
    )
    check_and_run_model(
        model_path, qual_model_name, inputs, expected_outputs, iterations, device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiler",
        default=None,
        help="Which compiler produced onnx model to run. "
        "By default None, which runs all models it can find under 'model-dir'.",
    )
    parser.add_argument(
        "--model-dir",
        "--model_dir",
        type=str,
        help="Path to onnx model directory.",
        required=True,
    )
    parser.add_argument(
        "--iteration", "-i", type=int, default=2, help="Number of iterations for bench."
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--log-level", "--log_level", type=int, default=logging.INFO)
    args = parser.parse_args()

    compiler = args.compiler
    iter_ = args.iteration
    model_dir = args.model_dir
    device = args.device
    log_level = args.log_level
    logging.basicConfig(level=log_level)

    if not os.path.exists(".logs"):
        os.makedirs(".logs")

    if compiler is None:
        compiler_model_folders = [
            folder for folder in pathlib.Path(model_dir).iterdir() if folder.is_dir()
        ]
        for compiler_model_folder in compiler_model_folders:
            compiler_name = compiler_model_folder.stem
            model_name = pathlib.Path(model_dir).stem
            with open(f".logs/stderr_{model_name}_{compiler_name}.log", "w") as stderr_file:
                # Capture stderr which contains ORT logs.
                subprocess_args = [
                    "python",
                    "bench_model.py",
                    "--compiler",
                    compiler_name,
                    "--iteration",
                    str(iter_),
                    "--model-dir",
                    model_dir,
                    "--device",
                    device,
                ]
                result = subprocess.run(
                    subprocess_args,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                try:
                    result.check_returncode()
                except subprocess.CalledProcessError:
                    print(f"========== {model_name} {compiler_name} failed")
                    print(result.stderr.decode())
                    raise

                stderr_output = result.stderr.decode()
                # Analyze ORT logs.
                print(f"========== Analyzing {model_name} {compiler_name} ORT Metrics")
                analysis = analyze_ort_logs(stderr_output)
                print(json.dumps(dataclasses.asdict(analysis), indent=4))
                stderr_file.write(f"========== {model_name} {compiler_name} ORT Logs\n")
                stderr_file.write(stderr_output)
    else:
        run_model(compiler, model_dir, iter_, device)
