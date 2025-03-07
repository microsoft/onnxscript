# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This script is an e2e tool to start a model run and profile the run.

It parses the analysis produced by onnxruntime/nsys profiling and prints out the result.

Example usage:

    CUDA_VISIBLE_DEVICES="2" python nsys_profile.py --compiler torchscript --model-dir onnx_models/Speech2Text2ForCausalLM --iteration 20

This runs the torchscript model in onnx_models/Speech2Text2ForCausalLM for 20 iterations and profile the run.

The model and test data must be in the same format specified in `examples/bench_model.py`.
"""

import argparse
import datetime
import os
import subprocess

import profile_analysis


def nsys_profile_benchmark(
    compiler: str, model_dir: str, iteration: int, report_name: str, nsys_path: str
):
    subprocess.check_call(
        [
            f"{nsys_path}/nsys",
            "profile",
            "--trace=cuda,nvtx",
            f"--output=.logs/{report_name}.nsys-rep",
            "--force-overwrite=true",
            "python",
            "bench_model.py",
            "--compiler",
            compiler,
            "--model-dir",
            model_dir,
            "--iteration",
            str(iteration),
            "--device",
            "cuda",
        ]
    )


def nsys_export_benchmark(report_name: str, nsys_path: str):
    subprocess.check_call(
        [
            f"{nsys_path}/nsys",
            "export",
            f"--output=.logs/{report_name}.json",
            "--type",
            "json",
            f".logs/{report_name}.nsys-rep",
        ]
    )


def analyze_nsys_json_report(
    iteration: int, report_name: str, model_name: str, compiler: str
) -> profile_analysis.ModelProfile:
    return profile_analysis.analyze_profile_nvtx(
        f".logs/{report_name}.json", iteration, model_name, compiler
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiler",
        default=[],
        help="Which compiler produced onnx model to run. "
        "By default None, which runs all models it can find under 'model-dir'.",
        action="append",
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
    parser.add_argument(
        "--nsys-path",
        "--nsys_path",
        type=str,
        default="/usr/local/bin",
        help="Path to nsys binary. The default might not match with your cuda version.",
    )
    args = parser.parse_args()

    compilers = args.compiler
    model_dir = args.model_dir
    iteration = args.iteration
    nsys_path = args.nsys_path
    model_name = os.path.basename(model_dir)

    if not os.path.exists(".logs"):
        os.makedirs(".logs")

    reports = []
    for compiler in compilers:
        report_name = f"{model_name}_{compiler}_{iteration}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        nsys_profile_benchmark(compiler, model_dir, iteration, report_name, nsys_path)
        nsys_export_benchmark(report_name, nsys_path)
        reports.append(analyze_nsys_json_report(iteration, report_name, model_name, compiler))

    if len(compilers) > 1:
        profile_analysis.compare_node_reports(
            reports[0],
            reports[1],
        )


if __name__ == "__main__":
    main()
