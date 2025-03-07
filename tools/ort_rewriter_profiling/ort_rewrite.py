# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Runs onnxruntime rewriter to optimize on the given onnx model.

Input:
    <model-dir>/<model>/<compiler>/<model>_<compiler>.onnx

Output:
    <model-dir>/<model>/<compiler>_<rewritten_name>/<model>_<compiler>_<rewritten_name>.onnx
"""

import argparse
import contextlib
import logging
import os
import shutil

import onnx

from onnxscript.rewriter import onnxruntime as ort_rewriter

logger = logging.getLogger(__name__)


def ort_rewrite(model_name: str, compiler_name: str, model_dir: str):
    old_model_folder = f"{model_dir}/{model_name}/{compiler_name}"
    old_model_name = f"{model_name}_{compiler_name}"

    post_process_name = "ort_rewritten"
    new_model_folder = f"{model_dir}/{model_name}/{compiler_name}_{post_process_name}"
    new_model_name = f"{old_model_name}_{post_process_name}"

    model = onnx.load(f"{old_model_folder}/{old_model_name}.onnx", load_external_data=True)
    ort_rewritten_model = ort_rewriter.rewrite(model)

    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(new_model_folder)

    if not os.path.exists(new_model_folder):
        os.mkdir(new_model_folder)
        shutil.copytree(
            f"{old_model_folder}/test_data_set_0",
            f"{new_model_folder}/test_data_set_0",
        )

    logger.debug("Model size: %s", ort_rewritten_model.ByteSize())
    onnx.save(
        ort_rewritten_model,
        f"{new_model_folder}/{new_model_name}.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--compiler", type=str, default="dynamo")
    parser.add_argument("--model-dir", "--model_dir", type=str, default="./onnx_models")
    parser.add_argument("--log-level", "--log_level", type=int, default=logging.WARNING)

    args = parser.parse_args()

    model_name = args.model
    compiler_name = args.compiler
    model_dir = args.model_dir

    log_level = args.log_level
    logging.basicConfig(level=log_level)

    ort_rewrite(model_name, compiler_name, model_dir)


if __name__ == "__main__":
    main()
