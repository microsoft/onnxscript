# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import subprocess
from os import makedirs
from pathlib import Path
from shutil import rmtree

from opgen.onnx_opset_builder import OpsetsBuilder

MIN_REQUIRED_ONNX_OPSET_VERSION = 14

self_dir = Path(__file__).parent
repo_root = self_dir.parent

module_base_names = ["onnxscript", "onnx_opset"]
opsets_path = repo_root.joinpath(*module_base_names)

try:
    rmtree(opsets_path)
except FileNotFoundError:
    pass  # if base_path doesn't exist, that's great

# need to generate a blank onnx_opset module since
# onnxscript/__init__.py will import it (and we deleted it above);
# it will be overridden with correct code as part of the generation
# below.
makedirs(opsets_path)
with open(opsets_path.joinpath("__init__.py"), "w", encoding="utf-8"):
    pass

builder = OpsetsBuilder(".".join(module_base_names), MIN_REQUIRED_ONNX_OPSET_VERSION)
paths = builder.write(repo_root)
subprocess.check_call(["black", "--quiet", *paths])
subprocess.check_call(["isort", "--quiet", *paths])

print(f"Generated Ops: {builder.all_ops_count}")

if len(builder.unsupported_ops) > 0:
    print("Unsupported Ops:")
    for key, unsupported_ops in sorted(builder.unsupported_ops.items()):
        print(f"  reason: {key}:")
        for unsupported_op in unsupported_ops:
            print(f"    - {unsupported_op.op}")
            print(f"      {unsupported_op.op.docuri}")
