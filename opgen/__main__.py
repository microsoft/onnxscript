# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import shutil
import subprocess
import textwrap
from pathlib import Path

from opgen.onnx_opset_builder import (
    OpsetId,
    OpsetsBuilder,
    format_opsetid,
    parse_opsetid,
)

MIN_REQUIRED_ONNX_OPSET_VERSION = 14

self_dir = Path(__file__).parent
repo_root = self_dir.parent

module_base_names = ["onnxscript", "onnx_opset"]
opsets_path = repo_root.joinpath(*module_base_names)

argparser = argparse.ArgumentParser("opgen")
argparser.add_argument(
    "-x",
    "--exclude",
    action="append",
    metavar="OPSET",
    dest="exclude_opsets",
    help="exclude an opset from generation; example: -x 19 -x ai.onnx.ml/3",
)
argparser.add_argument(
    "-i",
    "--include-only",
    action="append",
    metavar="OPSET",
    dest="include_opsets",
    help="include only these opsets; example: -i 19",
)
argparser.add_argument(
    "--min-opset-version",
    help="the minimum supported ONNX opset version",
    default=MIN_REQUIRED_ONNX_OPSET_VERSION,
    action="store",
    type=int,
)
args = argparser.parse_args()

try:
    shutil.rmtree(opsets_path)
except FileNotFoundError:
    pass  # if base_path doesn't exist, that's great

# need to generate a blank onnx_opset module since
# onnxscript/__init__.py will import it (and we deleted it above);
# it will be overridden with correct code as part of the generation
# below.

opsets_path.mkdir(parents=True)
with opsets_path.joinpath("__init__.py").open("w", encoding="utf-8"):
    pass


builder = OpsetsBuilder(
    module_base_name=".".join(module_base_names),
    min_default_opset_version=args.min_opset_version,
    include_opsets={parse_opsetid(opsetid) for opsetid in args.include_opsets or []},
    exclude_opsets={parse_opsetid(opsetid) for opsetid in args.exclude_opsets or []},
)
result = builder.build()
paths = result.write(repo_root)
subprocess.check_call(["black", "--quiet", *paths])
subprocess.check_call(["isort", "--quiet", *paths])

print(f"🎉 Generated Ops: {result.all_ops_count}")
print(f"   Minimum Opset Version: {args.min_opset_version}")
print()


def print_opsets(label: str, opsets: set[OpsetId]):
    if any(opsets):
        print(label)
        summary = ", ".join([format_opsetid(i) for i in sorted(opsets)])
        print("\n".join(textwrap.wrap(summary, initial_indent="   ", subsequent_indent="   ")))
        print()


print_opsets("🟢 Included Opsets:", result.included_opsets)
print_opsets("🔴 Excluded Opsets:", result.excluded_opsets)

if any(result.unsupported_ops):
    print("🟠 Unsupported Ops:")
    for key, unsupported_ops in sorted(result.unsupported_ops.items()):
        print(f"   reason: {key}:")
        for unsupported_op in unsupported_ops:
            print(f"     - {unsupported_op.op}")
            print(f"       {unsupported_op.op.docuri}")
