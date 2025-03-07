# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""NOTE: Put all metadata in pyproject.toml. Do not include complex logic in setup.py."""

import datetime
import os
import pathlib
import subprocess

import setuptools

# Logic for computing the development version number.
ROOT_DIR = pathlib.Path(__file__).parent
VERSION_FILE = ROOT_DIR / "VERSION"
version = VERSION_FILE.read_text().strip()

project_urls = {
    "Homepage": "https://onnxscript.ai/",
    "Repository": "https://github.com/microsoft/onnxscript",
}
if os.environ.get("ONNX_SCRIPT_RELEASE") != "1":
    date = datetime.date.today().strftime("%Y%m%d")
    version = f"{version}.dev{date}"

    commit_hash_cmd = subprocess.run(
        ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, check=False
    )
    if commit_hash_cmd.returncode == 0:
        project_urls["Commit"] = (
            f"https://github.com/microsoft/onnxscript/tree/{commit_hash_cmd.stdout.decode('utf-8').strip()}"
        )

# NOTE: Do not include other metadata in setup.py. Put it in pyproject.toml.
setuptools.setup(version=version, project_urls=project_urls)
