#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0


set +o errexit
set -o nounset


cd "$(git rev-parse --show-toplevel)"

err=0
trap 'err=1' ERR

echo -e "\n::group:: ===> check flake8..."
flake8 onnxscript --config .flake8
echo -e "::endgroup::"

echo -e "\n::group:: ===> check isort..."
isort onnxscript --color --diff --check
echo -e "::endgroup::"

echo -e "\n::group:: ===> check black format..."
black onnxscript --color --diff --check
echo -e "::endgroup::"

echo -e "\n::group:: ===> check mypy"
mypy onnxscript --config-file pyproject.toml
echo -e "::endgroup::"

git diff --exit-code

test $err = 0 # Return non-zero if any command failed