#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0


set +o errexit
set -o nounset


cd "$(git rev-parse --show-toplevel)"

err=0
trap 'err=1' ERR

echo -e "\n::group:: ===> check flake8..."
flake8 onnxscript
echo -e "::endgroup::"

echo -e "\n::group:: ===> check isort..."
isort . --color --diff --check
echo -e "::endgroup::"

echo -e "\n::group:: ===> check black format..."
black . --color --diff --check
echo -e "::endgroup::"

git diff --exit-code

test $err = 0 # Return non-zero if any command failed
