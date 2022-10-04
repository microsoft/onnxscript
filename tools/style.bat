:: SPDX-License-Identifier: Apache-2.0

@echo off
:: This script helps Windows user to check formatting
::before submitting the PR

for /f %%i in ('git rev-parse --show-toplevel')   do set root_path=%%i
ECHO "Git Root PATH: %root_path%"
CD /D %root_path%

ECHO "\n::group:: ===> check flake8..."
flake8 --docstring-convention google onnxscript/
ECHO "::endgroup::"

ECHO "\n::group:: ===> check isort..."
isort . --color --diff --check
ECHO "::endgroup::"

ECHO "\n::group:: ===> check black format..."
black . --color --diff --check
ECHO "::endgroup::"

PAUSE