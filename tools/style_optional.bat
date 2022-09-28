:: SPDX-License-Identifier: Apache-2.0

@echo off 
:: This script helps Windows user to check formatting 
::before submitting the PR

for /f %%i in ('git rev-parse --show-toplevel')   do set root_path=%%i
ECHO "Git Root PATH: %root_path%"
CD /D %root_path%

ECHO "\n::group:: ===> check pylint"
pylint onnxscript
ECHO "::endgroup::"

ECHO "\n::group:: ===> check mypy"
mypy onnxscript --config-file pyproject.toml
ECHO "::endgroup::"

PAUSE
