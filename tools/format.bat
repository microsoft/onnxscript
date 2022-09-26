:: SPDX-License-Identifier: Apache-2.0

@echo off 
:: This script helps Windows user to format the code 
::before submitting the PR

for /f %%i in ('git rev-parse --show-toplevel')   do set root_path=%%i
ECHO "Git Root PATH: %root_path%"
CD /D %root_path%

ECHO "\nblack reformatting..."
black . --color
ECHO "\nblack done!"

ECHO "\nisort reformatting..."
isort . --color
ECHO "\nisort done!"

PAUSE