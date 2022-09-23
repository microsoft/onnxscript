#!/usr/bin/env bash

cd "$(git rev-parse --show-toplevel)"

echo -e "\nblack reformatting..."
black onnxscript --color
echo -e "\nblack done!"

echo -e "\nisort reformatting..."
isort onnxscript --color
echo -e "\nisort done!"