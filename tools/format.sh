#!/usr/bin/env bash

cd "$(git rev-parse --show-toplevel)"

echo -e "\nblack reformatting..."
black . --color
echo -e "\nblack done!"

echo -e "\nisort reformatting..."
isort . --color
echo -e "\nisort done!"
