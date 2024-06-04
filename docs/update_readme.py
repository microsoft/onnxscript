# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Script to update end-to-end example in README.md.

updated_readme = []
with open("README.md", encoding="utf-8") as f:
    in_stub = False
    readme = f.readlines()
    for line in readme:
        if not in_stub:
            updated_readme.append(line)
        if line == "```python update-readme\n":
            in_stub = True
            with open(
                "docs/tutorial/examples/hardmax_end_to_end.py", encoding="utf-8"
            ) as example_f:
                example_code = example_f.readlines()
                updated_readme += example_code
        if line == "```\n" and in_stub:
            updated_readme.append(line)
            in_stub = False

with open("README.md", "w", encoding="utf-8") as f:
    f.writelines(updated_readme)
