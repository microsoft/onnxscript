# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Testing GQA fusion."""

from onnxscript import script


@script()
def _gqa_prompt_script(query, key, value):
    pass
