# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for the placeholder-domain ('this') export warning.

Covers the warning contract from https://github.com/microsoft/onnxscript/issues/3044:
exporting a model that bundles local functions still carrying the placeholder
domain "this" warns, while models that bundle no local functions at all, or only
explicitly-domained ones, do not.
"""
import unittest
import warnings

from onnxscript import script
from onnxscript.onnx_opset import opset17 as op
from onnxscript.onnx_types import FLOAT


@script()
def _default_domain_helper(x: FLOAT[4]) -> FLOAT[4]:
    return op.Add(x, x)


@script(op)
def _model_with_default_domain_helper(x: FLOAT[4]) -> FLOAT[4]:
    return _default_domain_helper(x)


@script(op)
def _explicit_domain_helper(x: FLOAT[4]) -> FLOAT[4]:
    return op.Add(x, x)


@script(op)
def _model_with_explicit_domain_helper(x: FLOAT[4]) -> FLOAT[4]:
    return _explicit_domain_helper(x)


@script()
def _plain_default_domain_model(x: FLOAT[4]) -> FLOAT[4]:
    return op.Add(x, x)


def _placeholder_domain_warnings(func):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        func.to_model_proto()
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning)
        and "placeholder domain 'this'" in str(w.message)
    ]


class ThisDomainWarningTest(unittest.TestCase):
    def test_warns_when_bundling_default_domain_helper(self):
        found = _placeholder_domain_warnings(_model_with_default_domain_helper)
        self.assertTrue(found, "expected a placeholder-domain warning on export")

    def test_no_warning_when_bundled_functions_use_explicit_domains(self):
        found = _placeholder_domain_warnings(_model_with_explicit_domain_helper)
        self.assertFalse(found, f"unexpected placeholder-domain warnings: {found}")

    def test_no_warning_when_no_local_functions_bundled(self):
        found = _placeholder_domain_warnings(_plain_default_domain_model)
        self.assertFalse(found, f"unexpected placeholder-domain warnings: {found}")


if __name__ == "__main__":
    unittest.main()
