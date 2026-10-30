# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator
from onnxruntime import InferenceSession

from onnxscript import version_converter
from onnxscript.version_converter import _version_converter


@pytest.mark.parametrize("target", range(21, 29))
@pytest.mark.parametrize("dtype", ["float", "float16", "double", "bfloat16"])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_group_normalization_preserves_dynamic_shapes_precision_and_epsilon(
    target, dtype, groups
):
    model = ir.from_onnx_text(
        f"""
        <ir_version: 10, opset_import: ["" : 20]>
        graph ({dtype}[N,C,H] input, {dtype}[{groups}] scale, {dtype}[{groups}] bias)
            => ({dtype}[N,C,H] y) {{
            x = Identity (input)
            y = GroupNormalization <num_groups={groups}, epsilon=0.25> (x, scale, bias)
        }}
        """
    )
    numpy_dtype = ir.DataType[{"float": "FLOAT"}.get(dtype, dtype.upper())].numpy()
    if dtype == "bfloat16":
        try:
            onnx.helper.np_dtype_to_tensor_dtype(np.dtype(numpy_dtype))
        except ValueError:
            pytest.skip(
                "Installed ONNX reference evaluator does not support ml_dtypes bfloat16."
            )
    feeds = {
        "input": np.linspace(-1, 1, 24).reshape(2, 4, 3).astype(numpy_dtype),
        "scale": np.arange(1, groups + 1).astype(numpy_dtype),
        "bias": np.arange(groups).astype(numpy_dtype),
    }
    assert model.graph.node(1).inputs[0].shape is None
    expected = ReferenceEvaluator(ir.to_proto(model)).run(None, feeds)
    version_converter.convert_version(model, target)
    assert all(node.op_type != "GroupNormalization" for node in model.graph)
    assert all(node.version == target for node in model.graph)
    converted = ir.to_proto(model)
    if target <= onnx.defs.onnx_opset_version():
        onnx.checker.check_model(converted, full_check=True)
    actual = ReferenceEvaluator(converted).run(None, feeds)
    np.testing.assert_array_equal(actual[0], expected[0])


@pytest.mark.parametrize("mode", [None, "bilinear", "bicubic", "nearest"])
@pytest.mark.parametrize("target", [20, 21, 28])
def test_grid_sample_preserves_modes_and_attributes(mode, target):
    attributes = 'align_corners=1, padding_mode="reflection"'
    if mode is not None:
        attributes += f', mode="{mode}"'
    model = ir.from_onnx_text(
        f"""
        <ir_version: 10, opset_import: ["" : 19]>
        graph (float[1,1,3,3] x, float[1,2,2,2] grid) => (float[1,1,2,2] y) {{
            y = GridSample <{attributes}> (x, grid)
        }}
        """
    )
    feeds = {
        "x": np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3),
        "grid": np.array([[[[-1, 0], [0.2, 0.3]], [[1, 1], [1.5, -1.5]]]], np.float32),
    }
    expected = InferenceSession(
        ir.to_proto(model).SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    version_converter.convert_version(model, target)
    converted = ir.to_proto(model)
    if target <= onnx.defs.onnx_opset_version():
        onnx.checker.check_model(converted, full_check=True)
    actual = ReferenceEvaluator(converted).run(None, feeds)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)


def test_local_function_uses_its_own_opset_and_preserves_axis_reference():
    model = ir.from_onnx_text(
        """
        <ir_version: 10, opset_import: ["" : 25, "local" : 1]>
        graph (float[2,3,4,2] x) => (float[2,3,4,2] y) {
            y = local.transform <axis=1> (x)
        }
        <domain: "local", opset_import: ["" : 19]>
        transform <axis> (x) => (y) {
            y = DFT <axis: int = @axis> (x)
        }
        """
    )
    _version_converter.convert_version(model, 28)
    function = model.functions[("local", "transform", "")]
    assert function.opset_imports[""] == 28
    constant, dft = function
    assert constant.op_type == "Constant"
    assert constant.attributes["value_int"].is_ref()
    assert constant.attributes["value_int"].ref_attr_name == "axis"
    assert len(dft.inputs) == 3
    assert constant.version == dft.version == 28
    onnx.checker.check_model(ir.to_proto(model))


@pytest.mark.parametrize("mode", ["bilinear", "bicubic", "nearest"])
def test_public_pass_resolves_grid_sample_mode_references(mode):
    model = ir.from_onnx_text(
        f"""
        <ir_version: 10, opset_import: ["" : 20, "local" : 1]>
        graph (float[1,1,3,3] x, float[1,2,2,2] grid) => (float[1,1,2,2] y) {{
            y = local.sample <mode="{mode}"> (x, grid)
        }}
        <domain: "local", opset_import: ["" : 19]>
        sample <mode> (x, grid) => (y) {{
            y = GridSample <mode: string = @mode, align_corners=1> (x, grid)
        }}
        """
    )
    feeds = {
        "x": np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3),
        "grid": np.array([[[[-1, 0], [0.2, 0.3]], [[1, 1], [1.5, -1.5]]]], np.float32),
    }
    original = ir.to_proto(model)
    # ORT inlines against the model imports rather than the function imports.
    # The root graph has no standard operators, so evaluate at the function's opset.
    next(opset for opset in original.opset_import if opset.domain == "").version = 19
    expected = InferenceSession(
        original.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    version_converter.convert_version(model, 28)
    actual = ReferenceEvaluator(ir.to_proto(model)).run(None, feeds)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("root_version", [18, 25, 28])
def test_group_normalization_function_preserves_reference_attributes(inline, root_version):
    model = ir.from_onnx_text(
        """
        <ir_version: 10, opset_import: ["" : 25, "local" : 1]>
        graph (float[1,4,2] x, float[2] scale, float[2] bias) => (float[1,4,2] y) {
            y = local.normalize <epsilon=0.25, groups=2> (x, scale, bias)
        }
        <domain: "local", opset_import: ["" : 20]>
        normalize <epsilon,groups> (x, scale, bias) => (y) {
            y = GroupNormalization <epsilon: float = @epsilon, num_groups: int = @groups>
                (x, scale, bias)
        }
        """
    )
    model.opset_imports[""] = root_version
    feeds = {
        "x": np.arange(8, dtype=np.float32).reshape(1, 4, 2),
        "scale": np.array([2, 3], np.float32),
        "bias": np.array([1, -1], np.float32),
    }
    grouped = feeds["x"].reshape(1, 2, -1)
    mean = grouped.mean(axis=-1, keepdims=True)
    variance = (grouped * grouped).mean(axis=-1, keepdims=True) - mean * mean
    normalized = (grouped - mean) / np.sqrt(variance + np.float32(0.25))
    expected = (
        normalized * feeds["scale"].reshape(1, 2, 1) + feeds["bias"].reshape(1, 2, 1)
    ).reshape(1, 4, 2)
    if inline:
        version_converter.convert_version(model, 28)
    else:
        _version_converter.convert_version(model, 28)
    assert bool(model.functions) != inline
    actual = ReferenceEvaluator(ir.to_proto(model)).run(None, feeds)
    np.testing.assert_allclose(actual[0], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("target", [20, 21, 28])
@pytest.mark.parametrize("axis", [None, 1, -2])
@pytest.mark.parametrize("length", [None, 2])
@pytest.mark.parametrize("inverse", [0, 1])
def test_dft_preserves_axis_length_and_inverse(target, axis, length, inverse):
    attributes = f"inverse={inverse}"
    if axis is not None:
        attributes += f", axis={axis}"
    inputs = "float[2,3,4,2] x"
    args = "x"
    if length is not None:
        inputs += ", int64[] length"
        args += ", length"
    model = ir.from_onnx_text(
        f"""
        <ir_version: 10, opset_import: ["" : 19]>
        graph ({inputs}) => (float[N,C,H,2] y) {{
            y = DFT <{attributes}> ({args})
        }}
        """
    )
    feeds = {"x": np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2) / 16}
    if length is not None:
        feeds["length"] = np.array(length, dtype=np.int64)
    expected = ReferenceEvaluator(ir.to_proto(model)).run(None, feeds)
    version_converter.convert_version(model, target)
    actual = ReferenceEvaluator(ir.to_proto(model)).run(None, feeds)
    np.testing.assert_array_equal(actual[0], expected[0])


@pytest.mark.parametrize("attr_type", [ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS])
def test_custom_operator_subgraphs_are_converted_even_at_target_opset(attr_type):
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3, 4, 2]))
    dft = ir.Node("", "DFT", [x], num_outputs=1, version=19)
    branch = ir.Graph([], [dft.outputs[0]], nodes=[dft], opset_imports={"": 19}, name="body")
    attr_value = branch if attr_type == ir.AttributeType.GRAPH else [branch]
    container = ir.Node(
        "custom",
        "Container",
        [x],
        [ir.Attr("body", attr_type, attr_value)],
        num_outputs=1,
        version=7,
    )
    model = ir.Model(
        ir.Graph(
            [x], [container.outputs[0]], nodes=[container], opset_imports={"": 28, "custom": 7}
        ),
        ir_version=10,
    )
    version_converter.convert_version(model, 28)
    assert container.domain == "custom"
    assert container.version == 7
    assert branch.opset_imports[""] == 28
    assert [node.op_type for node in branch] == ["Constant", "DFT"]
    assert all(node.version == 28 for node in branch)


def test_standard_domain_alias_is_normalized():
    model = ir.from_onnx_text(
        """
        <ir_version: 10, opset_import: ["ai.onnx" : 19]>
        graph (float[2,3,4,2] x) => (float[2,3,4,2] y) {
            y = ai.onnx.DFT (x)
        }
        """
    )
    assert _version_converter.version_supported(model, 28)
    _version_converter.convert_version(model, 28)
    assert model.opset_imports == {"": 28}
    assert [node.op_type for node in model.graph] == ["Constant", "DFT"]
    assert all(node.domain == "" and node.version == 28 for node in model.graph)
    onnx.checker.check_model(ir.to_proto(model))


@pytest.mark.parametrize("source,target", [(17, 18), (29, 28), (28, 27)])
def test_native_converter_rejects_unsupported_source_versions(source, target):
    model = ir.from_onnx_text(
        f"""
        <ir_version: 10, opset_import: ["" : {source}]>
        graph (float[2] x) => (float[2] y) {{ y = Identity (x) }}
        """
    )
    assert not _version_converter.version_supported(model, target)
    with pytest.raises(_version_converter.VersionConverterError, match="fallback=True"):
        version_converter.convert_version(model, target)
    assert model.opset_imports[""] == source


def test_invalid_adapter_attribute_raises_without_relabeling_model():
    model = ir.from_onnx_text(
        """
        <ir_version: 10, opset_import: ["" : 19]>
        graph (float[2,3,4,2] x) => (float[2,3,4,2] y) {
            y = DFT <axis: float = 1.0> (x)
        }
        """
    )
    with pytest.raises(_version_converter.VersionConverterError, match="attribute 'axis'"):
        version_converter.convert_version(model, 28)
    assert model.opset_imports[""] == 19


def test_noop_conversion_does_not_report_modification():
    model = ir.from_onnx_text(
        """
        <ir_version: 10, opset_import: ["" : 28]>
        graph (float[2] x) => (float[2] y) { y = Identity (x) }
        """
    )
    assert not version_converter.ConvertVersionPass(28)(model).modified


def test_native_converter_records_explicit_node_versions():
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
    identity = ir.Node("", "Identity", [x], num_outputs=1, version=18)
    model = ir.Model(ir.Graph([x], [identity.outputs[0]], nodes=[identity]), ir_version=10)
    _version_converter.convert_version(model, 28)
    assert model.opset_imports[""] == 28
    assert identity.version == 28


def test_fallback_still_supports_older_source_opsets():
    model = ir.from_onnx_text(
        """
        <ir_version: 10, opset_import: ["" : 17]>
        graph (float[2] x) => (float[2] y) { y = Identity (x) }
        """
    )
    version_converter.convert_version(model, 18, fallback=True)
    assert model.opset_imports[""] == 18
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
