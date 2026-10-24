# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from onnxscript import GraphBuilder, build_graph


@pytest.mark.parametrize("shape", [(), (2,), (2, 2)])
@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize("scope", ["", "layer"])
def test_promote_unnamed_tensor(shape, as_tensor, scope):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 23}, name="test")
    builder = GraphBuilder(graph)
    x = builder.input("const_tensor_1", ir.DataType.FLOAT, shape)
    builder.initializer(ir.tensor(np.float32(7)), name="const_tensor_0")
    if scope:
        builder.push_module(scope)
    bias = np.full(shape, 2, dtype=np.float32)
    constant = ir.tensor(bias) if as_tensor else bias
    first = builder.op.Add(x, constant)
    result = builder.op.Add(first, constant)
    builder.add_output(result, "result")

    assert list(result.shape) == list(shape)
    assert result.dtype == ir.DataType.FLOAT
    assert len(graph.initializers) == 3
    if as_tensor:
        assert constant.name is None
    model = ir.to_proto(ir.Model(graph, ir_version=10))
    onnx.checker.check_model(model, full_check=True)
    data = np.full(shape, 3, dtype=np.float32)
    (actual,) = ReferenceEvaluator(model).run(None, {x.name: data})
    np.testing.assert_array_equal(actual, data + 2 * bias)


def test_promote_named_tensor_preserves_name():
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 23})
    builder = GraphBuilder(graph)
    x = builder.input("x", ir.DataType.FLOAT, [2])
    builder.push_module("layer")
    bias = ir.tensor(np.array([1, 2], dtype=np.float32), name="bias")
    result = builder.op.Add(x, bias)
    assert result.producer().inputs[1].name == "layer.bias"
    assert bias.name == "bias"


def test_promote_unnamed_tensor_in_subgraph():
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 23}, name="test")
    builder = GraphBuilder(graph)
    x = builder.input("const_tensor_0", ir.DataType.FLOAT, [2])
    condition = builder.input("condition", ir.DataType.BOOL, [])

    def branch(op):
        return op.Add(x, ir.tensor(np.array([1, 2], dtype=np.float32)))

    branches = [
        builder.subgraph(branch, [], [ir.Value(name="y")], name=f"branch_{i}")
        for i in range(2)
    ]
    result = builder.op.If(condition, then_branch=branches[0], else_branch=branches[1])
    builder.add_output(result, "result")
    assert all(not branch_graph.initializers for branch_graph in branches)
    assert len(graph.initializers) == 2
    model = ir.to_proto(ir.Model(graph, ir_version=10))
    onnx.checker.check_model(model, full_check=True)
    for flag in (False, True):
        (actual,) = ReferenceEvaluator(model).run(
            None, {x.name: np.array([3, 4], dtype=np.float32), condition.name: np.array(flag)}
        )
        np.testing.assert_array_equal(actual, [4, 6])


def test_promote_unnamed_tensor_avoids_enclosing_subgraph_names():
    root = GraphBuilder(ir.Graph([], [], nodes=[], opset_imports={"": 23}))
    parent = GraphBuilder(ir.Graph([], [], nodes=[], opset_imports={"": 23}), parent=root)
    x = parent.input("const_tensor_0", ir.DataType.FLOAT, [2])
    child = GraphBuilder(ir.Graph([], [], nodes=[], opset_imports={"": 23}), parent=parent)
    result = child.op.Add(x, ir.tensor(np.array([1, 2], dtype=np.float32)))
    bias = result.producer().inputs[1]
    assert bias.name != x.name
    assert root.graph.initializers[bias.name] is bias
    assert not child.graph.initializers


@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize(
    "mode", ["direct", "multiple", "late", "sequential", "nested", "nested_late"]
)
def test_standalone_build_graph_captures(as_tensor, mode):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 23}, name="test")
    builder = GraphBuilder(graph)
    x = builder.input("const_tensor_0", ir.DataType.FLOAT, [2])
    y = builder.input("const_tensor_1", ir.DataType.FLOAT, [2])
    flag = builder.input("flag", ir.DataType.BOOL, [])
    array = np.array([1, 2], dtype=np.float32)

    def branch(op):
        bias = ir.tensor(array) if as_tensor else array
        if mode == "multiple":
            result = op.Sum(bias, x, y)
        elif mode == "late":
            first = op.Identity(bias)
            result = op.Add(x, first)
        elif mode == "sequential":
            first = op.Add(x, bias)
            result = op.Add(y, first)
        elif mode == "nested_late":
            first = op.Identity(bias)

            def capture(inner_op):
                return inner_op.Add(x, first)

            result = op.If(
                flag, then_branch=make_branch(capture), else_branch=make_branch(capture)
            )
        else:
            result = op.Add(x, bias)
        if as_tensor:
            assert bias.name is None
        return result

    def make_branch(trace):
        return build_graph(trace, [], [ir.Value(name="out")], opset_imports={"": 23})

    def nested(op):
        return op.If(flag, then_branch=make_branch(branch), else_branch=make_branch(branch))

    trace = nested if mode == "nested" else branch
    result = builder.op.If(
        flag, then_branch=make_branch(trace), else_branch=make_branch(trace)
    )
    builder.add_output(result, "result")
    assert x.name == "const_tensor_0"
    assert y.name == "const_tensor_1"
    proto = ir.to_proto(ir.Model(graph, ir_version=10))
    proto = onnx.load_model_from_string(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    expected = [9, 12] if mode in {"multiple", "sequential"} else [4, 6]
    for condition in (False, True):
        (actual,) = ReferenceEvaluator(proto).run(
            None,
            {
                x.name: np.array([3, 4], dtype=np.float32),
                y.name: np.array([5, 6], dtype=np.float32),
                flag.name: np.array(condition),
            },
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("automatic_first", [False, True])
@pytest.mark.parametrize("named", [False, True])
def test_nested_captures_with_distinct_same_name_values(depth, automatic_first, named):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 23}, name="test")
    builder = GraphBuilder(graph)
    x = builder.input("const_tensor_0", ir.DataType.FLOAT, [2])
    flag = builder.input("flag", ir.DataType.BOOL, [])

    def branch(op):
        data = np.array([1, 2], dtype=np.float32)
        data.setflags(write=False)
        tensor = ir.tensor(data, name="safe_bias" if named else None)
        first = op.Identity(tensor)
        initializer = first.producer().inputs[0]
        old_name = initializer.name

        def nested(level):
            def trace(inner_op):
                if level == 0:
                    args = [initializer, x] if automatic_first else [x, initializer]
                    return inner_op.Add(*args)
                return inner_op.If(
                    flag, then_branch=nested(level - 1), else_branch=nested(level - 1)
                )

            return build_graph(
                trace, [], [ir.Value(name=f"nested_{level}")], opset_imports={"": 23}
            )

        result = op.If(flag, then_branch=nested(depth - 1), else_branch=nested(depth - 1))
        assert initializer is first.producer().inputs[0]
        assert initializer is not x
        assert op.builder.graph.initializers[initializer.name] is initializer
        assert initializer.graph is op.builder.graph
        assert initializer.const_value is tensor
        assert tensor.name == ("safe_bias" if named else None)
        assert not data.flags.writeable
        np.testing.assert_array_equal(data, [1, 2])
        if named:
            assert initializer.name == old_name == "safe_bias"
        elif initializer.name != old_name:
            assert old_name not in op.builder.graph.initializers
        return result

    branches = [
        build_graph(branch, [], [ir.Value(name="out")], opset_imports={"": 23})
        for _ in range(2)
    ]
    builder.add_output(
        builder.op.If(flag, then_branch=branches[0], else_branch=branches[1]), "result"
    )
    assert x.name == "const_tensor_0"
    proto = ir.to_proto(ir.Model(graph, ir_version=10))
    proto = onnx.load_model_from_string(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    for condition in (False, True):
        (actual,) = ReferenceEvaluator(proto).run(
            None, {x.name: np.array([3, 4], dtype=np.float32), "flag": np.array(condition)}
        )
        np.testing.assert_array_equal(actual, [4, 6])


@pytest.mark.parametrize("named", [False, True])
def test_pending_manual_output_avoids_automatic_initializer_name(named):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 23}, name="test")
    builder = GraphBuilder(graph)
    x = builder.input("x", ir.DataType.FLOAT, [2])
    data = np.array([1, 2], dtype=np.float32)
    data.setflags(write=False)
    tensor = ir.tensor(data, name="safe_bias" if named else None)
    first = builder.op.Add(x, tensor)
    initializer = first.producer().inputs[1]
    output = ir.val("const_tensor_0", ir.DataType.FLOAT, [2])
    builder.add_node(ir.Node("", "Identity", [first], outputs=[output], version=23))
    builder.add_output(output, "const_tensor_0")
    assert output.name == "const_tensor_0"
    assert graph.initializers[initializer.name] is initializer
    assert initializer.graph is graph
    assert first.producer().inputs[1] is initializer
    assert initializer.const_value is tensor
    assert tensor.name == ("safe_bias" if named else None)
    assert not data.flags.writeable
    np.testing.assert_array_equal(data, [1, 2])
    if named:
        assert initializer.name == "safe_bias"
    proto = ir.to_proto(ir.Model(graph, ir_version=10))
    proto = onnx.load_model_from_string(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    (actual,) = ReferenceEvaluator(proto).run(None, {"x": np.array([3, 4], dtype=np.float32)})
    np.testing.assert_array_equal(actual, [4, 6])
