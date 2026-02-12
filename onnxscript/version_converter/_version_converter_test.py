# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx.defs
import pytest

from onnxscript import ir, version_converter


class AdapterCoverageTest(unittest.TestCase):
    def get_all_unique_schema_versions(self) -> dict[str, list]:
        """Collect all unique versions of ONNX standard domain ops"""
        op_version_dict = {}
        all_schemas = onnx.defs.get_all_schemas_with_history()
        for schema in all_schemas:
            if schema.name not in op_version_dict:
                op_version_dict[schema.name] = [schema.since_version]
            else:
                if schema.since_version not in op_version_dict[schema.name]:
                    op_version_dict[schema.name].append(schema.since_version)
        return op_version_dict

    # TODO(shubhambhokare1) : Using existing onnx testing suite to verify operator adapter's functionality
    def test_upstream_coverage(self):
        op_version_dict = self.get_all_unique_schema_versions()
        op_upgrades = []
        for op_type in op_version_dict:  # pylint: disable=consider-using-dict-items
            for opset_version in op_version_dict[op_type]:
                op_upgrades.append((op_type, opset_version))

        adapter_list = version_converter._version_converter.registry.op_adapters  # pylint: disable=protected-access
        for adapter_sig in adapter_list:
            adapter_info = list(adapter_sig)
            domain, name, upgrade_version = (
                adapter_info[0],
                adapter_info[1],
                adapter_info[2] + 1,
            )
            self.assertEqual(domain, "")
            self.assertIn((name, upgrade_version), op_upgrades)

    @pytest.mark.xfail(reason="TODO: Cleanup error status API.")
    def test_version_convert_no_source_version(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "local" : 1]>
            agraph (float[4, 512, 512] input_x, float[4, 1024, 1024] input_y) => (float[4, 1024, 1024] output)
            {
                shape_a = Constant<value: tensor = int64[4] {1, 4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[4] {1, 4, 1024, 1024}>()
                reshape_y = Reshape (input_x, shape_b)
                gridsample = GridSample <mode = "bilinear"> (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[3] {4, 1024, 1024}>()
                output = Reshape (gridsample, shape_c)
            }
        """
        )
        self.assertEqual(model.graph.node(4).op_type, "GridSample")
        self.assertEqual(model.graph.node(4).attributes["mode"].value, "bilinear")

        target_version = 20
        version_converter.convert_version(model, target_version=target_version)


class VersionConverter18to17Test(unittest.TestCase):
    @pytest.mark.xfail(strict=True, reason="Version downgrade not yet supported.")
    def test_version_convert_compatible(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[1, 4, 512, 512] input_x, float[1, 4, 512, 64] input_y) => (float[1, 4, 512, 64] output)
            {
                shape_a = Constant<value: tensor = int64[3] {4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[3] {4, 512, 64}>()
                reshape_y = Reshape (input_y, shape_b)
                matmul = MatMul (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[4] {1, 4, 512, 64}>()
                output = Reshape (matmul, shape_c)
            }
        """
        )
        target_version = 17
        version_converter.convert_version(model, target_version=target_version)


class VersionConverter18to19Test(unittest.TestCase):
    def test_version_convert_compatible(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[1, 4, 512, 512] input_x, float[1, 4, 512, 64] input_y) => (float[1, 4, 512, 64] output)
            {
                shape_a = Constant<value: tensor = int64[3] {4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[3] {4, 512, 64}>()
                reshape_y = Reshape (input_y, shape_b)
                matmul = MatMul (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[4] {1, 4, 512, 64}>()
                output = Reshape (matmul, shape_c)
            }
        """
        )
        target_version = 19
        version_converter.convert_version(model, target_version=target_version)
        self.assertEqual(model.opset_imports[""], target_version)

        self.assertEqual(model.graph.node(0).op_type, "Constant")
        self.assertEqual(model.graph.node(0).version, 19)
        self.assertEqual(model.graph.node(1).op_type, "Reshape")
        self.assertEqual(model.graph.node(1).version, 19)
        self.assertEqual(model.graph.node(4).op_type, "MatMul")
        self.assertEqual(model.graph.node(4).version, 19)


class VersionConverter19to20Test(unittest.TestCase):
    def test_version_convert_compatible(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[4, 512, 512] input_x) => (float[4, 257, 64, 2] output)
            {
                shape_a = Constant<value: tensor = int64[5] {1, 4, 512, 512, 1}>()
                reshape_x = Reshape (input_x, shape_a)
                dft = DFT <axis = 2, onesided = 1> (reshape_x)
                shape_c = Constant<value: tensor = int64[4] {4, 257, 64, 2}>()
                output = Reshape (dft, shape_c)
            }
        """
        )
        target_version = 20
        version_converter.convert_version(model, target_version=target_version)
        self.assertEqual(model.opset_imports[""], target_version)

        self.assertEqual(model.graph.node(0).op_type, "Constant")
        self.assertEqual(model.graph.node(0).version, 20)
        self.assertEqual(model.graph.node(1).op_type, "Reshape")
        self.assertEqual(model.graph.node(1).version, 20)
        self.assertEqual(model.graph.node(2).op_type, "Constant")
        self.assertEqual(model.graph.node(3).version, 20)
        self.assertEqual(model.graph.node(3).op_type, "DFT")
        self.assertEqual(model.graph.node(3).version, 20)
        self.assertEqual(len(model.graph.node(3).inputs), 3)

    def test_version_convert_gridsample_linear(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[4, 512, 512] input_x, float[4, 1024, 1024] input_y) => (float[4, 1024, 1024] output)
            {
                shape_a = Constant<value: tensor = int64[5] {1, 4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[5] {1, 4, 1024, 1024}>()
                reshape_y = Reshape (input_x, shape_b)
                gridsample = GridSample <mode = "bilinear"> (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[4] {4, 1024, 1024}>()
                output = Reshape (gridsample, shape_c)
            }
        """
        )
        self.assertEqual(model.graph.node(4).op_type, "GridSample")
        self.assertEqual(model.graph.node(4).attributes["mode"].value, "bilinear")

        target_version = 20
        version_converter.convert_version(model, target_version=target_version)
        self.assertEqual(model.opset_imports[""], target_version)

        self.assertEqual(model.graph.node(0).op_type, "Constant")
        self.assertEqual(model.graph.node(0).version, 20)
        self.assertEqual(model.graph.node(1).op_type, "Reshape")
        self.assertEqual(model.graph.node(1).version, 20)
        self.assertEqual(model.graph.node(4).op_type, "GridSample")
        self.assertEqual(model.graph.node(4).version, 20)
        self.assertEqual(model.graph.node(4).attributes["mode"].value, "linear")

    def test_version_convert_gridsample_cubic(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[4, 512, 512] input_x, float[4, 1024, 1024] input_y) => (float[4, 1024, 1024] output)
            {
                shape_a = Constant<value: tensor = int64[5] {1, 4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[5] {1, 4, 1024, 1024}>()
                reshape_y = Reshape (input_x, shape_b)
                gridsample = GridSample <mode = "bicubic"> (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[4] {4, 1024, 1024}>()
                output = Reshape (gridsample, shape_c)
            }
        """
        )
        self.assertEqual(model.graph.node(4).op_type, "GridSample")
        self.assertEqual(model.graph.node(4).attributes["mode"].value, "bicubic")

        target_version = 20
        version_converter.convert_version(model, target_version=target_version)
        self.assertEqual(model.opset_imports[""], target_version)

        self.assertEqual(model.graph.node(0).op_type, "Constant")
        self.assertEqual(model.graph.node(0).version, 20)
        self.assertEqual(model.graph.node(1).op_type, "Reshape")
        self.assertEqual(model.graph.node(1).version, 20)
        self.assertEqual(model.graph.node(4).op_type, "GridSample")
        self.assertEqual(model.graph.node(4).version, 20)
        self.assertEqual(model.graph.node(4).attributes["mode"].value, "cubic")

    def test_version_convert_function_nodes(self):
        """Test that version converter processes nodes inside model functions."""
        model = ir.from_onnx_text(
            """
            <ir_version: 8, opset_import: [ "" : 18, "pkg.custom": 1]>
            agraph (float[4, 512, 512] input_x) => (float[4, 257, 64, 2] output)
            {
                output = pkg.custom.dft_func (input_x)
            }

            <domain: "pkg.custom", opset_import: [ "" : 18]>
            dft_func (x) => (result) {
                shape_a = Constant<value: tensor = int64[5] {1, 4, 512, 512, 1}>()
                reshape_x = Reshape (x, shape_a)
                dft = DFT <axis = 2, onesided = 1> (reshape_x)
                shape_c = Constant<value: tensor = int64[4] {4, 257, 64, 2}>()
                result = Reshape (dft, shape_c)
            }
        """
        )
        # Verify the function exists with correct initial state
        self.assertEqual(len(model.functions), 1)
        func = model.functions[("pkg.custom", "dft_func", "")]
        self.assertEqual(len(func), 5)  # 5 nodes in the function

        target_version = 20
        version_converter.convert_version(model, target_version=target_version)
        self.assertEqual(model.opset_imports[""], target_version)

        # Verify that nodes inside the function were version-converted
        func = model.functions[("pkg.custom", "dft_func", "")]
        self.assertEqual(func[0].op_type, "Constant")
        self.assertEqual(func[0].version, 20)
        self.assertEqual(func[1].op_type, "Reshape")
        self.assertEqual(func[1].version, 20)
        # After DFT adapter, a new Constant node is inserted for dft_length
        self.assertEqual(func[2].op_type, "Constant")
        self.assertEqual(func[2].version, 20)
        self.assertEqual(func[3].op_type, "DFT")
        self.assertEqual(func[3].version, 20)
        self.assertEqual(len(func[3].inputs), 3)  # DFT 19->20 adds dft_length input

    def test_version_convert_function_with_control_flow_subgraph(self):
        """Test that version converter processes subgraphs inside control flow nodes in functions."""
        model = ir.from_onnx_text(
            """
            <ir_version: 8, opset_import: [ "" : 18, "pkg.custom": 1]>
            agraph (float[4, 512, 512] input_x, bool cond) => (float[4, 257, 64, 2] output)
            {
                output = pkg.custom.conditional_dft (input_x, cond)
            }

            <domain: "pkg.custom", opset_import: [ "" : 18]>
            conditional_dft (x, cond) => (result) {
                result = If (cond) <then_branch: graph = then_graph () => (out) {
                    shape_a = Constant<value: tensor = int64[5] {1, 4, 512, 512, 1}>()
                    reshape_x = Reshape (x, shape_a)
                    dft = DFT <axis = 2, onesided = 1> (reshape_x)
                    shape_c = Constant<value: tensor = int64[4] {4, 257, 64, 2}>()
                    out = Reshape (dft, shape_c)
                }, else_branch: graph = else_graph () => (out) {
                    shape_c = Constant<value: tensor = int64[4] {4, 257, 64, 2}>()
                    out = Reshape (x, shape_c)
                }>
            }
        """
        )
        # Verify the function exists with correct initial state
        self.assertEqual(len(model.functions), 1)
        func = model.functions[("pkg.custom", "conditional_dft", "")]
        self.assertEqual(len(func), 1)  # 1 node (If) in the function

        # Verify the If node has subgraphs
        if_node = func[0]
        self.assertEqual(if_node.op_type, "If")
        then_branch = if_node.attributes["then_branch"].as_graph()
        else_branch = if_node.attributes["else_branch"].as_graph()
        self.assertEqual(len(then_branch), 5)  # 5 nodes in then_branch
        self.assertEqual(len(else_branch), 2)  # 2 nodes in else_branch

        target_version = 20
        # Use internal API to test function version conversion without inlining
        version_converter.convert_version(model, target_version=target_version)
        self.assertEqual(model.opset_imports[""], target_version)

        # Verify nodes inside the function's If node subgraphs were version-converted
        func = model.functions[("pkg.custom", "conditional_dft", "")]
        if_node = func[0]
        self.assertEqual(if_node.op_type, "If")
        self.assertEqual(if_node.version, 20)

        # Check then_branch subgraph nodes
        then_branch = if_node.attributes["then_branch"].as_graph()
        # After DFT adapter, a new Constant node is inserted for dft_length
        self.assertEqual(len(then_branch), 6)  # 5 + 1 new Constant for DFT
        dft_node = None
        for node in then_branch:
            self.assertEqual(node.version, 20)
            if node.op_type == "DFT":
                dft_node = node
        self.assertIsNotNone(dft_node)
        self.assertEqual(len(dft_node.inputs), 3)  # DFT 19->20 adds dft_length input

        # Check else_branch subgraph nodes
        else_branch = if_node.attributes["else_branch"].as_graph()
        self.assertEqual(len(else_branch), 2)
        for node in else_branch:
            self.assertEqual(node.version, 20)


class VersionConverter20to21Test(unittest.TestCase):
    def test_version_groupnorm(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[1, 4, 512, 512] input_x, float[2] scale, float[2] bias) => (float[4, 512, 512] output)
            {
                groupnorm = GroupNormalization <num_groups = 2> (input_x, scale, bias)
                shape_c = Constant<value: tensor = int64[4] {4, 512, 512}>()
                output = Reshape (groupnorm, shape_c)
            }
        """
        )
        target_version = 21
        version_converter.convert_version(model, target_version=target_version)
        self.assertEqual(model.opset_imports[""], target_version)

        self.assertEqual(model.graph.node(3).op_type, "Reshape")
        self.assertEqual(model.graph.node(3).version, 21)
        self.assertEqual(model.graph.node(4).op_type, "Expand")
        self.assertEqual(model.graph.node(4).version, 21)
        self.assertEqual(model.graph.node(5).op_type, "Reshape")
        self.assertEqual(model.graph.node(5).version, 21)
        self.assertEqual(model.graph.node(6).op_type, "Reshape")
        self.assertEqual(model.graph.node(6).version, 21)
        self.assertEqual(model.graph.node(7).op_type, "Expand")
        self.assertEqual(model.graph.node(7).version, 21)
        self.assertEqual(model.graph.node(8).op_type, "Reshape")
        self.assertEqual(model.graph.node(8).version, 21)
        self.assertEqual(model.graph.node(9).op_type, "GroupNormalization")
        self.assertEqual(model.graph.node(9).version, 21)

    def test_version_groupnorm_no_bias(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[1, 4, 512, 512] input_x, float[2] scale) => (float[4, 512, 512] output)
            {
                groupnorm = GroupNormalization <num_groups = 2> (input_x, scale)
                shape_c = Constant<value: tensor = int64[4] {4, 512, 512}>()
                output = Reshape (groupnorm, shape_c)
            }
        """
        )
        target_version = 21
        version_converter.convert_version(model, target_version=target_version)
        self.assertEqual(model.opset_imports[""], target_version)

        self.assertEqual(model.graph.node(0).op_type, "GroupNormalization")
        self.assertEqual(model.graph.node(0).version, 20)


class VersionConverterMetadataMergeTest(unittest.TestCase):
    def test_metadata_is_copied_on_version_conversion(self):
        """Test that metadata is copied from original node to replacement nodes during version conversion."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[4, 512, 512] input_x) => (float[4, 257, 64, 2] output)
            {
                shape_a = Constant<value: tensor = int64[5] {1, 4, 512, 512, 1}>()
                reshape_x = Reshape (input_x, shape_a)
                dft = DFT <axis = 2, onesided = 1> (reshape_x)
                shape_c = Constant<value: tensor = int64[4] {4, 257, 64, 2}>()
                output = Reshape (dft, shape_c)
            }
        """
        )
        # Find the DFT node and add metadata to it
        dft_node = model.graph.node(2)
        self.assertEqual(dft_node.op_type, "DFT")
        dft_node.metadata_props["test_key"] = "test_value"
        dft_node.metadata_props["another_key"] = "another_value"

        target_version = 25
        version_converter.convert_version(model, target_version=target_version)
        self.assertEqual(model.opset_imports[""], target_version)

        # After conversion, DFT adapter adds a Constant node for axis and the DFT node is replaced
        # The replacement DFT node should have the metadata copied
        new_dft_node = model.graph.node(3)
        self.assertEqual(new_dft_node.op_type, "DFT")
        self.assertEqual(new_dft_node.version, 25)

        # Verify metadata was copied to the new DFT node
        self.assertEqual(new_dft_node.metadata_props.get("test_key"), "test_value")
        self.assertEqual(new_dft_node.metadata_props.get("another_key"), "another_value")

    def test_metadata_is_copied_to_multiple_replacement_nodes(self):
        """Test that metadata is copied to all replacement nodes when an adapter creates multiple nodes."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[1, 4, 512, 512] input_x, float[2] scale, float[2] bias) => (float[4, 512, 512] output)
            {
                groupnorm = GroupNormalization <num_groups = 2> (input_x, scale, bias)
                shape_c = Constant<value: tensor = int64[4] {4, 512, 512}>()
                output = Reshape (groupnorm, shape_c)
            }
        """
        )
        # Find the GroupNormalization node and add metadata to it
        groupnorm_node = model.graph.node(0)
        self.assertEqual(groupnorm_node.op_type, "GroupNormalization")
        groupnorm_node.metadata_props["source"] = "original_groupnorm"

        target_version = 21
        version_converter.convert_version(model, target_version=target_version)
        self.assertEqual(model.opset_imports[""], target_version)

        # GroupNormalization adapter creates multiple nodes (Reshape, Expand, etc.)
        # Verify that metadata was copied to the new nodes created by the adapter
        new_groupnorm_node = model.graph.node(9)
        self.assertEqual(new_groupnorm_node.op_type, "GroupNormalization")
        self.assertEqual(new_groupnorm_node.version, 21)

        # Verify metadata was copied to the new GroupNormalization node
        self.assertEqual(new_groupnorm_node.metadata_props.get("source"), "original_groupnorm")

        # Also check that intermediate nodes created by the adapter received the metadata
        # The adapter creates Reshape, Expand, Reshape nodes for scale and bias
        for i in range(9):
            node = model.graph.node(i)
            if node.version == 21 and node.op_type in ("Reshape", "Expand", "Constant"):
                self.assertEqual(
                    node.metadata_props.get("source"),
                    "original_groupnorm",
                    f"Node {i} ({node.op_type}) should have metadata copied",
                )

    def test_version_convert_raises_on_function_node_with_ref_attribute(self):
        """Test that version conversion raises when a function contains a node with a ref attribute."""
        # Build a function with a LeakyRelu node that uses a RefAttr for 'alpha'
        func_input = ir.Value(name="x")
        ref_attr = ir.RefAttr("alpha", "alpha", ir.AttributeType.FLOAT)
        func_output = ir.Value(name="result")
        leaky_relu_node = ir.Node(
            domain="",
            op_type="LeakyRelu",
            inputs=[func_input],
            outputs=[func_output],
            attributes=[ref_attr],
            version=18,
        )
        func_graph = ir.Graph(
            inputs=[func_input],
            outputs=[func_output],
            nodes=[leaky_relu_node],
            opset_imports={"": 18},
        )
        func_attr_param = ir.Attr("alpha", ir.AttributeType.FLOAT, 0.01)
        function = ir.Function(
            domain="pkg.custom",
            name="leaky_relu_func",
            graph=func_graph,
            attributes=[func_attr_param],
        )

        # Build a main graph that calls the function
        main_input = ir.Value(name="input_x")
        main_output = ir.Value(name="output")
        call_node = ir.Node(
            domain="pkg.custom",
            op_type="leaky_relu_func",
            inputs=[main_input],
            outputs=[main_output],
            version=18,
        )
        main_graph = ir.Graph(
            inputs=[main_input],
            outputs=[main_output],
            nodes=[call_node],
            opset_imports={"": 18, "pkg.custom": 1},
        )
        model = ir.Model(
            main_graph,
            ir_version=8,
            functions=[function],
        )

        target_version = 20
        with self.assertRaises(
            (
                version_converter._version_converter.VersionConverterError,  # pylint: disable=protected-access`
                ir.passes.PassError,
            )
        ) as ctx:
            version_converter.convert_version(model, target_version=target_version)
        # Check the error message, unwrapping PassError if needed
        error = ctx.exception
        if isinstance(error, ir.passes.PassError) and error.__cause__ is not None:
            error = error.__cause__
        self.assertIn(
            "has ref attribute, which is not supported by version converter",
            str(error),
        )


class VersionConverter25to26Test(unittest.TestCase):
    @pytest.mark.xfail(strict=True, reason="Version upgrade beyond 25 not yet supported.")
    def test_version_convert_compatible(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 25]>
            agraph (float[1, 4, 512, 512] input_x, float[1, 4, 512, 64] input_y) => (float[1, 4, 512, 64] output)
            {
                shape_a = Constant<value: tensor = int64[3] {4, 512, 512}>()
                reshape_x = Reshape (input_x, shape_a)
                shape_b = Constant<value: tensor = int64[3] {4, 512, 64}>()
                reshape_y = Reshape (input_y, shape_b)
                matmul = MatMul (reshape_x, reshape_y)
                shape_c = Constant<value: tensor = int64[4] {1, 4, 512, 64}>()
                output = Reshape (matmul, shape_c)
            }
        """
        )
        target_version = 26
        version_converter.convert_version(model, target_version=target_version)


if __name__ == "__main__":
    unittest.main()
