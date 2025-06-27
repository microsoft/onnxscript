# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import math
import unittest

import numpy as np
import onnx
import onnx_ir as ir
import onnx_ir.passes.common.shape_inference as shape_inference
import onnxruntime as ort
import torch

import onnxscript
import onnxscript.optimizer
from onnxscript import FLOAT, script
from onnxscript import opset18 as op
from onnxscript.rewriter.ort_fusions import optimize_for_ort
from onnxscript.rewriter.ort_fusions._test_utils import assert_allclose
from onnxscript.rewriter.ort_fusions.gqa import fuse_gqa
from onnxscript.rewriter.ort_fusions.models._phi4lm import phi4lm_test
from onnxscript.rewriter.ort_fusions.sdpa import fuse_sdpa

msft_op = onnxscript.values.Opset("com.microsoft", 1)

# Test case for GroupQueryAttention (GQA) fusion.


class GQAFusionTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Config parameters
        self.batchsize = 1  # Note: GQA (cpu) seems to require batch-size 1?
        self.seqlen = 8
        self.kv_seqlen = self.seqlen
        self.past_seqlen = 16
        self.head_size = 16
        self.num_heads = 20
        self.kv_num_heads = 10

        # Computed config parameters
        self.hidden_size = self.head_size * self.num_heads
        self.kv_hidden_size = self.head_size * self.kv_num_heads
        assert (self.num_heads % self.kv_num_heads) == 0, (
            "num_heads must be divisible by kv_num_heads"
        )
        self.num_groups = self.num_heads // self.kv_num_heads
        self.total_seqlen = self.seqlen + self.past_seqlen

        # Abbreviations
        B = self.batchsize
        S = self.seqlen
        P = self.past_seqlen
        D = self.hidden_size
        Dkv = self.kv_hidden_size
        Dh = self.head_size
        Hkv = self.kv_num_heads
        total_seqlen = S + P
        max_seqlen = total_seqlen

        # Input/output types have some dimensions as dynamic (even though the
        # test case instance has specific values above).
        self.input_types = (
            FLOAT["B", "S", D],  # query
            FLOAT["B", "S", Dkv],  # key
            FLOAT["B", "S", Dkv],  # value
            FLOAT["B", Hkv, "P", Dh],  # past_key
            FLOAT["B", Hkv, "P", Dh],  # past_value
            FLOAT["max_seqlen", Dh // 2],  # cos
            FLOAT["max_seqlen", Dh // 2],  # sin
        )
        self.output_types = (
            FLOAT["B", "S", D],  # attention
            FLOAT["B", Hkv, "T", Dh],  # present_key
            FLOAT["B", Hkv, "T", Dh],  # present_value
        )

        self.inputs = {
            "query": np.random.rand(B, S, D).astype(np.float32),
            "key": np.random.rand(B, S, Dkv).astype(np.float32),
            "value": np.random.rand(B, S, Dkv).astype(np.float32),
            "past_key": np.random.rand(B, Hkv, P, Dh).astype(np.float32),
            "past_value": np.random.rand(B, Hkv, P, Dh).astype(np.float32),
            "cos": np.random.rand(max_seqlen, Dh // 2).astype(np.float32),
            "sin": np.random.rand(max_seqlen, Dh // 2).astype(np.float32),
        }

    def target_model_script(self):
        H = self.num_heads
        Hkv = self.kv_num_heads

        @script()
        def gqa(query, key, value, past_key, past_value, cos, sin):
            # Generate seqlens_k and total_seqlen inputs for GQA:
            # In this test case, all batch elements have same sequence length.
            S = op.Shape(query, start=1, end=2)
            past_seq_length = op.Shape(past_key, start=2, end=3)
            total_seq_length = op.Add(past_seq_length, S)
            total_seqlen_int32 = op.Cast(total_seq_length, to=6)
            total_seqlen_int32_minus_1 = op.Sub(total_seqlen_int32, 1)
            batchsize = op.Shape(query, start=0, end=1)
            seqlens_k = op.Tile(total_seqlen_int32_minus_1, batchsize)

            attn, past_key, past_value = msft_op.GroupQueryAttention(
                query,
                key,
                value,
                past_key,
                past_value,
                seqlens_k,
                total_seqlen_int32,
                cos,
                sin,
                num_heads=H,
                kv_num_heads=Hkv,
                do_rotary=1,
            )
            return attn, past_key, past_value

        return gqa

    def source_model_script(self):
        scale_factor = math.sqrt(math.sqrt(self.head_size))
        minval = torch.finfo(torch.float32).min
        minval_tp = onnx.helper.make_tensor("minval", onnx.TensorProto.FLOAT, [1], [minval])
        H = [self.num_heads]
        Hkv = [self.kv_num_heads]
        Dh = [self.head_size]
        G = [self.num_groups]
        minus_1 = [-1]  # inferred dimension in Reshape op
        plus_1 = [1]

        @script()
        def gqa(query, key, value, past_key, past_value, cos, sin):
            # Shapes used for Reshape ops. Note that we have a few different options on how shapes are
            # specified in an ONNX Reshape op (which supports special values 0 and -1 to propagate
            # existing dimension and one inferred dimension respectively). The following shapes are
            # based on what is observed in Phi models generated by the exporter.
            B = op.Shape(query, start=0, end=1)
            S = op.Shape(query, start=1, end=2)
            past_seq_length = op.Shape(past_key, start=2, end=3)
            total_seq_length = op.Add(past_seq_length, S)
            # past_seq_length = op.Squeeze(past_seq_length_1D, [0])
            # S_0D = op.Squeeze(S,[0])

            shape_BSHDh = op.Concat(B, S, minus_1, Dh, axis=0)
            shape_BSHkvDh = op.Concat(B, S, minus_1, Dh, axis=0)
            shape_BSD = op.Concat(B, S, minus_1, axis=0)
            shape_BHkvGSDh = op.Concat(B, Hkv, G, total_seq_length, Dh, axis=0)

            shape_BHSDh = op.Concat(B, H, total_seq_length, Dh, axis=0)

            # First, get Q, K, V into right shapes. Inputs are 3D tensors in the BSD format.
            # D is different for Q and K/V (not reflected in the names, unfortunately).
            # We convert them into BHSDh (i.e., BHSd) format. In this version, we have only
            # one sequence length (S) for all Q, K, and V (with no cache).
            query_BSHDh = op.Reshape(query, shape_BSHDh)
            query_BHSDh = op.Transpose(query_BSHDh, perm=[0, 2, 1, 3])

            key_BSHkvDh = op.Reshape(key, shape_BSHkvDh)
            key_BHkvSDh = op.Transpose(key_BSHkvDh, perm=[0, 2, 1, 3])

            value_BSHkvDh = op.Reshape(value, shape_BSHkvDh)
            value_BHkvSDh = op.Transpose(value_BSHkvDh, perm=[0, 2, 1, 3])

            # Concat past and do rotary embedding
            position_ids_1d = op.Range(past_seq_length, total_seq_length, 1)
            position_ids_q = op.Unsqueeze(position_ids_1d, [0])
            position_ids_k = op.Unsqueeze(position_ids_1d, [0])

            # Note: The above code pattern for position-ids is from exported Phi model.
            # However, for use with ORT's RotaryEmbedding it needs the following for batchsize > 1
            # But we currently target batchsize=1 since GQA requires it when there is a past key/value.
            #
            # position_ids_2d = op.Unsqueeze(position_ids_1d, [0])
            # tile_B_1 = op.Concat(B, plus_1, axis=0)
            # position_ids = op.Tile(position_ids_2d, tile_B_1)

            query_BHSDh_rope = msft_op.RotaryEmbedding(
                query_BHSDh,
                position_ids_q,
                cos,
                sin,
            )
            key_BHkvSDh_rope = msft_op.RotaryEmbedding(
                key_BHkvSDh,
                position_ids_k,
                cos,
                sin,
            )
            key_seq_BHkvSkvDh = op.Concat(past_key, key_BHkvSDh_rope, axis=-2)

            value_seq_BHkvSkvDh = op.Concat(past_value, value_BHkvSDh, axis=-2)

            # Now, expand from shared heads to all heads
            key_BHkv1SDh = op.Unsqueeze(key_seq_BHkvSkvDh, 2)
            key_BHkvGSDh = op.Expand(key_BHkv1SDh, shape_BHkvGSDh)
            key_BHSDh = op.Reshape(key_BHkvGSDh, shape_BHSDh)

            value_BHkv1SDh = op.Unsqueeze(value_seq_BHkvSkvDh, 2)
            value_BHkvGSDh = op.Expand(value_BHkv1SDh, shape_BHkvGSDh)
            value_BHSDh = op.Reshape(value_BHkvGSDh, shape_BHSDh)

            # Generate causal mask:
            # where every row looks like [0, 0, ..., /*diagonal=*/ 0, minval, minval, ...]
            seq_len = op.Shape(query, end=2, start=1)
            seq_len_0D = op.Squeeze(seq_len)

            past_seq_len_0D = op.Squeeze(past_seq_length)

            total_seq_len_0D = op.Add(past_seq_len_0D, seq_len_0D)
            total_seq_len = op.Reshape(total_seq_len_0D, [-1])

            # The Phi modeling code generates the following +1 as the target-length, which seems
            # unnecessary in this context. But duplicating same logic here.
            total_seq_len_plus_1_0D = op.Add(total_seq_len_0D, 1)
            total_seq_len_plus_1 = op.Reshape(total_seq_len_plus_1_0D, [-1])

            current_range = op.Range(past_seq_len_0D, total_seq_len_0D, 1)
            mask_shape = op.Concat(seq_len, total_seq_len_plus_1, axis=0)
            min_val = op.Constant(value=minval_tp)
            mask_all_min = op.Expand(min_val, mask_shape)
            total_range_as_row = op.Range(0, total_seq_len_plus_1_0D, 1)
            current_range_as_column = op.Reshape(current_range, [-1, 1])
            boolean_mask = op.Greater(total_range_as_row, current_range_as_column)
            float_0_1_mask = op.Cast(boolean_mask, to=1)
            float_0_min_mask = op.Mul(mask_all_min, float_0_1_mask)
            mask_4d = op.Unsqueeze(float_0_min_mask, [0, 1])
            shape_B111 = op.Concat(B, plus_1, plus_1, plus_1, axis=0)
            mask_B1ST_plus = op.Expand(mask_4d, shape_B111)

            # Get rid of the extra +1 added above: total_seq_len is enough, no
            # need for total_seq_len+1.
            mask_B1ST = op.Slice(mask_B1ST_plus, [0], total_seq_len, [3], [1])

            # Now, compute attention:
            key_transposed = op.Transpose(key_BHSDh, perm=[0, 1, 3, 2])
            divisor = op.Constant(value_float=scale_factor)
            scaled_query = op.Div(query_BHSDh_rope, divisor)
            scaled_key = op.Div(key_transposed, divisor)
            attn_score = op.MatMul(scaled_query, scaled_key)
            masked_attn_score = op.Add(attn_score, mask_B1ST)
            attn_weight = op.Softmax(masked_attn_score, axis=-1)
            attention_BHSDh = op.MatMul(attn_weight, value_BHSDh)

            # Reshape back to BSD format
            attention_BSHDh = op.Transpose(attention_BHSDh, perm=[0, 2, 1, 3])
            attention_BSD = op.Reshape(attention_BSHDh, shape_BSD)

            return attention_BSD, key_seq_BHkvSkvDh, value_seq_BHkvSkvDh

        return gqa

    def test_equivalence(self):
        """Test that the source and target models produce the same outputs."""
        inputs = self.inputs

        source_model = self.source_model_script().to_model_proto(
            input_types=self.input_types,
            output_types=self.output_types,
        )
        session = ort.InferenceSession(
            source_model.SerializeToString(), providers=("CPUExecutionProvider",)
        )
        source_model_outputs = session.run(None, inputs)

        target_model = self.target_model_script().to_model_proto(
            input_types=self.input_types,
            output_types=self.output_types,
        )
        session = ort.InferenceSession(
            target_model.SerializeToString(), providers=("CPUExecutionProvider",)
        )
        target_model_outputs = session.run(None, inputs)

        self.assertEqual(len(source_model_outputs), len(target_model_outputs))
        assert_allclose(source_model_outputs, target_model_outputs)

    def test_fusion(self):
        """Test that GQA fusion is successful on source model and produces an equivalent model."""
        inputs = self.inputs

        source_model = self.source_model_script().to_model_proto(
            input_types=self.input_types,
            output_types=self.output_types,
        )
        session = ort.InferenceSession(
            source_model.SerializeToString(), providers=("CPUExecutionProvider",)
        )
        source_model_outputs = session.run(None, inputs)

        # Some shapes need to be present in input model for fusion to be successful.
        # (i) Shape inference doesn't handle handle ORT contrib ops.
        # (ii) TODO: investigate if Reshape(..., ["B", "S", -1, Dh]) handled precisely
        # by shape inference.
        query_BHSDh_rope_value_info = onnx.helper.make_tensor_value_info(
            "query_BHSDh_rope",
            onnx.TensorProto.FLOAT,
            ["B", self.num_heads, self.seqlen, self.head_size],
        )
        key_BHkvSDh_rope_value_info = onnx.helper.make_tensor_value_info(
            "key_BHkvSDh_rope",
            onnx.TensorProto.FLOAT,
            ["B", self.kv_num_heads, self.seqlen, self.head_size],
        )
        query_BSHDh_value_info = onnx.helper.make_tensor_value_info(
            "query_BSHDh",
            onnx.TensorProto.FLOAT,
            ["B", self.seqlen, self.num_heads, self.head_size],
        )
        key_BHSDh_value_info = onnx.helper.make_tensor_value_info(
            "key_BHSDh",
            onnx.TensorProto.FLOAT,
            ["B", self.num_heads, self.total_seqlen, self.head_size],
        )
        key_BSHkvDh_value_info = onnx.helper.make_tensor_value_info(
            "key_BSHkvDh",
            onnx.TensorProto.FLOAT,
            ["B", self.seqlen, self.kv_num_heads, self.head_size],
        )
        key_transposed_value_info = onnx.helper.make_tensor_value_info(
            "key_transposed",
            onnx.TensorProto.FLOAT,
            ["B", self.num_heads, self.head_size, self.total_seqlen],
        )
        value_BHSDh_value_info = onnx.helper.make_tensor_value_info(
            "value_BHSDh",
            onnx.TensorProto.FLOAT,
            ["B", self.num_heads, self.total_seqlen, self.head_size],
        )
        source_model.graph.value_info.extend(
            [
                query_BHSDh_rope_value_info,
                key_BHkvSDh_rope_value_info,
                query_BSHDh_value_info,
                key_BHSDh_value_info,
                key_BSHkvDh_value_info,
                key_transposed_value_info,
                value_BHSDh_value_info,
            ]
        )

        source_model_ir = ir.serde.from_proto(source_model)
        inferred_model = shape_inference.infer_shapes(source_model_ir)
        onnxscript.optimizer.optimize(inferred_model)

        count = fuse_sdpa(inferred_model, debug=True)
        self.assertGreater(count, 0)

        count = fuse_gqa(inferred_model, debug=True)
        self.assertGreater(count, 0)

        fused_model = ir.serde.to_proto(inferred_model)
        session = ort.InferenceSession(
            fused_model.SerializeToString(), providers=("CPUExecutionProvider",)
        )
        outputs3 = session.run(None, inputs)

        self.assertEqual(len(outputs3), len(source_model_outputs))
        assert_allclose(outputs3, source_model_outputs)


class GQAFusionTest2(unittest.TestCase):
    @unittest.skip("Needs too much memory.")
    def test_phi4lm(self):
        test_case = phi4lm_test()
        model = test_case.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        optimize_for_ort(model, debug=True)
        gqa_nodes = [n for n in model.graph if n.op_type == "GQA"]
        self.assertEqual(len(gqa_nodes), 2, "Expected 2i GQA nodes after fusion")


if __name__ == "__main__":
    unittest.main()
