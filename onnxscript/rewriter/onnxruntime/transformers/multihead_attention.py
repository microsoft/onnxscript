
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
r"""POC experimenting function aware pattern re-write.

In this case we don't want to spell-out the entire source pattern.
Instead, we want to replace an entire function call a new subgraph.

Source function: LlamaAttention
inputs (positional args, the names in function definition are unfortunately arbitrary and don't provide value):
    - hidden_states
    - position_id
    - attention_mask
    - q_proj.weight
    - k_proj.weight
    - v_proj.weight
    - cos_cached
    - sin_cached
    - o_proj.weight
outputs (similarly, positional)
    - present_value
    - present_key
    - attn_output (o_proj)

The rewriting algorithm is as follows:

The final new function graph should look like this:

    function_proj_q                     function_proj_k
            |                                   |
            |                                   |
com.microsoft::RotaryEmbedding     com.microsoft::RotaryEmbedding        function_proj_v
            \                                   /                              /
             \                                 /                              /
              \                               /                              /
               \---------------              /       -----------------------/
                        com.microsoft::MultiHeadAttention
                            |               |           |
                        attn_output   (present_key) (present_value)
                            |
                     function_proj_o
                            |
                        (output)

So all we need, is to locate 'function_proj_q', 'function_proj_k', 'function_proj_v', 'function_proj_o'.
Construct the 4 nodes with new contrib op nodes, and properly name their inputs/outputs.

"""

from __future__ import annotations

import abc
import dataclasses
import logging
import numpy as np
import onnx
from onnx import helper as onnx_helper
from onnx import NodeProto, TensorProto, helper, numpy_helper

import onnxscript
from onnxscript import ir
from onnxscript.ir import SymbolicDim
from onnxscript.rewriter import _ir_utils, function_rule

logger = logging.getLogger(__name__)

import dataclasses
import abc
from onnxscript import ir
from typing import List, Tuple, Union, Optional

from onnxscript.rewriter import _ir_utils, function_rule
import onnx.shape_inference

@dataclasses.dataclass
class AttnSizeConfig:
    num_attention_heads: int
    num_key_value_heads: int | None
    head_size: int
    hidden_size: int

class AttentionRewriteRule(function_rule.FunctionRewriteRule, abc.ABC):
    PACKAGE_NAME: str


    def infer_attn_size_config(self, function: ir.Function) -> AttnSizeConfig:
        if len(function.outputs) == 3 or len(function.outputs) == 4:
            if len(function.outputs) == 3:
                # Usually the Attention related modules have 3 outputs:
                present_value, _, attn_output = function.outputs
            else:
                # Some Attention related modules have 4 outputs:
                _, present_value, _, attn_output = function.outputs

            if present_value.shape is None:
                raise function_rule.FunctionRewriteError(
                    "Failed to find shape for present_value."
                )
            if attn_output.shape is None:
                raise function_rule.FunctionRewriteError(
                    "Failed to find shape for attn_output."
                )
            if len(present_value.shape) <= 3:
                raise function_rule.FunctionRewriteError(
                    "Expected present_value to have at least 4 dimensions."
                )
            head_size = present_value.shape[3]
            num_key_value_heads = present_value.shape[1]
            hidden_size = attn_output.shape[2]
            num_attention_heads = hidden_size // head_size

            return AttnSizeConfig(
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_size=head_size,
                hidden_size=hidden_size,
            )

        elif any("scaled_dot_product_attention" in node.op_type for node in function):
            # If the Attention related modules use scaled_dot_product_attention,
            # present_value and present_key are not present in the output.
            hidden_size = function.outputs[0].shape[2]
            # Get head size and number of heads from the Reshape node.
            # Reference:
            # https://github.com/huggingface/diffusers/blob/ae05050db9d37d5af48a6cd0d6510a5ffb1c1cd4/src/diffusers/models/attention_processor.py#L1269
            reshape_nodes = [node for node in function if node.op_type == "Reshape"]
            assert (
                len(reshape_nodes) == 4
            ), "Expected 3 Reshape nodes for Q, K and V, and 1 reshape node for output of scaled_dot_product_attention."
            for reshape_node in reshape_nodes:
                constant_node = reshape_node.inputs[1].producer()
                assert (
                    constant_node.op_type == "Constant"
                ), "Expected the second input to Reshape to be a Constant node."
                value = _ir_utils.propagate_const_value(reshape_node.inputs[1])
                constant_value = value.const_value
                if constant_value is None:
                    raise function_rule.FunctionRewriteError(
                        "Failed to propagate constant value for Reshape node."
                    )
                constant_numpy_value = constant_value.numpy()
                if constant_numpy_value.shape[0] == 4:
                    num_attention_heads = constant_numpy_value[2]
                    head_size = constant_numpy_value[3]
                    return AttnSizeConfig(
                        num_attention_heads=num_attention_heads,
                        num_key_value_heads=None,
                        head_size=head_size,
                        hidden_size=hidden_size,
                    )
            raise function_rule.FunctionRewriteError(
                "Failed to infer head size and number of heads from QKV Reshape nodes. \
                Expected 4D shape in the constant node (batch_size, seq_length, num_attention_heads, head_size)."
            )
        raise function_rule.FunctionRewriteError(
            f"Attenion modules should have 3 outputs or scaled_dot_product_attention node, "
            f"got output: {len(function.outputs)} and no scaled_dot_product_attention."
        )




class MHALlama2RewriteRule(AttentionRewriteRule):
    FUNCTION_KEYWORD = "LlamaAttention"
    PACKAGE_NAME = "transformers"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version(min_version="4.33", max_version="4.36")
    def _fusion_with_4d_cache(self, function: ir.Function) -> ir.Function:
        if len(function.inputs) != 9:
            raise function_rule.FunctionRewriteError(
                f"Unexpected number of inputs. Expected 9, got {len(function.inputs)}."
            )

        # Infer size configurations from the function.
        attn_size_config = self.infer_attn_size_config(function)

        # Code new pattern with onnxscript.
        op = onnxscript.opset18
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        # Workaround onnxscript error by specifying the output shape here.
        cos_sin_gather_size = [attn_size_config.head_size // 2]
        expand_shape = [1, attn_size_config.num_attention_heads, 1, 1]

        def mha(
            hidden_states,
            position_id,
            attention_mask,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            cos_cached,
            sin_cached,
            o_proj_weight,
        ):
            q = op.MatMul(hidden_states, op.Transpose(q_proj_weight, [1, 0]))
            k = op.MatMul(hidden_states, op.Transpose(k_proj_weight, [1, 0]))
            v = op.MatMul(hidden_states, op.Transpose(v_proj_weight, [1, 0]))

            # TODO(onnxscript)
            # ValueError: ERROR: Unsupported expression type <class 'ast.List'>.
            # at: Function 'mha', line 16
            #     cos = op.Slice(op.Squeeze(cos_cached, [0, 1]), [0], [cos_sin_gather_size], [1])
            # NOTE: Depending on transformers version, the shape of cos/sin is different.
            # In later version, the shape is [seq_len, head_size], so the Squeeze is not needed.
            # In this version, the shape is [1, 1, seq_len, head_size], hence the below Squeeze.
            cos = op.Slice(op.Squeeze(cos_cached, [0, 1]), [0], cos_sin_gather_size, [1])
            sin = op.Slice(op.Squeeze(sin_cached, [0, 1]), [0], cos_sin_gather_size, [1])

            q_rope = msft_op.RotaryEmbedding(q, position_id, cos, sin, interleaved=False)
            k_rope = msft_op.RotaryEmbedding(k, position_id, cos, sin, interleaved=False)

            # TODO(onnxscript)
            # ValueError: ERROR: Unsupported expression type <class 'ast.List'>.
            # expanded_mask = op.Expand(attention_mask, [1, self.num_heads, 1, 1])
            expanded_mask = op.Expand(attention_mask, expand_shape)

            mha_output, present_key, present_value = msft_op.MultiHeadAttention(
                q_rope,
                k_rope,
                v,
                None,
                None,
                expanded_mask,
                num_heads=attn_size_config.num_attention_heads,
            )
            attn_output = op.MatMul(mha_output, op.Transpose(o_proj_weight, [1, 0]))
            return present_value, present_key, attn_output

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(
            mha
        ).to_function_proto()
        return ir.serde.deserialize_function(function_proto)

    @_version_controller.register_version(min_version="4.36", max_version="4.38")
    def _fusion_with_2d_cache(self, function: ir.Function) -> ir.Function:
        # Infer size configurations from the function.
        attn_size_config = self.infer_attn_size_config(function)

        if len(function.inputs) != 9:
            raise function_rule.FunctionRewriteError(
                f"Unexpected number of inputs. Expected 9, got {len(function.inputs)}."
            )

        # Code new pattern with onnxscript.
        op = onnxscript.opset18
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        # Workaround onnxscript error by specifying the output shape here.
        cos_sin_gather_size = [attn_size_config.head_size // 2]
        expand_shape = [1, attn_size_config.num_attention_heads, 1, 1]

        def mha(
            hidden_states,
            position_id,
            attention_mask,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            cos_cached,
            sin_cached,
            o_proj_weight,
        ):
            q = op.MatMul(hidden_states, op.Transpose(q_proj_weight, [1, 0]))
            k = op.MatMul(hidden_states, op.Transpose(k_proj_weight, [1, 0]))
            v = op.MatMul(hidden_states, op.Transpose(v_proj_weight, [1, 0]))

            cos = op.Slice(cos_cached, [0], cos_sin_gather_size, [1])
            sin = op.Slice(sin_cached, [0], cos_sin_gather_size, [1])

            q_rope = msft_op.RotaryEmbedding(q, position_id, cos, sin, interleaved=False)
            k_rope = msft_op.RotaryEmbedding(k, position_id, cos, sin, interleaved=False)

            # TODO(onnxscript)
            # ValueError: ERROR: Unsupported expression type <class 'ast.List'>.
            # expanded_mask = op.Expand(attention_mask, [1, self.num_heads, 1, 1])
            expanded_mask = op.Expand(attention_mask, expand_shape)

            mha_output, present_key, present_value = msft_op.MultiHeadAttention(
                q_rope,
                k_rope,
                v,
                None,
                None,
                expanded_mask,
                num_heads=attn_size_config.num_attention_heads,
            )
            attn_output = op.MatMul(mha_output, op.Transpose(o_proj_weight, [1, 0]))
            return present_value, present_key, attn_output

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(
            mha
        ).to_function_proto()
        return ir.serde.deserialize_function(function_proto)


class GQALlama2RewriteRule(AttentionRewriteRule):
    FUNCTION_KEYWORD = "LlamaAttention"
    PACKAGE_NAME = "transformers"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version(min_version="4.33", max_version="4.36")
    def _fusion_with_4d_cache(self, function: ir.Function) -> ir.Function:
        if len(function.inputs) != 9:
            raise function_rule.FunctionRewriteError(
                f"Unexpected number of inputs. Expected 9, got {len(function.inputs)}."
            )

        # Infer size configurations from the function.
        attn_size_config = self.infer_attn_size_config(function)

        # Code new pattern with onnxscript.
        op = onnxscript.opset18
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        # Workaround onnxscript error by specifying the output shape here.
        cos_sin_gather_size = [attn_size_config.head_size // 2]

        def gqa(
            hidden_states,
            position_id,           attention_mask,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            cos_cached,
            sin_cached,
            o_proj_weight,
        ):
            q = op.MatMul(hidden_states, op.Transpose(q_proj_weight, [1, 0]))
            k = op.MatMul(hidden_states, op.Transpose(k_proj_weight, [1, 0]))
            v = op.MatMul(hidden_states, op.Transpose(v_proj_weight, [1, 0]))

            # NOTE: Depending on transformers version, the shape of cos/sin is different.
            # In later version, the shape is [seq_len, head_size], so the Squeeze is not needed.
            # In this version, the shape is [1, 1, seq_len, head_size], hence the below Squeeze.
            cos = op.Slice(op.Squeeze(cos_cached, [0, 1]), [0], cos_sin_gather_size, [1])
            sin = op.Slice(op.Squeeze(sin_cached, [0, 1]), [0], cos_sin_gather_size, [1])

            q_rope = msft_op.RotaryEmbedding(q, position_id, cos, sin, interleaved=False)
            k_rope = msft_op.RotaryEmbedding(k, position_id, cos, sin, interleaved=False)

            batch_size = op.Slice(op.Shape(hidden_states), [0], [1], [0])
            sequence_length = op.Slice(op.Shape(hidden_states), [1], [2], [0])
            past_seq_lengths = op.ConstantOfShape(
                batch_size,
                value=onnx_helper.make_tensor(
                    "past_seq_lengths", onnx.TensorProto.INT32, [1], [0]
                ),
            )
            total_seq_lengths = op.Cast(sequence_length, to=onnx.TensorProto.INT32)

            gqa_output, present_key, present_value = msft_op.GroupQueryAttention(
                q_rope,
                k_rope,
                v,
                None,
                None,
                past_seq_lengths,
                total_seq_lengths,
                kv_num_heads=attn_size_config.num_key_value_heads,
                num_heads=attn_size_config.num_attention_heads,
            )
            attn_output = op.MatMul(gqa_output, op.Transpose(o_proj_weight, [1, 0]))
            return present_value, present_key, attn_output

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(
            gqa
        ).to_function_proto()
        return ir.serde.deserialize_function(function_proto)

    @_version_controller.register_version(min_version="4.36", max_version="4.38")
    def _fusion_with_2d_cache(self, function: ir.Function) -> ir.Function:
        # Infer size configurations from the function.
        attn_size_config = self.infer_attn_size_config(function)

        if len(function.inputs) != 9:
            raise function_rule.FunctionRewriteError(
                f"Unexpected number of inputs. Expected 9, got {len(function.inputs)}."
            )

        # Code new pattern with onnxscript.
        op = onnxscript.opset18
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        # Workaround onnxscript error by specifying the output shape here.
        cos_sin_gather_size = [attn_size_config.head_size // 2]

        def gqa(
            hidden_states,
            position_id,
            attention_mask,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            cos_cached,
            sin_cached,
            o_proj_weight,
        ):
            q = op.MatMul(hidden_states, op.Transpose(q_proj_weight, [1, 0]))
            k = op.MatMul(hidden_states, op.Transpose(k_proj_weight, [1, 0]))
            v = op.MatMul(hidden_states, op.Transpose(v_proj_weight, [1, 0]))

            cos = op.Slice(cos_cached, [0], cos_sin_gather_size, [1])
            sin = op.Slice(sin_cached, [0], cos_sin_gather_size, [1])

            q_rope = msft_op.RotaryEmbedding(q, position_id, cos, sin, interleaved=False)
            k_rope = msft_op.RotaryEmbedding(k, position_id, cos, sin, interleaved=False)

            batch_size = op.Slice(op.Shape(hidden_states), [0], [1], [0])
            sequence_length = op.Slice(op.Shape(hidden_states), [1], [2], [0])
            past_seq_lengths = op.ConstantOfShape(
                batch_size,
                value=onnx_helper.make_tensor(
                    "past_seq_lengths", onnx.TensorProto.INT32, [1], [0]
                ),
            )
            total_seq_lengths = op.Cast(sequence_length, to=onnx.TensorProto.INT32)

            gqa_output, present_key, present_value = msft_op.GroupQueryAttention(
                q_rope,
                k_rope,
                v,
                None,
                None,
                past_seq_lengths,
                total_seq_lengths,
                kv_num_heads=attn_size_config.num_key_value_heads,
                num_heads=attn_size_config.num_attention_heads,
            )
            attn_output = op.MatMul(gqa_output, op.Transpose(o_proj_weight, [1, 0]))
            return present_value, present_key, attn_output

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(
            gqa
        ).to_function_proto()
        return ir.serde.deserialize_function(function_proto)


class GQALlamaSdpa2RewriteRule(AttentionRewriteRule):
    # TODO: There are a lot of duplicated code with `MHALlama2RewriteRule`.
    # The pitfall is that the source function signature is slightly different.
    # One has `attention_mask` as input while the other does not.
    # Possibly designing a function template system could help reduce the boilerplate.
    FUNCTION_KEYWORD = "LlamaSdpaAttention"
    PACKAGE_NAME = "transformers"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version(min_version="4.36", max_version="4.38")
    def _fusion(self, function: ir.Function) -> ir.Function:
        # Infer size configurations from the function.
        attn_size_config = self.infer_attn_size_config(function)

        # Code new pattern with onnxscript.
        op = onnxscript.opset18
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        cos_sin_gather_size = [attn_size_config.head_size // 2]

        def gqa(
            hidden_states,
            position_id,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            cos_cached,
            sin_cached,
            o_proj_weight,
        ):
            q = op.MatMul(hidden_states, op.Transpose(q_proj_weight, [1, 0]))
            k = op.MatMul(hidden_states, op.Transpose(k_proj_weight, [1, 0]))
            v = op.MatMul(hidden_states, op.Transpose(v_proj_weight, [1, 0]))

            cos = op.Slice(cos_cached, [0], cos_sin_gather_size, [1])
            sin = op.Slice(sin_cached, [0], cos_sin_gather_size, [1])

            q_rope = msft_op.RotaryEmbedding(q, position_id, cos, sin, interleaved=False)
            k_rope = msft_op.RotaryEmbedding(k, position_id, cos, sin, interleaved=False)

            batch_size = op.Slice(op.Shape(hidden_states), [0], [1], [0])
            sequence_length = op.Slice(op.Shape(hidden_states), [1], [2], [0])
            past_seq_lengths = op.ConstantOfShape(
                batch_size,
                value=onnx_helper.make_tensor(
                    "past_seq_lengths", onnx.TensorProto.INT32, [1], [0]
                ),
            )
            total_seq_lengths = op.Cast(sequence_length, to=onnx.TensorProto.INT32)

            gqa_output, present_key, present_value = msft_op.GroupQueryAttention(
                q_rope,
                k_rope,
                v,
                None,
                None,
                past_seq_lengths,
                total_seq_lengths,
                kv_num_heads=attn_size_config.num_key_value_heads,
                num_heads=attn_size_config.num_attention_heads,
            )
            attn_output = op.MatMul(gqa_output, op.Transpose(o_proj_weight, [1, 0]))
            return present_value, present_key, attn_output

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(
            gqa
        ).to_function_proto()
        return ir.serde.deserialize_function(function_proto)

    @_version_controller.register_version(min_version="4.38")
    def _fusion_without_cos_sin_cache(self, function: ir.Function) -> ir.Function:
        # Infer size configurations from the function.
        attn_size_config = self.infer_attn_size_config(function)
        # Code new pattern with onnxscript.
        op = onnxscript.opset18
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        cos_sin_gather_size = [attn_size_config.head_size // 2]

        def gqa(
            hidden_states,
            position_id,
            causal_mask,
            cache_position,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            inv_freq,
            o_proj_weight,
        ):
            q = op.MatMul(hidden_states, op.Transpose(q_proj_weight, [1, 0]))
            k = op.MatMul(hidden_states, op.Transpose(k_proj_weight, [1, 0]))
            v = op.MatMul(hidden_states, op.Transpose(v_proj_weight, [1, 0]))

            # In 4.38 and later, cos/sin are not cached, but computed on the fly.
            # This can be further optimized by constant folding for scenarios where
            # the position_id is known at compile time.
            seq_len = op.Slice(op.Shape(hidden_states), [1], [2], [0])
            seq_len_scalar = op.Squeeze(seq_len, [0])
            t = op.Unsqueeze(
                op.Cast(op.Range(0, seq_len_scalar, 1), to=onnx.TensorProto.FLOAT), [1]
            )
            inv_freq = op.Cast(op.Unsqueeze(inv_freq, [0]), to=onnx.TensorProto.FLOAT)
            freqs = op.MatMul(t, inv_freq)

            emb = op.Concat(freqs, freqs, axis=-1)
            cos = op.CastLike(op.Cos(emb), hidden_states)
            sin = op.CastLike(op.Sin(emb), hidden_states)
            cos = op.Slice(cos, [0], cos_sin_gather_size, [1])
            sin = op.Slice(sin, [0], cos_sin_gather_size, [1])

            q_rope = msft_op.RotaryEmbedding(q, position_id, cos, sin, interleaved=False)
            k_rope = msft_op.RotaryEmbedding(k, position_id, cos, sin, interleaved=False)

            batch_size = op.Slice(op.Shape(hidden_states), [0], [1], [0])
            sequence_length = op.Slice(op.Shape(hidden_states), [1], [2], [0])
            past_seq_lengths = op.ConstantOfShape(
                batch_size,
                value=onnx_helper.make_tensor(
                    "past_seq_lengths", onnx.TensorProto.INT32, [1], [0]
                ),
            )
            total_seq_lengths = op.Cast(sequence_length, to=onnx.TensorProto.INT32)

            gqa_output, present_key, present_value = msft_op.GroupQueryAttention(
                q_rope,
                k_rope,
                v,
                None,
                None,
                past_seq_lengths,
                total_seq_lengths,
                kv_num_heads=attn_size_config.num_key_value_heads,
                num_heads=attn_size_config.num_attention_heads,
            )
            attn_output = op.MatMul(gqa_output, op.Transpose(o_proj_weight, [1, 0]))
            return present_value, present_key, attn_output

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(
            gqa
        ).to_function_proto()
        return ir.serde.deserialize_function(function_proto)


class AttnPhi15RewriteRule(AttentionRewriteRule):
    FUNCTION_KEYWORD = "PhiAttention"
    PACKAGE_NAME = "transformers_modules"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version()
    def _fusion(self, function: ir.Function) -> ir.Function:
        # Infer size configurations from the function.
        attn_size_config = self.infer_attn_size_config(function)

        # Code new pattern with onnxscript.
        op = onnxscript.opset18
        msft_opset = onnxscript.values.Opset("com.microsoft", 1)

        def phi_attention(
            hidden_states,
            position_id,
            attention_mask,
            q_proj_weight,
            q_proj_bias,
            k_proj_weight,
            k_proj_bias,
            v_proj_weight,
            v_proj_bias,
            cos_cached,
            sin_cached,
            dense_weight,
            dense_bias,
        ):
            qkv_weight = op.Transpose(
                op.Concat(q_proj_weight, k_proj_weight, v_proj_weight, axis=0),
                perm=[1, 0],
            )
            qkv_bias = op.Concat(q_proj_bias, k_proj_bias, v_proj_bias, axis=0)

            # [batch_size, sequence_length]
            attention_mask_shape = op.Slice(op.Shape(hidden_states), [0], [2], [0])

            # Create 2d mask to mimic 4d causal mask.
            attention_mask = op.ConstantOfShape(
                attention_mask_shape,
                value=onnx_helper.make_tensor("mask_value", onnx.TensorProto.INT32, [1], [1]),
            )
            attn_output, present = msft_opset.Attention(
                hidden_states,
                qkv_weight,
                qkv_bias,
                attention_mask,
                unidirectional=1,
                do_rotary=1,
                # Attention.rotary_embedding_dim only supports 32, 64 or 128
                rotary_embedding_dim=attn_size_config.head_size // 2 // 32 * 32,
                num_heads=attn_size_config.num_attention_heads,
            )
            present_key = op.Gather(present, 0)
            present_value = op.Gather(present, 1)
            output = op.Add(
                op.MatMul(attn_output, op.Transpose(dense_weight, [1, 0])), dense_bias
            )

            return present_value, present_key, output

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(
            phi_attention
        ).to_function_proto()
        return ir.serde.deserialize_function(function_proto)


class MHAStableDiffusionUnetRewriteRule(AttentionRewriteRule):
    """Rewrite rule for Attention in diffusers."""

    FUNCTION_KEYWORD = "Attention"
    PACKAGE_NAME = "diffusers"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version()
    def _fusion(self, function: ir.Function) -> ir.Function:
        # Attention inputs could be 6 or 7:
        # hidden_states, encoder_hidden_states(optional), q_weight, k_weight, v_weight, o_weight, o_bias
        if len(function.inputs) != 6 and len(function.inputs) != 7:
            raise function_rule.FunctionRewriteError(
                f"Unexpected number of inputs. Expected 6 or 7, got {len(function.inputs)}."
            )

        # Infer size configurations from the function.
        attn_size_config = self.infer_attn_size_config(function)

        # Code new pattern with onnxscript.
        op = onnxscript.opset18
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        def attention(
            hidden_states,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            o_bias,
        ):
            qkv_weight = op.Transpose(
                op.Concat(q_weight, k_weight, v_weight, axis=0),
                perm=[1, 0],
            )

            # NOTE: MHA does not work when Q, K, and V has the same root inputs.
            attn_output, _ = msft_op.Attention(
                hidden_states,
                qkv_weight,
                None,
                None,
                num_heads=attn_size_config.num_attention_heads,
            )

            # linear projection
            output = op.Add(op.MatMul(attn_output, op.Transpose(o_weight, [1, 0])), o_bias)
            return output

        def mha(
            hidden_states,
            encoder_hidden_states,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            o_bias,
        ):
            q = op.MatMul(hidden_states, op.Transpose(q_weight, [1, 0]))
            k = op.MatMul(encoder_hidden_states, op.Transpose(k_weight, [1, 0]))
            v = op.MatMul(encoder_hidden_states, op.Transpose(v_weight, [1, 0]))

            # NOTE: Q and K needs to have the sequence length (dim 1) to use
            # GQA.
            mha_output, _, _ = msft_op.MultiHeadAttention(
                q,
                k,
                v,
                None,
                None,
                num_heads=attn_size_config.num_attention_heads,
            )
            attn_output = op.Add(op.MatMul(mha_output, op.Transpose(o_weight, [1, 0])), o_bias)
            return attn_output

        if len(function.inputs) == 6:
            function_proto = onnxscript.script(default_opset=onnxscript.opset18)(
                attention
            ).to_function_proto()
            return ir.serde.deserialize_function(function_proto)

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(
            mha
        ).to_function_proto()
        return ir.serde.deserialize_function(function_proto)


"""
Implementations of llama3b rules (MLP and LLamaAttention), only difference between llama2 and llama3 is the model inputs
These are for the newer transformer versions 4.39 to 4.42
"""
class MLP3RewriteRule(AttentionRewriteRule): # same logic as attention layer
    FUNCTION_KEYWORD = "LlamaMLP"
    PACKAGE_NAME = "transformers"
    _version_controller = function_rule.VersionController()

    def __init__(self, opset=onnxscript.opset18):
        super().__init__(opset)

    @_version_controller.register_version(min_version="4.39", max_version="4.42")
    def _optimize_mlp_layer(self, function: ir.Function) -> ir.Function:
        print("Optimizing MLP layer for function:", function.name)
        if len(function.inputs) != 5 and len(function.inputs) != 6:
            raise function_rule.FunctionRewriteError(
                f"Unexpected number of inputs. Expected 5 or 6, got {len(function.inputs)}."
            )
        op = onnxscript.opset18

        def optimized_mlp(
            sym_size_int,
            input_tensor,
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
        ):

            gate_proj_weight_t = op.Transpose(gate_proj_weight, perm=[1, 0])
            up_proj_weight_t = op.Transpose(up_proj_weight, perm=[1, 0])
            down_proj_weight_t = op.Transpose(down_proj_weight, perm=[1, 0])

            gate_proj_output = op.MatMul(input_tensor, gate_proj_weight_t)
            up_proj_output = op.MatMul(input_tensor, up_proj_weight_t)

            act_fn_output = op.Sigmoid(gate_proj_output)
            gate_times_act_output = op.Mul(gate_proj_output, act_fn_output)

            mul_output = op.Mul(gate_times_act_output, up_proj_output)
            down_proj_output = op.MatMul(mul_output, down_proj_weight_t)

            return down_proj_output # must return just one which is down proj

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(optimized_mlp).to_function_proto()
        print("Returning optimized MLP function proto")
        return ir.serde.deserialize_function(function_proto)


#llama3b attention layer rewrite rule
class GQALlama3RewriteRule(AttentionRewriteRule):
    FUNCTION_KEYWORD = "LlamaAttention"
    PACKAGE_NAME = "transformers"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version(min_version="4.39", max_version="4.42")
    def _fusion_with_4d_cache_12_inp(self, function: ir.Function) -> ir.Function:
        print("Applying fusion with 4D cache for function 12 inputs:", function.name)
        if len(function.inputs) != 12:
            raise function_rule.FunctionRewriteError(
                f"Unexpected number of inputs. Expected 12, got {len(function.inputs)}."
            )
        attn_size_config = self.infer_attn_size_config(function)
        op = onnxscript.opset18
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        def gqa(
            sym_size_int,
            hidden_states,
            sym_size_int_1,
            position_ids,
            key_states,
            value_states,
            attention_mask,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            inv_freq,
            o_proj_weight,
        ):
            q = op.MatMul(hidden_states, op.Transpose(q_proj_weight, [1, 0]))
            k = op.MatMul(hidden_states, op.Transpose(k_proj_weight, [1, 0]))
            v = op.MatMul(hidden_states, op.Transpose(v_proj_weight, [1, 0]))
            # Process the attention mask to derive seqlens_k and total_seq_lengths
            path = op.Shape(attention_mask)
            path2 = op.Gather(path, indices=[1], axis=0)
            total_seq_lengths = op.Cast(path2, to=onnx.TensorProto.INT32)

            temp = op.ReduceSum(attention_mask, axes=[1])
            temp2 = op.Sub(temp, op.CastLike(op.Constant(value_int=1), temp))
            seqlens_k = op.Cast(temp2, to=onnx.TensorProto.INT32)
            # Apply GroupQueryAttention
            gqa_output, present_key, present_value = msft_op.GroupQueryAttention(
                q,
                k,
                v,
                key_states,
                value_states,
                seqlens_k,
                total_seq_lengths,
                inv_freq,
                inv_freq,
                kv_num_heads=attn_size_config.num_key_value_heads,
                num_heads=attn_size_config.num_attention_heads,
                do_rotary=True,
                rotary_interleaved=False,
            )
            attn_output = op.MatMul(gqa_output, o_proj_weight)
            return present_value, present_key, attn_output
        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(gqa).to_function_proto()
        return ir.serde.deserialize_function(function_proto)

    @_version_controller.register_version(min_version="4.39", max_version="4.42")
    def _fusion_with_2d_cache_12_inp(self, function: ir.Function) -> ir.Function:
        print("Applying fusion with 2D cache for function 12 inputs:", function.name)

        if len(function.inputs) != 12:
            raise function_rule.FunctionRewriteError(
                f"Unexpected number of inputs. Expected 12, got {len(function.inputs)}."
            )
        attn_size_config = self.infer_attn_size_config(function)

        op = onnxscript.opset18
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        def gqa(
            sym_size_int,
            hidden_states,
            sym_size_int_1,
            position_ids,
            key_states,
            value_states,
            attention_mask,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            inv_freq,
            o_proj_weight,
        ):
            # Transpose the projection weights
            q = op.MatMul(hidden_states, op.Transpose(q_proj_weight, [1, 0]))
            k = op.MatMul(hidden_states, op.Transpose(k_proj_weight, [1, 0]))
            v = op.MatMul(hidden_states, op.Transpose(v_proj_weight, [1, 0]))

            # Process the attention mask to derive seqlens_k and total_seq_lengths
            path = op.Shape(attention_mask)
            path2 = op.Gather(path, indices=[1], axis=0)
            total_seq_lengths = op.Cast(path2, to=onnx.TensorProto.INT32)

            temp = op.ReduceSum(attention_mask, axes=[1])
            temp2 = op.Sub(temp, op.CastLike(op.Constant(value_int=1), temp))
            seqlens_k = op.Cast(temp2, to=onnx.TensorProto.INT32)

            # Apply GroupQueryAttention
            gqa_output, present_key, present_value = msft_op.GroupQueryAttention(
                q,
                k,
                v,
                key_states,
                value_states,
                seqlens_k,
                total_seq_lengths,
                inv_freq,
                inv_freq,
                kv_num_heads=attn_size_config.num_key_value_heads,
                num_heads=attn_size_config.num_attention_heads,
                do_rotary=True,
                rotary_interleaved=False,
            )
            attn_output = op.MatMul(gqa_output, o_proj_weight)

            return present_value, present_key, attn_output

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(gqa).to_function_proto()
        print("Returning new function proto")
        return ir.serde.deserialize_function(function_proto)


# first atttention
class GQALlama3RewriteRuleFirstAttention(AttentionRewriteRule): # same logic as attention layer
    FUNCTION_KEYWORD = "LlamaAttention"
    PACKAGE_NAME = "transformers"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version(min_version="4.39", max_version="4.42")
    def _fusion_with_2d_cache_11_inp(self, function: ir.Function) -> ir.Function:
        print("Applying fusion with 2D cache for function 11 inputs:", function.name)
        attn_size_config = self.infer_attn_size_config(function)

        if len(function.inputs) != 11:
            raise function_rule.FunctionRewriteError(
                f"Unexpected number of inputs. Expected 11, got {len(function.inputs)}."
            )

        op = onnxscript.opset18
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        def gqa(
            sym_size_int,
            hidden_states,
            position_id,
            key_states,
            value_states,
            attention_mask,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            inv_freq,
            o_proj_weight,
        ):
            q = op.MatMul(hidden_states, op.Transpose(q_proj_weight, [1, 0]))
            k = op.MatMul(hidden_states, op.Transpose(k_proj_weight, [1, 0]))
            v = op.MatMul(hidden_states, op.Transpose(v_proj_weight, [1, 0]))

            # from attention mask downwards, reversed, path and temp denote both left and right nodes

            path = op.Shape(attention_mask)
            path2 = op.Gather(path, 1, axis = 0)
            total_seq_lengths = op.Cast(path2, to=onnx.TensorProto.INT32) # <-

            temp = op.ReduceSum(attention_mask, [1])
            temp2 = op.Sub(temp, [1])
            seqlens_k = op.Cast(temp2, to=onnx.TensorProto.INT32) # <--

            gqa_output, present_key, present_value, _ = msft_op.GroupQueryAttention(
                q,
                k,
                v,
                key_states,
                value_states,
                seqlens_k,
                total_seq_lengths,
                inv_freq,
                inv_freq,
                kv_num_heads=attn_size_config.num_key_value_heads,
                num_heads=attn_size_config.num_attention_heads,
                do_rotary = True,
                rotary_interleaved = False,

            )
            attn_output = op.MatMul(gqa_output, o_proj_weight)
            return _, present_key, present_value, attn_output # for 11 inputs, we return 4 outputs but find the correct output

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(gqa).to_function_proto()
        print("Returning new function proto for 2D cache")
        return ir.serde.deserialize_function(function_proto)


"""
Implementations of llama2 rules (MLP and LLamaAttention Layers), only difference between llama2 and llama3 is the model inputs
These are for the newer transformer versions 4.39 to 4.42

"""

class MLPRewriteRule(AttentionRewriteRule): # same logic as attention layer
    FUNCTION_KEYWORD = "LlamaMLP"
    PACKAGE_NAME = "transformers"
    _version_controller = function_rule.VersionController()

    def __init__(self, opset=onnxscript.opset18):
        super().__init__(opset)

    @_version_controller.register_version(min_version="4.39", max_version="4.42")
    def _optimize_mlp_layer(self, function: ir.Function) -> ir.Function:
        print("Optimizing MLP layer for function:", function.name)
        if len(function.inputs) != 5 and len(function.inputs) != 6:
            raise function_rule.FunctionRewriteError(
                f"Unexpected number of inputs. Expected 6, got {len(function.inputs)}."
            )

        op = onnxscript.opset18

        def optimized_mlp(
            sym_size_int_2,
            sym_size_int,
            input_tensor,
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
        ):

            gate_proj_weight_t = op.Transpose(gate_proj_weight, perm=[1, 0])
            up_proj_weight_t = op.Transpose(up_proj_weight, perm=[1, 0])

            down_proj_weight_t = op.Transpose(down_proj_weight, perm=[1, 0])
            gate_proj_output = op.MatMul(input_tensor, gate_proj_weight_t)

            up_proj_output = op.MatMul(input_tensor, up_proj_weight_t)
            act_fn_output = op.Sigmoid(gate_proj_output)

            gate_times_act_output = op.Mul(gate_proj_output, act_fn_output)
            mul_output = op.Mul(gate_times_act_output, up_proj_output)

            down_proj_output = op.MatMul(mul_output, down_proj_weight_t)
            return down_proj_output # must return just one which is down proj

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(optimized_mlp).to_function_proto()
        print("Returning optimized MLP function proto")
        return ir.serde.deserialize_function(function_proto)


# first attention
class GQALlamaRewriteRuleFirstAttention(AttentionRewriteRule):
    FUNCTION_KEYWORD = "LlamaAttention"
    PACKAGE_NAME = "transformers"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version(min_version="4.39", max_version="4.42")
    def _fusion_with_2d_cache_12_inp(self, function: ir.Function) -> ir.Function:
        print("Applying fusion with 2D cache for function 12 inputs:", function.name)
        attn_size_config = self.infer_attn_size_config(function)
        # please refer to model for len of input size
        if len(function.inputs) != 12:
            raise function_rule.FunctionRewriteError(
                f"Unexpected number of inputs. Expected 12, got {len(function.inputs)}."
            )

        op = onnxscript.opset18
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        def gqa(
            sym_size_int,
            hidden_states,
            position_id,
            key_states,
            value_states,
            attention_mask,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            inv_freq,
            o_proj_weight,

        ):

            q = op.MatMul(hidden_states, op.Transpose(q_proj_weight, [1, 0]))
            k = op.MatMul(hidden_states, op.Transpose(k_proj_weight, [1, 0]))
            v = op.MatMul(hidden_states, op.Transpose(v_proj_weight, [1, 0]))

            # from attention mask downwards, reversed, path and temp denote both left and right nodes

            path = op.Shape(attention_mask)
            path2 = op.Gather(path, 1, axis = 0)
            total_seq_lengths = op.Cast(path2, to=onnx.TensorProto.INT32) # <-

            temp = op.ReduceSum(attention_mask, [1])
            temp2 = op.Sub(temp, [1])
            seqlens_k = op.Cast(temp2, to=onnx.TensorProto.INT32) # <--

            gqa_output, present_key, present_value, _ = msft_op.GroupQueryAttention(
                q,
                k,
                v,
                key_states,
                value_states,
                seqlens_k,
                total_seq_lengths,
                inv_freq,
                inv_freq,
                kv_num_heads=attn_size_config.num_key_value_heads,
                num_heads=attn_size_config.num_attention_heads,
                do_rotary = True,
                rotary_interleaved = False,

            )
            attn_output = op.MatMul(gqa_output, o_proj_weight)

            return _, present_key, present_value, attn_output # for 12 inputs, we return 4 outputs but find the correct output

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(gqa).to_function_proto()
        print("Returning new function proto for 2D cache")
        return ir.serde.deserialize_function(function_proto)


class GQALlamaRewriteRule(AttentionRewriteRule):
    FUNCTION_KEYWORD = "LlamaAttention"
    PACKAGE_NAME = "transformers"
    _version_controller = function_rule.VersionController()

    @_version_controller.register_version(min_version="4.39", max_version="4.42")
    def _fusion_with_4d_cache_13_inp(self, function: ir.Function) -> ir.Function:
        print("Applying fusion with 4D cache for function 12 inputs:", function.name)
        if len(function.inputs) != 13:
            raise function_rule.FunctionRewriteError(
                f"Unexpected number of inputs. Expected 13, got {len(function.inputs)}."
            )
        attn_size_config = self.infer_attn_size_config(function)
        op = onnxscript.opset18
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        def gqa(
            sym_size_int_2,
            sym_size_int,
            hidden_states,
            sym_size_int_1,
            position_ids,
            key_states,
            value_states,
            attention_mask,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            inv_freq,
            o_proj_weight,
        ):
            q = op.MatMul(hidden_states, op.Transpose(q_proj_weight, [1, 0]))
            k = op.MatMul(hidden_states, op.Transpose(k_proj_weight, [1, 0]))
            v = op.MatMul(hidden_states, op.Transpose(v_proj_weight, [1, 0]))
            # Process the attention mask to derive seqlens_k and total_seq_lengths
            path = op.Shape(attention_mask)
            path2 = op.Gather(path, indices=[1], axis=0)
            total_seq_lengths = op.Cast(path2, to=onnx.TensorProto.INT32)

            temp = op.ReduceSum(attention_mask, axes=[1])
            temp2 = op.Sub(temp, op.CastLike(op.Constant(value_int=1), temp))
            seqlens_k = op.Cast(temp2, to=onnx.TensorProto.INT32)
            # Apply GroupQueryAttention
            gqa_output, present_key, present_value = msft_op.GroupQueryAttention(
                q,
                k,
                v,
                key_states,
                value_states,
                seqlens_k,
                total_seq_lengths,
                inv_freq,
                inv_freq,
                kv_num_heads=attn_size_config.num_key_value_heads,
                num_heads=attn_size_config.num_attention_heads,
                do_rotary=True,
                rotary_interleaved=False,
            )
            attn_output = op.MatMul(gqa_output, o_proj_weight)
            return present_value, present_key, attn_output
        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(gqa).to_function_proto()
        return ir.serde.deserialize_function(function_proto)

    @_version_controller.register_version(min_version="4.39", max_version="4.42")
    def _fusion_with_2d_cache_12_inp(self, function: ir.Function) -> ir.Function:
        print("Applying fusion with 2D cache for function 12 inputs:", function.name)

        if len(function.inputs) != 12:
            raise function_rule.FunctionRewriteError(
                f"Unexpected number of inputs. Expected 12, got {len(function.inputs)}."
            )
        attn_size_config = self.infer_attn_size_config(function)

        op = onnxscript.opset18
        msft_op = onnxscript.values.Opset("com.microsoft", 1)

        def gqa(
            sym_size_int_2,
            sym_size_int,
            hidden_states,
            sym_size_int_1,
            position_ids,
            key_states,
            value_states,
            attention_mask,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            inv_freq,
            o_proj_weight,
        ):
            # Transpose the projection weights
            q = op.MatMul(hidden_states, op.Transpose(q_proj_weight, [1, 0]))
            k = op.MatMul(hidden_states, op.Transpose(k_proj_weight, [1, 0]))
            v = op.MatMul(hidden_states, op.Transpose(v_proj_weight, [1, 0]))

            # Process the attention mask to derive seqlens_k and total_seq_lengths
            path = op.Shape(attention_mask)
            path2 = op.Gather(path, indices=[1], axis=0)
            total_seq_lengths = op.Cast(path2, to=onnx.TensorProto.INT32)

            temp = op.ReduceSum(attention_mask, axes=[1])
            temp2 = op.Sub(temp, op.CastLike(op.Constant(value_int=1), temp))
            seqlens_k = op.Cast(temp2, to=onnx.TensorProto.INT32)

            # Apply GroupQueryAttention
            gqa_output, present_key, present_value = msft_op.GroupQueryAttention(
                q,
                k,
                v,
                key_states,
                value_states,
                seqlens_k,
                total_seq_lengths,
                inv_freq,
                inv_freq,
                kv_num_heads=attn_size_config.num_key_value_heads,
                num_heads=attn_size_config.num_attention_heads,
                do_rotary=True,
                rotary_interleaved=False,
            )

            # Compute the final attention output
            attn_output = op.MatMul(gqa_output, o_proj_weight)

            return present_value, present_key, attn_output

        function_proto = onnxscript.script(default_opset=onnxscript.opset18)(gqa).to_function_proto()
        print("Returning new function proto")
        return ir.serde.deserialize_function(function_proto)
