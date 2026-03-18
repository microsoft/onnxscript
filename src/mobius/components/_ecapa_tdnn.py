# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ECAPA-TDNN speaker encoder components for Qwen3-TTS.

Implements the "Emphasized Channel Attention, Propagation and Aggregation
in TDNN Based Speaker Verification" architecture (arXiv:2005.07143).

Components:
  - TimeDelayNetBlock: Conv1d (reflect padding) + ReLU
  - Res2NetBlock: Multi-scale feature extraction with scale-way split
  - SqueezeExcitationBlock: Channel attention via global mean pooling
  - SqueezeExcitationRes2NetBlock: TDNN → Res2Net → TDNN → SE + residual
  - AttentiveStatisticsPooling: Attention-weighted mean/std pooling
  - SpeakerEncoder: Full ECAPA-TDNN speaker encoder

Reference: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
HuggingFace class: Qwen3TTSSpeakerEncoder
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig

if TYPE_CHECKING:
    import onnx_ir as ir


def _reflect_pad_1d(op: builder.OpBuilder, x, padding: int):
    """Apply reflect padding along the last dimension (time axis).

    ONNX Conv does not support reflect padding natively, so we use
    an explicit Pad op before Conv.

    Args:
        op: ONNX op builder.
        x: Input tensor (batch, channels, time).
        padding: Number of positions to pad on each side.
    """
    if padding == 0:
        return x
    # Pad format: [dim0_begin, dim1_begin, dim2_begin, dim0_end, dim1_end, dim2_end]
    pads = op.Constant(value_ints=[0, 0, padding, 0, 0, padding])
    return op.Pad(x, pads, mode="reflect")


class TimeDelayNetBlock(nn.Module):
    """Conv1d with reflect padding + ReLU activation.

    HuggingFace class: ``TimeDelayNetBlock``

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor for the convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        # "same" padding: (kernel_size - 1) * dilation // 2 on each side
        self._padding = (kernel_size - 1) * dilation // 2
        self.conv = _TDNNConv1d(in_channels, out_channels, kernel_size, dilation)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # hidden_states: (batch, channels, time)
        hidden_states = _reflect_pad_1d(op, hidden_states, self._padding)
        hidden_states = self.conv(op, hidden_states)
        return op.Relu(hidden_states)


class _TDNNConv1d(nn.Module):
    """Conv1d without padding (padding done externally via reflect Pad)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        self.weight = nn.Parameter([out_channels, in_channels, kernel_size])
        self.bias = nn.Parameter([out_channels])
        self._kernel_size = kernel_size
        self._dilation = dilation

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x is already padded by reflect_pad_1d
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel_size],
            strides=[1],
            dilations=[self._dilation],
            pads=[0, 0],
        )


class Res2NetBlock(nn.Module):
    """Multi-scale feature extraction using split + accumulate pattern.

    Splits input into ``scale`` chunks along channel dim. Chunk 0 passes
    through unchanged. Chunk i (i>0) is processed by TDNN(chunk[i] + out[i-1]).
    All chunks are concatenated at the end.

    HuggingFace class: ``Res2NetBlock``

    Args:
        in_channels: Total input channels (must be divisible by scale).
        out_channels: Total output channels (must be divisible by scale).
        scale: Number of splits for multi-scale processing.
        kernel_size: Kernel size for the internal TDNN blocks.
        dilation: Dilation factor for the internal TDNN blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        # scale-1 TDNN blocks (first chunk passes through)
        self.blocks = nn.ModuleList(
            [
                TimeDelayNetBlock(in_channel, hidden_channel, kernel_size, dilation)
                for _ in range(scale - 1)
            ]
        )
        self._scale = scale
        self._in_channel = in_channel

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # hidden_states: (batch, channels, time)
        # Split into scale chunks along channel dim
        # Use Slice instead of Split since scale varies dynamically
        outputs = []
        output_part = None
        for i in range(self._scale):
            # Slice chunk i: channels [i*in_channel : (i+1)*in_channel]
            start = i * self._in_channel
            end = start + self._in_channel
            hidden_part = op.Slice(
                hidden_states,
                op.Constant(value_ints=[start]),
                op.Constant(value_ints=[end]),
                op.Constant(value_ints=[1]),  # axis=1
            )
            if i == 0:
                # First chunk passes through unchanged
                output_part = hidden_part
            elif i == 1:
                # Second chunk: just apply TDNN
                output_part = self.blocks[i - 1](op, hidden_part)
            else:
                # Subsequent chunks: add previous output, then apply TDNN
                output_part = self.blocks[i - 1](op, op.Add(hidden_part, output_part))
            outputs.append(output_part)

        # Concatenate all chunks along channel dim
        return op.Concat(*outputs, axis=1)


class SqueezeExcitationBlock(nn.Module):
    """Channel attention via global mean pooling + 2xConv1d + sigmoid.

    Computes channel-wise importance weights:
    mean(x, dim=time) → Conv1d → ReLU → Conv1d → Sigmoid → x * weights

    HuggingFace class: ``SqueezeExcitationBlock``

    Args:
        in_channels: Number of input channels.
        se_channels: Bottleneck channels in squeeze path.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, se_channels: int, out_channels: int):
        super().__init__()
        # 1x1 convs (kernel=1, no padding needed)
        self.conv1 = _SEConv1d(in_channels, se_channels)
        self.conv2 = _SEConv1d(se_channels, out_channels)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # hidden_states: (batch, channels, time)
        # Global mean pooling over time dimension → (batch, channels, 1)
        hidden_states_mean = op.ReduceMean(hidden_states, [2], keepdims=True)

        # Squeeze path: Conv1d → ReLU → Conv1d → Sigmoid
        hidden_states_mean = op.Relu(self.conv1(op, hidden_states_mean))
        hidden_states_mean = op.Sigmoid(self.conv2(op, hidden_states_mean))

        # Channel-wise gating
        return op.Mul(hidden_states, hidden_states_mean)


class _SEConv1d(nn.Module):
    """1x1 Conv1d for squeeze-excitation (no padding)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter([out_channels, in_channels, 1])
        self.bias = nn.Parameter([out_channels])

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[1],
            strides=[1],
            pads=[0, 0],
        )


class SqueezeExcitationRes2NetBlock(nn.Module):
    """ECAPA-TDNN building block: TDNN → Res2Net → TDNN → SE + residual.

    HuggingFace class: ``SqueezeExcitationRes2NetBlock``

    Args:
        in_channels: Input/output channels (residual connection requires equal).
        out_channels: Output channels (should equal in_channels for residual).
        res2net_scale: Scale factor for Res2NetBlock.
        se_channels: Bottleneck channels for SqueezeExcitationBlock.
        kernel_size: Kernel size for Res2Net TDNN blocks.
        dilation: Dilation for Res2Net TDNN blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res2net_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.tdnn1 = TimeDelayNetBlock(in_channels, out_channels, kernel_size=1, dilation=1)
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TimeDelayNetBlock(out_channels, out_channels, kernel_size=1, dilation=1)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, op: builder.OpBuilder, hidden_state: ir.Value):
        # hidden_state: (batch, channels, time)
        residual = hidden_state

        hidden_state = self.tdnn1(op, hidden_state)
        hidden_state = self.res2net_block(op, hidden_state)
        hidden_state = self.tdnn2(op, hidden_state)
        hidden_state = self.se_block(op, hidden_state)

        # Residual connection
        return op.Add(hidden_state, residual)


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling: attention-weighted mean + std.

    Computes attention over time frames using (hidden, mean, std) features,
    then produces weighted mean and std as the output embedding.

    HuggingFace class: ``AttentiveStatisticsPooling``

    Args:
        channels: Number of input channels.
        attention_channels: Hidden dimension for attention computation.
    """

    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        self._eps = 1e-12
        # Attention network: TDNN on (hidden, mean, std) concatenation
        self.tdnn = TimeDelayNetBlock(
            channels * 3, attention_channels, kernel_size=1, dilation=1
        )
        # Conv1d to produce per-channel attention weights
        self.conv = _SEConv1d(attention_channels, channels)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # hidden_states: (batch, channels, time)
        seq_length = op.Shape(hidden_states, start=2, end=3)  # (1,)
        seq_length_float = op.Cast(seq_length, to=1)  # float32

        # Compute global mean and std for temporal context
        # mean: (batch, channels, 1)
        total = op.ReduceSum(hidden_states, [2], keepdims=True)
        mean = op.Div(total, seq_length_float)

        # std: (batch, channels, 1)
        diff = op.Sub(hidden_states, mean)
        variance = op.Div(
            op.ReduceSum(op.Mul(diff, diff), [2], keepdims=True),
            seq_length_float,
        )
        eps_const = op.Constant(value_float=self._eps)
        std = op.Sqrt(op.Add(variance, eps_const))

        # Expand mean and std to (batch, channels, time) for concatenation
        mean_expanded = op.Expand(mean, op.Shape(hidden_states))
        std_expanded = op.Expand(std, op.Shape(hidden_states))

        # Concatenate (hidden, mean, std) → (batch, 3*channels, time)
        attention_input = op.Concat(hidden_states, mean_expanded, std_expanded, axis=1)

        # Attention computation: TDNN → Tanh → Conv1d → Softmax
        attention = self.tdnn(op, attention_input)
        attention = op.Tanh(attention)
        attention = self.conv(op, attention)  # (batch, channels, time)
        attention = op.Softmax(attention, axis=2)  # Softmax over time

        # Weighted statistics
        # Weighted mean: sum(attention * hidden_states, dim=time)
        weighted = op.Mul(hidden_states, attention)
        w_mean = op.ReduceSum(weighted, [2], keepdims=False)  # (batch, channels)

        # Weighted std: sqrt(sum(attention * (x - mean)^2, dim=time))
        w_diff = op.Sub(hidden_states, op.Unsqueeze(w_mean, [2]))
        w_var = op.ReduceSum(op.Mul(attention, op.Mul(w_diff, w_diff)), [2], keepdims=False)
        w_std = op.Sqrt(op.Add(w_var, eps_const))  # (batch, channels)

        # Concatenate mean and std → (batch, 2*channels)
        pooled_stats = op.Concat(w_mean, w_std, axis=1)
        # Add trailing dim → (batch, 2*channels, 1) for FC conv
        return op.Unsqueeze(pooled_stats, [2])


class SpeakerEncoder(nn.Module):
    """ECAPA-TDNN speaker encoder for Qwen3-TTS voice cloning.

    Extracts a fixed-length speaker embedding from a mel spectrogram.

    Architecture:
      1. Initial TDNN: mel_dim → enc_channels[0]
      2. N-1 SERes2Net blocks: enc_channels[i-1] → enc_channels[i]
      3. Multi-layer Feature Aggregation (MFA): concat blocks[1:] → TDNN
      4. Attentive Statistics Pooling → 2xchannels
      5. FC projection → enc_dim

    HuggingFace class: ``Qwen3TTSSpeakerEncoder``

    Args:
        config: Architecture config with speaker encoder fields.
        mel_dim: Input mel spectrogram dimension.
        enc_dim: Output speaker embedding dimension.
        enc_channels: Channel sizes for each layer.
        enc_kernel_sizes: Kernel sizes for each layer.
        enc_dilations: Dilation factors for each layer.
        enc_attention_channels: Attention channels for ASP.
        enc_res2net_scale: Scale factor for Res2Net blocks.
        enc_se_channels: SE bottleneck channels.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        mel_dim: int = 128,
        enc_dim: int = 1024,
        enc_channels: list[int] | None = None,
        enc_kernel_sizes: list[int] | None = None,
        enc_dilations: list[int] | None = None,
        enc_attention_channels: int = 128,
        enc_res2net_scale: int = 8,
        enc_se_channels: int = 128,
    ):
        super().__init__()
        if enc_channels is None:
            enc_channels = [512, 512, 512, 512, 1536]
        if enc_kernel_sizes is None:
            enc_kernel_sizes = [5, 3, 3, 3, 1]
        if enc_dilations is None:
            enc_dilations = [1, 2, 3, 4, 1]

        self._num_blocks = len(enc_channels)
        self._enc_dim = enc_dim

        # Block 0: Initial TDNN layer
        self.blocks = nn.ModuleList()
        self.blocks.append(
            TimeDelayNetBlock(mel_dim, enc_channels[0], enc_kernel_sizes[0], enc_dilations[0])
        )

        # Blocks 1..N-2: SE-Res2Net layers
        for i in range(1, len(enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    enc_channels[i - 1],
                    enc_channels[i],
                    res2net_scale=enc_res2net_scale,
                    se_channels=enc_se_channels,
                    kernel_size=enc_kernel_sizes[i],
                    dilation=enc_dilations[i],
                )
            )

        # Multi-layer Feature Aggregation (MFA)
        # Concatenates outputs from blocks[1:] along channel dim
        # Last enc_channels entry is the MFA output size
        self.mfa = TimeDelayNetBlock(
            enc_channels[-1],
            enc_channels[-1],
            enc_kernel_sizes[-1],
            enc_dilations[-1],
        )

        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling(
            enc_channels[-1],
            attention_channels=enc_attention_channels,
        )

        # Final projection: (2 * enc_channels[-1]) → enc_dim
        self.fc = _SEConv1d(enc_channels[-1] * 2, enc_dim)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # hidden_states: (batch, time, mel_dim) — mel spectrogram
        # Transpose to (batch, mel_dim, time) for Conv1d processing
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])

        # Process through blocks, collecting outputs for MFA
        hidden_states_list = []
        for i in range(self._num_blocks - 1):
            hidden_states = self.blocks[i](op, hidden_states)
            hidden_states_list.append(hidden_states)

        # Multi-layer Feature Aggregation: concatenate blocks[1:] outputs
        # blocks[0] output is NOT included in MFA (matches HF implementation)
        mfa_input = op.Concat(*hidden_states_list[1:], axis=1)
        hidden_states = self.mfa(op, mfa_input)

        # Attentive Statistics Pooling → (batch, 2*channels, 1)
        hidden_states = self.asp(op, hidden_states)

        # Final projection → (batch, enc_dim, 1)
        hidden_states = self.fc(op, hidden_states)

        # Squeeze time dim → (batch, enc_dim)
        return op.Squeeze(hidden_states, [2])
