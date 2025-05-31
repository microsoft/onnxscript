# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Fuses Pad nodes into preceding nodes. Supported fusion patterns:
- Pad ∘ Conv          -> Conv
- Pad ∘ ConvInteger   -> ConvInteger

To make some rules possible, we implicitly transform `auto_pad` attribute into its explicit list.
"""

import typing

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter import pattern as orp


def fill_pads_with_axes(
    pads: typing.Sequence[int], axes: typing.Sequence[int], rank: int
) -> typing.List[int]:
    new_pads = [0] * 2 * rank
    N = len(axes)
    for start_idx, axis in enumerate(axes):
        new_pads[axis] = pads[start_idx]
        new_pads[axis + rank] = pads[start_idx + N]
    return new_pads


def read_conv_attributes(ir_conv: ir.Node) -> dict[str, typing.Sequence[int] | str]:
    # Read attributes
    attributes = {}
    if (kernel_shape := ir_conv.attributes.get("kernel_shape", None)) is not None:
        attributes["kernel_shape"] = kernel_shape.as_ints()
    else:
        attributes["kernel_shape"] = ir_conv.inputs[1].shape[2:]
    if (strides := ir_conv.attributes.get("strides", None)) is not None:
        attributes["strides"] = strides.as_ints()
    else:
        attributes["strides"] = [1] * len(ir_conv.inputs[0].shape[2:])
    if (auto_pad := ir_conv.attributes.get("auto_pad", None)) is not None:
        attributes["auto_pad"] = auto_pad.as_string()
    else:
        attributes["auto_pad"] = "NOTSET"
    if (pads := ir_conv.attributes.get("pads", None)) is not None:
        attributes["pads"] = pads.as_ints()
    return attributes


class _FusePadConvBase(orp.RewriteRuleClassBase):
    """Interface for PadConv nodes fusion."""

    def __init__(self, name: str, as_function: bool = False):
        # Remove nodes is set to False to remove unused nodes after the rewrite.
        super().__init__(name=name, remove_nodes=False, as_function=as_function)

    def rewrite(
        self, op: ir.tape.Tape, x: ir.Value, pad: ir.Value, conv: ir.Value
    ) -> ir.Value:
        pnode = pad.producer()
        cnode = conv.producer()

        # Retrieve the padding and axes
        x_rank = len(x.shape)
        pad_pads = pnode.inputs[1].const_value.numpy().tolist()
        if len(pnode.inputs) > 3 and (axes := pnode.inputs[3]) is not None:
            axes = [x if x >= 0 else x_rank + x for x in axes.const_value.numpy()]
        else:
            axes = list(range(x_rank))

        # Fulfill pad_pads in every dimension (filling with zero the other ones)
        pad_pads = fill_pads_with_axes(pad_pads, axes, x_rank)

        # Get only spatial pads
        new_pads = pad_pads[2:x_rank] + pad_pads[x_rank + 2 :]

        # Replace conv pads = new + old
        conv_attr: typing.Mapping[str, ir.Attr] = cnode.attributes.copy()
        if "pads" in conv_attr:
            new_pads = [x + y for x, y in zip(conv_attr["pads"].as_ints(), new_pads)]
        conv_attr["pads"] = ir.convenience.convert_attribute("pads", new_pads)

        return op.op(
            cnode.op_type,
            inputs=(x, *cnode.inputs[1:]),
            attributes=conv_attr,
            domain=cnode.domain,
            name=cnode.name,
        )

    def check(self, context, x: ir.Value, pad: ir.Value, conv: ir.Value) -> orp.MatchResult:
        del context  # Unused
        check_result = orp.MatchResult()
        pnode = pad.producer()
        x_rank = len(x.shape)

        # Pad constraints: attributes
        if (mode := pnode.attributes.get("mode", None)) and mode.as_string() != "constant":
            return check_result.fail(f"{pnode.name} mode must be 'constant'.")

        # Pad constraints: inputs
        if (pads := pnode.inputs[1]).const_value is None:
            return check_result.fail(f"{pads.name} is not a constant/initializer.")
        if len(pnode.inputs) > 2 and (constant_value := pnode.inputs[2]) is not None:
            if constant_value.const_value is None:
                return check_result.fail(
                    f"{constant_value.name} is not a constant/initializer."
                )
            elif constant_value.const_value.numpy().item() != 0:
                return check_result.fail(f"{constant_value.name} must be equal to 0.")
        axes = list(range(x_rank))
        if len(pnode.inputs) > 3 and (axes := pnode.inputs[3]) is not None:
            if axes.const_value is None:
                return check_result.fail(f"{axes.name} is not a constant/initializer.")
            axes_list = [x if x >= 0 else x_rank + x for x in axes.const_value.numpy()]
        else:
            axes_list = list(range(x_rank))

        # Pad constraints: values
        pads_list = fill_pads_with_axes(pads.const_value.numpy(), axes_list, x_rank)
        if np.any(pads_list[:2] + pads_list[x_rank : x_rank + 2]):
            return check_result.fail(f"{pads.name} must be zero in non-spatial dimensions.")

        return check_result


class FusePadConv(_FusePadConvBase):
    """Replaces ``Pad(Conv(x))`` with ``Conv(x)``."""

    def __init__(self, as_function: bool = False):
        super().__init__(name="FusePadConv", as_function=as_function)

    def pattern(self, op: ir.tape.Tape, x: ir.Value) -> ir.Value:
        return op.Conv(
            op.Pad(x, _allow_other_inputs=True, _outputs=["pad"]),
            _allow_other_inputs=True,
            _outputs=["conv"],
        )

    def check(self, context, x: ir.Value, pad: ir.Value, conv: ir.Value) -> orp.MatchResult:
        check_result = super().check(context, x, pad, conv)
        if check_result.reason:
            return check_result

        # Conv constraints: attributes
        cnode = conv.producer()
        if (apad := cnode.attributes.get("auto_pad", None)) and apad.as_string() != "NOTSET":
            return check_result.fail(f"{cnode.name} auto_pad must be 'NOTSET'.")
        return check_result


class FusePadConvInteger(FusePadConv):
    """Replaces ``Pad(ConvInteger(x))`` with ``ConvInteger(x)``."""

    def __init__(self, as_function: bool = False):
        super(FusePadConv, self).__init__(name="FusePadConvInteger", as_function=as_function)

    def pattern(self, op: ir.tape.Tape, x: ir.Value) -> ir.Value:
        return op.ConvInteger(
            op.Pad(x, _allow_other_inputs=True, _outputs=["pad"]),
            _allow_other_inputs=True,
            _outputs=["conv"],
        )


class _NormalizePadFormatBase(orp.RewriteRuleClassBase):
    """Interface to normalize pad attributes in conv nodes."""

    def compute_pads(
        self,
        input_shape: typing.Sequence[int],
        output_shape: typing.Sequence[int],
        attributes: dict[str, typing.Sequence[int] | str],
    ) -> typing.Sequence[int]:
        raise NotImplementedError("Child have to implement this function")

    def rewrite(self, op: ir.tape.Tape, conv: ir.Value, **__) -> ir.Value:
        cnode = conv.producer()

        # Read spatial dimensions and attributes
        input_shape = cnode.inputs[0].shape[2:]
        output_shape = cnode.outputs[0].shape[2:]
        attributes = read_conv_attributes(cnode)

        # Convert auto_pad mode into an explicit list
        pads = self.compute_pads(input_shape, output_shape, attributes)

        # Replace auto_pad, forcing to the explicit list
        conv_attr: typing.Mapping[str, ir.Attr] = cnode.attributes.copy()
        conv_attr["auto_pad"] = ir.convenience.convert_attribute("auto_pad", "NOTSET")
        if any(x != 0 for x in pads):
            conv_attr["pads"] = ir.convenience.convert_attribute("pads", pads)

        return op.op(
            cnode.op_type,
            inputs=cnode.inputs,
            attributes=conv_attr,
            domain=cnode.domain,
            name=cnode.name,
        )

    def check(self, context, conv: ir.Value, **__) -> orp.MatchResult:
        del context
        check_result = orp.MatchResult()

        # Conv constraints: attributes
        cnode = conv.producer()
        auto_pad = cnode.attributes.get("auto_pad", None)
        if auto_pad is None or auto_pad.as_string() == "NOTSET":
            return check_result.fail(f"{cnode.name} auto_pad must be different to 'NOTSET'.")

        # Conv constraints: inputs/outputs
        if cnode.inputs[0].shape is None:
            return check_result.fail(f"Input shapes are not defined on {cnode.name}.")
        if cnode.outputs[0].shape is None:
            return check_result.fail(f"Output shapes are not defined on {cnode.name}.")
        return check_result


class NormalizePadFormatConv(_NormalizePadFormatBase):
    """Convert auto_pad attribute into 'NOTSET' in Conv nodes ."""

    def compute_pads(
        self,
        input_shape: typing.Sequence[int],
        output_shape: typing.Sequence[int],
        attributes: dict[str, typing.Sequence[int] | str],
    ) -> typing.Sequence[int]:
        # Compute pads, following auto_pad/pads attributes
        if attributes["auto_pad"] in ["NOTSET", "VALID"]:
            return attributes.get("pads", [0] * len(input_shape) * 2)

        bottom_pads, top_pads = [], []
        kernel_shape, strides = attributes["kernel_shape"], attributes["strides"]
        for x, y, k, s in zip(input_shape, output_shape, kernel_shape, strides):
            # Compute the output shape and the total padding to apply
            total_pads = max(0, (y - 1) * s + k - x)

            # Depending of mode, apply the padding to the upper or lower part
            pad1 = total_pads // 2
            pad2 = total_pads - pad1
            if attributes["auto_pad"] == "SAME_UPPER":
                bottom_pads.append(pad1)
                top_pads.append(pad2)
            else:
                top_pads.append(pad1)
                bottom_pads.append(pad2)
        return bottom_pads + top_pads

    def pattern(self, op: ir.tape.Tape, x: ir.Value) -> ir.Value:
        return op.Conv(x, _allow_other_inputs=True, _outputs=["conv"])


class NormalizePadFormatConvInteger(NormalizePadFormatConv):
    """Convert auto_pad attribute into 'NOTSET' in ConvInteger nodes ."""

    def pattern(self, op: ir.tape.Tape, x: ir.Value) -> ir.Value:
        return op.ConvInteger(x, _allow_other_inputs=True, _outputs=["conv"])


normalize_pad_format_conv = NormalizePadFormatConv.rule()
normalize_pad_format_conv_integer = NormalizePadFormatConvInteger.rule()
fuse_pad_into_conv = FusePadConv.rule()
fuse_pad_into_conv_integer = FusePadConvInteger.rule()


def fuse_pad_into_conv_rule_set() -> orp.RewriteRuleSet:
    """Returns a set of rewrite rules that fuse Pad nodes into preceding:
    - Conv
    - ConvInteger

    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet(
        [
            normalize_pad_format_conv,
            normalize_pad_format_conv_integer,
            fuse_pad_into_conv,
            fuse_pad_into_conv_integer,
        ]
    )
