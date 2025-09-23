# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value"
"""torchvision operators."""

from __future__ import annotations

import warnings

from onnxscript.function_libs.torch_lib.registration import torch_op
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import FLOAT, INT64

_INT64_MAX = 0x7FFFFFFFFFFFFFFF


@torch_op("torchvision::nms")
def torchvision_nms(boxes: FLOAT, scores: FLOAT, iou_threshold: float) -> INT64:
    # boxes: [num_batches, spatial_dimension, 4]
    boxes = op.Unsqueeze(boxes, [0])
    # scores: [num_batches, num_classes, spatial_dimension]
    scores = op.Unsqueeze(scores, [0, 1])
    # nms_out: [num_selected_indices, 3] where each column is [batch_index, class_index, box_index]
    nms_out = op.NonMaxSuppression(boxes, scores, _INT64_MAX, iou_threshold)
    return op.Reshape(op.Slice(nms_out, axes=[1], starts=[2], ends=[3]), [-1])


def _process_batch_indices_for_roi_align(rois):
    # Extract batch indices from the first column (index 0) of rois
    indices = op.Slice(rois, axes=[1], starts=[0], ends=[1])
    indices = op.Squeeze(indices, axes=[1])
    return op.Cast(indices, to=INT64.dtype)


def _process_rois_for_roi_align(rois):
    # Extract roi coordinates from columns 1, 2, 3, 4 (x1, y1, x2, y2)
    return op.Slice(rois, axes=[1], starts=[1], ends=[5])


def _process_sampling_ratio_for_roi_align(sampling_ratio: int):
    if sampling_ratio < 0:
        warnings.warn(
            "ONNX export for RoIAlign with a non-zero sampling_ratio is not supported. "
            "The model will be exported with a sampling_ratio of 0.",
            stacklevel=2,
        )
        sampling_ratio = 0
    return sampling_ratio


@torch_op("torchvision::roi_align")
def torchvision_roi_align(
    input,
    rois,
    spatial_scale: float,
    pooled_height: int,
    pooled_width: int,
    sampling_ratio: int = -1,
    aligned: bool = False,
):
    batch_indices = _process_batch_indices_for_roi_align(rois)
    rois_coords = _process_rois_for_roi_align(rois)
    coordinate_transformation_mode = "half_pixel" if aligned else "output_half_pixel"
    sampling_ratio = _process_sampling_ratio_for_roi_align(sampling_ratio)

    return op.RoiAlign(
        input,
        rois_coords,
        batch_indices,
        coordinate_transformation_mode=coordinate_transformation_mode,
        spatial_scale=spatial_scale,
        output_height=pooled_height,
        output_width=pooled_width,
        sampling_ratio=sampling_ratio,
    )


@torch_op("torchvision::roi_pool")
def torchvision_roi_pool(
    input, rois, spatial_scale: float, pooled_height: int, pooled_width: int
):
    # MaxRoiPool expects rois in format [batch_index, x1, y1, x2, y2]
    return op.MaxRoiPool(
        input,
        rois,
        pooled_shape=(pooled_height, pooled_width),
        spatial_scale=spatial_scale,
    )
