from onnx import TensorProto
from onnx.helper import make_tensor

from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import FLOAT, INT64

from typing import (
    Any,
    Callable,
    Mapping,
)

@script()
def extend_roi_indices(roi_indices: INT64["roi_D", "roi_H", "roi_W", 3], seg_C: INT64) -> INT64[1, "seg_C", "roi_D", "roi_H", "roi_W", 5]:
    """
    This function is used to extend the roi indices to the segment dimension.
    """
    zero = op.Constant(value_int=0)
    one = op.Constant(value_int=1)
    one_ = op.Constant(value_ints=[1])
    roi_D, roi_H, roi_W, _ = op.Split(op.Shape(roi_indices), num_outputs=4)
    shape_1_seg_C_roi_D_H_W_1 = op.Concat(one_, seg_C, roi_D, roi_H, roi_W, one_, axis=0)
    zeros_1_seg_C_D_H_W_1 = op.Cast(op.ConstantOfShape(shape_1_seg_C_roi_D_H_W_1), to=INT64.dtype) # seg_C, roi_D, roi_H, roi_W, 1

    seg_indices = op.Range(zero, seg_C, one)
    seg_indices_unsqueezed = op.Unsqueeze(seg_indices, op.Constant(value_ints=[0, 2, 3, 4, 5])) # 1, seg_C, 1, 1, 1, 1
    seg_indices_1_C_D_H_W_1 = seg_indices_unsqueezed + zeros_1_seg_C_D_H_W_1 # 1, seg_C, roi_D, roi_H, roi_W, 1
    roi_indices_unsqueezed = op.Unsqueeze(roi_indices, op.Constant(value_ints=[0, 1])) # 1, 1, roi_D, roi_H, roi_W, 3
    roi_indices_1_C_D_H_W_3 = roi_indices_unsqueezed + zeros_1_seg_C_D_H_W_1 # 1, seg_C, roi_D, roi_H, roi_W, 3 

    extended_roi_indices = op.Concat(zeros_1_seg_C_D_H_W_1, seg_indices_1_C_D_H_W_1, roi_indices_1_C_D_H_W_3, axis=-1)
    return extended_roi_indices

@script()
def roi_indices_3d(start: INT64[3], stop: INT64[3], step: INT64[3]) -> (INT64["roi_D", "roi_H", "roi_W"], INT64[3]):
    """
    This function is used to generate the indices for a 3d roi window.
    """
    start_d, start_h, start_w = op.Split(start, num_outputs=3)
    stop_d, stop_h, stop_w = op.Split(stop, num_outputs=3)
    step_d, step_h, step_w = op.Split(step, num_outputs=3)
    
    grid_d_0 = op.Range(start_d, stop_d, step_d)
    grid_h_0 = op.Range(start_h, stop_h, step_h)
    grid_w_0 = op.Range(start_w, stop_w, step_w)

    D = op.Shape(grid_d_0)
    H = op.Shape(grid_h_0)
    W = op.Shape(grid_w_0)

    roi_shape = op.Concat(D, H, W, axis=0)
    zeros_D_H_W = op.CastLike(op.ConstantOfShape(roi_shape), start)

    zeros_H_W_D = op.Transpose(zeros_D_H_W, perm=[1, 2, 0])
    grid_d_1 = zeros_H_W_D + grid_d_0
    grid_d = op.Transpose(grid_d_1, perm=[2, 0, 1])

    zeros_D_W_H = op.Transpose(zeros_D_H_W, perm=[0, 2, 1])
    grid_h_1 = zeros_D_W_H + grid_h_0
    grid_h = op.Transpose(grid_h_1, perm=[0, 2, 1])

    grid_w = grid_w_0 + zeros_D_H_W

    indices_seq = op.SequenceConstruct(grid_d, grid_h, grid_w)
    indices = op.ConcatFromSequence(indices_seq, axis=-1, new_axis=1)  # [D, H, W, 3]
    # output_shape = op.Concat(roi_shape, op.Constant(value_ints=[3]), axis=0)
    # indices = op.Reshape(op.Concat(grid_d, grid_h, grid_w, axis=-1), output_shape) # [D, H, W, 3]

    return indices, roi_shape

@script()
def aggrregate_predictor_output(
    pred: FLOAT["N", "seg_C", "roi_D", "roi_H", "roi_W"],
    start: INT64[3],
    stop: INT64[3],
    aggrregated_pred: FLOAT["N", "Seg_C", "D", "H", "W"],
    aggrregated_count: FLOAT["N", 1, "D", "H", "W"],
) -> (FLOAT["N", "Seg_C", "D", "H", "W"], INT64["N", 1, "D", "H", "W"]):
    """
    This function is used to aggregate the predictor output to the final output, count is used to record the number of
    pixels that are aggregated to the final output.
    """
    # r = 3,
    # q = 4,
    # k = indices.shape[-1] = 3
    # indices = [start_z:stop_z, start_y:stop_y, start_x:stop_x
    
    N, seg_C, _, _, _ = op.Split(op.Shape(pred), num_outputs=5)
    step_ones = op.Constant(value_ints=[1, 1, 1])
    indices, roi_shape = roi_indices_3d(start, stop, step_ones) # (D, H, W, 3), [D, H, W]
    indices_w_seg = extend_roi_indices(indices, seg_C) # (1, seg_C, D, H, W, 5)

    aggrregated_pred_out = op.ScatterND(aggrregated_pred, indices_w_seg, pred, reduction="add")

    indices_w_1_N = extend_roi_indices(indices, op.Constant(value_ints=[1]))  # 1, 1, D, H, W, 3
    count_ones = op.ConstantOfShape(op.Concat(op.Constant(value_ints=[1, 1]), roi_shape, axis=0), value=make_tensor("one", TensorProto.INT64, [1], [1])) # 1, 1, D, H, W
    aggrregated_count_out = op.ScatterND(aggrregated_count, indices_w_1_N, count_ones, reduction="add")
    return aggrregated_pred_out, aggrregated_count_out

@script()
def predict_mock(inputs: FLOAT["roi_D", "roi_H", "roi_W"]) -> FLOAT["roi_D", "roi_H", "roi_W"]:
    """
    This function is used to predict the output from the input.
    """
    return op.Add(inputs, inputs)

@script()
def predict_mock_2(inputs: FLOAT["N", 1, "roi_D", "roi_H", "roi_W"]) -> FLOAT["N", "seg_C", "roi_D", "roi_H", "roi_W"]:
    """
    This function is used to predict the output from the input.
    """

    c0 = op.Add(inputs, inputs)
    c1 = op.Add(inputs, inputs)
    c = op.Concat(c0, c1, axis=1)
    return c

@script()
def dense_patch_slices_script(image_size: INT64[3], patch_size: INT64[3], scan_interval: INT64[3]) -> INT64["N", 3, 2]:
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.
    """
    scan_num = op.CastLike(op.Ceil(op.Div(op.Cast(image_size - patch_size, FLOAT.dtype), op.Cast(scan_interval, FLOAT.dtype))), image_size) + 1
    D, H, W = op.Split(scan_num, num_outputs=3)
    step_D, step_H, step_W = op.Split(scan_interval, num_outputs=3)
    zeros_D_H_W = op.CastLike(op.ConstantOfShape(scan_num), image_size)
    
    grid_w_0 = op.Range(0, W * step_W, step_W)
    grid_w = grid_w_0 + zeros_D_H_W

    grid_h_0 = op.Range(0, H * step_H, step_H)
    zeros_D_W_H = op.Transpose(zeros_D_H_W, perm=[0, 2, 1])
    grid_h_1 = zeros_D_W_H + grid_h_0
    grid_h = op.Transpose(grid_h_1, perm=[0, 2, 1])

    grid_d_0 = op.Range(0, D * step_D, step_D)
    zeros_H_W_D = op.Transpose(zeros_D_H_W, perm=[1, 2, 0])
    grid_d_1 = zeros_H_W_D + grid_d_0
    grid_d = op.Transpose(grid_d_1, perm=[2, 0, 1])
                        
    original_grid_start_seq = op.SequenceConstruct(grid_d, grid_h, grid_w)
    original_grid_start_stack = op.ConcatFromSequence(original_grid_start_seq, axis=-1, new_axis=1)  # [D, H, W, 3]
    original_grid_start = op.Reshape(original_grid_start_stack, op.Constant(value_ints=[-1, 3]))  # [D*H*W, 3]
    original_grid_start = op.Min(original_grid_start, image_size - patch_size)
    original_grid_stop = original_grid_start + patch_size
    original_grid_seq = op.SequenceConstruct(original_grid_start, original_grid_stop)
    slices = op.ConcatFromSequence(original_grid_seq, axis=-1, new_axis=1)  # [D*H*W, 3, 2]
    return slices

@script()
def prepare_for_predictor_batch_size_is_1_script(inputs: FLOAT["N", 1, "D", "H", "W"], slice_g: INT64, slices: INT64["N", 3, 2]) -> (FLOAT["N", 1, "roi_D", "roi_H", "roi_W"], INT64[3], INT64[3]):
    # assume batch size N is 1
    num_win, _, _ = op.Split(op.Shape(slices), num_outputs=3)
    spatial_axes = op.Range(2, 5, 1)
    zero_int = op.Constant(value_ints=[0])
    one_int = op.Constant(value_ints=[1])
    two_int = op.Constant(value_ints=[2])
    slice_g_ = slice_g + zero_int
    slice_g_1 = slice_g + one_int
    start_ = op.Slice(slices, op.Concat(slice_g_, zero_int, axis=0), op.Concat(slice_g_1, one_int, axis=0), op.Constant(value_ints=[0, 2]))
    start = op.Reshape(start_, op.Constant(value_ints=[-1]))
    stop_ = op.Slice(slices, op.Concat(slice_g_, one_int, axis=0), op.Concat(slice_g_1, two_int, axis=0), op.Constant(value_ints=[0, 2]))
    stop = op.Reshape(stop_, op.Constant(value_ints=[-1]))
    win_data = op.Slice(inputs, start, stop, spatial_axes)
    # if sw_batch_size > 1:
    #     for idx in op.Range(slice_g + 1, op.Min(slice_g + sw_batch_size, num_win)):
    #         win_data = op.Concat(win_data, op.Slice(inputs, slices[idx, :, 0], slices[idx, :, 1], spatial_axes))
    return win_data, start, stop

@script()
def sliding_window_inference(inputs: FLOAT["N", 1, "D", "H", "W"], roi_size: INT64[3]) -> FLOAT["N", "Seg_C", "D", "H", "W"]:
    """
    for simplicity, we assume that the step size is the same as the roi size. D/H/W are multiple of roi_size in 3 dimensions,
    no overlay, no padding. weight is 1.
    """
    inputs_shape = op.Shape(inputs)
    inputs_spatial_shape = op.Slice(inputs_shape, op.Constant(value_ints=[2]), op.Constant(value_ints=[5]), op.Constant(value_ints=[0])) 
    N, _, D, H, W = op.Split(inputs_shape, num_outputs=5)
    roi_D, roi_H, roi_W = op.Split(roi_size, num_outputs=3)
    
    scan_interval = roi_size
    slices = dense_patch_slices_script(inputs_spatial_shape, roi_size, scan_interval)
    S_, _, _ = op.Split(op.Shape(slices), num_outputs=3)
    zero = op.Constant(value_ints=[0])
    S = op.Squeeze(S_, zero)

    seg_C = op.Constant(value_ints=[2])
    one = op.Constant(value_ints=[1])
    output_shape = op.Concat(one, seg_C, inputs_spatial_shape, axis=0)

    aggrregated_pred = op.CastLike(op.ConstantOfShape(output_shape), inputs)
    aggrregated_count = op.CastLike(op.ConstantOfShape(inputs_shape), roi_size)
    for slice_g in range(S):
        win_data, start, stop = prepare_for_predictor_batch_size_is_1_script(inputs, slice_g, slices)
        pred = predict_mock_2(win_data)
        aggrregated_pred, aggrregated_count = aggrregate_predictor_output(pred, start, stop, aggrregated_pred, aggrregated_count)
    
    return aggrregated_pred / op.CastLike(aggrregated_count, aggrregated_pred)

