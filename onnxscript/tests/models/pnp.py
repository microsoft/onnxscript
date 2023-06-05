from onnx import TensorProto
from onnx.helper import make_tensor

from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import FLOAT, INT64

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

    return indices, roi_shape

@script()
def aggrregate_predictor_output(
    pred: FLOAT["roi_D", "roi_H", "roi_W"],
    start: INT64[3],
    stop: INT64[3],
    aggrregated_pred: FLOAT["D", "H", "W"],
    aggrregated_count: INT64["D", "H", "W"],
) -> (FLOAT["D", "H", "W"], INT64["D", "H", "W"]):
    """
    This function is used to aggregate the predictor output to the final output, count is used to record the number of
    pixels that are aggregated to the final output.
    """
    # r = 3,
    # q = 4,
    # k = indices.shape[-1] = 3
    # indices = [start_z:stop_z, start_y:stop_y, start_x:stop_x
    
    step_ones = op.Constant(value_ints=[1, 1, 1])
    indices, roi_shape = roi_indices_3d(start, stop, step_ones)
    aggrregated_pred_out = op.ScatterND(aggrregated_pred, indices, pred, reduction="add")

    count_ones = op.ConstantOfShape(roi_shape, value=make_tensor("one", TensorProto.INT64, [1], [1]))
    aggrregated_count_out = op.ScatterND(aggrregated_count, indices, count_ones, reduction="add")
    return aggrregated_pred_out, aggrregated_count_out

@script()
def predict_mock(inputs: FLOAT["roi_D", "roi_H", "roi_W"]) -> FLOAT["roi_D", "roi_H", "roi_W"]:
    """
    This function is used to predict the output from the input.
    """
    return op.Add(inputs, inputs)

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
def prepare_for_predictor_batch_size_is_1_script(inputs: FLOAT["D", "H", "W"], slice_g: INT64, slices: INT64["N", 3, 2]) -> (FLOAT["D1", "H1", "W1"], INT64[3], INT64[3]):
    # assume batch size N is 1
    num_win, _, _ = op.Split(op.Shape(slices), num_outputs=3)
    spatial_axes = op.Range(0, 3, 1)
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
def sliding_window_inference(inputs: FLOAT["D", "H", "W"], roi_size: INT64[3]) -> FLOAT["D", "H", "W"]:
    """
    for simplicity, we assume that the step size is the same as the roi size. D is multiple of roi_size in 3 dimensions,
    no overlay, no padding. weight is 1.
    """
    inputs_shape = op.Shape(inputs)
    D, H, W = op.Split(inputs_shape, num_outputs=3)
    roi_D, roi_H, roi_W = op.Split(roi_size, num_outputs=3)
    
    scan_interval = roi_size
    slices = dense_patch_slices_script(inputs_shape, roi_size, scan_interval)
    S_, _, _ = op.Split(op.Shape(slices), num_outputs=3)

    # following code in stft
    zero = op.Constant(value=make_tensor("zero", TensorProto.INT64, [1], [0]))
    S = op.Squeeze(S_, zero)

    aggrregated_pred = op.CastLike(op.ConstantOfShape(op.Shape(inputs)), inputs)
    aggrregated_count = op.CastLike(op.ConstantOfShape(op.Shape(inputs)), roi_size)
    for slice_g in range(S):
        win_data, start, stop = prepare_for_predictor_batch_size_is_1_script(inputs, slice_g, slices)
        pred = predict_mock(win_data)
        aggrregated_pred, aggrregated_count = aggrregate_predictor_output(pred, start, stop, aggrregated_pred, aggrregated_count)
    
    return aggrregated_pred / op.CastLike(aggrregated_count, aggrregated_pred)
