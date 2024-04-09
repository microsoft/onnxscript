## Workflow

### Setup Environment

1. Setup access to vm with GPU.
2. Install PyTorch, ONNX, ONNXScript, etc.
3. Build ONNXRuntime from source, with nvtx profiling enabled.
    ```
    # Under onnxruntime root folder
    ./build.sh --config RelWithDebInfo --parallel 0 --use_cuda --build_wheel --skip_tests --enable_nvtx_profile
    # After build complete
    pip install build/Linux/RelWithDebInfo/dist/onnxruntime_gpu-*.whl
    ```
<!-- 3. (Optional) Install torchbenchmark and related dependencies. -->
<!-- 4. (Optional) git clone ONNXConverter. -->

### Model Optimization

1. Identify model with performance gap from torchbench dashboard.
2. Prepare ONNX model.
    <!-- - (Optional) Run torchbench locally to retrieve exported model.  -->
    - Download ONNX model from [benchmark pipeline](https://dev.azure.com/onnxconverter/ONNXConverter/_build?definitionId=7&_a=summary). Navigate to the latest run, and download selected models from the artifacts.
    - The models should be put under `onnx-script/tools/onnx_models/`. It is **important** to follow the exact folder format, since it is assumed by many analysis tools. It should look like:
        ```
        onnx-script/tools/onnx_models/<model_name>
        ├── dynamo
        │   ├── <model_name>_dynamo.onnx
        │   ├── test_data_set_0
        ├── torchscript
        │   ├── <model_name>_torchscript.onnx
        │   ├── test_data_set_0
        ```

3. Run optimization. Example command to apply optimizations on `stable_diffusion_unet` `dynamo` model and produces `dynamo_ort_rewritten` model under the same folder. `CUDA_VISIBLE_DEVICES` is used to specify GPU device to avoid the device others are using.
    ```
    CUDA_VISIBLE_DEVICES="3" python ort_rewrite.py.py  --model stable_diffusion_unet --model-dir ./onnx_models/ --log-level 20 --fusion
    ```

4. Investigate the root cause of performance gap.
    - Run nsys profiling analysis for detailed performance per operator. For example:
        ```
        # Under onnx-script/tools
        CUDA_VISIBLE_DEVICES="3" python nsys_profile.py --compiler torchscript --compiler dynamo_ort_rewritten --model-dir onnx_models/stable_diffusion_unet/ --iteration 20
        ```
        This script generates a sorted report of per operation performance. (WIP: more features to come including comparison table)
        ```
        INFO:__main__:Running llama_v2_7b_16h_torchscript with CUDAExecutionProvider
        Loading stable_diffusion_unet_torchscript model took 4.818927253130823 seconds.
        Running stable_diffusion_unet_torchscript model took 0.21183541482314466 seconds.
        ========== stable_diffusion_unet_torchscript passed
        Generating '/tmp/nsys-report-11e4.qdstrm'
        [1/1] [========================100%] _torchscript_20_20240408_231410.nsys-rep
        Generated:
            /root/onnx-script/tools/.logs/_torchscript_20_20240408_231410.nsys-rep
        Processing 147374 events: [================================================100%]
        Node Batch- Forward has 1.0 instances and total duration 75.96109084999999 ms
        Node MatMul has 256.0 instances and total duration 19.619553800000002 ms
        Node Add has 265.0 instances and total duration 14.37154445 ms
        Node Transpose has 160.0 instances and total duration 12.79009095 ms
        Node Conv has 66.0 instances and total duration 9.02397235 ms
        Node InstanceNormalization has 61.0 instances and total duration 3.5543551000000004 ms
        Node Mul has 142.0 instances and total duration 3.4500595499999998 ms
        Node QuickGelu has 47.0 instances and total duration 1.8921348500000001 ms
        Node LayerNormalization has 48.0 instances and total duration 1.65210675 ms
        Node Softmax has 32.0 instances and total duration 1.62622545 ms
        Node Gemm has 24.0 instances and total duration 1.3390768 ms
        Node Split has 17.0 instances and total duration 0.84570395 ms
        Node Concat has 14.0 instances and total duration 0.70874125 ms
        Node Reshape has 282.0 instances and total duration 0.5502581 ms
        Node Gelu has 16.0 instances and total duration 0.37270075 ms
        Node Resize has 3.0 instances and total duration 0.29633954999999995 ms
        Node Unsqueeze has 45.0 instances and total duration 0.10421114999999999 ms
        Node Sqrt has 2.0 instances and total duration 0.04319565 ms
        Node Cast has 4.0 instances and total duration 0.0345756 ms
        Node Expand has 1.0 instances and total duration 0.020945400000000003 ms
        Node Div has 1.0 instances and total duration 0.014120299999999999 ms
        Node Cos has 1.0 instances and total duration 0.00973515 ms
        Node Sin has 1.0 instances and total duration 0.007655 ms
        Node Batch- Backward has 1.0 instances and total duration 0.000475 ms
        Total duration: 72.32730190000002 ms
        INFO:__main__:Running stable_diffusion_unet_dynamo_ort_rewritten with CUDAExecutionProvider
        Loading stable_diffusion_unet_dynamo_ort_rewritten model took 3.7717798189260066 seconds.
        Running stable_diffusion_unet_dynamo_ort_rewritten model took 0.20689857872202994 seconds.
        ========== stable_diffusion_unet_dynamo_ort_rewritten passed
        Generating '/tmp/nsys-report-0832.qdstrm'
        [1/1] [========================100%] _dynamo_ort_rewritten_20_20240408_231449.nsys-rep
        Generated:
            /root/onnx-script/tools/.logs/_dynamo_ort_rewritten_20_20240408_231449.nsys-rep
        Processing 138089 events: [================================================100%]
        Node Batch- Forward has 1.0 instances and total duration 71.85810679999999 ms
        Node GroupNorm has 61.0 instances and total duration 17.643069600000004 ms
        Node MatMul has 224.0 instances and total duration 14.10612605 ms
        Node Conv has 66.0 instances and total duration 13.55052805 ms
        Node LayerNormalization has 48.0 instances and total duration 8.91337515 ms
        Node Transpose has 288.0 instances and total duration 4.5095647 ms
        Node Add has 204.0 instances and total duration 4.198386 ms
        Node FusedMatMul has 32.0 instances and total duration 1.17028955 ms
        Node Gemm has 24.0 instances and total duration 0.9632798 ms
        Node Concat has 14.0 instances and total duration 0.7315115 ms
        Node Softmax has 32.0 instances and total duration 0.5601483 ms
        Node Split has 17.0 instances and total duration 0.53355725 ms
        Node Reshape has 224.0 instances and total duration 0.46099255 ms
        Node Gelu has 16.0 instances and total duration 0.39153615 ms
        Node Mul has 17.0 instances and total duration 0.1747685 ms
        Node Unsqueeze has 45.0 instances and total duration 0.10806160000000001 ms
        Node GatherND has 3.0 instances and total duration 0.10457174999999999 ms
        Node Cast has 12.0 instances and total duration 0.1037319 ms
        Node Expand has 1.0 instances and total duration 0.027625499999999997 ms
        Node QuickGelu has 2.0 instances and total duration 0.0184254 ms
        Node Cos has 1.0 instances and total duration 0.0116252 ms
        Node Sin has 1.0 instances and total duration 0.0094251 ms
        Node Batch- Backward has 1.0 instances and total duration 0.0004 ms
        Total duration: 68.2905996 ms
        ```
    - Run benchmark for high level performance metrics. For example:
        ```
        # Under onnx-script/tools
        CUDA_VISIBLE_DEVICES="2" python bench_model.py --model-dir ./onnx_models/stable_diffusion_unet --device cuda -i 20 --compiler torchscript
        ```
        Or omit `--compiler` to run all compilers.
        ```
        # Under onnx-script/tools
        CUDA_VISIBLE_DEVICES="2" python bench_model.py --model-dir ./onnx_models/stable_diffusion_unet --device cuda -i 20
        ```
    - Load and inspect model in Netron.
        - Load `dynamo` model to inspect exported structured graph (optimizer already included).
        - Load `dynamo_ort_rewritten` model to inspect inlined graph after onnxruntime rewriter.

5. Develop optimization code.
    - `onnx-script/onnxscript/optimizer`: Optimizations such as constant folding, inlining, dead code elimination etc.
    - `onnx-script/onnxscript/rewriter`: Pattern based fusions.
    - `onnx-script/onnxscript/rewriter/functions`: Function based fusions.
        - Use function unittest producer tool to create function fusion unittest. Example command to distill 4 unittests for function `LlamaSdpaAttention` from `llama_v2_7b` `dynamo` model. The unittest models are named with prefix `sdpa_llama2`:
            ```
            # Under onnx-script/onnxscript/rewriter/transformers
            CUDA_VISIBLE_DEVICES="3" python tools/function_unittest_producer.py --model-path ../../../tools/onnx_models/llama_v2_7b_16h/dynamo_ort_rewritten/llama_v2_7b_16h_dynamo_ort_rewritten.onnx --function LlamaSdpaAttention --output-dir ../../testing/rewriter/transformers/unittest_models/ --max-outputs 4 --name sdpa_llama2
            ```
        - Create new testcase under `onnx-script/onnxscript/rewriter/transformers` with the generated unittest models.
            ```python
                def test_sdpa_llama2(self):
                    common.test_function_rewrite("sdpa_llama2", 4)
            ```

6. Repeat step 3 to step 5 to verify performance improvement as well as parity after new optimization.
