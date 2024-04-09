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

### Model Optimization

1. Identify model with performance gap from torchbench dashboard.
2. Prepare ONNX model.
    <!-- - (Optional) Run torchbench locally to retrieve exported model.  -->
    - Download ONNX model from [benchmark pipeline](https://dev.azure.com/onnxconverter/ONNXConverter/_build?definitionId=7&_a=summary). Navigate to the latest run, and download selected models from the artifacts.
    - The models should be put under `onnx-script/tools/ort_rewrite_profiling/onnx_models/`. It is **important** to follow the exact folder format, since it is assumed by many analysis tools. It should look like:
        ```
        onnx-script/tools/ort_rewrite_profiling/onnx_models/<model_name>
        ├── dynamo
        │   ├── <model_name>_dynamo.onnx
        │   ├── test_data_set_0
        ├── torchscript
        │   ├── <model_name>_torchscript.onnx
        │   ├── test_data_set_0
        ```

3. Run optimization. Example command to apply optimizations on `stable_diffusion_unet` `dynamo` model and produces `dynamo_ort_rewritten` model under the same folder. `CUDA_VISIBLE_DEVICES` is used to specify GPU device to avoid the device others are using.
    ```
    CUDA_VISIBLE_DEVICES="3" python ort_rewrite.py  --model stable_diffusion_unet --model-dir ./onnx_models/ --log-level 20 --fusion
    ```

4. Investigate the root cause of performance gap.
    - Run nsys profiling analysis for detailed performance per operator. For example:
        ```
        # Under onnx-script/tools
        CUDA_VISIBLE_DEVICES="3" python nsys_profile.py --compiler torchscript --compiler dynamo_ort_rewritten --model-dir onnx_models/stable_diffusion_unet/ --iteration 20
        ```
        This script generates a sorted report of per operation performance. (WIP: more features to come including comparison table)
        ```
        INFO:__main__:Running stable_diffusion_unet_torchscript with CUDAExecutionProvider
        Loading stable_diffusion_unet_torchscript model took 4.721486468333751 seconds.
        Running stable_diffusion_unet_torchscript model took 0.21081756763160228 seconds.
        ========== stable_diffusion_unet_torchscript passed
        Generating '/tmp/nsys-report-1be0.qdstrm'
        [1/1] [========================100%] _torchscript_20_20240409_015235.nsys-rep
        Generated:
            /root/onnx-script/tools/ort_rewriter_profiling/.logs/_torchscript_20_20240409_015235.nsys-rep
        Processing 147386 events: [================================================100%]
        Node Batch- Forward has 1.0 instances and total duration 75.48273005 ms
        Node MatMul has 256.0 instances and total duration 19.3666577 ms
        Node Add has 265.0 instances and total duration 14.356723699999998 ms
        Node Transpose has 160.0 instances and total duration 12.77478685 ms
        Node Conv has 66.0 instances and total duration 9.054065399999999 ms
        Node Mul has 142.0 instances and total duration 3.7753493 ms
        Node InstanceNormalization has 61.0 instances and total duration 3.52974435 ms
        Node QuickGelu has 47.0 instances and total duration 1.88163455 ms
        Node LayerNormalization has 48.0 instances and total duration 1.63459615 ms
        Node Softmax has 32.0 instances and total duration 1.5771648 ms
        Node Gemm has 24.0 instances and total duration 1.3238012 ms
        Node Split has 17.0 instances and total duration 0.8454728499999999 ms
        Node Concat has 14.0 instances and total duration 0.6981809499999999 ms
        Node Reshape has 282.0 instances and total duration 0.52096305 ms
        Node Gelu has 16.0 instances and total duration 0.37213579999999996 ms
        Node Resize has 3.0 instances and total duration 0.29573455000000004 ms
        Node Unsqueeze has 45.0 instances and total duration 0.09547169999999999 ms
        Node Sqrt has 2.0 instances and total duration 0.036865550000000004 ms
        Node Cast has 4.0 instances and total duration 0.0323905 ms
        Node Expand has 1.0 instances and total duration 0.01918515 ms
        Node Div has 1.0 instances and total duration 0.01271035 ms
        Node Cos has 1.0 instances and total duration 0.009415149999999999 ms
        Node Sin has 1.0 instances and total duration 0.0077702 ms
        Node Batch- Backward has 1.0 instances and total duration 0.00051 ms
        Total duration: 72.2208198 ms
        INFO:__main__:Running stable_diffusion_unet_dynamo_ort_rewritten with CUDAExecutionProvider
        Loading stable_diffusion_unet_dynamo_ort_rewritten model took 3.878677878063172 seconds.
        Running stable_diffusion_unet_dynamo_ort_rewritten model took 0.20752966087311506 seconds.
        ========== stable_diffusion_unet_dynamo_ort_rewritten passed
        Generating '/tmp/nsys-report-9e9a.qdstrm'
        [1/1] [========================100%] _dynamo_ort_rewritten_20_20240409_015314.nsys-rep
        Generated:
            /root/onnx-script/tools/ort_rewriter_profiling/.logs/_dynamo_ort_rewritten_20_20240409_015314.nsys-rep
        Processing 138097 events: [================================================100%]
        Node Batch- Forward has 1.0 instances and total duration 72.16954654999999 ms
        Node GroupNorm has 61.0 instances and total duration 17.78595435 ms
        Node MatMul has 224.0 instances and total duration 14.365499199999999 ms
        Node Conv has 66.0 instances and total duration 13.386948199999999 ms
        Node LayerNormalization has 48.0 instances and total duration 8.9019881 ms
        Node Transpose has 288.0 instances and total duration 4.80908445 ms
        Node Add has 204.0 instances and total duration 4.21939795 ms
        Node FusedMatMul has 32.0 instances and total duration 1.1677278999999998 ms
        Node Gemm has 24.0 instances and total duration 0.9268493000000001 ms
        Node Concat has 14.0 instances and total duration 0.6874159000000001 ms
        Node Softmax has 32.0 instances and total duration 0.5510288499999999 ms
        Node Split has 17.0 instances and total duration 0.52731265 ms
        Node Reshape has 224.0 instances and total duration 0.50016315 ms
        Node Gelu has 16.0 instances and total duration 0.38526135 ms
        Node Mul has 17.0 instances and total duration 0.16483735000000002 ms
        Node Unsqueeze has 45.0 instances and total duration 0.1144019 ms
        Node Cast has 12.0 instances and total duration 0.09924124999999999 ms
        Node GatherND has 3.0 instances and total duration 0.09330664999999999 ms
        Node Expand has 1.0 instances and total duration 0.022125250000000003 ms
        Node QuickGelu has 2.0 instances and total duration 0.017725150000000002 ms
        Node Cos has 1.0 instances and total duration 0.010325149999999998 ms
        Node Sin has 1.0 instances and total duration 0.0091601 ms
        Node Batch- Backward has 1.0 instances and total duration 0.000555 ms
        Total duration: 68.74575414999998 ms
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
    - `onnx-script/onnxscript/rewriter/onnxruntime`: Onnxruntime specific pattern based fusions.
    - `onnx-script/onnxscript/rewriter/onnxruntime/transformers`: Onnxruntime specific function based fusions.
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
