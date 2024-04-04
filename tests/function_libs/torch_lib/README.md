# Test op correctness by comparing with PyTorch results

## Usage

```bash
# All
pytest onnxscript/tests/function_libs/torch_lib/ops_test.py

# To run tests on a specific operator (e.g. torch.ceil):
pytest onnxscript/tests/function_libs/torch_lib/ops_test.py -k ceil

# To run tests on a nn operator (e.g. nn.functional.scaled_dot_product_attention):
pytest onnxscript/tests/function_libs/torch_lib/ops_test.py -k nn_functional_scaled_dot_product_attention
```

### Environment variables

1. Set environment variable `CATCH_ORT_SEGFAULT=1` to catch segmentation faults
in onnxruntime by running the inference sessions in a separate process.
2. Set `CREATE_REPRODUCTION_REPORT=1` to create markdown files for reproduction of errors. E.g.

    ```bash
    CREATE_REPRODUCTION_REPORT=1 python -m pytest onnxscript/tests/function_libs/torch_lib/ops_test.py -k div_mode_int
    ```

## How to add a new operator test

See _usage_ in [ops_test_data.py](./ops_test_data.py)
