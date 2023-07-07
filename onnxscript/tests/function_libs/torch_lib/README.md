# Test op correctness by comparing with PyTorch results.

## Usage

```bash
# All
pytest onnxscript/tests/function_libs/torch_lib/ops_test.py

# To run tests on a specific operator (e.g. torch.ceil):
pytest onnxscript/tests/function_libs/torch_lib/ops_test.py -k ceil

# To run tests on a nn operator (e.g. nn.functional.scaled_dot_product_attention):
pytest onnxscript/tests/function_libs/torch_lib/ops_test.py -k nn_functional_scaled_dot_product_attention
```

## How to add a new operator test

See _usage_ in [ops_test_data.py](./ops_test_data.py)
