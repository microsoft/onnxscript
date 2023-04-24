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

This test use PyTorch's OpInfo mechanism to generate test cases for each operator.
You may find all OpInfos in https://github.com/pytorch/pytorch/blob/7ec0d6f006fdd2c9b978dc6aa4923144684a3f51/torch/testing/_internal/common_methods_invocations.py#L8804

1. To enable test cases for an operator, add the op name in ops_test_data.py
    1a. If the op is not `trace_only`, add an entry to the
    `OPINFO_FUNCTION_MAPPING_SCRIPTED` map.
    1b. If the op is `trace_only`, add an entry to the
    `OPINFO_FUNCTION_MAPPING_TRACE_ONLY` map.

    The entries are <op_info_name: function> pairs.
2. Edit `EXPECTED_SKIPS_OR_FAILS` and/or `SKIP_SUBTESTS` to skip or xfail tests.
Prefer xfail over skip when possible.
    2a. If a test is now failing because of xpass, because some previous errors
    are now fixed, removed the corresponding xfail.
3. If sample inputs of the OpInfo needs to be adjusted to fit the aten signature, create an input
wrangler function. See `_cat_input_wrangler` for an example.
4. To test different ONNX functions that are registered as overloads of the same
    op, use `duplicate_opinfo` to create new OpInfo with new names and map each
    to one overload.
