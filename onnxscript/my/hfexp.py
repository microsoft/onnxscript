import transformers
from transformers import AutoModel


import torch
import transformers
from transformers import LlamaConfig

# from: https://github.com/pytorch/pytorch/issues/117752

kwargs = {}
device = "cpu"
max_length = 512
batch_size = 1
config = LlamaConfig(num_hidden_layers=1)
model = transformers.AutoModelForCausalLM.from_config(config, **kwargs).to(device)

eval_context = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(device)
# example_inputs = {'input_ids': eval_context, }
model.eval()

# module = torch.jit.trace(model, example_kwarg_inputs=example_inputs)
onnx_program = torch.onnx.dynamo_export(model, eval_context)
onnx_program.save(r"/home/grama/models/llama.onnx")

print('done')