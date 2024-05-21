import torch
import onnxruntime
from torch.onnx import (
    _OrtBackend as OrtBackend,
    _OrtBackendOptions as OrtBackendOptions,
    ExportOptions,
)


def _make_aot_ort(dynamic: bool = False) -> tuple:
    export_options = ExportOptions(dynamic_shapes=dynamic)
    options = OrtBackendOptions(export_options=export_options)
    ort_backend = OrtBackend(options=options)
    return ort_backend

class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 10)
        self.activation = torch.nn.ReLU()

    def forward(self, *inputs):
        input = self.linear(inputs[0])
        input = self.activation(input)
        return input

model = Linear()
model.train()
loss_fn = torch.nn.MSELoss()

input = torch.randn((64, 128), requires_grad=True)
labels = torch.randn((64, 10), requires_grad=True)

compiled_model = torch.compile(model, backend=_make_aot_ort())
output = compiled_model(*input)
loss = loss_fn(output, labels)
loss.backward()