from onnxscript import ir
import onnx

model_path=[
    "attn_stable_diffusion_unet_0/attn_stable_diffusion_unet_0.onnx",
    "attn_stable_diffusion_unet_1/attn_stable_diffusion_unet_1.onnx",
    "attn_stable_diffusion_unet_without_encoder_hidden_states_0/attn_stable_diffusion_unet_without_encoder_hidden_states_0.onnx",
    "attn_stable_diffusion_unet_without_encoder_hidden_states_1/attn_stable_diffusion_unet_without_encoder_hidden_states_1.onnx"
]

for model_name in model_path:
    model = onnx.load(model_name)
    model_ir = ir.serde.deserialize_model(model)
    model = ir.serde.serialize_model(model_ir)
    onnx.save(model, model_name)
