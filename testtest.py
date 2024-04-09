import onnx
import onnx.inliner
from pyinstrument import Profiler

from onnxscript import ir
from onnxscript.ir import serde

profiler = Profiler()


model_name = "llama2_0"
model = onnx.load(model_name + ".onnx", load_external_data=False)
# model = onnx.inliner.inline_local_functions(model)
# model = onnx.shape_inference.infer_shapes(model, data_prop=True)

# model = onnxscript imize(model)
# model = onnxscript runtime.rewrite(model)

profiler.start()
m: ir.Model = serde.deserialize_model(model)
profiler.stop()

print("Graph size: ", len(m.graph.nodes))
# inspect(m)
# inspect(m.graph)


profiler.start()
serialized = serde.serialize_model(m)
profiler.stop()

profiler.print()

# onnx.save(serialized, model_name + "_IR.onnx")

# onnxscript t_onnx_proto_equal(serialized, model)
