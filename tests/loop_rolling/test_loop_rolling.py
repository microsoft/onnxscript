import argparse

import numpy as np

import onnx
import os

from onnxscript import ir
from onnxscript.rewriter import pattern, rewrite
import onnxruntime as onnxrt

from onnxscript.utils.pattern_builder import build_loop_replace_pattern
from onnxscript.utils.pattern_builder import normalize_io_for_loop_rolling
from onnxscript.utils.pattern_builder import LoopBodyTemplate

from onnx import shape_inference

def remove_existing_data_file(filename):
    if os.path.exists(filename):
        print(f"Removing existing data file: {filename}")
        os.remove(filename)

def open_ir(filename):
    print('loading onnx')
    f = onnx.load(filename)
    print('deserializing')
    return ir.serde.deserialize_model(f)

# Handle Parser Stuff
parser = argparse.ArgumentParser(description="A simple argparse example")
parser.add_argument("filename", type=str, help="The name of the input onnx file")
parser.add_argument("patternfilename", type=str, help="The name of the input onnx file that is a single layer")
args = parser.parse_args()

def ort_make_rand_io(filename):

    session = onnxrt.InferenceSession(filename)

    input_name  = session.get_inputs()[0].name
    input_type  = session.get_inputs()[0].type
    input_shape = session.get_inputs()[0].shape

    if input_type == 'tensor(int64)':
        np_type = np.int64
    elif input_type == 'tensor(float)':
        np_type = np.float32
    else:
        raise Exception("unsupported type {input_type}")

    input_data  = np.random.random(input_shape).astype(np_type)

    return ({input_name: input_data}, session.get_outputs())


def ort_run_graph(filename, input_dict, output_name):
    sess_options = onnxrt.SessionOptions()
    sess_options.log_severity_level = 0
    sess_options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_BASIC

    return onnxrt.InferenceSession(filename, sess_options=sess_options).run([output_name], input_dict)

input_dict, outputs = ort_make_rand_io(args.filename)
golden_results      = ort_run_graph(args.filename, input_dict, outputs[0].name)


LoopBody = LoopBodyTemplate(args.patternfilename)

change_layers_to_function_calls = pattern.RewriteRule(
    LoopBody.pattern,
    LoopBody.function_replace
)


print("Find and Replace layers with single layer op.")

mypipeline = onnx.load(args.filename)

print('applying rewrite rule')

mypipeline_layers_replaced = rewrite(
    mypipeline,
    pattern_rewrite_rules = [change_layers_to_function_calls]
)


mypipeline_model = ir.serde.deserialize_model(mypipeline_layers_replaced)
mypipeline_model.functions[LoopBody.function.identifier()] = LoopBody.function
mypipeline_model.graph.opset_imports['loop']=0
mypipeline_layers_replaced = ir.serde.serialize_model(mypipeline_model)


replaced_filename = "replaced_"+args.filename
print(f"Writing Updated Graph to {replaced_filename}")
remove_existing_data_file(replaced_filename+'.data')
onnx.save(mypipeline_layers_replaced,
          replaced_filename, save_as_external_data=True, location= replaced_filename+'.data')

print("Replace Layer Ops with Loop Body")

normalized_graph = normalize_io_for_loop_rolling(mypipeline_model.graph, LoopBody)



model = ir.serde.serialize_model(mypipeline_model)
remove_existing_data_file('normalized.onnx.data')
onnx.save(model, 'normalized.onnx',  save_as_external_data=True, location='normalized.onnx.data')


LoopMatchPattern,nodes = LoopBody.build_function_match_pattern(normalized_graph)


loop_replace_pattern = build_loop_replace_pattern(normalized_graph, LoopBody)


change_function_calls_to_loop = pattern.RewriteRule(
    LoopMatchPattern,
    loop_replace_pattern
)

class AllTracer(pattern.MatchingTracer):
    def __init__(self):
        super().__init__()

    def log(
        self,
        rule: pattern.RewriteRule,
        container: ir.Graph | ir.Function,
        node: ir.Node,
        match_result: pattern.MatchResult,
        status: pattern.MatchStatus,
    ) -> None:
        this_match = pattern.MatchInfo(match_result, node, container, status)
        best_matches = self._best_matches_map[rule]
        best_matches.append(this_match)


tracer = pattern.MatchingTracer()
rewrite_set = pattern.RewriteRuleSet([change_function_calls_to_loop])
count = rewrite_set.apply_to_model(mypipeline_model, verbose=None)
print(f"Count {count}")

# tracer.report()
# for rule in tracer.best_matches_map:
#     matches = tracer.best_matches_map[rule]
#     for match in matches:
#         print(f'Reason: {match.match_result.reason}')
#         print(f'root_node: {match.root_node}')
#         pdb.set_trace()

#loop_added = rewrite (
#    mypipeline_layers_replaced,
#    pattern_rewrite_rules = [change_function_calls_to_loop]
#)

#mypipeline_model.opset_imports.pop('')
mypipeline_model.opset_imports.pop('loop')
mypipeline_model._functions = {}



# scanning for empty domains
# for node in ir.traversal.RecursiveGraphIterator(mypipeline_model.graph):
#     if node.domain == '':
#         print(node)


#mypipeline_model.opset_imports['main'] = 13

loop_added = ir.serde.serialize_model(mypipeline_model)




remove_existing_data_file('loop_added.onnx.data')
onnx.save(loop_added, 'loop_added.onnx', save_as_external_data=True, location='loop_added.onnx.data')

onnx.checker.check_model(loop_added)
loop_added = shape_inference.infer_shapes(loop_added)




transformed_results = ort_run_graph('loop_added.onnx', input_dict, outputs[0].name)


if (np.isclose(transformed_results, golden_results[0], rtol=1e-5, atol=1e-6)).all():
    print("Results Equal!!")
else:
    print("errors found")
    print(transformed_results[0] == golden_results[0])
    print(transformed_results[0])
    print(golden_results[0])

print('done')
