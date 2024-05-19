import time

import onnx

from onnxscript import optimizer

# from onnxscript.rewriter.rules import all_rules


def timeit(f, message):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"{message} time: {te-ts}")
        return result

    return timed


load = timeit(onnx.load, "Load")

save = timeit(onnx.save, "Save")

infer = timeit(onnx.shape_inference.infer_shapes, "Infer")

fold_constants = timeit(optimizer.fold_constants, "Fold Constants")

remove_unused = timeit(optimizer.remove_unused_nodes, "Remove Unused")

optimize = timeit(optimizer.optimize, "Optimize")

# rewrite = timeit(all_rules.apply_to_model, "Rewrite")
