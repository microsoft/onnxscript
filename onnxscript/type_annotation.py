
import onnx
import onnxscript

pytype_to_attrtype_map = {
    float: onnx.AttributeProto.FLOAT,
    int: onnx.AttributeProto.INT,
    str: onnx.AttributeProto.STRING,
}


def is_attr(typeinfo):
    return typeinfo in {float, int, str}
    # (typeinfo is float) or (typeinfo is str) or (typeinfo is int)


def is_tensor(typeinfo):
    return hasattr(typeinfo, "to_type_proto")
    # return isinstance(typeinfo, onnxscript.Tensor)  # TODO


def is_valid(typeinfo):
    return is_attr(typeinfo) or is_tensor(typeinfo)
