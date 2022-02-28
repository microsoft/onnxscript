from onnxscript.types import FLOAT, INT64

# simple loop
def sumprod (x : FLOAT['N'], N : INT64) -> (FLOAT['N'], FLOAT['N']) :
   sum = onnx.Identity(x)
   prod = onnx.Identity(x)
   for i in range(N):
      sum = sum + x
      prod = prod * x
   return sum, prod

# loop: loop-bound as an expression
def sumprod2 (x : FLOAT['N'], N : INT64) -> (FLOAT['N'], FLOAT['N']) :
   sum = onnx.Identity(x)
   prod = onnx.Identity(x)
   for i in range(2*N+1):
      sum = sum + x
      prod = prod * x
   return sum, prod