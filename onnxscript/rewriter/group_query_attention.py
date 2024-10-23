"""
MultiHeadAttention:

for Q, K, V:
   MatMul
   Reshape to B, S, 32, 64
   Transpose to B, 32, S, 64

Here, 32 is the number of heads and 64 is the head size

Embed Q and K

One of the embeddings (namely ) is also output of layer
and last two axes are transposed for SDPA

"""