import torch
import einops
from einops import rearrange

a = torch.randn(100,64)
b = torch.randn(100,64,10)

a = rearrange(a, "n d -> n 1 d")

print(a.shape)
print((a*b).shape)