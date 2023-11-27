import torch
from long_net.attention import DilatedAttention

b, n, d = 128, 7000, 512
h = 2
x = torch.randn(b, n, d).to("cuda")


# create model and data
model = DilatedAttention(d, h, 3, 64).to("cuda")

output = model(x)
print(output.shape)
