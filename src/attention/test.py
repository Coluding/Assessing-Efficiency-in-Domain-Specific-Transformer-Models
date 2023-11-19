import torch


from sliding_window_attention import MultiHeadDilatedLocalAttention

d_model = 512
s = MultiHeadDilatedLocalAttention(d_model=d_model,dilation_rate=5, window_size=49, num_heads=4).to("cuda")
q,k,v = (torch.randn(128, 5000, d_model).to("cuda") for _ in range(3))
att = torch.nn.MultiheadAttention(num_heads=4, embed_dim=d_model).to("cuda")

import utils

utils.track_cuda_memory("custom", att, q,k,v)

