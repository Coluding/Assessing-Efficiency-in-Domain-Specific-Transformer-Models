import torch.nn as nn
from xformers.components.attention.local import LocalAttention
from xformers.components.attention.scaled_dot_product import ScaledDotProduct
from xformers.components.multi_head_dispatch import MultiHeadDispatch
from xformers.ops import memory_efficient_attention
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from memory_efficient_attention_pytorch import Attention

import torch
import torch.nn as nn
from utils import evaluate_cuda_memory

b, n, d = 128, 4096, 512
h = 2
x = torch.randn(b, n, d).to("cuda")
mh = nn.MultiheadAttention(embed_dim=d, num_heads=2).to("cuda")
att = LocalAttention(window_size=201).to("cuda")
s = ScaledDotProduct()
#att.requires_input_projection = False   # verbraucht sehr viel
#mh_att = MultiHeadDispatch(attention=s, dim_model=d, num_heads=2).to("cuda")
#att(x,x,x) #374
#mh(x,x,x)

attn = Attention(
    dim = d,
    dim_head = d//h,                # dimension per head
    heads = h,                    # number of attention heads
    causal = False,                # autoregressive or not
    memory_efficient = True,      # whether to use memory efficient attention (can be turned off to test against normal attention)
    q_bucket_size = 128,         # bucket size along queries dimension
    k_bucket_size = 256          # bucket size along key / values dimension
).cuda()
#x_m = x.view(b,n, h, d//h)
#evaluate_cuda_memory(mh_att, x,x,x) #6806
#evaluate_cuda_memory(mh, x,x,x)
#evaluate_cuda_memory(att, x,x,x)
#evaluate_cuda_memory(memory_efficient_attention,x_m,x_m,x_m)

#print(memory_efficient_attention(x,x,x))
#torch.backends.cuda.enable_mem_efficient_sdp(True)
#print(nn.functional.scaled_dot_product_attention(x,x,x))
#mask = (torch.randn(b,n) ).to("cuda")
#print((mask.sum()) / (b*n))
o = attn(x)
#torch.sum(o).backward()