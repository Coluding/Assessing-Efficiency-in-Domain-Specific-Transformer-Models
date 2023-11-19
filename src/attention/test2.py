from xformers.components.attention import RandomAttention, GlobalAttention, LocalAttention, ScaledDotProduct, Attention
from xformers.components.multi_head_dispatch import MultiHeadDispatch

import torch
from sliding_window_attention import LocalAttentionDilation
import utils

d_model = 256
q,k,v = (torch.randn(64, 2000, d_model).to("cuda") for _ in range(3))
glob_mask = torch.zeros(2000, 1) == 1
glob_mask[[0,5,20,100,200],:] = torch.tensor(True)

local_att = LocalAttentionDilation(window_size=21).to("cuda")
local_att.set_dilation(5)
local_att.set_global_indices(torch.Tensor([0,5,20,100,200]).int())
mh = MultiHeadDispatch(attention=local_att, dim_model=d_model, num_heads=2).to("cuda")
utils.track_cuda_memory("glob", att, q,k,v)
