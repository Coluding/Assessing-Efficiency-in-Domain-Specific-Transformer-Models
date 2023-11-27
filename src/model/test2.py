import torch
#from dilated_attention_pytorch.dilated_attention import DilatedAttention
from xformers.components.reversible import ReversibleBlock, ReversibleSequence
from xformers.components.feedforward import MLP
from xformers.components.activations import Activation
from sliding_window_attention import DilatedAttention, MultiheadDilatedAttention



b,n,d = (128,2048,512)
h = 8
x = torch.randn(b, n, d, dtype=torch.float32, requires_grad=True).to("cuda")

# shape: (batch_size, seq_len, num_heads, embed_dim)
# NOTE: 'seq_len' must be a multiple of 8192 (the largest segment length)
# NOTE: For best performance, use 'dtype=torch.float16' or `dtype=torch.bfloat16`
feedforward_part = MLP(dim_model=d//2, dropout=0., activation=Activation.ReLU, hidden_layer_multiplier=2).to("cuda")
query = torch.randn(b, n, 8, 64, device="cuda", dtype=torch.float16, requires_grad=True)
key = torch.randn(b, n, 8, 64, device="cuda", dtype=torch.float16, requires_grad=True)
value = torch.randn(b, n,  8, 64, device="cuda", dtype=torch.float16,  requires_grad=True)

class Projector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dilated_attention = MultiheadDilatedAttention(d_model=d//2, num_heads=h,
                                                          dilation_rates=[1,3,5], segment_lengths=[128,512,1024])


    def forward(self, x):

        return self.dilated_attention(x)

#out = dilated_attention(query, key, value, is_causal=False)  # default: causal=False
#print(out.shape)
#out.sum().backward()

#x = dilated_attention(query, key, value)
#x.sum().backward()
#xx = torch.randn(b,n,512).to("cuda")
p = Projector().to("cuda")
#y = p(xx)
#y[0].sum().backward()

rev = ReversibleSequence(torch.nn.ModuleList([torch.nn.Sequential(p, feedforward_part),
                                             torch.nn.Sequential(p, feedforward_part)]))

#non_rev_sequence = torch.nn.Sequential(p, feedforward_part, p, feedforward_part)
#y = non_rev_sequence(key)
y = rev(x)
y.sum().backward()