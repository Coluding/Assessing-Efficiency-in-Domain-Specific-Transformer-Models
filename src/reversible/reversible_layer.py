from xformers.components.reversible import ReversibleBlock
from xformers.components.feedforward import MLP, FusedMLP
from xformers.components.activations import Activation
import torch
import torch.nn as  nn
from src.attention.sliding_window_attention import MultiHeadDilatedLocalAttention
from src.attention.utils import track_cuda_memory, evaluate_cuda_memory
d_model = 512
s = MultiHeadDilatedLocalAttention(d_model=d_model, dilation_rate=5, window_size=49, num_heads=4).to("cuda")
q,k,v = (torch.randn(64, 1000, d_model).to("cuda") for _ in range(3))
att = torch.nn.MultiheadAttention(num_heads=4, embed_dim=d_model).to("cuda")

activation = Activation("relu")
print(activation)

class Projector(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.q_proj = torch.nn.Linear(d_model//2,d_model//2)
        self.w_proj = torch.nn.Linear(d_model//2, d_model//2)
        self.k_proj = torch.nn.Linear(d_model//2, d_model//2)
        self.s = MultiHeadDilatedLocalAttention(d_model=d_model, dilation_rate=5, window_size=49, num_heads=4).to("cuda")
    def forward(self, emb):
        q = self.q_proj(emb)
        k = self.q_proj(emb)
        v = self.q_proj(emb)

        out = self.s(q,k,v)

        return out

proj = Projector().to("cuda")
mlp = FusedMLP(dim_model=d_model//2, dropout=0.1, activation=activation, hidden_layer_multiplier=2).to("cuda")
rev_layer = ReversibleBlock(proj, mlp).to("cuda")

track_cuda_memory("rev", rev_layer, q)
evaluate_cuda_memory(rev_layer, q)

class ReversibleResidualBlock(nn.Module):
    def __init__(self, F: nn.Module, G: nn.Module):
        super().__init__()
        assert isinstance(F, nn.Module)
        assert isinstance(G, nn.Module)


