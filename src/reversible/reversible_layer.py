from xformers.components.reversible import ReversibleBlock
from xformers.components.feedforward import MLP
from xformers.components.activations import Activation
import torch
import torch.nn as nn
from src.attention.sliding_window_attention import MultiHeadDilatedLocalAttention
from src.attention.utils import track_cuda_memory, evaluate_cuda_memory
d_model = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
s = MultiHeadDilatedLocalAttention(d_model=d_model, dilation_rate=5, window_size=49, num_heads=4).to(device)
q,k,v = (torch.randn(64, 1000, d_model).to(device) for _ in range(3))
att = torch.nn.MultiheadAttention(num_heads=4, embed_dim=d_model).to(device)

activation = Activation("relu")
print(activation)

class Projector(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.q_proj = torch.nn.Linear(d_model//2, d_model//2)
        self.w_proj = torch.nn.Linear(d_model//2, d_model//2)
        self.k_proj = torch.nn.Linear(d_model//2, d_model//2)
        self.s = MultiHeadDilatedLocalAttention(d_model=d_model, dilation_rate=5, window_size=49, num_heads=4).to(device)

    def forward(self, emb):
        q = self.q_proj(emb)
        k = self.q_proj(emb)
        v = self.q_proj(emb)

        out = self.s(q,k,v)

        return out

proj = Projector().to(device)
mlp = MLP(dim_model=d_model//2, dropout=0.1, activation=activation, hidden_layer_multiplier=2).to(device)
rev_layer = ReversibleBlock(proj, mlp).to(device)

#track_cuda_memory("rev", rev_layer, q)
#evaluate_cuda_memory(rev_layer, q)

rev_layer(q)


class ReversibleResidualBlock(nn.Module):
    def __init__(self, F: nn.Module, G: nn.Module):
        """
        Reversible Layer which avoids storing activations. Activations are recomputed during backward pass.
        Refer to equations (31) to (43) and algorithm 1 for an understanding of the process.
        :param F: Function F which should ideally be some kind of attention.
        :param G: Function F which should ideally be a feedforward layer.
        """
        super().__init__()
        assert isinstance(F, nn.Module)
        assert isinstance(G, nn.Module)

        self.rev_layer = ReversibleBlock(F, G)

    def forward(self, x: torch.Tensor):
        return self.rev_layer(x)




