import warnings

from xformers.components.reversible import ReversibleBlock
from xformers.components.feedforward import MLP
from xformers.components.activations import Activation
from xformers.components.attention import Attention
import torch
import torch.nn as nn
from typing import Type

from src.attention.sliding_window_attention import MultiHeadDilatedLocalAttention
from src.attention.utils import track_cuda_memory, evaluate_cuda_memory

"""
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
"""

class ReversibleWrapper(ReversibleBlock):
    """
        A wrapper class for the ReversibleBlock that incorporates layer normalization.

        This class extends the functionality of ReversibleBlock by optionally adding
        a layer normalization step in the forward pass. This can help stabilize
        the learning process in deep neural networks.

        :param f: A neural network module to be used as the 'f' function in the reversible block.
        :param g: A neural network module to be used as the 'g' function in the reversible block.
        :param split_dim: The dimension along which the input tensor should be split. Default is -1.
        """
    def __init__(self, f: nn.Module, g: nn.Module, split_dim: int = -1):
        super().__init__(f, g, split_dim)
        self.layer_norm = nn.Identity()

    def apply_layer_norm(self, model_dim: int):
        """
        Applies layer normalization to the reversible block.

        This method replaces the identity layer with a layer normalization layer.
        It should be called before the forward pass if layer normalization is desired.

        :param model_dim: The dimension of the model over which normalization is to be applied.
        """
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor, f_args={}, g_args={}):
        """
        Defines the forward pass with optional layer normalization.

        Splits the input tensor into two parts, processes them with the 'f' and 'g' functions,
        applies layer normalization if it's not set to identity, and then concatenates the outputs.

        :param x: The input tensor to the reversible block.
        :param f_args: Optional arguments for the 'f' function.
        :param g_args: Optional arguments for the 'g' function.
        :return: The output tensor after processing and recombination.
        """
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1, y2 = None, None

        if self.layer_norm == nn.Identity:
            warnings.warn("No layer norm applied, if not desired then call apply_layer_norm() before forward pass")

        with torch.no_grad():
            y1 = self.layer_norm(x1 + self.f(x2, record_rng=self.training, **f_args))
            y2 = self.layer_norm(x2 + self.g(y1, record_rng=self.training, **g_args))

        return torch.cat([y1, y2], dim=self.split_dim)


class ReversibleResidualBlock(nn.Module):
    def __init__(self, f: Type[nn.Module], g: MLP, dim_model: int):
        """
        Reversible Layer which avoids storing activations. Activations are recomputed during backward pass.
        Refer to equations (31) to (43) and algorithm 1 for an understanding of the process.
        :param F: Function F which should ideally be some kind of attention.
        :param G: Function F which should ideally be a feedforward layer.
        """
        super().__init__()
        assert isinstance(f, nn.Module)
        assert isinstance(g, nn.Module)

        self.rev_layer = ReversibleWrapper(f, g)
        self.rev_layer.apply_layer_norm(dim_model)

    def forward(self, x: torch.Tensor):
        return self.rev_layer(x)





