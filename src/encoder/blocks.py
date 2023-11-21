import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from xformers.components.feedforward import  MLP
from xformers.components.activations import Activation

from src.attention.sliding_window_attention import MultiHeadDilatedLocalAttention
from src.reversible.reversible_layer import ReversibleResidualBlock


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int,  window_size: int,
                 dilation_rate: int, global_token_indices: Optional[List[int]] = None,
                 dropout: float = 0.,
                 reversible: bool = True):
        super().__init__()
        # assert wether model dimension is a power of 2
        if np.log2(d_model) != int(np.log2(d_model)):
            raise ValueError("d_model has to be a power of 2")

        self.reversible = reversible
        if reversible:
            d_model /= 2

        self.q_proj = MLP(dim_model=d_model, dropout=0., activation=None, hidden_layer_multiplier=1)
        self.w_proj = MLP(dim_model=d_model, dropout=0., activation=None, hidden_layer_multiplier=1)
        self.k_proj = MLP(dim_model=d_model, dropout=0., activation=None, hidden_layer_multiplier=1)

        attention_part = MultiHeadDilatedLocalAttention(d_model, num_heads, window_size,
                                                    torch.Tensor(global_token_indices).int(),
                                                    dilation_rate)
        feedforward_part = MLP(dim_model=d_model,dropout=dropout, activation=Activation.ReLU, hidden_layer_multiplier=2)


    def forward(self, x: torch.Tensor):
        pass