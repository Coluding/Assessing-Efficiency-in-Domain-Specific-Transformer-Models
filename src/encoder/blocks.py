import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Type
from xformers.components.feedforward import  MLP
from xformers.components.activations import Activation

from src.attention.sliding_window_attention import (
    MultiHeadDilatedLocalAttention,
    AttentionProjector,
    QKVProjectionOption
)
from src.reversible.reversible_layer import ReversibleResidualBlock
from src.attention.utils import (
    track_cuda_memory,
    evaluate_cuda_memory
)


class ResidualBlock(nn.Module):
    def __init__(self, f: Type[nn.Module]):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x) + x


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int,  window_size: int,
                 dilation_rate: int, global_token_indices: Optional[List[int]] = None,
                 dropout: float = 0., reversible: bool = True,
                 projection_option: QKVProjectionOption = QKVProjectionOption.INDIVIDUAL):
        super().__init__()
        # assert wether model dimension is a power of 2
        if np.log2(d_model) != int(np.log2(d_model)):
            raise ValueError("d_model has to be a power of 2")

        if reversible:
            d_model //= 2

        attention_part = MultiHeadDilatedLocalAttention(
                                           d_model, num_heads, window_size,
                                           torch.Tensor(global_token_indices).int(),
                                           dilation_rate, projection_option)

        feedforward_part = MLP(dim_model=d_model, dropout=dropout, activation=Activation.ReLU, hidden_layer_multiplier=2)

        if reversible:
            self.block = ReversibleResidualBlock(attention_part, feedforward_part, d_model)

        else:
            residual1 = ResidualBlock(attention_part)
            residual2 = ResidualBlock(feedforward_part)
            layer_norm1 = nn.LayerNorm(d_model)
            layer_norm2 = nn.LayerNorm(d_model)

            self.block = nn.Sequential(
                residual1,
                layer_norm1,
                residual2,
                layer_norm2
            )

    def forward(self, x: torch.Tensor):
        return self.block(x)



def main():
    b, n, d = 1, 64, 128
    x = torch.randn(b, n, d).to("cuda")
    att = AttentionBlock(d, 2, 5, 1, [1, 10, 20]).to("cuda")
    print(att(x).shape)


if __name__ == "__main__":
    main()
