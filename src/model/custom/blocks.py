import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import yaml
from typing import Sequence

import sys
sys.path.insert(0, "../../../")

from src.model.custom.sliding_window_attention import (
    MultiheadDilatedAttention,
    AttentionProjector,
    QKVProjectionOption
)
from src.model.custom.reversible_layer import ReversibleResidualBlock

activation_mapper = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "mish": nn.Mish,
    "elu": nn.ELU,
}

class ResidualBlock(nn.Module):
    def __init__(self, f: nn.Module):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x) + x


class ProjectionAndAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.,
                 projection_option: QKVProjectionOption = QKVProjectionOption.INDIVIDUAL):
        super().__init__()
        self.projection = AttentionProjector(d_model=d_model, projection_option=projection_option)
        self.attention = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout,)

    def forward(self, x: torch.Tensor):
        q, k, v = self.projection(x)
        return self.attention(q, k, v, need_weights=False)[0]


class ReversibleFullAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int,  dropout: float = 0., reversible: bool = True,
                 projection_option: QKVProjectionOption = QKVProjectionOption.INDIVIDUAL,
                 activation: str = "relu"):
        super().__init__()
        # Ensure that d_model is a power of 2 for compatibility with certain optimizations
        if (d_model % 2) != 0:
            raise ValueError("d_model has to be divisible by 2")

        # Prepare for reversible configuration by adjusting d_model if necessary
        self.reversible = False
        if reversible:
            d_model //= 2
            self.reversible = True

        # Initialize the attention mechanism
        attention_with_projection = ProjectionAndAttention(d_model=d_model, num_heads=num_heads, dropout=dropout,
                                                           projection_option=projection_option)

        # Initialize the feedforward part
        try:
            activation_fn = activation_mapper[activation]
        except KeyError:
            raise ValueError(f"Activation {activation} not supported")
        feedforward_part = nn.Sequential(nn.Linear(d_model, d_model), activation_fn(), nn.Dropout(dropout),
                                         nn.Linear(d_model, d_model), nn.Dropout(dropout))

        # Construct the block, either reversible or standard
        if reversible:
            self.block = ReversibleResidualBlock(attention_with_projection, feedforward_part, d_model, layer_norm=True)

        else:
            residual1 = ResidualBlock(attention_with_projection)
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
        # Ensure gradient computation for reversible layers
        if self.reversible and not x.requires_grad:
            x.requires_grad = True
        # Apply the block to the input tensor
        return self.block(x)


class AttentionBlock(nn.Module):
    """
     A customizable attention block with optional reversible layers.

    This class defines an attention block which can be configured to use either standard or reversible layers.
    It includes a multi-head dilated attention mechanism and a feedforward network. The reversible configuration
    is particularly memory-efficient for training large models.
    """
    def __init__(self, d_model: int, num_heads: int,  dilation_rates: Sequence[int],
                 segment_lengths: Sequence[int], dropout: float = 0., reversible: bool = True,
                 projection_option: QKVProjectionOption = QKVProjectionOption.INDIVIDUAL,
                 activation: str = "relu"):
        """
        :param d_model: The dimensionality of the input and output features of the block.
        :param num_heads: The number of heads in the multi-head attention mechanism.
        :param dilation_rates: Dilation rates for each attention head.
        :param segment_lengths:  Segment lengths for attention calculation in each head.
        :param dropout: Dropout rate for the feedforward network. Defaults to 0.
        :param reversible: Flag to use reversible layers. Defaults to True.
        :param projection_option: Option for how queries, keys, and values are projected in the attention mechanism.
        Defaults to QKVProjectionOption.INDIVIDUAL.
        """
        super().__init__()
        # Ensure that d_model is a power of 2 for compatibility with certain optimizations
        if (d_model % 2) != 0:
            raise ValueError("d_model has to be divisible by 2")

        # Prepare for reversible configuration by adjusting d_model if necessary
        self.reversible = False
        if reversible:
            d_model //= 2
            self.reversible = True

        # Initialize the attention mechanism
        attention_part = MultiheadDilatedAttention(
            d_model=d_model, num_heads=num_heads,
            dilation_rates= dilation_rates,
            segment_lengths=segment_lengths,
            projection_option=projection_option
        )

        # Initialize the feedforward part
        try:
            activation_fn = activation_mapper[activation]
        except KeyError:
            raise ValueError(f"Activation {activation} not supported")
        feedforward_part = nn.Sequential(nn.Linear(d_model, d_model), activation_fn(), nn.Dropout(dropout),
                                         nn.Linear(d_model, d_model), nn.Dropout(dropout))


        # Construct the block, either reversible or standard
        if reversible:
            self.block = ReversibleResidualBlock(attention_part, feedforward_part, d_model, layer_norm=True)

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
        # Ensure gradient computation for reversible layers
        if self.reversible and not x.requires_grad:
            x.requires_grad = True
        # Apply the block to the input tensor
        return self.block(x)


def build_model_from_config(config_path: str) -> AttentionBlock:
    projection_mapper = {
        0: QKVProjectionOption.INDIVIDUAL,
        1: QKVProjectionOption.QK,
        2: QKVProjectionOption.SAME
    }
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config["encoder"]["num_blocks"] != len(config["attention_blocks"]):
        raise ValueError("num_block and number of blocks specified in config must match!")

    blocks = []

    for context in config["attention_blocks"]:
        block = AttentionBlock(context["model_dim"], context["num_heads"],
                               context["window_size"], context["dilation_rate"],
                               context["global_tokens"], context["dropout"],
                               context["reversible"], projection_mapper[context["projection_option"]])
        blocks.append(block)

    model = nn.Sequential(
        *blocks
    )

    return model

def main():
    # NOTE: When reversible set to False, the max sequence length halves
    b, n, d = 128, 4096, 512
    x = torch.randn(b, n, d, requires_grad=True).to("cuda")
    h = 8
    att = AttentionBlock(d, h, dilation_rates=[1,3,5], segment_lengths=[2048,2048,2048],
                         reversible=True).to("cuda")

    #att2 = ReversibleFullAttentionBlock(d,h).to("cuda")
    att(x).sum().backward()


if __name__ == "__main__":
    main()
