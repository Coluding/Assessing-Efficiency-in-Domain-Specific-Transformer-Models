import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Type
from xformers.components.feedforward import  MLP
from xformers.components.activations import Activation
import yaml
from typing import Sequence

import sys
sys.path.insert(0, "../../")

from src.model.sliding_window_attention import (
    MultiheadDilatedAttention,
    AttentionProjector,
    QKVProjectionOption
)
from src.model.reversible_layer import ReversibleResidualBlock, reversible_layer_constructor, ReversibleSequenceWrapper
from src.model.utils import track_cuda_memory, evaluate_cuda_memory

class ResidualBlock(nn.Module):
    def __init__(self, f: Type[nn.Module]):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x) + x


class AttentionBlock(nn.Module):
    """
     A customizable attention block with optional reversible layers.

    This class defines an attention block which can be configured to use either standard or reversible layers.
    It includes a multi-head dilated attention mechanism and a feedforward network. The reversible configuration
    is particularly memory-efficient for training large models.
    """
    def __init__(self, d_model: int, num_heads: int,  dilation_rates: Sequence[int],
                 segment_lengths: Sequence[int], dropout: float = 0., reversible: bool = True,
                 projection_option: QKVProjectionOption = QKVProjectionOption.INDIVIDUAL):
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
        if np.log2(d_model) != int(np.log2(d_model)):
            raise ValueError("d_model has to be a power of 2")

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
        feedforward_part = MLP(dim_model=d_model, dropout=dropout, activation=Activation.ReLU, hidden_layer_multiplier=2)

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
    b, n, d = 128, 2048, 512
    x = torch.randn(b, n, d, requires_grad=True).to("cuda")
    h = 8
    att = AttentionBlock(d, h, dilation_rates=[1,3,5], segment_lengths=[128,512,1024],
                         reversible=True).to("cuda")
    att(x).sum().backward()


if __name__ == "__main__":
    main()
