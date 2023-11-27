import torch
import torch.nn as nn
from typing import Sequence, List, Tuple, Union, Optional

from src.model.sliding_window_attention import QKVProjectionOption
from src.model.blocks import ReversibleResidualBlock, AttentionBlock


class ReversibleLongBertConfig:
    pass


class ReversibleLongBert(nn.Module):
    def __init__(self, num_blocks: int, num_heads: Union[int, List[int]], d_model: int,
                 segment_lengths: List[List[int]],
                 dilation_rates: List[List[int]], dropout: float = 0., reversible: bool = True,
                 projection_option: QKVProjectionOption = QKVProjectionOption.INDIVIDUAL, ):
        super().__init__()

        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_blocks

        block_intermediate: List[nn.Module] = []
        for i in range(num_blocks):
            block_intermediate.append(AttentionBlock(d_model, num_heads[i], dilation_rates[i], segment_lengths[i],
                                                     dropout, reversible, projection_option))

        self.attention_blocks: nn.Sequential = nn.Sequential(*block_intermediate)

    def forward(self, x):
        return self.attention_blocks(x)
