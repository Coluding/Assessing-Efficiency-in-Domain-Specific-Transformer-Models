import torch
import torch.nn as nn
import torchinfo
from dataclasses import dataclass, field
from typing import Sequence, List, Tuple, Union, Optional

from src.model.sliding_window_attention import QKVProjectionOption
from src.model.blocks import AttentionBlock
from src.model.embeddings import ReversibleLongFinBertEmbedding



@dataclass
class ReversibleLongBertConfig:
    num_blocks: int
    num_heads: Union[int, List[int]]
    d_model: int
    segment_lengths: List[List[int]]
    dilation_rates: List[List[int]]
    dropout: float = 0.
    reversible: bool = True
    projection_option: QKVProjectionOption = QKVProjectionOption.INDIVIDUAL
    vocab_size: int = 50000
    use_pretrained_embeddings: bool = False
    train_size: float = 0.9


class ReversibleLongBert(nn.Module):
    def __init__(self, config: ReversibleLongBertConfig):
        super().__init__()

        assert len(config.segment_lengths) == config.num_blocks
        assert len(config.dilation_rates) == config.num_blocks

        if isinstance(config.num_heads, int):
            config.num_heads = [config.num_heads] * config.num_blocks

        self.embedding = ReversibleLongFinBertEmbedding(config.d_model, config.use_pretrained_embeddings,
                                                        config.vocab_size, dropout=0.)

        block_intermediate: List[nn.Module] = []
        for i in range(config.num_blocks):
            block_intermediate.append(AttentionBlock(config.d_model, config.num_heads[i], config.dilation_rates[i],
                                                     config.segment_lengths[i], config.dropout, config.reversible,
                                                     config.projection_option))

        self.attention_blocks: nn.Sequential = nn.Sequential(*block_intermediate)

        self.prediction_head = nn.Linear(config.d_model, config.vocab_size)

    def inject_pretrained_embeddings(self, pretrained_embeddings: torch.Tensor):
        self.embedding.inject_pretrained_embeddings(pretrained_embeddings)

    def forward(self, x, segment_ids=None):
        x = self.embedding(x, segment_ids)
        x = self.attention_blocks(x)
        return self.prediction_head(x)


def main():
    from src.model.functions import MaskedCrossEntropyLoss
    b, n, d = 128, 1500, 512
    # x = torch.randn(b, n, d, requires_grad=True).to("cuda")
    h = 8
    x = torch.randint(0, 10000, (b, n)).to("cuda")
    segment_ids = torch.randint(0, 3, (b, n)).to("cuda")
    blocks = 2
    dilation_rates = [[1, 3, 5]] * blocks
    segment_lengths = [[250, 500, 1500]] * blocks
    config: ReversibleLongBertConfig = ReversibleLongBertConfig(blocks, h, d, dilation_rates=dilation_rates,
                                                                segment_lengths=segment_lengths,
                                                                reversible=True, use_pretrained_embeddings=False,
                                                                vocab_size=10000)
    att = ReversibleLongBert(config).to("cuda")
    att(x, segment_ids).sum().backward()


if __name__ == "__main__":
    main()
