import enum

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Union, Optional
import sys
from transformers import (
    AutoModelForMaskedLM,
    RobertaConfig,
    ReformerConfig,
    LongformerConfig,
    BigBirdConfig)
import yaml

sys.path.insert(0, '../../..')

from src.model.custom.sliding_window_attention import QKVProjectionOption
from src.model.custom.blocks import AttentionBlock, ReversibleFullAttentionBlock, activation_mapper
from src.model.custom.embeddings import ReversibleLongFinBertEmbedding


class YAMLRobertaElectraConfig(RobertaConfig):
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        roberta_config = config["electra_roberta"]
        super().__init__(**roberta_config)
        self.layer_norm_eps = float(self.layer_norm_eps)

class RobertConfigOfYAML(RobertaConfig):
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        roberta_config = config["roberta"]
        super().__init__(**roberta_config)
        self.layer_norm_eps = float(self.layer_norm_eps)


class ReformConfigOfYAML(ReformerConfig):
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        reformer_config = config["reformer"]
        super().__init__(**reformer_config)
        self.layer_norm_eps = float(self.layer_norm_eps)


class LongFormerConfigOfYAML(LongformerConfig):
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        longformer_config = config["longformer"]
        super().__init__(**longformer_config)
        self.layer_norm_eps = float(self.layer_norm_eps)

class BigBirdConfigOfYAML(BigBirdConfig):
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        bigbird_config = config["bigbird"]
        super().__init__(**bigbird_config)
        self.layer_norm_eps = float(self.layer_norm_eps)


class AttentionBlockType(enum.Enum):
    FULL = "full"
    DILATED = "dilated"

@dataclass
class ReversibleLongBertConfig:
    num_blocks: int
    num_heads: Union[int, List[int]]
    d_model: int
    segment_lengths: List[List[int]]
    dilation_rates: List[List[int]]
    attention_type: AttentionBlockType.DILATED
    dropout: float = 0.
    reversible: bool = True
    projection_option: QKVProjectionOption = QKVProjectionOption.INDIVIDUAL
    vocab_size: int = 50000
    use_pretrained_embeddings: bool = False
    train_size: float = 0.9
    activation: str = "relu"


class ReversibleBert(nn.Module):
    def __init__(self, config: ReversibleLongBertConfig):
        super().__init__()

        assert len(config.segment_lengths) == config.num_blocks
        assert len(config.dilation_rates) == config.num_blocks

        if isinstance(config.num_heads, int):
            config.num_heads = [config.num_heads] * config.num_blocks

        self.embedding = ReversibleLongFinBertEmbedding(config.d_model, config.use_pretrained_embeddings,
                                                        config.vocab_size, config.dropout)

        block_intermediate: List[nn.Module] = []
        for i in range(config.num_blocks):
            if config.attention_type == AttentionBlockType.DILATED:
                block_intermediate.append(AttentionBlock(config.d_model, config.num_heads[i], config.dilation_rates[i],
                                                         config.segment_lengths[i], config.dropout, config.reversible,
                                                         config.projection_option, config.activation))
            elif config.attention_type == AttentionBlockType.FULL:
                block_intermediate.append(ReversibleFullAttentionBlock(config.d_model, config.num_heads[i],
                                                                       config.dropout, config.reversible,
                                                                       config.projection_option,
                                                                       config.activation))

        self.attention_blocks: nn.Sequential = nn.Sequential(*block_intermediate)


    def inject_pretrained_embeddings(self, pretrained_embeddings: torch.Tensor, layer_norm: Optional[nn.Module] = None):
        self.embedding.inject_pretrained_embeddings(pretrained_embeddings)
        if layer_norm is not None:
            self.embedding.inject_layer_norm(layer_norm)

    def forward(self, x, segment_ids=None):
        x = self.embedding(x, segment_ids)
        x = self.attention_blocks(x)
        if x.isnan().any():
            print("Output is NaN. Replacing.")
            x[x.isnan()] = 0
            return x
        return x


class ReversibleBertForMaskedLM(ReversibleBert):
    def __init__(self, config: ReversibleLongBertConfig):
        super().__init__(config)
        self.prediction_head = MaskedHead(config.d_model, config.vocab_size, config.activation)

    def forward(self, x, segment_ids=None):
        x = super().forward(x, segment_ids)
        return self.prediction_head(x)


class MaskedHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, activation: str = "relu"):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)

        self.decoder = nn.Linear(d_model, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

        self.activation = activation_mapper[activation]()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        x = self.layer_norm(x)

        x = self.decoder(x)

        return x

def main():
    b, n, d = 8, 2048, 768
    # x = torch.randn(b, n, d, requires_grad=True).to("cuda")
    h = 8
    vocab_size = 30873
    x = torch.randint(0, vocab_size, (b, n)).to("cuda")
    segment_ids = torch.randint(0, 3, (b, n)).to("cuda")
    blocks = 12
    dilation_rates = [[1, 3, 5]] * blocks
    segment_lengths = [[2048, 2048, 2048]] * blocks
    config: ReversibleLongBertConfig = ReversibleLongBertConfig(blocks, h, d, dilation_rates=dilation_rates,
                                                                segment_lengths=segment_lengths,
                                                                reversible=True, use_pretrained_embeddings=False,
                                                                vocab_size=vocab_size,
                                                                attention_type=AttentionBlockType.FULL)

    att = ReversibleBertForMaskedLM(config)
    model = AutoModelForMaskedLM.from_pretrained("yiyanghkust/finbert-pretrain")
    weights = list(model.bert.embeddings.word_embeddings.parameters())[0]
    layer_norm = model.bert.embeddings.LayerNorm
    att.inject_pretrained_embeddings(weights, layer_norm)
    att = att.to("cuda")
    print(att(x).shape) #sum().backward()


if __name__ == "__main__":
    main()
