import torch
import unittest

from src.model.custom import ReversibleLongBert, ReversibleLongBertConfig
from src.utils.utils import YamlConfigLoader


class TestModel(unittest.TestCase):

    def setUp(self):
        device = "cuda"
        yaml_loader = YamlConfigLoader("../src/config.yml")
        config: ReversibleLongBertConfig = ReversibleLongBertConfig(num_blocks=yaml_loader.num_blocks,
                                                                    num_heads=yaml_loader.num_heads,
                                                                    d_model=yaml_loader.d_model,
                                                                    dilation_rates=[
                                                                                       yaml_loader.dilation_rates] *
                                                                                   yaml_loader.num_blocks,
                                                                    segment_lengths=[
                                                                                        yaml_loader.segment_lengths] *
                                                                                    yaml_loader.num_blocks,
                                                                    reversible=yaml_loader.reversible,
                                                                    use_pretrained_embeddings=yaml_loader.use_pretrained_embeddings,
                                                                    vocab_size=yaml_loader.vocab_size,
                                                                    train_size=yaml_loader.train_size)
        self.model = ReversibleLongBert(config).to(device)
        self.yaml_loader = yaml_loader
        self.device = device

    def test_forward_pass(self):
        x = torch.randint(0, self.yaml_loader.vocab_size, (self.yaml_loader.batch_size,
                                                           self.yaml_loader.context_length,
                                                           self.yaml_loader.d_model)).to(self.device)
        out = self.model(x)
        self.assertEquals(out.shape, torch.Tensor((self.yaml_loader.batch_size,
                                                   self.yaml_loader.context_length,
                                                   self.yaml_loader.vocab_size)))