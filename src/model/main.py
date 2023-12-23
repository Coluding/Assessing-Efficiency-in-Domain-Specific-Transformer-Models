import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.optim import Adam
import sys

sys.path.insert(0, '../..')

from src.data.dataset import FinDataset
from src.model.model import ReversibleLongBert, ReversibleLongBertConfig
from src.model.functions import MaskedCrossEntropyLoss, fit
from src.utils.utils import YamlConfigLoader
from src.utils.torch_utils import DeviceDataLoader


def main():
    device = "cuda"
    yaml_loader = YamlConfigLoader("../config.yml")
    config: ReversibleLongBertConfig = ReversibleLongBertConfig(num_blocks=yaml_loader.num_blocks,
                                                                num_heads=yaml_loader.num_heads,
                                                                d_model=yaml_loader.d_model,
                                                                dilation_rates=[yaml_loader.dilation_rates] * yaml_loader.num_blocks,
                                                                segment_lengths=[yaml_loader.segment_lengths] * yaml_loader.num_blocks,
                                                                reversible=yaml_loader.reversible,
                                                                use_pretrained_embeddings=yaml_loader.use_pretrained_embeddings,
                                                                vocab_size=yaml_loader.vocab_size,
                                                                train_size=yaml_loader.train_size)
    dataset = FinDataset("../config.yml")
    random_seed = 42
    torch.manual_seed(random_seed)
    train_size = int(config.train_size * len(dataset))
    indices = list(range(len(dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_data = Subset(dataset, train_indices)
    train_loader = DeviceDataLoader(DataLoader(train_data, batch_size=yaml_loader.batch_size, shuffle=True), device)
    val_data = Subset(dataset, val_indices)
    val_loader = DeviceDataLoader(DataLoader(val_data, batch_size=yaml_loader.batch_size, shuffle=False), device)

    loss = MaskedCrossEntropyLoss(dataset.mask_token_id).to(device)
    model = ReversibleLongBert(config).to(device)

    loggable_params = {"hparam/batch_size": yaml_loader.batch_size,
                     "hparam/num_blocks": yaml_loader.num_blocks,
                     "hparam/num_heads": yaml_loader.num_heads,
                     "hparam/d_model": yaml_loader.d_model,
                     "hparam/reversible": yaml_loader.reversible,
                     "hparam/projection_option": yaml_loader.projection_option,
                     "hparam/vocab_size": yaml_loader.vocab_size,
                     "hparam/use_pretrained_embeddings": yaml_loader.use_pretrained_embeddings,
                     "hparam/train_size": yaml_loader.train_size}

    fit(1, model, loss, train_loader, yaml_loader.learning_rate, val_loader, Adam,
        CosineAnnealingWarmRestarts, loggable_params=loggable_params, save_path="../checkpoints/finbert",
        save_best=True, verbose=True, lrs_params={"T_0": 10, "T_mult": 2, "eta_min": 0.00001}, iters_to_accumulate=1,
        mixed_precision=yaml_loader.mixed_precision, grad_clipping_norm=1.0)


if __name__ == "__main__":
    main()