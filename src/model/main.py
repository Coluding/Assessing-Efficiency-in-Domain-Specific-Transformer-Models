from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from transformers import (
    ReformerForMaskedLM,
    LongformerForMaskedLM,
    BigBirdForMaskedLM)
import sys

sys.path.insert(0, '../..')

from src.data.dataset import FinDataset, WikiDataset
from src.model.electra import (
    YAMLElectraConfig,
    ElectraModelWrapper
)
from src.model.functions.functions import (
    MaskedCrossEntropyLoss,
    fit_n_batch
)
from src.utils.utils import YamlConfigLoader
from src.utils.torch_utils import DeviceDataLoader


def prepare_data(yaml_loader: YamlConfigLoader, dataset: str = "fin"):
    if dataset == "fin":
        dataset = FinDataset("../config.yml")
    else:
        dataset = WikiDataset("../config.yml")
    random_seed = 42
    torch.manual_seed(random_seed)
    train_size = int(yaml_loader.train_size * len(dataset))
    indices = list(range(len(dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_data = Subset(dataset, train_indices)
    train_loader = DeviceDataLoader(DataLoader(train_data, batch_size=yaml_loader.batch_size, shuffle=True),
                                    yaml_loader.device)
    val_data = Subset(dataset, val_indices)
    val_loader = DeviceDataLoader(DataLoader(val_data, batch_size=yaml_loader.batch_size, shuffle=False),
                                  yaml_loader.device)

    return dataset, train_loader, val_loader


def train_custom_model():
    yaml_loader = YamlConfigLoader("../config.yml")
    #pretrained_finbert = AutoModelForMaskedLM.from_pretrained("yiyanghkust/finbert-pretrain")
    #weights = list(pretrained_finbert.bert.embeddings.word_embeddings.parameters())[0]
    #layer_norm = pretrained_finbert.bert.embeddings.LayerNorm
    config: ReversibleLongBertConfig = ReversibleLongBertConfig(num_blocks=yaml_loader.num_blocks,
                                                                num_heads=yaml_loader.num_heads,
                                                                d_model=yaml_loader.d_model,
                                                                dilation_rates=[
                                                                                   yaml_loader.dilation_rates] * yaml_loader.num_blocks,
                                                                segment_lengths=[
                                                                                    yaml_loader.segment_lengths] * yaml_loader.num_blocks,
                                                                reversible=yaml_loader.reversible,
                                                                use_pretrained_embeddings=yaml_loader.use_pretrained_embeddings,
                                                                vocab_size=yaml_loader.vocab_size,
                                                                train_size=yaml_loader.train_size,
                                                                dropout=yaml_loader.dropout,
                                                                activation=yaml_loader.activation,
                                                                attention_type= AttentionBlockType.FULL,
                                                                )

    dataset, train_loader, val_loader = prepare_data(yaml_loader)
    loss = MaskedCrossEntropyLoss(dataset.mask_token_id).to(yaml_loader.device)
    model = ReversibleBertForMaskedLM(config)
    #model.inject_pretrained_embeddings(weights, layer_norm)
    model = model.to(yaml_loader.device)


    loggable_params = {"hparam/batch_size": yaml_loader.batch_size,
                       "hparam/num_blocks": yaml_loader.num_blocks,
                       "hparam/num_heads": yaml_loader.num_heads,
                       "hparam/d_model": yaml_loader.d_model,
                       "hparam/reversible": yaml_loader.reversible,
                       "hparam/projection_option": yaml_loader.projection_option,
                       "hparam/vocab_size": yaml_loader.vocab_size,
                       "hparam/use_pretrained_embeddings": yaml_loader.use_pretrained_embeddings,
                       "hparam/train_size": yaml_loader.train_size}

    fit_n_batch(10000, model, loss, train_loader, yaml_loader.learning_rate, val_loader, AdamW,
                CosineAnnealingWarmRestarts, loggable_params=loggable_params, save_path="data/checkpoints/finbert",
                save_best=True, verbose=True, lrs_params={"T_0": 5, "T_mult": 1, "eta_min": 0.0000001},
                iters_to_accumulate=yaml_loader.accumulate_steps, mixed_precision=yaml_loader.mixed_precision,
                grad_clipping_norm=yaml_loader.grad_clip, batch_share=yaml_loader.batch_sample_size,
                n_batches=80000)


def train_roberta(dataset: str = "fin"):
    yaml_loader = YamlConfigLoader("../config.yml")
    config: RobertConfigOfYAML = RobertConfigOfYAML("../config.yml")
    #########cpom
    model = RobertaForMaskedLM(config).to(yaml_loader.device)
    #model.load_state_dict(torch.load("/src/model/checkpoints/electra_v1"))
    dataset, train_loader, val_loader = prepare_data(yaml_loader, dataset)
    loss = MaskedCrossEntropyLoss(dataset.mask_token_id).to(yaml_loader.device)
    loggable_params = {"hparam/batch_size": yaml_loader.batch_size,
                       }

    fit_n_batch(10000, model, loss, train_loader, yaml_loader.learning_rate, val_loader, AdamW,
                CosineAnnealingWarmRestarts, loggable_params=loggable_params, save_path="data/checkpoints/roberta",
                save_best=True, verbose=True, lrs_params={"T_0": 5, "T_mult": 1, "eta_min": 0.0000001},
                iters_to_accumulate=yaml_loader.accumulate_steps, mixed_precision=yaml_loader.mixed_precision,
                grad_clipping_norm=yaml_loader.grad_clip, batch_share=yaml_loader.batch_sample_size, n_batches=100000,
                mask_token_index=dataset.mask_token_id, pad_token_index=dataset.pad_token_id)


def train_reformer():
    yaml_loader = YamlConfigLoader("../config.yml")
    config: ReformConfigOfYAML = ReformConfigOfYAML("../config.yml")
    model = ReformerForMaskedLM(config).to(yaml_loader.device)
    #model.load_state_dict(torch.load("/src/model/checkpoints/electra_v1"))
    dataset, train_loader, val_loader = prepare_data(yaml_loader)
    loss = MaskedCrossEntropyLoss(dataset.mask_token_id).to(yaml_loader.device)
    loggable_params = {"hparam/batch_size": yaml_loader.batch_size,
                       }
    fit_n_batch(10000, model, loss, train_loader, yaml_loader.learning_rate, val_loader, AdamW,
                CosineAnnealingWarmRestarts, loggable_params=loggable_params, save_path="data/checkpoints/roberta",
                save_best=True, verbose=True, lrs_params={"T_0": 5, "T_mult": 1, "eta_min": 0.0000001},
                iters_to_accumulate=yaml_loader.accumulate_steps, mixed_precision=yaml_loader.mixed_precision,
                grad_clipping_norm=yaml_loader.grad_clip, batch_share=yaml_loader.batch_sample_size, n_batches=100000,
                mask_token_index=dataset.mask_token_id, pad_token_index=dataset.pad_token_id)


def train_longformer():
    yaml_loader = YamlConfigLoader("../config.yml")
    config: LongFormerConfigOfYAML = LongFormerConfigOfYAML("../config.yml")
    model = LongformerForMaskedLM(config).to(yaml_loader.device)
    #model.load_state_dict(torch.load("/src/model/checkpoints/electra_v1"))
    dataset, train_loader, val_loader = prepare_data(yaml_loader)
    loss = MaskedCrossEntropyLoss(dataset.mask_token_id).to(yaml_loader.device)
    loggable_params = {"hparam/batch_size": yaml_loader.batch_size,
                       }
    fit_n_batch(10000, model, loss, train_loader, yaml_loader.learning_rate, val_loader, AdamW,
                CosineAnnealingWarmRestarts, loggable_params=loggable_params, save_path="data/checkpoints/roberta",
                save_best=True, verbose=True, lrs_params={"T_0": 5, "T_mult": 1, "eta_min": 0.0000001},
                iters_to_accumulate=yaml_loader.accumulate_steps, mixed_precision=yaml_loader.mixed_precision,
                grad_clipping_norm=yaml_loader.grad_clip, batch_share=yaml_loader.batch_sample_size, n_batches=10000,
                mask_token_index=dataset.mask_token_id, pad_token_index=dataset.pad_token_id)

def train_electra(dataset: str = "fin"):
    config = YAMLElectraConfig("../config.yml")
    roberta_config = YAMLRobertaElectraConfig("../config.yml")
    generator = RobertaForMaskedLM(roberta_config)
    yaml_loader = YamlConfigLoader("../config.yml")
    dataset, train_loader, val_loader = prepare_data(yaml_loader, dataset=dataset)
    model = ElectraModelWrapper(config, generator=generator, tokenizer=dataset.tokenizer).to(yaml_loader.device)
    loss = MaskedCrossEntropyLoss(dataset.mask_token_id).to(yaml_loader.device)
    loggable_params = {"hparam/batch_size": yaml_loader.batch_size,
                       }

    fit_n_batch(10000, model, loss, train_loader, yaml_loader.learning_rate, val_loader, AdamW,
                CosineAnnealingWarmRestarts, loggable_params=loggable_params, save_path="data/checkpoints/roberta",
                save_best=True, verbose=True, lrs_params={"T_0": 5, "T_mult": 1, "eta_min": 0.0000001},
                iters_to_accumulate=yaml_loader.accumulate_steps, mixed_precision=yaml_loader.mixed_precision,
                grad_clipping_norm=yaml_loader.grad_clip, batch_share=yaml_loader.batch_sample_size, n_batches=10000,
                mask_token_index=dataset.mask_token_id, pad_token_index=dataset.pad_token_id)


def train_bigbird():
    config = BigBirdConfigOfYAML("../config.yml")
    yaml_loader = YamlConfigLoader("../config.yml")
    model = BigBirdForMaskedLM(config).to(yaml_loader.device)
    dataset, train_loader, val_loader = prepare_data(yaml_loader)
    loss = MaskedCrossEntropyLoss(dataset.mask_token_id).to(yaml_loader.device)
    loggable_params = {"hparam/batch_size": yaml_loader.batch_size,
                       }

    fit_n_batch(10000, model, loss, train_loader, yaml_loader.learning_rate, val_loader, AdamW,
                CosineAnnealingWarmRestarts, loggable_params=loggable_params, save_path="data/checkpoints/roberta",
                save_best=True, verbose=True, lrs_params={"T_0": 5, "T_mult": 1, "eta_min": 0.0000001},
                iters_to_accumulate=yaml_loader.accumulate_steps, mixed_precision=yaml_loader.mixed_precision,
                grad_clipping_norm=yaml_loader.grad_clip, batch_share=yaml_loader.batch_sample_size, n_batches=10000,
                mask_token_index=dataset.mask_token_id, pad_token_index=dataset.pad_token_id)


def main():
    train_roberta()


if __name__ == "__main__":
    main()
