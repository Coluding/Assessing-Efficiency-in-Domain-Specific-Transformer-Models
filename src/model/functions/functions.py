import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
import logging
import transformers
import numpy as np
from datetime import datetime
from tqdm import tqdm
from enum import Enum
from einops import rearrange
from transformers.trainer import Trainer, is_torch_tpu_available


import sys

sys.path.append("../../..")

from src.utils.torch_utils import DeviceDataLoader
from src.data.dataset import FinDataset
from src.model.electra import ElectraModelWrapper


class LossReduction(Enum):
    NONE = "none"
    SUM = "sum"
    MEAN = "mean"


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, mask_token_index: int, logits: bool = True, reduction: LossReduction = LossReduction.MEAN):
        super().__init__()
        self.mask_token_index = mask_token_index
        self.logits = logits
        self.loss = nn.CrossEntropyLoss(reduction=reduction.value)

    def forward(self, predictions: torch.Tensor, masked_ground_truth: torch.Tensor,
                unmasked_ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Computes the masked cross entropy loss for a given prediction and ground truth.
        :param predictions: Shape (batch_size, sequence_length, vocab_size)
        :param masked_ground_truth: Shape (batch_size, sequence_length) with masked tokens
        :param unmasked_ground_truth:  Shape (batch_size, sequence_length) without masked tokens
        :return: The masked cross entropy loss
        """
        predictions = rearrange(predictions, 'b n c -> (b n) c')
        unmasked_ground_truth[masked_ground_truth != self.mask_token_index] = -100
        unmasked_ground_truth = rearrange(unmasked_ground_truth, 'b n -> (b n)')
        loss = self.loss(predictions, unmasked_ground_truth)

        if loss.isnan().any():
            print("Loss is NaN. Skipping batch.")
            logging.info("Loss is NaN. Skipping batch.")
            return torch.tensor(10)

        return loss


def fit(epochs: int, model: nn.Module, loss_fn: nn.Module, train_loader: DeviceDataLoader,
        learning_rate: float, val_loader: DeviceDataLoader, optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler, save_path: str,
        save_best: bool = True, verbose: bool = True, loggable_params: dict = None, lrs_params: dict = None,
        iters_to_accumulate: int = 1, mixed_precision: bool = False, grad_clipping_norm: float = 1.0,
        batch_share: float = 1):
    """
    Fits a given model to the given data.
    :param epochs:
    :param model:
    :param loss:
    :param train_loader:
    :param learning_rate:
    :param val_loader:
    :param optimizer:
    :param scheduler:
    :param save_path:
    :param save_best:
    :param verbose:
    :param loggable_params:
    :param lrs_params:
    :param iters_to_accumulate:
    :param mixed_precision:
    :param grad_clipping_norm:
    :return:
    """
    try:
        torch.autograd.set_detect_anomaly(False)
        best_loss = float("inf")
        best_model = None
        history = []
        run_name = np.random.randint(0, 1000000000)
        id_full = datetime.now().strftime("%Y-%m-%d %H-%M") + str(run_name)
        logging.basicConfig(
            filename='data/logging/run.log',
            filemode='w',
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%d.%m.%y %H:%M:%S',
            level=logging.DEBUG
        )
        writer = SummaryWriter(log_dir=f'data/logging/runs/{id_full}')
        writer.add_hparams({}, loggable_params)
        scaler = GradScaler()
        optimizer = optimizer(model.parameters(), lr=learning_rate)
        lrs = scheduler(optimizer, **lrs_params)

        epoch_loader = tqdm(range(epochs), desc="Epochs")
        for epoch in epoch_loader:
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}")
            model.train()
            train_loss = 0
            optimizer.zero_grad()
            data = {}

            for batch_num, batch in enumerate(train_loader):
                inputs, labels = batch

                # Enable/Disable Mixed Precision
                with torch.autocast(device_type=inputs[0].device.type, dtype=torch.float16, enabled=mixed_precision):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, inputs, labels)
                    scaler.scale(loss).backward()

                    if loss.isnan():
                        print("Loss is NaN. Skipping batch.")
                        logging.info("Loss is NaN. Skipping batch.")
                        continue

                    # Gradient accumulation
                    if (batch_num + 1) % iters_to_accumulate == 0 or (batch_num + 1) == len(train_loader):
                        if grad_clipping_norm:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    torch.cuda.empty_cache()
                train_loss += loss.item()
                if verbose:
                    print(f"Batch {batch_num + 1}/{len(train_loader)} loss: {loss.item()}")
                    logging.info(f"Batch {batch_num + 1}/{len(train_loader)} loss: {loss.item()}")

                if batch_num == int(batch_share * len(train_loader)):
                    # early leave of epoch
                    break

            lrs.step()
            if verbose:
                print(f"Epoch {epoch} learning rate: {lrs.get_last_lr()[0]}")
                logging.info(f"Epoch {epoch} learning rate: {lrs.get_last_lr()[0]}")
            train_loss /= int(batch_share * len(train_loader))
            writer.add_scalar('Loss/train', train_loss, epoch)
            if verbose:
                print(f"Epoch [{epoch}] train loss: {train_loss}")
                logging.info(f"Epoch [{epoch}] train loss: {train_loss}")

            data['train_loss'] = train_loss

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_num, batch in enumerate(val_loader):
                    inputs, labels = batch
                    outputs = model(inputs)
                    loss = loss_fn(outputs, inputs, labels)
                    val_loss += loss.item()
                    if batch_num == int(batch_share * len(val_loader)):
                        # early leave of epoch
                        break
            val_loss /= int(batch_share * len(val_loader))
            writer.add_scalar('Loss/val', val_loss, epoch)
            data['val_loss'] = val_loss
            writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)
            if verbose:
                print(f"Epoch {epoch} validation loss: {val_loss}")
                logging.info(f"Epoch {epoch} validation loss: {val_loss}")

            if data['val_loss'] < best_loss:
                best_loss = data['val_loss']
                best_model = model
                if save_best:
                    torch.save(best_model.state_dict(), save_path)

            history.append(data)

    except KeyboardInterrupt:
        torch.save(model.state_dict(), "../data/checkpoints/electra_v1")


def fit_n_batch(epochs: int, model: nn.Module, loss_fn: nn.Module, train_loader: DeviceDataLoader,
                learning_rate: float, val_loader: DeviceDataLoader, optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler, save_path: str,
                save_best: bool = True, verbose: bool = True, loggable_params: dict = None, lrs_params: dict = None,
                iters_to_accumulate: int = 1, mixed_precision: bool = False, grad_clipping_norm: float = 1.0,
                batch_share: float = 1, n_batches: int = 1, mask_token_index: int = 50264, pad_token_index: int = 0):
    """
    Fits a given model to the given data.
    :param epochs:
    :param model:
    :param loss:
    :param train_loader:
    :param learning_rate:
    :param val_loader:
    :param optimizer:
    :param scheduler:
    :param save_path:
    :param save_best:
    :param verbose:
    :param loggable_params:
    :param lrs_params:
    :param iters_to_accumulate:
    :param mixed_precision:
    :param grad_clipping_norm:
    :return:
    """
    try:
        torch.autograd.set_detect_anomaly(False)
        best_loss = float("inf")
        best_model = None
        history = []
        run_name = np.random.randint(0, 1000000000)
        id_full = datetime.now().strftime("%Y-%m-%d %H-%M") + str(run_name)
        logging.basicConfig(
            filename='data/logging/run.log',
            filemode='w',
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%d.%m.%y %H:%M:%S',
            level=logging.DEBUG
        )
        writer = SummaryWriter(log_dir=f'data/logging/runs/{id_full}')
        writer.add_hparams({}, loggable_params)
        scaler = GradScaler()
        optimizer = optimizer(model.parameters(), lr=learning_rate)
        lrs = scheduler(optimizer, **lrs_params)

        epoch_loader = tqdm(range(epochs), desc="Epochs")
        for epoch in epoch_loader:
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}")
            model.train()
            train_loss = 0
            optimizer.zero_grad()
            data = {}

            for batch_num, batch in enumerate(train_loader):
                inputs, labels = batch

                # Enable/Disable Mixed Precision
                with ((torch.autocast(device_type=inputs[0].device.type,
                                      dtype=torch.float16 if inputs[0].device.type == "cuda" else torch.bfloat16,
                                      enabled=mixed_precision))):

                    if (isinstance(model, transformers.RobertaForMaskedLM)
                            or isinstance(model, transformers.ReformerForMaskedLM)
                            or isinstance(model, transformers.BigBirdForMaskedLM)):
                        labels[inputs != mask_token_index] = -100
                        attention_mask = (inputs != pad_token_index).long()
                        outputs = model(inputs, labels=labels, attention_mask=attention_mask)
                        loss = outputs.loss
                    elif isinstance(model, ElectraModelWrapper):
                        labels[inputs != mask_token_index] = -100
                        attention_mask = (inputs != pad_token_index).long()
                        outputs = model(inputs, generator_labels=labels, attention_mask=attention_mask)
                        loss = outputs[0]
                    else:
                        outputs = model(inputs)
                        loss = loss_fn(outputs, inputs, labels)

                    scaler.scale(loss).backward() if mixed_precision else loss.backward()

                    if loss.isnan():
                        print("Loss is NaN. Skipping batch.")
                        logging.info("Loss is NaN. Skipping batch.")
                        continue

                    # Gradient accumulation
                    if (batch_num + 1) % iters_to_accumulate == 0 or (batch_num + 1) == len(train_loader):
                        if grad_clipping_norm:
                            scaler.unscale_(optimizer) if mixed_precision else None
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer) if mixed_precision else optimizer.step()
                        scaler.update() if mixed_precision else None
                        optimizer.zero_grad()
                    torch.cuda.empty_cache()
                train_loss += loss.item()

                if verbose and batch_num % 500 == 0:
                    print(f"Batch {batch_num + 1}/{len(train_loader)} loss: {loss.item()}")
                    logging.info(f"Batch {batch_num + 1}/{len(train_loader)} loss: {loss.item()}")

                if batch_num == n_batches:
                    break

            train_loss /= n_batches
            lrs.step()
            if verbose:
                print(f"Epoch {epoch} learning rate: {lrs.get_last_lr()[0]}")
                logging.info(f"Epoch {epoch} learning rate: {lrs.get_last_lr()[0]}")
            writer.add_scalar('Loss/train', train_loss, epoch)
            if verbose:
                print(f"Epoch [{epoch}] train loss: {train_loss}")
                logging.info(f"Epoch [{epoch}] train loss: {train_loss}")

            data['train_loss'] = train_loss

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_num, batch in enumerate(val_loader):
                    inputs, labels = batch
                    if isinstance(model, transformers.RobertaForMaskedLM) or isinstance(model,
                                                                                        transformers.ReformerForMaskedLM):
                        labels[inputs != mask_token_index] = -100
                        outputs = model(inputs, labels=labels)
                        loss = outputs.loss
                    else:
                        outputs = model(inputs)
                        loss = loss_fn(outputs, inputs, labels)
                    val_loss += loss.item()
                    if batch_num == int(batch_share * len(val_loader)):
                        # early leave of epoch
                        break
                    if batch_num == n_batches:
                        break
            val_loss /= n_batches
            writer.add_scalar('Loss/val', val_loss, epoch)
            data['val_loss'] = val_loss
            writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)
            if verbose:
                print(f"Epoch {epoch} validation loss: {val_loss}")
                logging.info(f"Epoch {epoch} validation loss: {val_loss}")

            if data['val_loss'] < best_loss:
                best_loss = data['val_loss']
                best_model = model
                if save_best:
                    torch.save(best_model.state_dict(), save_path)

            history.append(data)


    except KeyboardInterrupt:
        torch.save(model.state_dict(), "../data/checkpoints/electra_v1")


def main():
    dataset = FinDataset("../../config.yml")
    print(dataset[0][0].shape)
    l = MaskedCrossEntropyLoss(50264)
    print(l(torch.randn((1600, 50000)), dataset[0][0], dataset[0][1]))


if __name__ == "__main__":
    main()
