import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
import logging
import numpy as np
from datetime import datetime
from tqdm import tqdm
from enum import Enum
from einops import rearrange

import sys
sys.path.append("../..")

from src.utils.torch_utils import DeviceDataLoader
from src.data.dataset import FinDataset


class LossReduction(Enum):
    NONE = "none"
    SUM = "sum"
    MEAN = "mean"


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, mask_token_index: int, logits: bool = True, reduction: LossReduction = LossReduction.MEAN):
        super().__init__()
        self.mask_token_index = mask_token_index
        self.logits = logits
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, masked_ground_truth: torch.Tensor, unmasked_ground_truth: torch.Tensor ) -> torch.Tensor:
        """
        Computes the masked cross entropy loss for a given prediction and ground truth.
        :param predictions: Shape (batch_size, sequence_length, vocab_size)
        :param masked_ground_truth: Shape (batch_size, sequence_length) with masked tokens
        :param unmasked_ground_truth:  Shape (batch_size, sequence_length) without masked tokens
        :return: The masked cross entropy loss
        """
        predictions = rearrange(predictions, 'b n c -> (b n) c')
        mask = (masked_ground_truth == self.mask_token_index).float()
        mask = rearrange(mask, 'b n -> (b n)')
        unmasked_ground_truth = rearrange(unmasked_ground_truth, 'b n -> (b n)')
        if self.logits:
            predictions = F.log_softmax(predictions, dim=-1)

        unreduced_loss = F.nll_loss(predictions, unmasked_ground_truth, reduction="none")

        if unreduced_loss.isnan().any():
            print("Loss is NaN. Skipping batch.")
            logging.info("Loss is NaN. Skipping batch.")
            return torch.tensor(10)

        if self.reduction == LossReduction.NONE:
            return unreduced_loss * mask
        elif self.reduction == LossReduction.SUM:
            return torch.sum(unreduced_loss * mask)
        elif self.reduction == LossReduction.MEAN:
            return torch.sum(unreduced_loss * mask) / torch.sum(mask)


def fit(epochs: int, model: nn.Module, loss_fn: nn.Module, train_loader: DeviceDataLoader,
        learning_rate: float, val_loader: DeviceDataLoader, optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler, save_path: str,
        save_best: bool = True, verbose: bool = True, loggable_params: dict = None, lrs_params: dict = None,
        iters_to_accumulate: int = 1, mixed_precision: bool = False, grad_clipping_norm: float = 1.0):
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
    torch.autograd.set_detect_anomaly(False)
    best_loss = float("inf")
    best_model = None
    history = []
    run_name = np.random.randint(0, 1000000000)
    id_full = datetime.now().strftime("%Y-%m-%d %H-%M") + str(run_name)
    logging.basicConfig(
        filename='./logging/run.log',
        filemode='w',
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S',
        level=logging.DEBUG
    )
    writer = SummaryWriter(log_dir=f'logging/runs/{id_full}')
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

        lrs.step()
        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        if verbose:
            print(f"Epoch [{epoch}] train loss: {train_loss}")
            logging.info(f"Epoch [{epoch}] train loss: {train_loss}")

        data['train_loss'] = train_loss

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                outputs = model(inputs)
                loss = loss(inputs, outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
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



def main():
    dataset = FinDataset("../config.yml")
    print(dataset[0][0].shape)
    l = MaskedCrossEntropyLoss(50264)
    print(l(torch.randn((1600, 50000)), dataset[0][0], dataset[0][1]))


if __name__ == "__main__":
    main()
