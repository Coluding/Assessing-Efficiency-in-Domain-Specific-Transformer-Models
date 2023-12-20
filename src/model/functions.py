import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import FinDataset
from src.data.preprocessing import Preprocessing, Database


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, mask_token_index: int, logits: bool = True):
        super().__init__()
        self.mask_token_index = mask_token_index
        self.logits = logits

    def forward(self, predictions: torch.Tensor, ground_truth: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cross entropy loss for masked tokens.
       :param predictions: Float tensor of shape (batch_size, sequence_length, vocab_size)
       :param ground_truth: Long tensor of shape (batch_size, sequence_length)
       :param labels: Long tensor of shape (batch_size, num_masked_tokens)
       :return: Cross entropy loss for masked tokens.
       """
        mask = ground_truth == self.mask_token_index
        predictions = predictions[mask]

        if self.logits:
            predictions = F.log_softmax(predictions, dim=-1)

        return F.nll_loss(predictions, labels)




def main():
    database = Database("../config.yml")
    preprocessor = Preprocessing("../config.yml", debug=True)
    dataset = FinDataset("../config.yml", database, preprocessor)
    print(dataset[0][0].shape)
    l = MaskedCrossEntropyLoss(50264)
    print(l(torch.randn((1600,50000)), dataset[0][0], dataset[0][1]))

if __name__ == "__main__":
    main()