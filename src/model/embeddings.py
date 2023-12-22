import torch
import torch.nn as nn

from xformers.components.positional_embedding import SinePositionalEmbedding


class ReversibleLongFinBertEmbedding(nn.Module):
    """
    Custom embedding layer for a BERT-like model with reversible layers.

    This embedding layer combines token embeddings, segment embeddings, and positional embeddings (sine-based).
    It supports the use of pretrained embeddings and is designed to work with models that utilize reversible layers.

    Attributes:
        positional_embedding (SinePositionalEmbedding): Sine-based positional embeddings.
        token_embedding (nn.Embedding or None): Embedding layer for tokens. If pretrained embeddings are used, this is initially None.
        segment_embedding (nn.Embedding): Embedding layer for segment ids.
        dropout (float): Dropout rate, not currently applied in the embedding.
        vocab_size (int or None): The size of the vocabulary.
        d_model (int): The dimensionality of the embeddings.
    """
    def __init__(self, d_model: int, use_pretrained_embeddings: bool = True, vocab_size: int = None, dropout: float = 0.):
        """
        Initialization of ReversibleLongFinBertEmbedding.

        :param d_model: The dimensionality of the embeddings.
        :param use_pretrained_embeddings: Flag indicating whether to use pretrained embeddings. Defaults to True.
        :param vocab_size: The size of the vocabulary. Required if not using pretrained embeddings.
        :param dropout: Dropout rate. Defaults to 0.0 (no dropout).
        """
        super().__init__()
        self.positional_embedding: SinePositionalEmbedding = SinePositionalEmbedding(d_model)

        if vocab_size is None and not use_pretrained_embeddings:
            raise ValueError("vocab_size has to be set if use_pretrained_embeddings is False")

        self.token_embedding: nn.Module = nn.Embedding(vocab_size, d_model, padding_idx=0) if not use_pretrained_embeddings else None
        self.segment_embedding: nn.Module = nn.Embedding(3, d_model, padding_idx=0)
        self.dropout: nn.Module = nn.Dropout(dropout)
        self.vocab_size: int = vocab_size
        self.d_model: int = d_model
        self.use_pretrained_embeddings: bool = use_pretrained_embeddings

        self.injected: bool = False

    def inject_pretrained_embeddings(self, pretrained_embeddings: torch.Tensor):
        """
         Injects pretrained token embeddings into the embedding layer.

        :param pretrained_embeddings: A tensor of pretrained token embeddings.
        :return: None
        """

        if pretrained_embeddings.shape[1] != self.vocab_size:
            raise ValueError("vocab_size does not match the size of the pretrained embeddings")

        if pretrained_embeddings.shape[2] != self.d_model:
            raise ValueError("d_model does not match the size of the pretrained embeddings")

        self.token_embedding = nn.Embedding.from_pretrained(pretrained_embeddings)
        self.injected = True

    def forward(self, sequence: torch.Tensor, segment_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the embeddings for a given sequence and segment IDs.

        :param sequence: input token ids of shape (batch_size, sequence_length)
        :param segment_ids: segment ids of shape (batch_size, sequence_length)
        :return: embeddings of shape (batch_size, sequence_length, d_model)
        """
        if self.use_pretrained_embeddings and not self.injected:
            raise ValueError("Pretrained embeddings have not been injected yet. "
                             "Call inject_pretrained_embeddings() first.")

        if segment_ids is None:
            segment_ids = torch.zeros_like(sequence)

        return self.dropout(
            self.positional_embedding(self.token_embedding(sequence)) + self.segment_embedding(segment_ids)
        )




def main():
    x = torch.randint(0,10000, (8,128)).to("cuda")
    segment_ids = torch.randint(0,3, (8,128)).to("cuda")
    emb = ReversibleLongFinBertEmbedding(512, use_pretrained_embeddings=False, vocab_size=10000,).to("cuda")

    print(emb(x, segment_ids).shape)


if __name__ == "__main__":
    main()
