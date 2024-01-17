from tokenizers import Tokenizer


def get_tokenizer(tokenizer_path: str) -> Tokenizer:
    """
    Helper function to load Huggingface tokenizers
    :param tokenizer_path:
    :return: tokenizer instance
    """

    tokenizer = Tokenizer.from_file(path=tokenizer_path)

    tokenizer.add_special_tokens([
        "<PAD>",  # Padding values must be 0
        "<MASK>",  # Masked tokens must be 1
        "<BOS>",  # BOS must be 2
        "<EOS>",  # EOS must be 3
        "<SEP>",  # SEP must be 4
        "<UNK>",  # UNK must be 5 (not relevant for BBPE but still present in vocab)
    ])

    tokenizer.no_truncation()

    return tokenizer
