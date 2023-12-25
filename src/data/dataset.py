import torch
from torch.utils.data import Dataset, DataLoader
from logging.config import dictConfig
import logging
import transformers
import yaml
import logging
import numpy as np
from typing import List, Dict

from src.data.preprocessing import Preprocessing, Database
from src.utils.utils import timing_decorator, exception_decorator
from src.external.finroberta.finroberta.finroberta import FinRobertaTokenizer

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='app.log',
                    filemode='w')


class FinDataset(Dataset):
    __DEBUG__ = False

    @exception_decorator(exception_type=KeyError, message="Check if all fields are listed in the config file.")
    def __init__(self, config_file_path: str):

        with open(config_file_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.__class__.__DEBUG__ = self.config["debug"]
        self.database = Database(config_file_path)
        self.logger = logging
        self.database.inject_loggers(self.logger)

        if self.config["use_local_tokenizer"]:
            self.tokenizer = FinRobertaTokenizer(self.config["local_tokenizer"], self.config["context_length"])
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config["tokenizer"])
        self.masked_probability = self.config["masked_probability"]
        self.preprocessor = Preprocessing(config_file_path)
        self.preprocessor.set_tokenizer(self.tokenizer)
        self.dynamic_masking = self.config["dynamic_masking"]
        self.lazy_loading = self.config["lazy_loading"]
        try:
            self.mask_token_id = self.tokenizer.mask_token_id
        except AttributeError:
            self.mask_token_id = self.tokenizer.get_vocab()["<mask>"]
        self.large_memory = self.config["large_memory"]
        self.pad_token_id = self.tokenizer.pad_token_id

        self.data: List[str] = None
        self.tokenized_data: Dict[torch.Tensor] = None
        if self.large_memory:
            print("Loading chunked data into memory...")
            self.chunked_data: List[str] = self.load_chunked_data(0, 0)
            print("Chunked data loaded into memory.")
        self.tokenized_chunked_data: torch.Tensor = None
        self.domain_words: List[str] = None
        self.synonym_map: Dict[str, List[str]] = None
        self.token_to_vocab: Dict[str, int] = None

        self.create_token_to_vocab()
        self.non_maskable_tokens = list(range(self.config["number_of_ignorable_tokens_for_masking"]))
        self.non_maskable_tokens.extend([self.tokenizer.get_vocab()[","], self.tokenizer.get_vocab()["."],
                                         self.tokenizer.get_vocab()["'"],self.tokenizer.get_vocab()["!"],
                                         self.tokenizer.get_vocab()["?"]])

        self.seeds = [i for i in range(self.__len__())]

    def load_data(self):
        return [x[0] for x in self.database.read_all_rows() if x[0] is not None]

    def load_data_limited(self, limit: int, offset: int = 0):
        return [x[0] for x in self.database.read_limited_rows(limit, offset)]

    def load_chunked_data(self, limit: int = 0, offset: int = 0):
        if self.data is not None:
            raise ValueError("This instance of the dataset is not set up for lazy loading. "
                             "Please set lazy_loading to True in the config file.")

        if self.config["data"]["use_csv"]:
            return [x[-1] for x in self.database.read_data_from_csv()]

        return [x[2] for x in self.database.read_chunked_rows(limit, offset)]

    def setup_data(self):
        """
        Sets up the data for the dataset.
        :return: None
        """
        if self.config["data"]["limit"]:
            self.data = self.load_data_limited(self.config["data"]["limit"], self.config["data"]["offset"])
        else:
            self.data = self.load_data()

    def inject_domain_specific_words(self, domain_specific_words: List[str], synonym_map: Dict[str, List[str]]):
        """
        Injects domain specific words and a synonym map into the preprocessor.
        :param domain_specific_words: Domain speciifc words as a list of words
        :param synonym_map: Synomym map as a dictionary of words to a list of synonyms.
        :return: None
        """
        self.domain_words = domain_specific_words
        self.synonym_map = synonym_map

    def setup_tokenized_and_chunked_data(self):
        """
        Tokenizes and chunks the data.
        :return: None
        """
        self.tokenized_data = self.preprocessor.tokenize_and_pad_list_of_reports(self.data)
        self.tokenized_chunked_data = self.preprocessor.chunk_tokenized_reports(self.tokenized_data)

    def create_token_to_vocab(self):
        self.token_to_vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def map_token_sequence_to_vocab(self, tokens: torch.Tensor):
        """
        Maps a sequence of tokens to the vocabulary.
        :param tokens: Sequence of tokens.
        :return: Sequence of tokens mapped to the vocabulary.
        """
        if self.token_to_vocab is None:
            self.create_token_to_vocab()
        return [self.token_to_vocab[token.item()] if not isinstance(tokens, list) else self.token_to_vocab[token]
                for token in tokens]

    def mask_sequence(self, sequence: torch.Tensor, seed: int = None):
        """
        Masks a sequence with a given probability while preserving padding structure.
        :param sequence: Sequence to be masked.
        :param seed: Seed for the random number generator.
        :return: Masked sequence.
        """

        # TODO: Inject domain words

        possible_masked_indices = torch.nonzero(torch.isin(sequence, torch.tensor(self.non_maskable_tokens)).bitwise_not().long())
        if not self.dynamic_masking:
            np.random.seed(seed)
        random_indices = np.random.choice(possible_masked_indices.squeeze(),
                                          int(len(possible_masked_indices) * self.masked_probability),
                                          replace=False)

        labels = sequence[random_indices]
        sequence[random_indices] = self.mask_token_id
        return sequence, labels

    def __len__(self):
        if self.data is None:
            return self.database.chunked_len()
        else:
            return self.database.original_len()

    def create_token_to_vocab(self):
        self.token_to_vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def __getitem__(self, index):
        if self.lazy_loading:
            if self.large_memory:
                report = self.chunked_data[index]
            else:
                report = self.load_chunked_data(limit=1, offset=index)[0]
            tokenized_report: torch.Tensor = self.preprocessor.tokenize_and_pad_max_length(report)[0]


        else:
            if self.data is None:
                raise ValueError("Data has not been set up yet. Please call setup_data() first.")
            if self.tokenized_data is None:
                raise ValueError("Tokenized data has not been set up yet. "
                                 "Please call setup_tokenized_and_chunked_data() first.")
            if self.chunked_data is None:
                raise ValueError("Chunked data has not been set up yet. "
                                 "Please call setup_tokenized_and_chunked_data() first.")
            tokenized_report = self.tokenized_chunked_data[index]

        tokenized_report_to_mask = tokenized_report.clone()
        if self.dynamic_masking:
            sequence, labels = self.mask_sequence(tokenized_report_to_mask)
        else:
            sequence, labels = self.mask_sequence(tokenized_report_to_mask, self.seeds[index])

        return sequence, tokenized_report


def main():
    dataset = FinDataset("../config.yml")
    #    dataset.mask_sequence(dataset[0])
    print(dataset[0][0].shape)


if __name__ == "__main__":
    main()
