import torch
from torch.utils.data import Dataset
from logging.config import dictConfig
import logging
import transformers
import yaml
import logging
import numpy as np
from typing import List, Dict

from src.data.preprocessing import Preprocessing, Database
from src.utils.utils import timing_decorator, exception_decorator


class FinDataset(Dataset):
    __DEBUG__ = False

    @exception_decorator(exception_type=KeyError, message="Check if all fields are listed in the config file.")
    def __init__(self, config_file_path: str, database: Database,
                 preprocessor: Preprocessing):

        with open(config_file_path, "r") as file:
            self.config = yaml.safe_load(file)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename='app.log',
                            filemode='w')

        self.__class__.__DEBUG__ = self.config["debug"]
        self.database = database
        self.logger = logging
        self.database.inject_loggers(self.logger)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config["tokenizer"])
        self.masked_probability = self.config["masked_probability"]
        self.preprocessor = preprocessor
        self.preprocessor.set_tokenizer(self.tokenizer)
        self.dynamic_masking = self.config["dynamic_masking"]

        self.data: List[str] = None
        self.tokenized_data: Dict[torch.Tensor] = None
        self.chunked_data: torch.Tensor = None
        self.domain_words: List[str] = None
        self.synonym_map: Dict[str, List[str]] = None
        self.token_to_vocab: Dict[str, int] = None

        self.setup_data()
        self.setup_tokenized_and_chunked_data()

        self.seeds = [i for i in range(self.__len__())]

    def load_data(self):
        return [x[0] for x in self.database.read_all_rows()]

    def load_data_limited(self, limit: int, offset: int = 0):
        return [x[0] for x in self.database.read_limited_rows(limit, offset)]

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
        self.token_to_vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.chunked_data = self.preprocessor.chunk_tokenized_reports(self.tokenized_data)

    def mask_sequence(self, sequence: torch.Tensor, seed: int = None):
        """
        Masks a sequence with a given probability.
        :param sequence: Sequence to be masked.
        :param seed: Seed for the random number generator.
        :return: Masked sequence.
        """
        padding_index: int = (sequence == 1).sum().item()
        if padding_index == 0:
            possible_masked_words = sequence
        else:
            possible_masked_words = sequence[: -padding_index]
        length_of_possible_masked_words = possible_masked_words.shape[0]

        if not self.dynamic_masking:
            np.random.seed(seed)
        random_indices = np.random.choice(np.arange(1, length_of_possible_masked_words),
                                          int(length_of_possible_masked_words * self.masked_probability))

        labels = sequence[random_indices]
        possible_masked_words[random_indices] = self.tokenizer.mask_token_id
        if padding_index == 0:
            masked_sequence = possible_masked_words
        else:
            masked_sequence = torch.cat([possible_masked_words, sequence[-padding_index:]])
        return masked_sequence, labels

    def __len__(self):
        return len(self.chunked_data)

    @timing_decorator(active=True if __DEBUG__ else False)
    def __getitem__(self, index):
        report = self.chunked_data[index]
        if self.dynamic_masking:
            sequence, labels = self.mask_sequence(report, self.seeds[index])
        else:
            sequence, labels = self.mask_sequence(report)

        return sequence, labels


def main():
    database = Database("../config.yml")
    preprocessor = Preprocessing("../config.yml", debug=True)
    dataset = FinDataset("../config.yml", database, preprocessor)
    dataset.mask_sequence(dataset.chunked_data[0])
    print(dataset[0][0].shape)


if __name__ == "__main__":
    main()