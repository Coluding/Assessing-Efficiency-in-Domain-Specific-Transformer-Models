import sqlite3
from typing import List
from nltk import sent_tokenize, word_tokenize
from transformers import AutoTokenizer
import torch
import yaml

from src.utils.utils import timing_decorator


class Database:
    __DEBUG__ = False

    def __init__(self, config_path: str, debug: bool = False):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.conn = sqlite3.connect(self.config["data"]["path"])
        self.table_name = self.config["data"]["table_name"]
        self.logger = None
        self.__class__.__DEBUG__ = debug


    def inject_loggers(self, logger):
        self.logger = logger

    @timing_decorator(active=True if __DEBUG__ else False)
    def read_all_rows(self) -> List:
        """
        Reads all rows from the database.
        :return: List of rows.
        """
        try:
            if self.logger:
                self.logger.info("Reading all rows from database.")
            query = f"SELECT * FROM {self.table_name}"
            cursor = self.conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            if self.logger:
                self.logger.info("Finished reading all rows from database.")
            return results

        except sqlite3.Error as e:
            if self.logger:
                self.logger.error(f"SQLite error: {e}")
            return None

        finally:
            if self.logger:
                self.logger.info("Closing database connection.")
            self.conn.close()

    @timing_decorator(active=True if __DEBUG__ else False)
    def read_limited_rows(self, limit: int, offset: int = 0) -> List:
        """
        Reads a limited number of rows from the database.
        :param limit: Limit of the query.
        :param offset: Offset of the query
        :return: List of the items with length limit.
        """

        try:
            query = f"SELECT report FROM {self.table_name} LIMIT {limit} OFFSET {offset}"
            cursor = self.conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            return results

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return None

        finally:
            self.conn.close()


class Preprocessing:
    __DEBUG__ = False

    def __init__(self, config_path: str, debug: bool = False):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.context_length = self.config["context_length"]

        self.text = None
        self.texts = None
        self.tokenizer = None
        self.__class__.__DEBUG__ = debug

    def set_texts(self, text: str):
        self.text = text

    def set_texts(self, text: List[str]):
        self.text = text

    def set_tokenizer(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    @timing_decorator(active=True if __DEBUG__ else False)
    def tokenize_and_pad_list_of_reports(self, texts: List[str]) -> torch.Tensor:
        """
        Tokenizes and pads a list of reports.
        :param texts: List of reports
        :return: Tensor of tokenized and padded reports.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been set yet.")

        tokenized_texts: torch.Tensor = self.tokenizer(texts, padding=True,
                                                       return_tensors="pt")

        padding_length: int = ((self.context_length - (tokenized_texts["input_ids"].shape[1] % self.context_length))
                               % self.context_length)

        # Texts will be split in contexts of length context_length. Therefore, the padding length has to be a multiple
        # of context_length. The ending of each report should not be mixed with the beginning of the next report.
        # Therefore, padding is added to the end of each report.
        padded_texts: torch.Tensor = torch.nn.functional.pad(tokenized_texts["input_ids"], (0, padding_length),
                                                             value=1)

        return padded_texts

    @timing_decorator(active=True if __DEBUG__ else False)
    def chunk_tokenized_reports(self, tokenized_reports: torch.Tensor) -> torch.Tensor:
        """
        Chunks a tensor of tokenized reports into contexts of length context_length.
        :param tokenized_reports: Tensor of tokenized reports.
        :return: Tensor of tokenized contexts.
        """
        if tokenized_reports.shape[1] % self.context_length != 0:
            raise ValueError("The number of tokens in the reports is not a multiple of the context length.")
        return tokenized_reports.reshape(-1,  self.context_length)

    @timing_decorator(active=True if __DEBUG__ else False)
    def split_text_into_sentences(self) -> List[str]:
        """
        Splits a text into sentences.
        :param text: Text to split.
        :return: List of sentences.
        """
        if self.text is None:
            raise ValueError("Text has not been set yet.")
        return sent_tokenize(self.text, language="english")

    @timing_decorator(active=True if __DEBUG__ else False)
    def split_into_words(self, text: str) -> List[str]:
        """
        Tokenizes a text into words.
        :param text: Text to tokenize.
        :return: List of words.
        """
        if self.text is None:
            raise ValueError("Text has not been set yet.")
        return word_tokenize(text)

    @timing_decorator(active=True if __DEBUG__ else False)
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes a text into tokens
        :param text: Text to tokenize.
        :return: List of tokens.
        """
        return self.tokenizer(text)

    def split_into_contexts(self, text: str) -> List[str]:
        """
        Splits a text into contexts of length context_length.
        :param text: Text to split.
        :return: List of contexts.
        """
        pass



"""
from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
db = Database("../config.yml")
rows = [x[0] for x in db.read_limited_rows(10,0)]

t = tokenizer(rows, return_tensors="pt")
print("Hello")
"""