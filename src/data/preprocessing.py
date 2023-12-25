import sqlite3
from typing import List, Iterable, Generator, Union
from nltk import sent_tokenize, word_tokenize
from transformers import AutoTokenizer
import torch
import yaml
import csv
from tqdm import tqdm
from time import time

import sys
sys.path.insert(0, "../../")

from src.utils.utils import timing_decorator, remove_none_from_list_decorator


class Database:
    __DEBUG__ = False

    def __init__(self, config_path: str, debug: bool = False):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.use_csv = self.config["data"]["use_csv"]
        if not self.use_csv:
            self.conn = sqlite3.connect(self.config["data"]["path"])
        self.table_name = self.config["data"]["table_name"]
        self.chunk_table_name = self.config["data"]["chunk_table_name"]
        self.logger = None
        self.__class__.__DEBUG__ = debug
        self.is_closed = False

    def __del__(self):
        if not self.use_csv:
            self.conn.close()

    def chunked_len(self):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.chunk_table_name}")
        return cursor.fetchone()[0]

    def original_len(self):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        return cursor.fetchone()[0]

    def inject_loggers(self, logger):
        self.logger = logger

    def read_data_from_csv(self) -> List[str]:
        if self.config["data"]["csv_path"] is None:
            raise ValueError("Csv path must not be empty")

        with open(self.config["data"]["csv_path"], 'r') as file:
            reader = csv.reader(file)
            data = [row for row in reader]
        return data


    #@remove_none_from_list_decorator(index_to_check=2)
    def read_chunked_rows(self, limit: int = 0, offset: int = 0) -> List[List[str]]:

        try:
            if self.logger:
                self.logger.info("Reading chunked rows from database.")

            if limit == 0 and offset == 0:
                query = f'SELECT chunk_id, original_id, text FROM {self.chunk_table_name}'
            else:
                query = f'SELECT chunk_id, original_id, text FROM {self.chunk_table_name} LIMIT {limit} OFFSET {offset}'
            cursor = self.conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            if self.logger:
                self.logger.info("Finished reading chunked rows from database.")
            return results

        except sqlite3.Error as e:
            print("SQLite error: %s" % e)
            if self.logger:
                self.logger.error(f"SQLite error: {e}")
            return None

    @timing_decorator(active=True if __DEBUG__ else False)
    def read_all_rows(self) -> List:
        """
        Reads all rows from the database.
        :return: List of rows.
        """

        try:
            if self.logger:
                self.logger.info("Reading all rows from database.")
            query = f'SELECT "index", report FROM {self.table_name}'
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

    @timing_decorator(active=True if __DEBUG__ else False)
    def read_limited_rows(self, limit: int, offset: int = 0) -> List:
        """
        Reads a limited number of rows from the database.
        :param limit: Limit of the query.
        :param offset: Offset of the query
        :return: List of the items with length limit.
        """

        try:
            query = f'SELECT "index", report FROM {self.table_name} LIMIT {limit} OFFSET {offset}'
            cursor = self.conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            return results

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return None

    def _search_for_sentence_end(self, words: List[str], end_index: int) -> int:
        """
        Searches for the end of a sentence in a list of words.
        :param words: List of words to search in.
        :param end_index: Starting index for the search.
        :return: Index of the end of the sentence (index of the sentence-ending punctuation).
        """
        sentence_end_punctuations = {".", "!", "?"}

        add_on = 0
        while end_index < len(words):
            if words[end_index] in sentence_end_punctuations:
                # Check if it's a typical sentence end or a special case like an abbreviation/number
                if end_index < len(words) - 1:
                    if not (words[end_index] == "." and words[end_index - 1].isnumeric()):
                        return add_on
                else:
                    # If it's the last word, and it's a punctuation, it's the end of a sentence
                    return end_index
            end_index += 1
            add_on += 1

        return add_on

    def _split_original_report_into_contexts_and_write_to_db(self, text: str, context_length: int, original_id: int):

        """
        Splits a text into contexts of length context_length.
        :param text: Text to split.
        :return: List of contexts.
        """
        chunks = self._split_into_contexts(text, context_length)
        self._insert_chunk_into_db(chunks, original_id)

    def _split_into_contexts(self, text, context_length) -> List[str]:
        """
        Splits a text into contexts of length context_length.
        :param text: Text to split.
        :return: List of contexts.
        """

        if text is None:
            return

        words = word_tokenize(text)
        for i in range(0, len(words), context_length):
            sentence_add_on = self._search_for_sentence_end(words, i + context_length)
            if i == 0:
                start = 0
                prev_sentence_add_on = sentence_add_on
            else:
                start = i + prev_sentence_add_on + 1
                prev_sentence_add_on = sentence_add_on
            yield " ".join(words[start:(i + context_length + sentence_add_on)])

    def _insert_chunk_into_db(self, chunks: Iterable[str], original_id):
        """
        Inserts a chunk into the database.
        :param chunk: Chunk to insert.
        :return: None
        """

        for chunk in chunks:
            try:
                query = f"INSERT INTO {self.chunk_table_name} (original_id, text) VALUES (?, ?)"
                cursor = self.conn.cursor()
                cursor.execute(query, (original_id, chunk))


            except sqlite3.Error as e:
                raise e
                print(f"SQLite error: {e}")

    def create_chunked_table(self):
        cursor = self.conn.cursor()
        cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.chunk_table_name} (
                        original_id INTEGER,
                        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT
                    )
                """)
        self.conn.commit()
        self.conn.close()

    @timing_decorator(active=True)
    def split_original_reports_into_contexts_and_write_to_db(self, reports: List[List[Union[str, int]]],
                                                             context_length: int) -> None:
        """
        Splits a text into contexts of length context_length.
        :param text: Text to split.
        :return: List of contexts.
        """

        print("Writing split reports to database.")
        for report in tqdm(reports):
            if report[1] is not None:
                self._split_original_report_into_contexts_and_write_to_db(report[1], context_length, report[0])

        if self.logger:
            self.logger.info("Finished writing split reports to database.")

        print("Finished writing split reports to database.")

        self.conn.commit()


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
        return tokenized_reports.reshape(-1, self.context_length)

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

    def tokenize_and_pad_max_length(self, text: str):
        return self.tokenizer(text, padding="max_length", max_length=self.context_length,
                              return_tensors="pt", truncation=True)["input_ids"]

    def split_into_contexts(self, text: str) -> List[str]:
        """
        Splits a text into contexts of length context_length.
        :param text: Text to split.
        :return: List of contexts.
        """
        pass



def main():
    db = Database("../config.yml")
    db2 = Database("../config.yml")
    rows = db.read_all_rows()
    db.create_chunked_table()
    db2.split_original_reports_into_contexts_and_write_to_db(rows, 1300)


if __name__ == "__main__":
    main()
