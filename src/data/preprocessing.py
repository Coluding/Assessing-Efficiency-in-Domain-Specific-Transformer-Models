import sqlite3
from typing import List, Iterable, Generator, Union
from nltk import sent_tokenize, word_tokenize
from transformers import AutoTokenizer
import torch
import yaml
import csv
from tqdm import tqdm
import random
import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from tokenizers import Tokenizer
from typing import *
from spacy.lang.en import English
import logging
from pathlib import Path
from functools import partial
import numpy as np

import sys
sys.path.insert(0, "../../")

from src.utils.utils import timing_decorator, remove_none_from_list_decorator
from src.data.utils import DatasetInfo

logger = logging.getLogger(__name__)


class HFDatasetUtils:
    @staticmethod
    def make_datasets_document_electra(tokenizer: Tokenizer,
                                       dataset_info: Union[DatasetInfo, List[DatasetInfo]],
                                       dataset_sampling: Union[float, List[float]],
                                       cache_dir: str,
                                       dataset_dir: str,
                                       validation_set_size: Optional[int] = None,
                                       test_set_size: Optional[int] = None,
                                       training_set_size: int = -1,
                                       training_set_random_seed: int = 42,
                                       valid_test_split_random_seed: int = 42,
                                       num_proc: int = None,
                                       subset_ratio: Optional[float] = None,
                                       ) -> (Tuple)[
        torch.utils.data.dataset.Dataset,
        torch.utils.data.dataset.Dataset,
        torch.utils.data.dataset.Dataset
    ]:
        """
        Create the Pytorch datasets from HuggingFace.

        :param tokenizer: Already trained tokenizer
        :param dataset_info: DatasetInfo or list of DatasetInfo to be used to create the train, val and test sets
        :param dataset_sampling:
        :param dataset_dir: Directory path for the cache for the HuggingFace datasets library
        :param cache_dir: Directory to store the cache for the processed datasets.
        :param validation_set_size: Validation set size.
        :param test_set_size: Test set size. If None, then use the dataset["test"]. Default None.
        :param training_set_size: Default -1 to use all possible samples
        :param training_set_random_seed: Seed used only for shuffle the training set
        :param valid_test_split_random_seed: Seed used only for the split between test and validation sets.
            Required to ensure the validation set remains the same if seed is used.
        :param num_proc: Number of processor for the creation of the dataset
        :return: train dataset, validation dataset, test dataset
        """

        if isinstance(dataset_info, DatasetInfo):
            dataset_info = [dataset_info]
        HFDatasetUtils._validate_dataset_infos(dataset_infos=dataset_info)

        if isinstance(dataset_sampling, float):
            dataset_sampling = [dataset_sampling]

        assert len(dataset_info) == len(dataset_sampling)

        logger.info(f"Dataset preprocessing started")
        combined_datasets = []
        for d_info, d_sampling in zip(dataset_info, dataset_sampling):
            cache_path: Path = HFDatasetUtils._get_cache_path(dataset_info=d_info, cache_dir=cache_dir)

            if cache_path.exists():
                logger.info(f"Load a cached dataset from {cache_path}")
                processed_datasets = DatasetDict.load_from_disk(dataset_dict_path=str(cache_path))
                logger.info(f"Load completed")

            else:
                if d_info.save_local_path is  None:
                    logger.info(f"Load unprocessed dataset from {d_info.name} using local path {d_info.save_local_path}")
                    datasets: DatasetDict = load_from_disk(d_info.save_local_path)
                    logger.info(f"Load unprocessed dataset from disk completed")

                elif dataset_dir:
                    logger.info(f"Load unprocessed dataset from {d_info.name} using cache {dataset_dir}")
                    datasets: DatasetDict = load_dataset(d_info.name, d_info.subset,
                                                         cache_dir=f"{dataset_dir}\\.cache\\huggingface\\datasets",)
                else:
                    logger.info(f"Load unprocessed dataset from {d_info.name} "
                                f"{d_info.subset if d_info.subset else ''}")
                    datasets: DatasetDict = load_dataset(d_info.name, d_info.subset)

                assert isinstance(datasets, DatasetDict)
                logger.info(f"Load unprocessed dataset completed")

                # Encode the dataset
                # WARNING this step takes 8.30 for sentence segmentation + tokenization of Wikipedia dataset
                logger.info(f"Dataset preprocessing")

                if subset_ratio is not None:
                    logger.info(f"Dataset subsampling started")
                    for d_name in datasets:
                        datasets[d_name] = datasets[d_name].select(
                            np.arange(int(len(datasets[d_name]) * subset_ratio)))
                    logger.info(f"Dataset subsampling completed")

                # adjust to split the text into n words instead of sentences
                datasets: DatasetDict = datasets.map(function=partial(HFDatasetUtils._encode_by_batch,
                                                                      tokenizer=tokenizer,
                                                                      nlp=HFDatasetUtils._get_sentence_segmentation_model(),
                                                                      dataset_info=d_info
                                                                      ),
                                                     batched=True,
                                                     num_proc=num_proc,
                                                     remove_columns=d_info.text_columns)

                processed_datasets = DatasetDict()
                processed_datasets["train"] = datasets["train"]

                if d_info.validation_set_names:
                    processed_datasets["validation"] = concatenate_datasets([datasets[name]
                                                                             for name in d_info.validation_set_names])

                if d_info.test_set_names:
                    processed_datasets["test"] = concatenate_datasets([datasets[name]
                                                                       for name in d_info.test_set_names])

                # Remove all features which are not required by the models
                # to allow the concatenation across different datasets
                nested_features = [v for _, v in processed_datasets.column_names.items()]
                flatten_features = [item for items in nested_features for item in items]
                extra_cols = set(flatten_features) - {"input_ids", "label"}
                processed_datasets.remove_columns(list(extra_cols))

                logger.info(f"Cache this processed dataset into {cache_path}")
                processed_datasets.save_to_disk(dataset_dict_path=str(cache_path))
                # Workaround to force the processed_dataset to remove the extra columns
                processed_datasets = DatasetDict.load_from_disk(dataset_dict_path=str(cache_path))
                logger.info(f"Cache completed")

            if d_sampling < 1.0:
                logger.info(f"Dataset downsampling started")
                for d_name in processed_datasets:
                    processed_datasets[d_name] = processed_datasets[d_name].select(
                        np.arange(int(len(processed_datasets[d_name]) * d_sampling)))
                logger.info(f"Dataset downsampling completed")

            combined_datasets += [processed_datasets]

        logger.info(f"Dataset preprocessing completed")

        train_set = [combined_dataset["train"] for combined_dataset in combined_datasets]
        if len(train_set) > 1:
            train_set = concatenate_datasets(train_set)
        else:
            train_set = train_set[0]

        val_combined_dataset = [combined_dataset["validation"] for combined_dataset in combined_datasets
                                if "validation" in combined_dataset]
        if len(val_combined_dataset) > 1:
            val_set = concatenate_datasets(val_combined_dataset)
        elif len(val_combined_dataset) == 1:
            val_set = val_combined_dataset[0]
        else:
            val_set = None

        test_combined_dataset = [combined_dataset["test"] for combined_dataset in combined_datasets
                                 if "test" in combined_dataset]
        if len(test_combined_dataset) > 1:
            test_set = concatenate_datasets(test_combined_dataset)
        elif len(test_combined_dataset) == 1:
            test_set = test_combined_dataset[0]
        else:
            test_set = None

        assert len(train_set) > 0, "Your train set is empty"

        if test_set_size is None and validation_set_size is not None:
            # Case you extract a validation set from train set and no test set
            subset_indices = [validation_set_size, len(train_set) - validation_set_size]

            generator = torch.Generator().manual_seed(valid_test_split_random_seed)
            val_set, train_set = torch.utils.data.random_split(dataset=train_set,
                                                               lengths=subset_indices,
                                                               generator=generator)
        elif test_set_size is not None and validation_set_size is not None:
            # Case you extract a test and validation set from train set
            subset_indices = [test_set_size,
                              test_set_size + validation_set_size]  # 100 for Electra in their code source
            subset_indices += [len(train_set) - sum(subset_indices)]

            generator = torch.Generator().manual_seed(valid_test_split_random_seed)
            test_set, val_set, train_set = torch.utils.data.random_split(dataset=train_set,
                                                                         lengths=subset_indices,
                                                                         generator=generator)
        else:
            # Case you don't need to use data from train set for validation and/or test sets
            train_set = train_set.shuffle(seed=training_set_random_seed)

        # Ability to use smaller dataset
        if training_set_size != -1:
            if isinstance(training_set_size, float):
                lengths = [int(len(train_set) * training_set_size),
                           len(train_set) - int(len(train_set) * training_set_size)]
            generator = torch.Generator().manual_seed(training_set_random_seed)
            train_set, _ = torch.utils.data.random_split(dataset=train_set,
                                                         lengths=lengths,
                                                         generator=generator)

        assert val_set is not None and len(val_set) > 0, "Your validation set is empty"

        logger.info(f"Training size: {len(train_set)}")
        logger.info(f"Valid size: {len(val_set)}")

        if test_set:
            logger.info(f"Test size: {len(test_set)}")

        return train_set, val_set, test_set

    @staticmethod
    def _get_sentence_segmentation_model():
        """
        Util function to create the sentence segmentation model
        :return:
        """
        # Load Spacy model
        nlp = English()
        # nlp.max_length = 1000000 * 8 # Equivalent of 8 GB of memory / Required to have all text in memory
        # sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe("sentencizer")
        return nlp

    @staticmethod
    def _encode_by_batch(documents: Dict[str, List],
                         tokenizer: Tokenizer,
                         nlp: English,
                         dataset_info: DatasetInfo,
                         bos_token="<BOS>",
                         sep_token="<SEP>",
                         ) -> Dict[str, List]:
        """
        Perform sentence segmentaton and tokenization.

        :param documents: List of all documents (string)
        :param tokenizer: Tokenizer
        :param bos_token: BOS token to be added
        :param sep_token: SEP token to be added
        :return:
        """
        encoded_docs = []

        assert len(dataset_info.text_columns) >= 1
        # get length of dataset of the text column
        # insert the correct column in dataset info object
        for i_d in range(len(documents[dataset_info.text_columns[0]])):
            if dataset_info.sentence_segmentation:
                assert len(dataset_info.text_columns) == 1, dataset_info
                # sents creates a generator
                d = [s.text for s in nlp(documents[dataset_info.text_columns[0]][i_d]).sents]
            else:
                d = [documents[text_column][i_d] for text_column in dataset_info.text_columns]

            texts = [s + sep_token for s in d]
            # add CLS token to the beginning
            texts[0] = bos_token + texts[0]
            # the encoded doc contains of n senteces, the first sentence has a BOS/CLS token at the beginning
            encoded_doc = [encoding.ids for encoding in tokenizer.encode_batch(texts)]
            encoded_docs += [encoded_doc]

        if "label" in documents:
            return {"input_ids": encoded_docs,
                    "label": documents["label"]}
        else:
            return {"input_ids": encoded_docs}

    @staticmethod
    def _context_encode_by_batch(documents: Dict[str, List],
                                 tokenizer: Tokenizer,
                                 nlp: English,
                                 dataset_info: DatasetInfo,
                                 bos_token="<BOS>",
                                 sep_token="<SEP>",
                                 ) -> Dict[str, List]:
        pass


    @staticmethod
    def _validate_dataset_infos(dataset_infos: List[DatasetInfo]):
        """
            Validate if the list is correct
        :param dataset_infos:
        :return:
        """

        assert isinstance(dataset_infos, List)
        assert len(dataset_infos) >= 1

        assert all([dataset_info.is_pretraining for dataset_info in dataset_infos]) or \
               (len(dataset_infos) == 1 and dataset_infos[0].is_downstream)

    @staticmethod
    def _get_cache_path(dataset_info: DatasetInfo,
                        cache_dir: str) -> Path:
        """
        :param dataset_info:
        :param cache_dir:
        :return: the path for the cache file for the processed dataset
        """
        cache_path: Path = Path(cache_dir if cache_dir else "")
        cache_path /= "data"
        cache_path /= f"{dataset_info.name}-{dataset_info.subset}.cache"

        return cache_path

    @staticmethod
    def _get_sentence_segmentation_model():
        """
        Util function to create the sentence segmentation model
        :return:
        """
        # Load Spacy model
        nlp = English()
        # nlp.max_length = 1000000 * 8 # Equivalent of 8 GB of memory / Required to have all text in memory
        # sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe("sentencizer")
        return nlp


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
        csv.field_size_limit(sys.maxsize)
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


class PreprocessingUtils:
    @staticmethod
    def search_for_sentence_end(words: List[str], end_index: int) -> int:
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

    @staticmethod
    def split_hf_dataset_into_chunks(dataset: datasets.arrow_dataset.Dataset | datasets.DatasetDict, chunk_size: int,
                                     subset_size: int | float | None = None,  key: str = "train",):
        """
        Splits a huggingface dataset into chunks of size chunk_size.
        :param key:
        :param subset_size:
        :param dataset: Dataset to split.
        :param chunk_size: Size of the chunks.
        :return: List of chunks.
        """

        if isinstance(dataset, datasets.DatasetDict):
            if key not in dataset:
                raise ValueError(f"Dataset does not contain key {key}")
            dataset = dataset[key]

        if subset_size is not None:
            dataset = dataset.shuffle(seed=random.randint(0, 100))
            if isinstance(subset_size, float):
                subset_size = int(subset_size * len(dataset))
            dataset = dataset.select(range(subset_size))

        def split_text_into_chunks(text: str, chunk_size: int):
            words = word_tokenize(text)
            for i in range(0, len(words), chunk_size):
                yield " ".join(words[i:i + chunk_size])

        def process_dataset(sample):
            text = sample if isinstance(sample, str) else sample["text"]
            return [{"chunk": chunk} for chunk in split_text_into_chunks(text, chunk_size)]

        all_chunks = []
        for sample in dataset:
            all_chunks.extend(process_dataset(sample))

        # Create a new dataset from the list of chunks
        processed_dataset = datasets.Dataset.from_dict({"chunk": all_chunks})

        return processed_dataset







def main():
    from transformers import AutoTokenizer
    from src.data.utils import get_dataset_info
    from src.tokenizer.utils import get_tokenizer
    from src.data.collator import DataCollatorForDocumentElectra
    from src.config.electra_config import DocumentElectraConfig

    tokenizer = get_tokenizer("/home/lubi/Documents/Projects/Electra/rc2020_electra/models/ByteLevelBPETokenizer-vocab_size=30522-min_frequency=2")
    dataset_name = "JanosAudran/financial-reports-sec"
    #dataset_name = dataset_name.split("-")
    #dataset_name, dataset_subset = (dataset_name[0], dataset_name[1] if len(dataset_name) > 1 else None)

    dataset_infos = [get_dataset_info(dataset_name=dataset_name, dataset_subset="large-lite")]
    config = DocumentElectraConfig(
        vocab_size=30522,
        embedding_size=512,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size= 2048,
        max_sentence_length= 100,
        max_sentences= 10,
        max_position_embeddings= 256,
        max_length= 256,
    )
    train_set, val_set, test_set = HFDatasetUtils.make_datasets_document_electra(
        tokenizer=tokenizer,
        dataset_info=dataset_infos,
        cache_dir=None,
        dataset_dir=None,
        dataset_sampling=[1.0],
        training_set_size=-1,
        validation_set_size=100 * 1,
        test_set_size=0,

        training_set_random_seed=123,
        valid_test_split_random_seed=123,
        num_proc=1,
    subset_ratio=0.001)

    collator = DataCollatorForDocumentElectra(config=config)
    collator([train_set[0], train_set[1]])
    print(2)


if __name__ == "__main__":
    main()
