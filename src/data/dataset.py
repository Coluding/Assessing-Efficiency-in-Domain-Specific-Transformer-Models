from torch.utils.data import Dataset
from logging.config import dictConfig
import logging
import transformers

from src.data.preprocessing import Database
from src.utils.logging import LogConfigBase




class FinDataset(Dataset):
    def __init__(self, database: Database, tokenizer: transformers.PreTrainedTokenizer):
        self.database = database
        dictConfig(LogConfigBase().dict())
        self.logger = logging.getLogger("Dataset Logger")
        self.database.inject_loggers(self.logger)
        self.tokenizer = tokenizer

        self.data = self.load_data()
        # TODO: check whether tokenizing all data at once is feasible or if it should be done in the __getitem__ method

    def load_data(self):
        return self.database.read_all_rows()

    def tokenize_data(self, data: str):
        return self.tokenizer(data, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass