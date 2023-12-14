from torch.utils.data import Dataset
from logging.config import dictConfig
import logging
import transformers
import yaml
import logging

from src.data.preprocessing import Preprocessing, Database
from src.utils.utils import timing_decorator


class FinDataset(Dataset):
    __DEBUG__ = False

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
        limited_loader = self.config["data"]["limited_loader"]
        self.logger = logging
        self.database.inject_loggers(self.logger)
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.config["tokenizer"])
        self.preprocessor = preprocessor
        self.preprocessor.set_tokenizer(self.tokenizer)
        if limited_loader or limited_loader == 0:
            self.data = self.load_data_limited(limited_loader)
        else:
            self.data = self.load_data()
        self.tokenized_data = self.preprocessor.tokenize_and_pad_list_of_reports(self.data)
        self.chunked_data = self.preprocessor.chunk_tokenized_reports(self.tokenized_data)

    def load_data(self):
        return [x[0] for x in self.database.read_all_rows()]

    def load_data_limited(self, limit: int, offset: int = 0):
        return [x[0] for x in self.database.read_limited_rows(limit, offset)]

    def __len__(self):
        return len(self.data)

    @timing_decorator(active=True if __DEBUG__ else False)
    def __getitem__(self, index):
        report = self.chunked_data[index]
        return report



def main():
    database = Database("../config.yml")
    preprocessor = Preprocessing("../config.yml", debug=True)
    dataset = FinDataset("../config.yml", database, preprocessor)
    print(dataset[0].shape)


if __name__ == "__main__":
    main()