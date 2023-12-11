import sqlite3
from typing import List


class Database:
    def __init__(self, db_path: str, table_name: str = "K_reports_full"):
        self.conn = sqlite3.connect(db_path)
        self.table_name = table_name
        self.logger = None

    def inject_loggers(self, logger):
        self.logger = logger

    def read_all_rows(self) -> List:
        """
        Reads all rows from the database.
        :return: List of rows.
        """
        try:
            self.logger.info("Reading all rows from database.")
            query = f"SELECT * FROM {self.table_name}"
            cursor = self.conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            self.logger.info("Finished reading all rows from database.")
            return results

        except sqlite3.Error as e:
            self.logger.error(f"SQLite error: {e}")
            return None

        finally:
            self.logger.info("Closing database connection.")
            self.conn.close()

    def read_limited_rows(self, limit: int, offset: int = 0) -> List:
        """
        Reads a limited number of rows from the database.
        :param limit: Limit of the query.
        :param offset: Offset of the query
        :return: List of the items with length limit.
        """

        try:
            query = f"SELECT * FROM {self.table_name} LIMIT {limit} OFFSET {offset}"
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
    """
    Utility class for preprocessing data.
    """
    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """
        Tokenizes a given text.

        :param text: Text to tokenize.
        :return: List of tokens.
        """
        return [token.text for token in nlp(text)]


from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
db = Database("/home/lubi/Desktop/K_reports_full.sqlite")
print(len(tokenizer(db.read_limited_rows(5)[0][-10])["input_ids"]))