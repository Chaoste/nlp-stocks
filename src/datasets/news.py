import os
import logging

import pandas as pd


NEWS_FILE = os.path.join("data", "preprocessed", "news-v2.csv")


class NewsDataset():

    def __init__(self, name: str, file_path: str):
        self.name = name
        self._data = None
        self._new_to_old_idx = None
        self._old_to_new_idx = None
        self.file_path = file_path or NEWS_FILE
        self.logger = logging.getLogger(__name__)

    def __str__(self) -> str:
        return self.name

    def load(self):
        """Load data"""
        self.logger.debug('Reading news data (takes about 45 seconds)...')
        # Columns: date[datetime64[ns]], filename, content, reuters[bool]
        df_news = pd.read_csv(self.file_path, index_col=0)
        self._data = df_news
        self.new_to_old_idx = df_news.old_idx.to_dict()
        self.old_to_new_idx = {v: k for k, v in self.new_to_old_idx.items()}

    def get_new_index(self, old_idx):
        self.data()
        return self.old_to_new_idx[old_idx]

    def get_old_index(self, new_idx):
        self.data()
        return self.new_to_old_idx[new_idx]

    # Return DataFrame with indexed rows and the column 'content'
    def get_articles(self):
        indexed_contents = self.data()[['content']]
        return indexed_contents

    def data(self) -> (pd.DataFrame):
        """Return data, load if necessary"""
        if self._data is None:
            self.load()
        return self._data
