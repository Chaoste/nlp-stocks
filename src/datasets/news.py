import os
import logging

import pandas as pd


NEWS_FILE = os.path.join("data", "preprocessed", "news.csv")


class NewsDataset():

    def __init__(self, name: str, file_path: str):
        self.name = name
        self._data = None
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

    # Return DataFrame with indexed rows and the column 'content'
    def get_articles(self):
        indexed_contents = self.data()[['content']]
        return indexed_contents

    def data(self) -> (pd.DataFrame):
        """Return data, load if necessary"""
        if self._data is None:
            self.load()
        return self._data
