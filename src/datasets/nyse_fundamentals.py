import os
import logging

import pandas as pd

DATA_DIR = "data"
NYSE_FUNDAMENTALS = os.path.join(DATA_DIR, 'nyse', 'fundamentals.csv')


class NyseFundamentalsDataset():

    def __init__(self, name: str = 'NyseFundamentalsDataset',
                 file_path: str = NYSE_FUNDAMENTALS):
        self.name = name
        self._data = None
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)

    def __str__(self) -> str:
        return self.name

    def load(self):
        """Load data"""
        self.logger.debug(
            'Reading NYSE fundamentals data (takes about ? seconds)...')
        fundamentals = pd.read_csv(self.file_path, index_col=0)
        self._data = fundamentals

    def data(self) -> (pd.DataFrame):
        """Return data, load if necessary"""
        if self._data is None:
            self.load()
        return self._data
