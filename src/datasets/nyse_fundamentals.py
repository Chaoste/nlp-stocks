import os
import logging

import pandas as pd
import numpy as np

DATA_DIR = "data"
NYSE_FUNDAMENTALS = os.path.join(DATA_DIR, 'nyse', 'fundamentals.csv')


class NyseFundamentalsDataset():

    def __init__(self, name: str = 'NyseFundamentalsDataset',
                 file_path: str = NYSE_FUNDAMENTALS,
                 load: bool = False):
        self.name = name
        self._data = None
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)
        if load:
            self.load()

    def __str__(self) -> str:
        return self.name

    def load(self):
        """Load data"""
        self.logger.debug(
            'Reading NYSE fundamentals data (takes about ? seconds)...')
        fundamentals = pd.read_csv(self.file_path, index_col=0)
        assert fundamentals.iloc[854]['For Year'] == 1215, 'Year Bug for IPG is not present'
        fundamentals.loc[854, 'For Year'] = int(fundamentals.iloc[854]['Period Ending'][:4])
        self._data = fundamentals

    def get_revenue(self, sym) -> str:
        comp = self.get_fundamentals(sym)
        if comp is None:
            return np.nan
        return comp.sort_values(by='For Year').iloc[0]['Total Revenue']

    def get_fundamentals(self, sym) -> str:
        fundamentals = self.data()
        comp = fundamentals[fundamentals['Ticker Symbol'] == sym]
        if len(comp) == 0:
            return None
        # assert len(comp) > 1, f'No fundamentals available for company {sym}'
        return comp

    def data(self) -> (pd.DataFrame):
        """Return data, load if necessary"""
        if self._data is None:
            self.load()
        return self._data
