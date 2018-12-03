import os

import pandas as pd

from .dataset import Dataset

DATA_DIR = "data"
NYSE_PRICES = os.path.join(DATA_DIR, 'nyse', 'prices.csv')
FEATURES = ['open', 'close', 'low', 'high', 'volume']  # 'date' , 'symbol'

START_DATE = pd.to_datetime('2010-01-04')
TRAIN_VAL_SPLIT = pd.to_datetime('2014-01-04')
VAL_TEST_SPLIT = pd.to_datetime('2016-01-04')
END_DATE = pd.to_datetime('2016-12-30')


class NyseStocksDataset(Dataset):
    def __init__(self, name: str = 'NyseStocksDataset',
                 file_path: str = NYSE_PRICES):
        super(NyseStocksDataset, self).__init__(name)
        self.prices = None
        self.file_path = file_path

    def load(self):
        """Load data"""
        self.logger.debug(
            'Reading NYSE stocks data (takes about ? seconds)...')
        prices = pd.read_csv(self.file_path)
        prices['date'] = pd.to_datetime(prices['date'], errors='coerce')
        assert all((prices['date'] >= START_DATE) &
                   (prices['date'] <= END_DATE))
        self.prices = prices
        self._data = prices

    def get_prices(self, stock_symbol):
        self.data()
        return self.prices[self.prices['symbol'] == stock_symbol]
