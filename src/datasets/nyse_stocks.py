import os

import pandas as pd

from .dataset import Dataset

DATA_DIR = "data"
NYSE_PRICES = os.path.join(DATA_DIR, 'nyse', 'prices.csv')
# TODO: Currently only using open for one less dimension
# FEATURES = ['date', 'symbol', 'open', 'close', 'low', 'high', 'volume']
FEATURES = ['date', 'symbol']

START_DATE = pd.to_datetime('2010-01-04')
TRAIN_VAL_SPLIT = pd.to_datetime('2014-01-04')
TRAIN_TEST_SPLIT = pd.to_datetime('2016-01-04')
END_DATE = pd.to_datetime('2016-12-30')


class NyseStocksDataset(Dataset):
    def __init__(self, name: str = 'NyseStocksDataset',
                 file_path: str = NYSE_PRICES,
                 epsilon: int = 0.003,
                 forecast_out: int = 7):
        super().__init__(name)
        self.prices = None
        self.file_path = file_path
        self.epsilon = epsilon
        self.forecast_out = forecast_out

    def load(self):
        """Load data"""
        self.logger.debug(
            'Reading NYSE stocks data (takes about 6 seconds)...')
        prices = pd.read_csv(self.file_path)
        prices['date'] = pd.to_datetime(prices['date'], errors='coerce')
        assert all((prices['date'] >= START_DATE) &
                   (prices['date'] <= END_DATE))

        self.prices = prices
        X, y = self.shape_data()
        X_train = X[X['date'] < TRAIN_TEST_SPLIT]
        y_train = y[X['date'] < TRAIN_TEST_SPLIT]
        X_test = X[X['date'] >= TRAIN_TEST_SPLIT]
        y_test = y[X['date'] >= TRAIN_TEST_SPLIT]
        # Or use cross_validation.train_test_split(X, y, test_size = 0.2)

        self._data = (X_train, y_train, X_test, y_test)

    def shape_data(self):
        prices = self.prices.copy()
        # Efficiently calculate relative distance
        prices['rel_dist'] = (1 - (self.prices['close'] / self.prices['open']))
        merged_X, merged_y = [], []
        for sym, comp_prices in prices.groupby(prices.symbol, sort=False):
            previous = pd.concat([comp_prices.open.shift(i).rename(f'day_{i}')
                                  for i in range(1, self.forecast_out + 1)],
                                 axis=1)
            comp_X = pd.concat([comp_prices[FEATURES], previous], axis=1)\
                .iloc[self.forecast_out:]
            comp_y = comp_prices.rel_dist.iloc[self.forecast_out:]\
                .apply(self.get_label)
            merged_X.append(comp_X)
            merged_y.append(comp_y)
        X = pd.concat(merged_X)
        # Symbols stay randomly positioned, not by their initial position
        X.sort_values(by='date', inplace=True)
        y = pd.concat(merged_y)
        return X, y

    def get_label(self, rel_dist):
        return -1 if rel_dist < -self.epsilon else 1 \
            if rel_dist > self.epsilon else 0

    def get_all_prices(self):
        self.data()
        return self.prices

    def get_prices(self, stock_symbol):
        self.data()
        return self.prices[self.prices['symbol'] == stock_symbol]
