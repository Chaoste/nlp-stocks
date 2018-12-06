import os
from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

from .dataset import Dataset

DATA_DIR = "data"
NYSE_PRICES = os.path.join(DATA_DIR, 'nyse', 'prices.csv')
# TODO: Currently only using open for one less dimension
# FEATURES = ['date', 'symbol', 'open', 'close', 'low', 'high', 'volume']
FEATURES = ['date', 'symbol']
TIME_FEATURES = ['open', 'close', 'low', 'high', 'volume']

START_DATE = pd.to_datetime('2010-01-04')
TRAIN_VAL_SPLIT = pd.to_datetime('2014-01-04')
TRAIN_TEST_SPLIT = pd.to_datetime('2016-01-04')
END_DATE = pd.to_datetime('2016-12-30')


class NyseStocksDataset(Dataset):
    def __init__(self, name: str = 'NyseStocksDataset',
                 file_path: str = NYSE_PRICES,
                 epsilon: int = 0.004,
                 look_back: int = 7,
                 forecast_out: int = 1,
                 features: List[str] = TIME_FEATURES):
        super().__init__(name)
        self.prices = None
        self.file_path = file_path
        self.epsilon = epsilon
        assert look_back > 0
        self.look_back = look_back
        assert forecast_out > 0
        self.forecast_out = forecast_out
        self.features = features

    def load(self):
        """Load data"""
        self.logger.debug(
            'Reading NYSE stocks data (takes about 43 seconds)...')
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

    def shape_company_data(self, comp_prices):
        # Last days features
        previous = pd.concat([
            comp_prices[self.features].shift(i).rename(
                lambda c: f'day_{i}_{c[0].upper()}', axis=1)
            for i in range(1, self.look_back + 1)], axis=1)
        # Store in MultiIndex DataFrame
        previous.columns = pd.MultiIndex.from_tuples(
            [(c[:-2], c[-1:]) for c in previous.columns],
            names=('day', 'feature'))
        # Add other features and store them at day column 'other'
        other_features = comp_prices[FEATURES].copy()
        other_features.columns = pd.MultiIndex.from_tuples(
            [(c, '') for c in other_features.columns],
            names=('day', 'feature'))
        comp_data = pd.concat([other_features, previous], axis=1)
        # Add label for sorting (splitted afterwards)
        comp_data['label'] = self.calculate_labels(comp_prices)
        if self.forecast_out == 1:
            return comp_data.iloc[self.look_back:]
        return comp_data.iloc[self.look_back:-self.forecast_out+1]

    def calculate_labels(self, prices):
        rel_dist = prices.shift(-self.forecast_out+1).close / prices.open - 1
        if self.epsilon is not None:
            labels = np.zeros(len(rel_dist))
            labels[rel_dist < -self.epsilon] = -1
            labels[rel_dist > self.epsilon] = 1
        else:
            labels = np.ones(len(rel_dist))
            labels[rel_dist < 0] = -1
        return labels

    def shape_data(self):
        prices = self.prices.copy()
        merged = []
        for _, comp_prices in tqdm(prices.groupby(prices.symbol, sort=False)):
            comp_data = self.shape_company_data(comp_prices)
            merged.append(comp_data)
        data = pd.concat(merged).sort_index()
        X = data.drop(columns='label', level=0)
        X.columns = X.columns.remove_unused_levels()
        y = data['label']
        return X, y

    def get_all_prices(self):
        self.data()
        return self.prices

    def get_prices(self, stock_symbol):
        self.data()
        return self.prices[self.prices['symbol'] == stock_symbol]
