import os
from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

from .dataset import Dataset

DATA_DIR = "data"
NYSE_PRICES = os.path.join(DATA_DIR, 'nyse', 'prices-split-adjusted.csv')
# TODO: Currently only using open for one less dimension
# FEATURES = ['date', 'symbol', 'open', 'close', 'low', 'high', 'volume']
FEATURES = ['date', 'symbol']
TIME_FEATURES = ['open', 'close', 'low', 'high', 'volume', 'movement', 'gspc', 'vix']
DEFAULT_TIME_FEATURES = ['open', 'close']

START_DATE = pd.to_datetime('2010-01-04')
TRAIN_VAL_SPLIT = pd.to_datetime('2014-01-04')
TRAIN_TEST_SPLIT = pd.to_datetime('2016-01-04')
END_DATE = pd.to_datetime('2016-12-30')

FINAL_TEST_SPLIT = pd.to_datetime('2012-12-31')


class NyseStocksDataset(Dataset):
    def __init__(self, name: str = 'NyseStocksDataset',
                 file_path: str = NYSE_PRICES,
                 epsilon: int = 0.01,  # Good classes distribution: 0.004
                 look_back: int = 7,
                 forecast_out: int = 1,
                 features: List[str] = DEFAULT_TIME_FEATURES,
                 companies: List[int] = None,
                 incl_test: bool = False):
        super().__init__(name)
        self.prices = None
        self.file_path = file_path
        self.file_dir, _ = os.path.split(self.file_path)
        self.epsilon = epsilon
        self.incl_test = incl_test
        assert look_back > 0
        self.look_back = look_back
        assert forecast_out > 0
        self.forecast_out = forecast_out
        self.features = features
        self.companies = companies

    def load(self):
        """Load data"""
        self.logger.debug(
            'Reading NYSE stocks data (takes about 43 seconds)...')
        prices = pd.read_csv(self.file_path)
        prices['date'] = pd.to_datetime(prices['date'], errors='coerce')
        assert all((prices['date'] >= START_DATE) &
                   (prices['date'] <= END_DATE))
        if not self.incl_test:
            prices = prices[prices.date <= FINAL_TEST_SPLIT]
        self.prices = prices
        X, y = self.shape_data()
        X_train = X[X['date'] < TRAIN_TEST_SPLIT]
        y_train = y[X['date'] < TRAIN_TEST_SPLIT]
        X_test = X[X['date'] >= TRAIN_TEST_SPLIT]
        y_test = y[X['date'] >= TRAIN_TEST_SPLIT]
        # Or use cross_validation.train_test_split(X, y, test_size = 0.2)

        self._data = (X_train, y_train, X_test, y_test)

    def _shorten_feature_name(self, day, feature):
        if feature[:5] == 'gspc_':
            return f'd{day:02d}_g{feature[5].upper()}'
        if feature[:4] == 'vix_':
            return f'd{day:02d}_v{feature[4].upper()}'
        return f'd{day:02d}_{feature[0].upper()}'

    def _split_into_multiindex(self, columns):
        for c in columns:
            yield c[:3], c[4:]  # (day, feature)

    def shape_company_data(self, comp_prices):
        # Last days features
        previous = pd.concat([
            comp_prices[self.features].shift(i).rename(
                lambda c: self._shorten_feature_name(i, c), axis=1)
            for i in range(1, self.look_back + 1)], axis=1)
        # Store in MultiIndex DataFrame
        previous.columns = pd.MultiIndex.from_tuples(
            list(self._split_into_multiindex(previous.columns)),
            names=('day', 'feature'))
        # Add other features and store them at day column 'other'
        other_features = comp_prices[FEATURES].copy()
        other_features.columns = pd.MultiIndex.from_tuples(
            [(c, '') for c in other_features.columns],
            names=('day', 'feature'))
        comp_data = pd.concat([other_features, previous], axis=1)
        # Add label for sorting (splitted afterwards)
        comp_data['label'] = self.calculate_labels(comp_prices)
        # assert False
        if self.forecast_out == 1:
            return comp_data.iloc[self.look_back:]
        return comp_data.iloc[self.look_back:-self.forecast_out+1]

    def calculate_labels(self, prices):
        rel_dist = prices.shift(-self.forecast_out+1).close / prices.open - 1
        return self.get_movement(rel_dist)

    def get_movement(self, rel_dist):
        if self.epsilon is not None:
            labels = np.zeros(len(rel_dist))
            labels[rel_dist < -self.epsilon] = -1
            labels[rel_dist > self.epsilon] = 1
        else:
            labels = np.ones(len(rel_dist))
            labels[rel_dist < 0] = -1
        return labels

    def load_vix(self):
        vix = pd.read_csv(os.path.join(self.file_dir, 'vix.csv'), skiprows=1)
        vix.columns = [x.lower() for x in vix.columns]
        vix.columns = [x.replace(" ", "_") for x in vix.columns]
        vix.date = pd.to_datetime(vix.date, errors='coerce')
        vix = vix[vix.date.between(START_DATE, END_DATE)]
        return vix

    def load_gspc(self):
        gspc = pd.read_csv(os.path.join(self.file_dir, 'gspc.csv'))
        gspc.columns = [x.lower() for x in gspc.columns]
        gspc.columns = [f'gspc_{x.replace(" ", "_")}' if x != 'date' else x
                        for x in gspc.columns]
        gspc.date = pd.to_datetime(gspc.date, errors='coerce')
        gspc = gspc[gspc.date.between(START_DATE, END_DATE)]
        return gspc

    def enhance_features(self, prices):
        if 'movement' in self.features:
            rel_dist = prices.close / prices.open - 1
            prices['movement'] = self.get_movement(rel_dist)
        if any([x[:4] == 'gspc' for x in self.features]):
            gspc = self.load_gspc()
            prices = prices.merge(gspc)
        if any([x[:3] == 'vix' for x in self.features]):
            vix = self.load_vix()
            prices = prices.merge(vix, how='left', sort=False)
            # assert False
        return prices

    def shape_data(self):
        prices = self.prices.copy()
        prices = self.enhance_features(prices)
        merged = []
        grouped = prices.groupby(prices.symbol, sort=False)
        # companies = pd.Series(grouped.keys())
        for comp_symbol, comp_prices in tqdm(grouped):
            if self.companies is None or comp_symbol in self.companies:
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
