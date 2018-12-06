import logging

import numpy as np
import pandas as pd

from src.datasets import NyseStocksDataset


def find_cut(y, c, n_samples):
    class_matches = (y == c)
    matches_count = np.add.accumulate(class_matches.values) == n_samples
    first_match = matches_count & ~np.roll(matches_count, 1)
    if not any(first_match):
        return len(y)
    return y[first_match].index[0]


def downsample(X, y, n_samples=None):
    classes = pd.unique(y)
    counts = [sum(y == c) for c in classes]
    n_samples = n_samples or min(counts)
    selectors = [find_cut(y, c, n_samples) for c in classes]
    y_resampled = pd.concat([
        y[y == c].loc[:s]
        for c, s in zip(classes, selectors)])
    y_resampled.sort_index(inplace=True)
    X_resampled = pd.concat([
        X[y == c].loc[:s]
        for c, s in zip(classes, selectors)])
    X_resampled.sort_index(inplace=True)
    return X_resampled, y_resampled


def select_only_numerical(X):
    X = X.drop(columns=['symbol', 'date'])
    X.columns = X.columns.remove_unused_levels()
    return X


def prepare_data(stocks_ds, train_size=20000, test_size=2000):
    logger = logging.getLogger(__name__)
    stocks_ds = stocks_ds or NyseStocksDataset()
    X_train, y_train, X_test, y_test = stocks_ds.data()
    small_X_train, small_y_train = downsample(X_train, y_train, train_size)
    small_X_test, small_y_test = downsample(X_test, y_test, test_size)
    counts = small_y_train.groupby(small_y_train).count()
    logger.info(f"Train Labels --> {'; '.join([f'{x}: {counts[x]}' for x in counts.index])}")
    logger.info(f'Training range: {small_X_train.date.min()} to {small_X_train.date.max()}')
    counts = small_y_test.groupby(small_y_test).count()
    logger.info(f"Test Labels --> {'; '.join([f'{x}: {counts[x]}' for x in counts.index])}")
    logger.info(f'Testing range: {small_X_test.date.min()} to {small_X_test.date.max()}')
    small_X_train = select_only_numerical(small_X_train)
    small_X_test = select_only_numerical(small_X_test)
    logger.info("Done preparing data")

    return small_X_train, small_y_train, small_X_test, small_y_test
