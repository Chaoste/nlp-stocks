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


def balance_by_downsampling(X, y, n_samples=None):
    classes = pd.unique(y)
    counts = [sum(y == c) for c in classes]
    n_samples = min(n_samples or np.inf, *counts)
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


def random_choice(X, y, n_samples):
    idx = np.random.choice(np.arange(len(y)), n_samples, replace=False)
    return X.iloc[idx], y.iloc[idx]


def prepare_data(stocks_ds, train_size=60000, test_size=6000, downsample=True):
    logger = logging.getLogger(__name__)
    stocks_ds = stocks_ds or NyseStocksDataset()
    X_train, y_train, X_test, y_test = stocks_ds.data()
    train_size = min(train_size, len(y_train))
    test_size = min(test_size, len(y_test))
    n_classes = len(pd.unique(y_train))
    class_train_size = train_size // n_classes
    # class_test_size = test_size // n_classes
    if downsample:
        X_train, y_train = balance_by_downsampling(X_train, y_train, class_train_size)
        # X_test, y_test = balance_by_downsampling(X_test, y_test, class_test_size)
    else:
        X_train, y_train = random_choice(X_train, y_train, train_size)
    X_test, y_test = random_choice(X_test, y_test, test_size)
    counts = y_train.groupby(y_train).count()
    logger.info(f"Train Labels --> {'; '.join([f'{x}: {counts[x]}' for x in counts.index])}")
    logger.info(f'Training range: {X_train.date.min()} to {X_train.date.max()}')
    counts = y_test.groupby(y_test).count()
    logger.info(f"Test Labels --> {'; '.join([f'{x}: {counts[x]}' for x in counts.index])}")
    logger.info(f'Testing range: {X_test.date.min()} to {X_test.date.max()}')
    X_train = select_only_numerical(X_train)
    X_test = select_only_numerical(X_test)
    logger.info("Done preparing data")

    return X_train, y_train, X_test, y_test
