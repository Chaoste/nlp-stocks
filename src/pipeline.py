import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.metrics import classification_report, matthews_corrcoef,\
    accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC

from src.datasets import NyseStocksDataset


def from_3d_to_2d(X):
    # First transform pandas MultiIndex to numpy (sklearn doesn't support pandas)
    X = X_to_numpy(X)
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def from_2d_to_3d(X, n_timestamps, time_dim):
    if not time_dim:
        return X
    return X.reshape(X.shape[0], n_timestamps, X.shape[1] // n_timestamps)


def run_pipeline(predictor, data, time_dim=True):
    X_train, y_train, X_test, y_test = data
    n_timestamps = len(X_test.columns.levels[0])
    pipeline = Pipeline([
        # ("sel", FunctionTransformer(select_only_numerical, validate=True)),
        ("pre-scaling", FunctionTransformer(from_3d_to_2d, validate=False)),
        ("scaler", RobustScaler()),  # feature_range=(0, 1)
        ("post-scaling", FunctionTransformer(lambda X: from_2d_to_3d(X, n_timestamps, time_dim),
                                             validate=False)),
        ("predictor", predictor),
    ])
    # pipeline.set_params(predictor__h=0)
    # Keras model will create validation set on its own
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return pipeline, y_pred


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
    stocks_ds = stocks_ds or NyseStocksDataset()
    X_train, y_train, X_test, y_test = stocks_ds.data()
    small_X_train, small_y_train = downsample(X_train, y_train, train_size)
    small_X_test, small_y_test = downsample(X_test, y_test, test_size)
    counts = small_y_train.groupby(small_y_train).count()
    print(f"Train Labels --> {'; '.join([f'{x}: {counts[x]}' for x in counts.index])}")
    print(f'Training range: {small_X_train.date.min()} to {small_X_train.date.max()}')
    counts = small_y_test.groupby(small_y_test).count()
    print(f"Test Labels --> {'; '.join([f'{x}: {counts[x]}' for x in counts.index])}")
    print(f'Testing range: {small_X_test.date.min()} to {small_X_test.date.max()}')
    small_X_train = select_only_numerical(small_X_train)
    small_X_test = select_only_numerical(small_X_test)
    print("Done preparing data")

    return small_X_train, small_y_train, small_X_test, small_y_test


def X_to_numpy(X):
    # Transform from MultiIndex to numpy 3d array
    n_samples = X.shape[0]
    n_features = len(X.columns.levels[1])
    n_timestamps = len(X.columns) // n_features
    return X.values.reshape(n_samples, n_timestamps, n_features)


def evaluate_results(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    target_names = np.array(['Down', 'Still', 'Up'])[np.unique(y_true)+1]
    print(classification_report(y_true, y_pred, target_names=target_names))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f'Precision={prec}')
    print(f'Recall={rec}')
    print(f'F1-Score={f1}')
    acc = accuracy_score(y_true, y_pred)
    print(f'Accuracy={acc}')
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f'MCC={mcc}')
    return {
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'acc': acc,
        'mcc': mcc
    }


if __name__ == "__main__":
    data = prepare_data()
    predictor = SVC(verbose=True, gamma='scale')
    run_pipeline(predictor, data)
