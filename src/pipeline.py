from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler  # FunctionTransformer
from sklearn.svm import SVC

from datasets import NyseStocksDataset


def select_only_numerical(X):
    return X[sorted(set(X.columns) - set(('date', 'symbol')))]


def run(stocks_ds=None):
    stocks_ds = stocks_ds or NyseStocksDataset()

    X_train, y_train, X_test, y_test = stocks_ds.data()
    X_train = X_train.iloc[:10000]
    y_train = y_train.iloc[:10000]
    X_test = X_test.iloc[:2000]
    y_test = y_test.iloc[:2000]
    counts = y_train.groupby(y_train).count()
    print(f"Train Labels --> {'; '.join([f'{x}: {counts[x]}' for x in counts.index])}")
    print(f'Training range: {X_train.date.min()} to {X_train.date.max()}')
    counts = y_test.groupby(y_test).count()
    print(f"Test Labels --> {'; '.join([f'{x}: {counts[x]}' for x in counts.index])}")
    print(f'Testing range: {X_test.date.min()} to {X_test.date.max()}')
    X_train = select_only_numerical(X_train)
    X_test = select_only_numerical(X_test)
    print("Done preparing data")


    # sklearn.neural_network.MLPClassifier
    predictor = SVC(verbose=True, gamma='scale')  # kernel='linear'

    pipeline = Pipeline([
        # ("selector", FunctionTransformer(select_only_numerical, validate=True)),
        ("scaler", MinMaxScaler()),  # feature_range=(0, 1)
        ("predictor", predictor),
    ])
    # pipeline.set_params(predictor__h=0)
    # Keras model will create validation set on its own
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    hits = sum(prediction == y_test)
    print(f'Accuracy = {hits / len(prediction)} ({hits} out of {len(prediction)})')
    return pipeline, prediction

if __name__ == "__main__":
    run()
