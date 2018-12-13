from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.svm import SVC

from .preparation import prepare_data
from .algorithms import Algorithm


def from_3d_to_2d(X):
    # First transform pandas MultiIndex to numpy (sklearn doesn't support pandas)
    X = X_to_numpy(X)
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def from_2d_to_3d(X, n_timestamps, time_dim):
    if not time_dim:
        return X
    return X.reshape(X.shape[0], n_timestamps, X.shape[1] // n_timestamps)


def X_to_numpy(X):
    # Transform from MultiIndex to numpy 3d array
    n_samples = X.shape[0]
    n_features = len(X.columns.levels[1])
    n_timestamps = len(X.columns) // n_features
    return X.values.reshape(n_samples, n_timestamps, n_features)


def run_pipeline(predictor, data):
    can_handle_time_dim = isinstance(predictor, Algorithm) and\
        predictor.can_handle_time_dim()
    X_train, y_train, X_test, y_test = data
    n_timestamps = len(X_test.columns.levels[0])
    pipeline = Pipeline([
        # ("sel", FunctionTransformer(select_only_numerical, validate=True)),
        ("pre-scaling", FunctionTransformer(from_3d_to_2d, validate=False)),
        ("scaler", RobustScaler()),  # feature_range=(0, 1)
        ("post-scaling", FunctionTransformer(
            lambda X: from_2d_to_3d(X, n_timestamps, can_handle_time_dim), validate=False)),
        ("predictor", predictor),
    ])
    # pipeline.set_params(predictor__h=0)
    # Keras model will create validation set on its own
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return pipeline, y_pred


if __name__ == "__main__":
    data = prepare_data()
    predictor = SVC(verbose=True, gamma='scale')
    run_pipeline(predictor, data)
