from sklearn.svm import SVC

from .algorithms import SimpleLSTM, MLPClassifier
from .evaluation import Evaluator


def get_predictors(n_features, n_timestamps=7, n_classes=3):
    return [
        SimpleLSTM(n_timestamps=n_timestamps, n_features=n_features, n_classes=n_classes,
                   epochs=10, batch_size=32, n_units=[64, 64]),
        MLPClassifier(activation='tanh', hidden_layer_sizes=(32, 32, 32, 3), epochs=150,
                      solver='adam', verbose=True, seed=42),
        # tol=1e-4, learning_rate_init=1e-4, alpha=0.0001
        SVC(verbose=True, gamma='scale'),
    ]


def run_experiment(datasets, reports_dir='reports', seed=42, store=False):
    evaluator = Evaluator('feature_selection', datasets, get_predictors, reports_dir, seed=seed)
    metrics = evaluator(store=store)
    return metrics
