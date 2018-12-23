import numpy as np

from ..algorithm_utils import Algorithm


class Heuristic(Algorithm):
    def __init__(self, name_suffix='', n_features=3, n_timestamps=7, mapping=None, seed=42):
        super().__init__('heuristic', f'Heu{name_suffix}', f'Heuristic{name_suffix}', seed)
        self.n_features = n_features
        self.n_timestamps = n_timestamps
        self.mapping = mapping or Heuristic.last_movement

    def can_handle_time_dim(self):
        return False

    def __call__(self):
        return None

    def fit(self, X, y, **kwargs):
        self.history = None
        return None

    def predict(self, X, **kwargs):
        return self.mapping(X, self.n_features, self.n_timestamps)

    # ----- Heuristics ------------------------------------------------------- #

    @staticmethod
    def last_movement(X, *args):
        # Assuming the last feature is the movement
        return X[:, -1].round()

    @staticmethod
    def mean_movement(X, n_features, n_timestamps):
        # Assuming the last feature is the movement
        all_movements = X[:, n_features-1::n_features]
        return all_movements.mean(axis=1).round()

    @staticmethod
    def last_three_movements(X, n_features, n_timestamps):
        # Assuming the last feature is the movement
        all_movements = X[:, n_features-1::n_features]
        return all_movements[:, -3:].mean(axis=1).round()

    @staticmethod
    def last_vix_movement(X, n_features, n_timestamps):
        # Assuming the third last and second last features are vix_open and vix_close
        # all_open = X[:, n_features-2::n_features]
        # all_close = X[:, n_features-1::n_features]
        rel_dist = X[:, -2] / X[:, -3] - 1
        classes = np.array([-1, 0, 1])
        # TODO: Classes split should be adjusted in fit method
        pred = np.array([np.abs(classes - 1.65*d).argmin() - 1 for d in rel_dist])
        return pred
