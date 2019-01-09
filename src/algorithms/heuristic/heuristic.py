import numpy as np

from ..algorithm_utils import Algorithm


class Heuristic(Algorithm):
    def __init__(self, name_suffix='', n_features=3, n_timestamps=7, mapping=None, seed=42,
                 movement_feat=-3, vix_feats=[-2, -1], one_class=0):
        super().__init__('heuristic', f'Heu{name_suffix}', f'Heuristic{name_suffix}', seed)
        self.n_features = n_features
        self.n_timestamps = n_timestamps
        self.movement_feat = movement_feat
        self.vix_feats = vix_feats
        self.one_class = one_class
        self.mapping = (mapping or Heuristic.last_movement).__get__(self, Heuristic)

    def can_handle_time_dim(self):
        return False

    def __call__(self):
        return None

    def fit(self, X, y, **kwargs):
        self.history = None
        return None

    def predict(self, X, **kwargs):
        return self.mapping(X)

    def transform_index(self, idx):
        if idx > 0:
            return idx
        return self.n_features + idx

    def get_mov_index(self):
        return self.transform_index(self.movement_feat)

    def get_vix_indices(self):
        return [self.transform_index(x) for x in self.vix_feats]

    def for_last_day(self, idx):
        return self.n_features * (self.n_timestamps-1) + idx

    # ----- Heuristics ------------------------------------------------------- #

    def one_class(self, X):
        assert self.one_class in [-1, 0, 1]
        return np.ones(X.shape[0]) * self.one_class

    def random(self, X):
        return np.random.randint(-1, 2, size=X.shape[0])

    def last_movement(self, X):
        last_mov = self.for_last_day(self.get_mov_index())
        return X[:, last_mov].round()

    def mean_movement(self, X):
        all_movements = X[:, self.get_mov_index()::self.n_features]
        return all_movements.mean(axis=1).round()

    def last_three_movements(self, X):
        all_movements = X[:, self.get_mov_index()::self.n_features]
        return all_movements[:, -3:].mean(axis=1).round()

    def last_vix_movement(self, X):
        # all_open = X[:, n_features-2::n_features]
        # all_close = X[:, n_features-1::n_features]
        last_vix_open, last_vix_close = [self.for_last_day(x) for x in self.get_vix_indices()]
        rel_dist = X[:, last_vix_close] / X[:, last_vix_open] - 1
        classes = np.array([-1, 0, 1])
        # TODO: Classes split should be adjusted in fit method
        pred = np.array([np.abs(classes - 1.65*d).argmin() - 1 for d in rel_dist])
        return pred
