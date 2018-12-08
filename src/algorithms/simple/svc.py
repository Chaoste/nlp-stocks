# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import SVC as SKL_SVC

from ..algorithm_utils import Algorithm


class SVC(Algorithm):
    def __init__(self, *args, seed=42, epochs=100, **kwargs):
        super().__init__('svc', 'SVC', 'SVC', seed)
        self._args = args
        self._kwargs = kwargs
        self._kwargs['gamma'] = self._kwargs.get('gamma', 'scale')

    def can_handle_time_dim(self):
        return False

    def __call__(self):
        model = SKL_SVC(*self._args, **self._kwargs)
        model.loss = None
        return model

    def fit(self, X, y, **kwargs):
        # 60k -> 8min
        # 80k -> ?
        # 600k -> didn't converge after 16h calculations
        X = X.iloc[:80000]
        y = y.iloc[:80000]
        return super(SVC, self).fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        # super().predict not working: github.com/keras-team/keras/issues/11818
        pred = self.model.predict(X, **kwargs)
        return self.classes_[pred]
