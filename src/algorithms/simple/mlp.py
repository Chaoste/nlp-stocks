import sys
import re

import numpy as np
from tqdm import tqdm_notebook
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
from sklearn.neural_network import MLPClassifier as SKL_MLPClassifier

from ..algorithm_utils import Algorithm


class MLPClassifier(Algorithm):
    def __init__(self, *args, seed=42, epochs=200, **kwargs):
        super().__init__('mlp', 'MLPClassifier', 'MLP', seed)
        self._args = args
        self._kwargs = kwargs
        self._kwargs['hidden_layer_sizes'] = self._kwargs.get('hidden_layer_sizes', (32, 32, 32, 3))
        self._kwargs['activation'] = self._kwargs.get('activation', 'tanh')
        self._kwargs['solver'] = self._kwargs.get('solver', 'adam')
        self._kwargs['batch_size'] = self._kwargs.get('batch_size', 256)
        self._kwargs['random_state'] = seed
        self._kwargs['max_iter'] = epochs
        self._kwargs['verbose'] = True
        # tol=1e-4, learning_rate_init=1e-4, alpha=0.0001

    def can_handle_time_dim(self):
        return False

    def __call__(self):
        model = SKL_MLPClassifier(*self._args, **self._kwargs)
        return model

    def transform(self, y):
        classes = np.unique(y)
        assert min(classes) == -1
        return y + 1

    def reverse_transform(self, y):
        classes = np.unique(y)
        assert min(classes) == 0
        return y - 1

    def fit(self, X, y, **kwargs):
        wrapper = MLPOutputToProgressBar(total=self._kwargs['max_iter'])
        with wrapper:
            super(MLPClassifier, self).fit(X, y, **kwargs)
        self.history = wrapper.history

    def predict(self, X, **kwargs):
        # super().predict not working: github.com/keras-team/keras/issues/11818
        pred = self.model.predict(X, **kwargs)
        return self.classes_[pred]

    def clone(self, **kwargs):
        return MLPClassifier(
            *self._args, seed=self._kwargs['random_state'], epochs=self._kwargs['max_iter'],
            **self._kwargs, **kwargs)


class History():
    def __init__(self):
        self.history = {
            'loss': [],
        }

    def insert(self, loss, acc=None):
        self.history['loss'].append(loss)


class MLPOutputToProgressBar():
    def __init__(self, total):
        self.tqdm_bar = tqdm_notebook(total=total)
        self.mlp_progress_regex = re.compile('Iteration ([0-9]+), loss = ([0-9\.]+)',
                                             flags=re.IGNORECASE)
        self.print_mocker = PrintMocker(self.on_listen_predictors)
        self.was_last_input_match = False
        self.history = History()

    def on_listen_predictors(self, data):
        if self.was_last_input_match:
            self.was_last_input_match = False
            if data == '\n':
                return None
        match = self.mlp_progress_regex.match(data)
        if not match:
            return data
        iteration = int(match.group(1))
        loss = float(match.group(2))
        self.history.insert(loss)
        self.tqdm_bar.set_description(f'Loss={loss:.5f}')
        self.tqdm_bar.update(iteration - self.tqdm_bar.n)
        self.was_last_input_match = True
        return None

    def __enter__(self):
        self.tqdm_bar.__enter__()
        self.print_mocker.__enter__()

    def __exit__(self, *args):
        self.tqdm_bar.__exit__(*args)
        self.print_mocker.__exit__(*args)


class Unbuffered(object):
    def __init__(self, stream, on_handle_data):
        self.stream = stream
        self.on_handle_data = on_handle_data

    def write(self, data):
        result = self.on_handle_data(data)
        if result is not None:
            self.stream.write(result)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


class PrintMocker():
    def __init__(self, on_handle_data):
        self.orig_stdout = sys.stdout
        self.mocked_stdout = Unbuffered(sys.stdout, on_handle_data)

    def __enter__(self):
        sys.stdout = self.mocked_stdout

    def __exit__(self, type, value, traceback):
        sys.stdout = self.orig_stdout
