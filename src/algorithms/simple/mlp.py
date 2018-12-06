import sys
import re

from tqdm import tqdm_notebook
from sklearn.neural_network import MLPClassifier as SKL_MLPClassifier

from ..algorithm_utils import Algorithm


class MLPClassifier(Algorithm):
    def __init__(self, *args, seed=42, epochs=100, **kwargs):
        super().__init__('mlp', 'MLPClassifier', 'MLP', seed, can_handle_time_dim=False)
        self._args = args
        self._kwargs = kwargs
        self._kwargs['random_state'] = seed
        self._kwargs['max_iter'] = epochs

    def __call__(self):
        model = SKL_MLPClassifier(*self._args, **self._kwargs)
        return model

    def fit(self, X, y, **kwargs):
        wrapper = MLPOutputToProgressBar(total=self._kwargs['max_iter'])
        with wrapper:
            super().fit(X, y, **kwargs)
        self.history = wrapper.history

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)


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
