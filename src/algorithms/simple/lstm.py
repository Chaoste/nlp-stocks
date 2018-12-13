from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import np_utils
from keras import optimizers
from sklearn.model_selection import train_test_split

from ..algorithm_utils import Algorithm, TQDMNotebookCallback


class SimpleLSTM(Algorithm):
    def __init__(self, name_suffix='', n_timestamps=7, n_features=5, n_classes=3, shuffle=True,
                 n_units=[64, 64], lstm_dropout=0.2, rec_dropout=0.2, lr=0.01, **kwargs):
        kwargs['batch_size'] = kwargs.get('batch_size', 512)
        kwargs['epochs'] = kwargs.get('epochs', 200)
        super().__init__('simple_lstm', f'SimpleLSTM{name_suffix}', f'SLSTM{name_suffix}',
                         **kwargs)
        self.n_timestamps = n_timestamps
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_units = n_units
        self.lr = lr
        self.lstm_dropout = lstm_dropout
        self.rec_dropout = rec_dropout
        self.shuffle = shuffle

    def can_handle_time_dim(self):
        return True

    def __call__(self):
        model = Sequential()
        for i, layer_units in enumerate(self.n_units):
            is_last_one = i == len(self.n_units) - 1
            model.add(LSTM(
                layer_units, input_shape=(self.n_timestamps, self.n_features), stateful=False,
                dropout=self.lstm_dropout, recurrent_dropout=self.rec_dropout,
                return_sequences=not is_last_one))
        model.add(Dense(self.n_classes, activation='softmax'))
        # rmsprop, adadelta, adam
        optimizer = optimizers.Adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['categorical_accuracy'])
        return model

    def transform(self, X=None, y=None):
        if X is not None:
            # _X = X.reshape((*X.shape, 1))
            _X = X
        if y is not None:
            _y = np_utils.to_categorical(y + 1)
        if X is None:
            return _y
        if y is None:
            return _X
        return _X, _y

    def undo_transform(self, X=None, y=None):
        if X is not None:
            # _X = X.reshape(X.shape[:2])
            _X = X
        if y is not None:
            _y = y - 1  # y.argmax(axis=1) - 1
        if X is None:
            return _y
        if y is None:
            return _X
        return _X, _y

    def fit(self, X, y, **kwargs):
        # For statefule training (would require seperating by company)
        # for i in range(self.sk_params['epochs']):
        #     history = self.model.fit(X, y, epochs=1, batch_size=self.sk_params['batch_size'],
        #                              verbose=0, shuffle=False)
        #     self.model.reset_states()
        #     print(f'Epoch {i}: '
        #           f'{"; ".join([f"{m}={history.history[m]}" for m in history.history])}')
        val_split = kwargs.get('validation_split', 0.2)
        if not self.shuffle:
            kwargs['validation_split'] = val_split
            X, y = self.transform(X, y)
        else:
            X, y = self.transform(X, y)
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=val_split, random_state=self.seed)
            kwargs['validation_split'] = 0
            kwargs['validation_data'] = (X_val, y_val)

        return super().fit(X, y, shuffle=False, verbose=0,
                           callbacks=[TQDMNotebookCallback()], **kwargs)

    def predict(self, X, **kwargs):
        pred = super().predict(self.transform(X), **kwargs)
        return self.undo_transform(y=pred)
