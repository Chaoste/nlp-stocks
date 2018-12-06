from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import np_utils

from ..algorithm_utils import Algorithm


class SimpleLSTM(Algorithm):
    def __init__(self, n_timestamps=7, n_features=5, n_classes=3,
                 n_units=[64, 32], lstm_dropout=0.2, rec_dropout=0.0, **kwargs):
        super().__init__('simple_lstm', 'SimpleLSTM', 'SLSTM', 42, **kwargs)
        self.n_timestamps = n_timestamps
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_units = n_units
        self.lstm_dropout = lstm_dropout
        self.rec_dropout = rec_dropout

    def __call__(self):
        # model = Sequential()
        # model.add(LSTM(return_sequences=True, input_shape=(None, 1), units=50))
        # model.add(Dropout(0.2))
        # model.add(LSTM(100, return_sequences=False))
        # model.add(Dropout(0.2))
        # model.add(Dense(3))
        # model.add(Activation('softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='adam',
        #               metrics=['categorical_accuracy'])
        # return model
        model = Sequential()
        for i, layer_units in enumerate(self.n_units):
            is_last_one = i == len(self.n_units) - 1
            model.add(LSTM(
                layer_units, input_shape=(self.n_timestamps, self.n_features), stateful=False,
                dropout=self.lstm_dropout, recurrent_dropout=self.rec_dropout,
                return_sequences=not is_last_one))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',  # rmsprop, adadelta, adam
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
        return super().fit(*self.transform(X, y), shuffle=True, **kwargs)

    def predict(self, X, **kwargs):
        pred = super().predict(self.transform(X), **kwargs)
        return self.undo_transform(y=pred)
