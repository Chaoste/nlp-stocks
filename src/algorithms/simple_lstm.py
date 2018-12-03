import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

from .algorithm_utils import Algorithm


class SimpleLSTM(Algorithm):
    def __init__(seed:int = 42):
        super().__init__('simple_lstm', 'SimpleLSTM', 'SLSTM')

    def fit(self, X):
        """
        Train the algorithm on the given dataset
        """

    def predict(self, X):
        """
        :return anomaly score
        """
