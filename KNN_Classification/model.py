from collections import Counter
from typing import NoReturn

import numpy as np
import pandas as pd


class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = 'euclidean',
                 weight: str = 'uniform'):
        assert metric in ['euclidean', 'chebyshev', 'manhattan', 'cosine']
        assert weight in ['uniform', 'rank', 'distance']
        assert k > 0

        self.k = k
        self._train_size = None
        self._data = None
        self._data_label = None
        self._n_labels = None
        self.metric = metric
        self.weight = weight

    def __str__(self):
        return f'MyKNNClf class: k={self.k}'

    def _calculate_metric(self, x: np.array) -> np.array:
        if self.metric == 'euclidean':
            return np.sum((self._data - x) ** 2, axis=1).squeeze() ** 0.5
        elif self.metric == 'chebyshev':
            return np.max(abs(self._data - x), axis=1).squeeze()
        elif self.metric == 'manhattan':
            return np.sum(abs(self._data - x), axis=1).squeeze()
        elif self.metric == 'cosine':
            return (1 - (self._data.dot(x.T)).squeeze() /
                    (np.apply_along_axis(lambda y: y.dot(y.T) ** 0.5, 1,
                                         self._data).squeeze() *
                     (x.dot(x.T) ** 0.5).squeeze()))

    def _calculate_weights(self, metric: np.array,
                           indexes: np.array) -> np.array:
        r = np.zeros(self._n_labels)
        if self.weight == 'uniform':
            c = Counter(self._data_label[indexes])

            for count in c.values():
                r[count[0]] += count[1]

            r /= self.k
        elif self.weight == 'rank':
            for s, j in enumerate(indexes):
                r[self._data_label[j]] += 1 / (s + 1)

            r /= np.sum(1 / np.arange(1, self.k + 1))
        elif self.weight == 'distance':
            dist = metric[indexes]

            for s, p in enumerate(dist):
                r[self._data_label[indexes[s]]] += 1 / p

            r /= np.sum(1 / dist)

        return r

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NoReturn:
        n_data = len(X)
        n_features = len(X.columns)

        assert n_data > self.k
        assert n_features > 0

        self._data = X.values
        self._data_label = y.values
        self._train_size = (n_data, n_features)
        self._n_labels = len(set(y.values))

    def predict(self, X: pd.DataFrame) -> np.array:
        assert self._data is not None and \
               self._data_label is not None and \
               self._train_size is not None

        n = len(X)

        X = X.values
        labels = np.zeros(n, dtype=int)
        for i, x in enumerate(X):
            metric = self._calculate_metric(x)
            indexes = np.argsort(metric)[:self.k]

            r = self._calculate_weights(metric, indexes)
            label = np.argmax(r)

            labels[i] = label

        return labels

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        assert self._data is not None and \
               self._data_label is not None and \
               self._train_size is not None

        n = len(X)

        X = X.values
        labels = [[0 for j in range(self.k)] for i in range(n)]
        for i, x in enumerate(X):
            metric = self._calculate_metric(x)
            indexes = np.argsort(metric)[:self.k]

            r = self._calculate_weights(metric, indexes)

            labels[i] = list(r)
        labels = np.array(labels)

        return labels