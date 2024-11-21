import random
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from Tree_Regression import MyTreeReg


class MyForestReg:
    def __init__(self, n_estimators: int = 10, max_features: float = 0.5,
                 max_samples: float = 0.5, random_state: int = 42,
                 max_depth: int = 5, min_samples_split: int = 2,
                 max_leafs: int = 20, bins: int = 16, oob_score: str = None,
                 multiprocess: bool = True):
        assert n_estimators > 0
        assert max_features > 0 and max_features <= 1
        assert max_samples > 0 and max_samples <= 1
        assert random_state > 0
        assert max_depth > 0
        assert min_samples_split > 0
        assert max_leafs > 0
        assert bins > 0
        assert oob_score is None or oob_score in ['mae', 'mse', 'rmse',
                                                  'mape', 'r2']

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.oob_score = oob_score
        self.multiprocess = multiprocess

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.leafs_cnt = None
        self._forest = None
        self.fi = None
        self.oob_score_ = None

    def __str__(self):
        return f'MyForestReg class: n_estimators={self.n_estimators}, ' \
               f'max_features={self.max_features}, ' \
               f'max_samples={self.max_samples}, ' \
               f'max_depth={self.max_depth}, ' \
               f'min_samples_split={self.min_samples_split}, ' \
               f'max_leafs={self.max_leafs}, ' \
               f'bins={self.bins}, ' \
               f'random_state={self.random_state}'

    def _get_score(self, y_pred: np.array, y_true: np.array) -> float:
        n_data = len(y_pred)

        if self.oob_score == 'mae':
            return np.sum(np.abs(y_true - y_pred)) / n_data
        elif self.oob_score == 'mse':
            return np.sum((y_true - y_pred) ** 2) / n_data
        elif self.oob_score == 'rmse':
            return (np.sum((y_true - y_pred) ** 2) / n_data) ** 0.5
        elif self.oob_score == 'mape':
            return 100 * np.sum(np.abs((y_true - y_pred) / y_true)) / n_data
        elif self.oob_score == 'r2':
            return 1 - np.sum((y_true - y_pred) ** 2) / \
                   np.sum((y_true - np.mean(y_true)) ** 2)

        return 0

    def _fit_tree(self, X, y) -> (MyTreeReg, int):
        tree = MyTreeReg(self.max_depth,
                         self.min_samples_split,
                         self.max_leafs,
                         self.bins)
        tree.fit(X, y)

        return tree, len(y)

    def _map_fit_tree(self, x: tuple) -> (MyTreeReg, int):
        return self._fit_tree(x[0], x[1])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        random.seed(self.random_state)

        self._forest = []
        self.leafs_cnt = 0
        self.fi = {feature: 0 for feature in X.columns}

        n = len(X)

        n_cols = round(len(X.columns) * self.max_features)
        n_rows = round(n * self.max_samples)
        oob_predicts = np.zeros(n)
        oob_mask = np.zeros(n)

        data_tarin = []
        for i in range(self.n_estimators):
            cols_idx = random.sample(list(X.columns), n_cols)
            rows_idx = random.sample(range(len(X)), n_rows)

            data_tarin.append((X.iloc[rows_idx][cols_idx], y[rows_idx]))

        if self.multiprocess:
            with Pool(processes=cpu_count()) as pool:
                result = pool.map(self._map_fit_tree, data_tarin)
        else:
            result = map(self._map_fit_tree, data_tarin)

        for k, (tree, num) in enumerate(result):
            oob = np.setdiff1d(np.arange(n), data_tarin[k][0].index.values)
            predict = tree.predict(X.iloc[oob])

            oob_predicts = np.zeros(n)
            oob_mask = np.zeros(n)
            for k, id in enumerate(oob):
                oob_mask[id] += 1
                oob_predicts[id] += predict[k]

            for feature, fi in tree.fi.items():
                self.fi[feature] += fi * num / n
            self.leafs_cnt += tree.leafs_cnt
            self._forest.append(tree)

        oob_predicts[oob_mask == 0] = y[oob_mask == 0]
        oob_mask[oob_mask == 0] = 1
        predicts = oob_predicts / oob_mask
        self.oob_score_ = self._get_score(predicts, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        return sum([tree.predict(X) for tree in
                    self._forest]) / self.n_estimators
