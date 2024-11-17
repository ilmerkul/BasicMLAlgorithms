from collections import deque
from typing import NoReturn

import numpy as np
import pandas as pd


class MyTreeReg:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2,
                 max_leafs: int = 20, bins: int = None):
        assert max_depth > 0
        assert min_samples_split >= 2
        assert max_leafs > 0
        assert bins is None or bins > 0

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.fi = None
        self.leafs_cnt = None
        self._tree = None
        self._hists = None

    def __str__(self):
        return f'MyTreeReg class: max_depth={self.max_depth}, ' \
               f'min_samples_split={self.min_samples_split}, ' \
               f'max_leafs={self.max_leafs}'

    def _calculate_mse(self, x):
        return np.sum((x - np.mean(x)) ** 2) / len(x)

    def _get_best_split(self, X: pd.DataFrame, y: pd.Series) \
            -> (str, float, float):
        col_name = 0
        split_value = 0
        ig = float('-inf')

        y = y.values

        n = len(X)

        criterion = self._calculate_mse

        S0 = criterion(y)

        for feature in X.columns:
            indexes = np.argsort(X[feature].values)

            for i in range(1, n):
                S1 = criterion(y[indexes[:i]])

                S2 = criterion(y[indexes[i:]])

                ig_new = S0 - i * S1 / n - (n - i) * S2 / n
                if ig_new > ig:
                    ig = ig_new
                    col_name = feature
                    split_value = (X[feature].iloc[indexes[i - 1]] +
                                   X[feature].iloc[indexes[i]]) / 2

        return col_name, split_value, ig

    def _get_best_split_bins(self, X: pd.DataFrame, y: pd.Series) \
            -> (str, float, float):
        col_name = None
        split_value = None
        ig = float('-inf')

        n = len(X)

        criterion = self._calculate_mse

        S0 = criterion(y)

        for feature in X.columns:
            bins = self._hists[feature]

            c = 0
            b_last = float('-inf')
            for b in bins:
                labels = y[
                    X[(X[feature] <= b) & (X[feature] > b_last)].index].values
                b_last = b
                c += len(labels)
                if n == c or len(labels) == 0:
                    continue

                S1 = criterion(y[X[(X[feature] <= b)].index].values)

                S2 = criterion(y[X[(X[feature] > b)].index].values)

                ig_new = S0 - c * S1 / n - (n - c) * S2 / n
                if ig_new > ig:
                    ig = ig_new
                    col_name = feature
                    split_value = b

        return col_name, split_value, ig

    def fit(self, X: pd.DataFrame, y: pd.Series):
        split_func = self._get_best_split
        if self.bins is not None:
            self._hists = dict()
            for feature in X.columns:
                count_bins = 0
                i = 0
                bins = np.zeros(self.bins)
                indexes = np.argsort(X[feature].values)
                while count_bins < self.bins and i < len(X) - 1:
                    idx_i = indexes[i]
                    idx_i_1 = indexes[i + 1]
                    if X[feature].iloc[idx_i] < X[feature].iloc[idx_i_1]:
                        bins[count_bins] = (X[feature].iloc[idx_i] +
                                            X[feature].iloc[idx_i_1]) / 2
                        count_bins += 1
                    i += 1

                if count_bins >= self.bins:
                    hist, bins = np.histogram(X[feature], bins=self.bins)
                    bins = bins[1:-1]

                self._hists[feature] = bins

        self.fi = dict([(feature, 0) for feature in X.columns])
        self.leafs_cnt = 1
        self._tree = dict()

        deq = deque()
        deq.append((np.ones(len(X), dtype=bool), 1, self._tree))

        while len(deq) or self.leafs_cnt == 1:
            idx, depth, tree = deq.pop()

            if depth > self.max_depth or \
                    np.sum(idx) < self.min_samples_split or \
                    len(set(y[idx])) <= 1 or \
                    self.leafs_cnt >= self.max_leafs and \
                    self.leafs_cnt != 1:
                tree['value'] = np.mean(y[idx])
                continue

            col_name, split_value, ig = split_func(X[idx], y[idx])
            if col_name is None:
                tree['value'] = np.mean(y[idx])
                continue

            tree['fi'] = np.sum(idx) * ig / len(X)
            tree['feature'] = (col_name, split_value)
            self.leafs_cnt += 1

            idx_left = idx & (X[col_name] <= split_value)
            idx_right = idx & (X[col_name] > split_value)

            tree['right'] = dict()
            deq.append((idx_right, depth + 1, tree['right']))

            tree['left'] = dict()
            deq.append((idx_left, depth + 1, tree['left']))

        self.dfs_tree(print_node=False)
        return self

    def dfs_tree(self, print_node=False) -> NoReturn:
        assert self._tree is not None

        deq = deque()
        deq.append((self._tree, 0))

        while len(deq):
            tree, depth = deq.pop()

            feature = tree.get('feature', False)
            if print_node:
                print('\t' * depth, end=' ')
            if feature:
                if print_node:
                    print(str(tree['feature']), end=' ')
                self.fi[tree['feature'][0]] += tree['fi']
                deq.append((tree['right'], depth + 1))
                deq.append((tree['left'], depth + 1))
            else:
                if print_node:
                    print(str(tree['value']), end=' ')
            if print_node:
                print()

    def predict(self, X: pd.DataFrame) -> np.array:
        assert self._tree is not None

        predict = np.zeros(len(X))
        for i in range(len(X)):
            x = X.iloc[i]
            node = self._tree
            while True:
                feature, value = node.get('feature', (False, False))
                if not feature:
                    predict[i] = node['value']
                    break

                if x.loc[feature] <= value:
                    node = node['left']
                else:
                    node = node['right']

        return predict
