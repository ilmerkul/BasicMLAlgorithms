import numpy as np
import pandas as pd

from Tree_Regression import MyTreeReg


class MyBoostReg:
    def __init__(self, n_estimators: int = 10, learning_rate: float = 0.1,
                 max_depth: int = 5, min_samples_split: int = 2,
                 max_leafs: int = 20, bins: int = 16):
        assert n_estimators > 0
        assert learning_rate > 0

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.pred_0 = None
        self.trees = None

    def __str__(self):
        return f'MyBoostReg class: n_estimators={self.n_estimators}, ' \
               f'learning_rate={self.learning_rate}, ' \
               f'max_depth={self.max_depth}, ' \
               f'min_samples_split={self.min_samples_split}, ' \
               f'max_leafs={self.max_leafs}, bins={self.bins}'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pred_0 = np.mean(y)
        self.trees = []

        y -= self.pred_0

        for i in range(self.n_estimators):
            estimator = MyTreeReg(self.max_depth, self.min_samples_split,
                                  self.max_leafs, self.bins).fit(X, y)

            self.trees.append(estimator)
            y -= self.learning_rate * estimator.predict(X)

        return self

    def predict(self, X: pd.DataFrame):
        return self.pred_0 + self.learning_rate * np.sum(
            [tree.predict(X) for tree in self.trees], axis=0)
