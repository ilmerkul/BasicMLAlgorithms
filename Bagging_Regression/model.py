import pandas as pd
import numpy as np
import random
import copy
from Linear_Regression import MyLineReg
from KNN_Regression import MyKNNReg
from Tree_Regression import MyTreeReg


class MyBaggingReg:
    def __init__(self, estimator=MyLineReg(), n_estimators: int = 10, max_samples: float = 1.0, random_state: int = 42):
        assert n_estimators > 0
        assert max_samples >= 0.0 and max_samples <= 1.0
        assert random_state > 0

        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.estimators = None

    def __str__(self):
        return f'MyBaggingReg class: estimator={self.estimator}, n_estimators={self.n_estimators}, max_samples={self.max_samples}, random_state={self.random_state}'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        assert self.estimator is not None

        random.seed(self.random_state)

        k = round(self.max_samples * len(X))

        sample_rows_idx = [random.choices(range(len(X)), k=k) for _ in range(self.n_estimators)]

        self.estimators = [copy.deepcopy(self.estimator).fit(X.loc[sample_rows_idx[i]].reset_index(drop=True),
                                                             y.loc[sample_rows_idx[i]].reset_index(drop=True)) for i in
                           range(self.n_estimators)]

    def predict(self, X: pd.DataFrame):
        return np.mean([estimator.predict(X) for estimator in self.estimators], axis=0)
