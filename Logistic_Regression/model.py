import random
from types import FunctionType
from typing import Callable, NoReturn

import numpy as np
import pandas as pd


class MyLogReg:
    def __init__(self, n_iter: int = 10, learning_rate: float | Callable = 0.1,
                 metric: str = None, reg: str = None, l1_coef: float = 0,
                 l2_coef: float = 0, sgd_sample: int | float = None,
                 random_state: int = 42):
        assert metric is None or metric in ['accuracy', 'precision',
                                            'recall', 'f1', 'roc_auc']
        assert reg is None or reg in ['l1', 'l2', 'elasticnet']
        assert l1_coef >= 0.0 and l1_coef <= 1.0
        assert l2_coef >= 0.0 and l2_coef <= 1.0
        assert sgd_sample is None or \
               type(sgd_sample) == int or \
               type(sgd_sample) == float and \
               sgd_sample >= 0.0 and \
               sgd_sample <= 1.0

        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = np.array([])
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, ' \
               f'learning_rate={self.learning_rate}'

    def _get_accuracy(self, y_pred: np.array, y: np.array) -> float:
        return np.sum(((y_pred.T > 0.5) == y)) / len(y)

    def _get_precision(self, y_pred: np.array, y: np.array) -> float:
        y_pred = (y_pred.T > 0.5)
        TP = np.sum((y_pred == y) & (y == 1))
        FP = np.sum((y_pred != y) & (y == 0))
        return TP / (TP + FP)

    def _get_recall(self, y_pred: np.array, y: np.array) -> float:
        y_pred = (y_pred.T > 0.5)
        TP = np.sum((y_pred == y) & (y == 1))
        FN = np.sum((y_pred != y) & (y == 1))
        return TP / (TP + FN)

    def _get_f1(self, y_pred: np.array, y: np.array) -> float:
        precision = self._get_precision(y_pred, y)
        recall = self._get_recall(y_pred, y)
        return 2 * precision * recall / (precision + recall)

    def _get_auc(self, y_pred: np.array, y: np.array) -> float:
        desc_score_indices = np.argsort(y_pred)[::-1]
        y_score = y[desc_score_indices]
        y_true = y[desc_score_indices]

        distinct_indices = np.where(np.diff(y_score))[0]
        end = np.array([y_true.size - 1])
        threshold_indices = np.hstack((distinct_indices, end))

        tps = np.cumsum(y_true)[threshold_indices]

        fps = (1 + threshold_indices) - tps

        tpr = tps / tps[-1]
        fpr = fps / fps[-1]

        zero = np.array([0])
        tpr_diff = np.hstack((np.diff(tpr), zero))
        fpr_diff = np.hstack((np.diff(fpr), zero))
        auc = np.dot(tpr, fpr_diff) + np.dot(tpr_diff, fpr_diff) / 2

        return auc

    def _get_evaluate(self, X: np.array, y: np.array) -> float:
        y_pred = 1 / (1 + np.exp(-1 * np.dot(X, self.weights.T)))
        evaluations = {'accuracy': self._get_accuracy,
                       'precision': self._get_precision,
                       'recall': self._get_recall,
                       'f1': self._get_f1,
                       'roc_auc': self._get_auc}

        return evaluations[self.metric](y_pred, y)

    def _get_grad(self, X: np.array, y: np.array) -> float:
        n_data = len(X)

        n = self.sgd_sample
        if self.sgd_sample is None:
            n = n_data
        elif self.sgd_sample <= 1:
            n = int(n_data * self.sgd_sample)
        sample_rows_idx = random.sample(range(n_data), n)

        y_pred = 1 / (1 + np.exp(-1 * np.dot(X[sample_rows_idx, :],
                                             self.weights.T)))

        grad = (y_pred.T - y[sample_rows_idx]).dot(X[sample_rows_idx, :]) / n
        if self.reg == 'l1' or self.reg == 'elasticnet':
            l1 = np.array([1 if w > 0 else (-1 if w != 0 else 0)
                           for w in self.weights])
            grad += self.l1_coef * l1
        if self.reg == 'l2' or self.reg == 'elasticnet':
            grad += 2 * self.l2_coef * self.weights

        return grad

    def _get_loss(self, X: np.array, y: np.array, eps: float = 1e-15) -> float:
        n_data = len(X)

        y_pred = 1 / (1 + np.exp(-1 * np.dot(X, self.weights.T)))
        loss = -1 * (np.dot(y, np.log(y_pred + eps)) +
                     np.dot(1 - y, np.log(1 - y_pred + eps))) / n_data
        if self.reg == 'l1' or self.reg == 'elasticnet':
            loss += self.l1_coef * np.sum(np.abs(self.weights[:-1]))
        if self.reg == 'l2' or self.reg == 'elasticnet':
            loss += self.l2_coef * np.sum(self.weights[:-1] ** 2)

        return loss

    def _print_train(self, X: np.array, y: np.array, i: int = 0,
                     start: bool = False) -> NoReturn:
        loss = self._get_loss(X, y)

        step = str(i)
        if start:
            step = 'start'

        r = f'{step} | loss: {loss}'
        if self.metric is not None:
            eval = self._get_evaluate(X, y)
            r += f' | {self.metric}: {eval}'
        print(r)

    def fit(self, X: pd.DataFrame, y: pd.Series,
            verbose: int = False) -> NoReturn:
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        n_features = len(X.columns)
        n_data = len(X)

        assert n_features > 0
        assert n_data > 1

        X['bias_feature'] = 1
        X = X.values
        y = y.values

        # self.weights = np.random.uniform(-1, 1, n_features + 1)
        self.weights = np.ones(n_features + 1)

        if verbose:
            self._print_train(X, y, start=True)

        for i in range(1, self.n_iter + 1):
            grad = self._get_grad(X, y)

            if type(self.learning_rate) == FunctionType:
                self.weights -= self.learning_rate(i) * grad
            else:
                self.weights -= self.learning_rate * grad

            if verbose and i > 0 and (i % verbose) == 0:
                self._print_train(X, y, i)

        self.metric_value = self._get_evaluate(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        assert len(self.weights) > 0

        X['bias_feature'] = 1
        X = X.values

        y_pred = 1 / (1 + np.exp(-1 * np.dot(X, self.weights.T)))

        return y_pred.T

    def predict(self, X: pd.DataFrame) -> np.array:
        assert len(self.weights) > 0

        X['bias_feature'] = 1
        X = X.values

        y_pred = 1 / (1 + np.exp(-1 * np.dot(X, self.weights.T)))

        y_pred = y_pred > 0.5

        return y_pred.T

    def get_coef(self) -> np.array:
        assert len(self.weights) > 0

        return self.weights[:-1]

    def get_best_score(self) -> float:
        assert self.metric is not None

        return self.metric_value
