import numpy as np
import pandas as pd
from numpy import linalg


class MyPCA:
    def __init__(self, n_components: int = 3):
        assert n_components > 0

        self.n_components = n_components

    def __str__(self):
        return f"MyPCA class: n_components={self.n_components}"

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.values
        X = X - np.mean(X, axis=0)

        cov = np.cov(X.T)

        eigenvalues, eigenvectors = linalg.eigh(cov)

        n_eigen_indexes = np.argsort(eigenvalues)[::-1][: self.n_components]
        eigenvectors = eigenvectors.T[n_eigen_indexes]

        return pd.DataFrame(data=X.dot(eigenvectors.T))
