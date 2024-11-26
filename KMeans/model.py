import numpy as np
import pandas as pd


class MyKMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 10,
                 n_init: int = 3, random_state: int = 42):
        assert n_clusters > 1
        assert max_iter > 0
        assert n_init > 0
        assert random_state > 0

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.cluster_centers_ = None
        self.inertia_ = None

    def __str__(self):
        return f'MyKMeans class: n_clusters={self.n_clusters}, ' \
               f'max_iter={self.max_iter}, ' \
               f'n_init={self.n_init}, ' \
               f'random_state={self.random_state}'

    def fit(self, X: pd.DataFrame):
        assert len(X) > self.n_clusters

        np.random.seed(self.random_state)

        variant_clusters = [[] for _ in range(self.n_init)]
        wcss = [0 for _ in range(self.n_init)]
        for i in range(self.n_init):
            cluster_centers = np.array([
                [np.random.uniform(np.min(X[feature].values),
                                   np.max(X[feature].values))
                 for feature in X.columns] for _ in range(self.n_clusters)])

            j = 0
            while j < self.max_iter:
                distances = np.apply_along_axis(lambda cl:
                                                np.sum((X.values - cl) ** 2,
                                                       axis=1).T, 1,
                                                cluster_centers)

                labels = np.argmin(distances, axis=0)
                wcss[i] = np.sum(distances[labels, np.arange(labels.shape[0])])

                new_cluster_centers = np.array([np.mean(X[labels == k].values,
                                                        axis=0)
                                                if len(X[labels == k])
                                                else cluster_centers[k]
                                                for k in
                                                range(self.n_clusters)])

                if np.all(new_cluster_centers == cluster_centers):
                    break

                cluster_centers = new_cluster_centers
                j += 1

            variant_clusters[i] = cluster_centers

        self.inertia_ = np.min(wcss)
        self.cluster_centers_ = variant_clusters[np.argmin(wcss)]

        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        assert self.cluster_centers_ is not None

        distances = np.apply_along_axis(lambda cl: np.sum((X.values - cl) ** 2,
                                                          axis=1).T, 1,
                                        self.cluster_centers_)

        return np.argmin(distances, axis=0)
