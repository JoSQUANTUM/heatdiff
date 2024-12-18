"""The following classes are adaptations of the k-means class from 
https://github.com/aihubprojects/Machine-Learning-From-Scratch/blob/master/K-Means%20from%20Scratch.ipynb
"""

import abc
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import iv, eval_hermitenorm


class BaseKMeans:
    def __init__(self, k, tolerance, max_iter):
        self.k = k
        self.max_iterations = max_iter
        self.tolerance = tolerance
        self.centroids = None

    @abc.abstractmethod
    def kernel(self):
        return

    def weighted_distance(self, points, centroids):
        distances = cdist(points, centroids)
        return distances * self.kernel(distances)

    def predict(self, data):
        return np.argmin(self.weighted_distance(data, self.centroids), axis=1)

    def fit(self, data):
        data = np.asarray(data)
        self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]

        for _ in range(self.max_iterations):
            distances = self.weighted_distance(data, self.centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array(
                [
                    (
                        data[labels == i].mean(axis=0)
                        if np.sum(labels == i) > 0
                        else self.centroids[i]
                    )
                    for i in range(self.k)
                ]
            )

            if np.all(np.abs(new_centroids - self.centroids) <= self.tolerance):
                break

            self.centroids = new_centroids

        return self


class GaussianKMeans(BaseKMeans):
    def __init__(self, k, tolerance, max_iter, mu, sigma):
        super().__init__(k, tolerance, max_iter)
        self.mu = mu
        self.sigma = sigma

    def kernel(self, x):
        return np.exp(-(((x - self.mu) / self.sigma) ** 2))


class JacobiWeightedKMeans(BaseKMeans):
    def __init__(self, k, tolerance, max_iter, t):
        super().__init__(k, tolerance, max_iter)
        self.t = t

    def kernel(self, x):
        def infinite_sum(n):
            return np.exp(-(n**2) * np.pi * self.t) * np.cos(2 * np.pi * n * x)

        f = 1 + 2 * np.sum(infinite_sum(n) for n in range(1, 100))
        return f


class BesselWeightedKMeans(BaseKMeans):
    def __init__(self, k, tolerance, max_iter, centroids, t):
        super().__init__(k, tolerance, max_iter, centroids)
        self.t = t

    def kernel(self, x):
        f = iv(np.square(x), self.t)
        return f


class HermiteWeightedKMeans(BaseKMeans):
    def __init__(self, k, tolerance, max_iter, n):
        super().__init__(k, tolerance, max_iter)
        self.n = n

    def kernel(self, x):
        f = eval_hermitenorm(self.n, x, out=None)
        return f
