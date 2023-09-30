from math import sqrt
from random import randint
from typing import Any, NoReturn, Optional

import numpy as np


class KMeans:
    def __init__(
            self,
            data: np.ndarray,
            n_clusters: int = 5,
            max_iter: int = 200,
    ):
        self.data = data
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids: Optional[np.ndarray] = None

    @staticmethod
    def _check_type(att_name: str, att_val: Any, att_exp: type) -> None:
        """Raise TypeError message if unexpected attribute type.

        Args:
            att_name: Attribute name
            att_val: Attribute value
            att_exp: Expected attribute data type
        """
        att_type = type(att_val).__name__
        if not isinstance(att_val, att_exp):
            raise TypeError(
                f"Expected type '{att_exp}' for '{att_name}' but got '{att_type}'"
            )

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, n_clusters: int) -> None:
        self._check_type(att_name='n_clusters', att_val=n_clusters, att_exp=int)
        self._n_clusters = n_clusters

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter: int) -> None:
        self._check_type(att_name='max_iter', att_val=max_iter, att_exp=int)
        self._max_iter = max_iter

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data: np.ndarray) -> None:
        self._check_type(att_name='data', att_val=data, att_exp=np.ndarray)
        self._data = data

    def initialize_centroids(self) -> np.ndarray:
        """Implement k-means++ method."""
        centroids = [randint(a=0, b=self._data.shape[0] - 1)]
        while len(centroids) != self._n_clusters:
            mask = np.ones(self._data.shape[0], dtype=bool)
            mask[centroids] = False

            # Calculate squared distances
            distances = np.sum(
                (self._data[mask, :] - self._data[centroids[-1], :]) ** 2,
                axis=1
            )

            # Normalize distances to probabilities
            probabilities = distances / np.sum(distances)

            # Select next centroid
            next_centroid = np.random.choice(
                np.arange(self._data.shape[0])[mask],
                p=probabilities
            )
            centroids.append(next_centroid)
        self.centroids = self._data[centroids, :]

        # Add new column for centroids
        self._data = np.c_[self._data, [None] * len(self._data)]
        return self.centroids

    def assign_centroids(self, obs: int) -> NoReturn:
        """Retrieve the closest centroid to an observation.

        Args:
            obs: Index of an obversation

        Returns:
            Closes centroid to observation
        """
        dists = []

        # Calculate L2 distance
        for centroid in range(self._n_clusters):
            data_points = self._data[obs, :-1]
            data_centroids = self.centroids[centroid]
            distance = sum((data_points - data_centroids)**2)
            dists.append(sqrt(distance))
        self._data[obs, -1] = str(dists.index(min(dists)))

    def get_clusters(self) -> np.ndarray:
        """Recalculate clusters until convergence."""
        n_iter = 0

        # Initialize centroids
        self.initialize_centroids()

        # Compute new centroids until convergence
        diff_clusters = True
        diff_centroids = True
        while n_iter < self._max_iter and diff_clusters and diff_centroids:
            for row in range(len(self._data)):
                self.assign_centroids(row)
            past_clusters = self._data[: -1].copy()

            # Update centroids
            past_centroids = self.centroids.copy()
            for centroid in range(self._n_clusters):
                centroid_filter = self._data[:, -1] == str(centroid)
                avg = np.mean(
                    self._data[centroid_filter, :-1],
                    axis=0
                )
                self.centroids[centroid] = avg

            # Check if clusters didn't change
            if np.array_equal(past_clusters, self._data[:, -1]):
                diff_centroids = False

            # Check if changes in new centroids
            if np.array_equal(past_centroids, self.centroids):
                diff_clusters = False

            # Update number of iterations
            n_iter += 1
        return self._data[:, -1]
