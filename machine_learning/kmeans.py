from typing import Any, Optional, Union

import numpy as np
from scipy.spatial.distance import cdist


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
        self.wcss = Union[int, float]

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
        centroids = [np.random.choice(self._data.shape[0])]
        while len(centroids) != self._n_clusters:
            mask = np.ones(self._data.shape[0], dtype=bool)
            mask[centroids] = False

            # Calculate squared distances
            distances = np.sum(
                (self._data[mask, :] - self._data[centroids[-1], :]) ** 2,
                axis=1
            )

            # Normalize distances to probabilities
            probabilities = distances / distances.sum(axis=0, keepdims=True)

            # Select next centroid
            next_centroid = np.random.choice(
                np.arange(self._data.shape[0])[mask],
                p=probabilities
            )
            centroids.append(next_centroid)

        centroids = self._data[centroids, :]
        self.centroids = centroids
        return centroids

    def get_clusters(self) -> np.ndarray[int]:
        """Recalculate clusters until convergence."""
        n_iter = 0

        # Initialize centroids
        self.centroids = self.initialize_centroids().astype(float)

        # Add new column for centroids
        total = len(self._data)
        max_limit = self._n_clusters - 1
        rand_pts = [np.random.randint(low=0, high=max_limit) for _ in range(total)]
        self._data = np.c_[self._data, rand_pts]

        # Compute new centroids until convergence
        diff_clusters = True
        diff_centroids = True
        while n_iter < self._max_iter and diff_clusters and diff_centroids:
            # Store previous iterations
            former_clusters = self._data[:, -1].copy()
            former_centroids = self.centroids.copy()

            # Calculate distances for all observations and centroids
            distances = cdist(self._data[:, :-1], self.centroids)

            # Assign each observation to the closest centroid
            new_clusters = np.argmin(distances, axis=1)

            # Update clusters
            self._data[:, -1] = new_clusters

            # Update centroids
            self.centroids = np.array(
                [
                    np.mean(self._data[self._data[:, -1] == k, :-1], axis=0)
                    for k in range(self._n_clusters)
                ]
            )

            # Check if points remain in the same cluster
            diff_clusters = not all(np.equal(former_clusters, self._data[:, -1]))

            # Check if centroids didn't change significantly
            tolerance = 1e-8
            diff_centroids = not np.allclose(
                self.centroids.astype(float),
                former_centroids.astype(float),
                atol=tolerance
            )

            # Update number of iterations
            n_iter += 1
        return self._data[:, -1]

    def get_wcss(self) -> Union[int, float]:
        """Retrieve WCSS."""
        self.wcss = 0
        for centroid in range(self._n_clusters):
            data_points = self._data[self._data[:, -1] == centroid, :-1]
            centroid = self.centroids[centroid, :]
            distance = np.sum(np.power(data_points - centroid, 2))
            self.wcss += round(distance, 2)
        return self.wcss
