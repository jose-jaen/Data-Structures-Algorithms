from math import sqrt
from random import randint
from typing import Any, NoReturn, Optional

import numpy as np
import pandas as pd


class KMeans:
    def __init__(
            self,
            data: pd.DataFrame,
            n_clusters: int = 5,
            max_iter: int = 200,
    ):
        self.data = data
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids: Optional[pd.DataFrame] = None

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
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        self._check_type(att_name='data', att_val=data, att_exp=pd.DataFrame)
        self._data = data
        self._data['cluster'] = None

    def initialize_centroids(self) -> NoReturn:
        """Randomly initialize centroids."""
        centroids = []
        while len(centroids) != self._n_clusters:
            candidate = randint(a=0, b=len(self._data) - 1)

            # Check if centroid was already chosen
            if candidate not in centroids:
                centroids.append(candidate)

        # Convert into numpy array
        cols = self._data.columns[self._data.columns != 'cluster']
        self.centroids = self._data.loc[centroids, cols].to_numpy()

    def assign_centroids(self, obs: int) -> NoReturn:
        """Retrieve the closest centroid to an observation

        Args:
            obs: Index of an obversation

        Returns:
            Closes centroid to observation
        """
        dists = []

        # Get numerical columns
        cols = self._data.columns[self._data.columns != 'cluster']
        for centroid in range(self._n_clusters):
            distance = sum((self._data.loc[obs, cols] - self.centroids[centroid])**2)
            dists.append(sqrt(distance))
        self._data.loc[obs, 'cluster'] = str(dists.index(min(dists)))

    def get_clusters(self) -> pd.Series:
        """Recalculate clusters until convergence."""
        n_iter = 0
        cols = self._data.columns[self._data.columns != 'cluster']

        # Initialize centroids
        self.initialize_centroids()

        # Compute new centroids until convergence
        diff_clusters = True
        diff_centroids = True
        while n_iter < self._max_iter and diff_clusters and diff_centroids:
            # Assign centroids
            past_clusters = self._data.copy()
            for row in range(len(self._data)):
                self.assign_centroids(row)

            # Update centroids
            past_centroids = self.centroids.copy()
            for centroid in range(self._n_clusters):
                avg = np.mean(
                    self._data.loc[self._data['cluster'] == str(centroid), cols],
                    axis=0
                )
                self.centroids[centroid] = avg

            # Check if clusters didn't change
            if past_clusters['cluster'].equals(self._data['cluster']):
                diff_centroids = False

            # Check if changes in new centroids
            if np.array_equal(past_centroids, self.centroids):
                diff_clusters = False

            # Update number of iterations
            n_iter += 1
        return self._data['cluster']
