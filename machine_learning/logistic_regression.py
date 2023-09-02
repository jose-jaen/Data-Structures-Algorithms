from typing import List, Optional, Union, Tuple, Any, Type

import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(
            self,
            intercept: bool = True,
            learning_rate: float = 0.01,
            max_iter: int = 1000,
            tol: float = 0.001,
            l2_penalty: float = 0.0,
            solver: str = 'newton'
    ):
        self.learning_rate = learning_rate
        self.intercept = intercept
        self.max_iter = max_iter
        self.tol = tol
        self.l2_penalty = l2_penalty
        self.solver = solver
        self.coef_: Optional[List[float]] = None
        self.intercept_: Optional[float] = None

    @staticmethod
    def _type_error(att_name: str, att_value: Any, att_type: Type) -> None:
        """Raise TypeError if attribute has unexpected type."""
        expected = att_type.__name__
        mistyped = type(att_value).__name__
        if not isinstance(att_value, att_type):
            raise TypeError(
                f"'{att_name}' must be '{expected}' but got '{mistyped}'"
            )

    @property
    def intercept(self) -> bool:
        return self._intercept

    @intercept.setter
    def intercept(self, intercept: bool) -> None:
        self._type_error(
            att_name='intercept',
            att_value=intercept,
            att_type=bool
        )
        self._intercept = intercept

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self._type_error(
            att_name='learning_rate',
            att_value=learning_rate,
            att_type=float
        )
        if learning_rate <= 0:
            raise ValueError("'learning_rate' must be strictly positive")
        self._learning_rate = learning_rate

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter: int):
        self._type_error(
            att_name='max_iter',
            att_value=max_iter,
            att_type=int
        )
        if max_iter <= 0:
            raise ValueError("'max_iter' must be strictly positive")
        self._max_iter = max_iter

    @property
    def tol(self) -> float:
        return self._tol

    @tol.setter
    def tol(self, tol: float):
        self._type_error(
            att_name='tol',
            att_value=tol,
            att_type=float
        )
        if tol <= 0:
            raise ValueError("'tol' must be strictly positive")
        self._tol = tol

    @property
    def l2_penalty(self) -> float:
        return self._l2_penalty

    @l2_penalty.setter
    def l2_penalty(self, l2_penalty: float):
        self._type_error(
            att_name='l2_penalty',
            att_value=l2_penalty,
            att_type=float
        )
        if l2_penalty < 0:
            raise ValueError("'l2_penalty' must be strictly positive")
        self._l2_penalty = l2_penalty

    @property
    def solver(self) -> str:
        return self._solver

    @solver.setter
    def solver(self, solver: str) -> None:
        self._type_error(
            att_name='solver',
            att_value=solver,
            att_type=str
        )
        if solver not in ['gradient', 'newton']:
            raise ValueError(
                f"'solver' must be either 'gradient' or 'newton' but got '{solver}'"
            )
        self._solver = solver

    @staticmethod
    def _sigmoid(z: np.ndarray) -> float:
        """Stable sigmoid activation function."""
        z = np.clip(z, -709, 709)
        return 1 / (1 + np.exp(-z))

    def gradient_ascent(
            self,
            regressors: Union[pd.DataFrame, np.ndarray],
            target: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """Implement Gradient Ascent algorithm for learning.

        Args:
            - regressors (Union[pd.DataFrame, np.ndarray]): Feature matrix
            - target (Union[pd.DataFrame, np.ndarray]: Target vector

        Returns:
            - np.ndarray: Coefficients
        """
        z = np.dot(regressors, self.coef_)
        preds = self._sigmoid(z)

        # Account for regularization
        penalty = -2 * self.l2_penalty * np.array(self.coef_)
        sample_size = regressors.shape[0]

        # Get stable gradient
        gradient = (1 / sample_size) * (regressors.T @ (target - preds)) - penalty
        self.coef_ += self.learning_rate * gradient
        return np.array(self.coef_)

    def newton_raphson(
            self,
            regressors: Union[pd.DataFrame, np.ndarray],
            target: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """Implement Newton-Raphson algorithm for learning.

        Args:
            - regressors (Union[pd.DataFrame, np.ndarray]): Feature matrix
            - target (Union[pd.DataFrame, np.ndarray]: Target vector

        Returns:
            - np.ndarray: Coefficients
        """
        z = np.dot(regressors, self.coef_)
        predictions = self._sigmoid(z)

        # Compute derivatives
        gradient = regressors.T @ (target - predictions)
        diagonal = np.diag(predictions * (1 - predictions))
        hessian = -regressors.T @ diagonal @ regressors

        # Control for singularity
        try:
            hessian_inv = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            hessian_inv = np.linalg.pinv(hessian)

        # Update coefficients
        self.coef_ -= hessian_inv @ gradient
        return np.array(self.coef_)

    def fit(
            self,
            regressors: Union[pd.DataFrame, np.ndarray],
            target: Union[pd.Series, np.ndarray],
            standardize: bool = False
    ) -> Tuple[Optional[float], List[float]]:
        """Train a Logistic Regression model.

        Args:
            - regressors (Union[pd.DataFrame, np.ndarray]): Feature matrix
            - target (Union[pd.DataFrame, np.ndarray]: Target vector
            - standardize (bool): Whether to standardize the feature matrix or not

        Returns:
            - Tuple(Optional[float], List[float]): Coefficients
        """
        # Convert to numpy array for enhanced performance
        if isinstance(regressors, pd.DataFrame):
            regressors = regressors.to_numpy()

        if standardize:
            mean = regressors.mean(axis=0)
            std = regressors.std(axis=0)
            regressors = (regressors - mean) / std

        if self._intercept:
            regressors = np.c_[np.ones((regressors.shape[0],)), regressors]

        if isinstance(target, pd.DataFrame):
            target = target.to_numpy()

        # Initialize variables for iterating
        iteration = 0
        self.coef_ = np.random.normal(
            loc=0.0,
            scale=0.001,
            size=regressors.shape[1]
        )
        past = np.zeros((regressors.shape[1],))
        distance = np.linalg.norm(self.coef_ - past)

        # Iterate until convergence
        while iteration < self._max_iter and distance > self._tol:
            iteration += 1
            past = self.coef_.copy()
            if self.solver == 'gradient':
                self.coef_ = self.gradient_ascent(regressors, target)
            else:
                self.coef_ = self.newton_raphson(regressors, target)
            distance = np.linalg.norm(self.coef_ - past)

        self.intercept_ = self.coef_[0] if self._intercept else None
        self.coef_ = self.coef_ if not self._intercept else self.coef_[1:]
        return self.intercept_, self.coef_
