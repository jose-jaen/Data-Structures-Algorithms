import warnings
from typing import Optional, Any, Type, Union, Tuple, List

import numpy as np
import pandas as pd

warnings.simplefilter('always', UserWarning)


class LinearRegression:
    def __init__(
            self,
            intercept: bool = True,
            learning_rate: float = 1.0,
            max_iter: int = 0,
            tol: float = 0.0,
            l2_penalty: float = 0.0,
            solver: str = 'ols'
    ):
        self.intercept = intercept
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.l2_penalty = l2_penalty
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    @staticmethod
    def _type_error(
            att_name: str,
            att_value: Any,
            att_type: Union[Tuple[Type[Any], Type[Any]], Type[Any]]
    ) -> None:
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
    def solver(self) -> str:
        return self._solver

    @solver.setter
    def solver(self, solver: str) -> None:
        self._type_error(
            att_name='solver',
            att_value=solver,
            att_type=str
        )

        # Only OLS and Gradient Descent implementations
        if solver not in ['ols', 'gradient']:
            raise ValueError(
                f"'solver' must be either 'ols' or 'gradient' but got '{solver}'"
            )
        self._solver = solver

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float) -> None:
        self._type_error(
            att_name='learning_rate',
            att_value=learning_rate,
            att_type=float
        )

        # Warn user and check for incorrect values
        if learning_rate != 1.0 and self._solver == 'ols':
            warnings.warn(
                "'ols' estimation method does not use 'learning_rate', so it will be ignored",
                UserWarning
            )
        elif learning_rate <= 0.0 and self._solver != 'ols':
            raise ValueError("'learning_rate' must be strictly positive")
        self._learning_rate = learning_rate

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter: int) -> None:
        self._type_error(
            att_name='max_iter',
            att_value=max_iter,
            att_type=int
        )

        # Warn user and check for incorrect values
        if max_iter and self._solver == 'ols':
            warnings.warn(
                "'ols' estimation method does not use 'max_iter', so it will be ignored",
                UserWarning
            )
        elif max_iter <= 0 and self._solver != 'ols':
            raise ValueError("'max_iter' must be strictly positive")
        self._max_iter = max_iter

    @property
    def tol(self) -> float:
        return self._tol

    @tol.setter
    def tol(self, tol: float) -> None:
        self._type_error(
            att_name='tol',
            att_value=tol,
            att_type=float
        )

        # Warn user and check for incorrect values
        if tol and self._solver == 'ols':
            warnings.warn(
                "'ols' estimation method does not use 'tol', so it will be ignored",
                UserWarning
            )
        elif tol <= 0.0 and self._solver != 'ols':
            raise ValueError("'tol' must be strictly positive")
        self._tol = tol

    @property
    def l2_penalty(self) -> float:
        return self._l2_penalty

    @l2_penalty.setter
    def l2_penalty(self, l2_penalty: float) -> None:
        self._type_error(
            att_name='l2_penalty',
            att_value=l2_penalty,
            att_type=float
        )

        if l2_penalty < 0:
            raise ValueError("'l2_penalty' must be strictly positive")
        self._l2_penalty = l2_penalty

    def ols(
            self,
            regressors: Union[np.ndarray, pd.DataFrame],
            target: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Estimate a Linear Regression model with OLS.

        Args:
            - regressors (Union[np.ndarray, pd.DataFrame]): Feature matrix
            - target (Union[np.ndarray, pd.DataFrame]): Target vector

        Returns:
            - np.ndarray: Coefficients
        """
        if self._l2_penalty:
            penalty = np.zeros(shape=(regressors.shape[1], regressors.shape[1]))
            penalty[1:, 1:] = self._l2_penalty * np.identity(n=regressors.shape[1] - 1)
        else:
            penalty = 0
        cross_product = regressors.T @ regressors + penalty

        # Control for singularity
        try:
            cross_product_inv = np.linalg.inv(cross_product)
        except np.linalg.LinAlgError:
            cross_product_inv = np.linalg.pinv(cross_product)

        # Get closed-form solution
        self.coef_ = cross_product_inv @ regressors.T @ target
        return np.array(self.coef_)

    def fit(
            self,
            regressors: Union[np.ndarray, pd.DataFrame],
            target: Union[np.ndarray, pd.Series]
    ) -> Tuple[Optional[float], List[float]]:
        """Train a Linear Regression model.

        Args:
            - regressors (Union[np.ndarray, pd.DataFrame]): Feature matrix
            - target (Union[np.ndarray, pd.DataFrame]): Target vector

        Returns:
            - Tuple(Optional[float], List[float]): Coefficients
        """
        # Convert to numpy arrays
        if isinstance(regressors, pd.DataFrame):
            regressors = regressors.to_numpy()

        if isinstance(target, pd.Series):
            target = target.to_numpy()

        # Add intercept term if required
        if self._intercept:
            regressors = np.c_[np.ones((regressors.shape[0],)), regressors]

        # Get coefficients
        self.coef_ = self.ols(regressors, target)

        # Separate intercept and coefficients
        self.intercept_ = self.coef_[0] if self._intercept else None
        self.coef_ = self.coef_[1:] if self._intercept else self.coef_
        return self.intercept_, self.coef_
