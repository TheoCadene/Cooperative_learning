"""
Centralized kernel ridge regression (reference alpha*).
(sigma^2 K_mm + K_nm^T K_nm + nu I) alpha* = K_nm^T y
"""
import numpy as np
from utils import Cov, Cov2


def solve_centralized(x_data, y_data, x_m, sigma=0.5, nu=1.0):
    """
    x_data, y_data: n training points
    x_m: m Nyström centers (1d array length m)
    """
    x_data = np.asarray(x_data, dtype=float).ravel()
    y_data = np.asarray(y_data, dtype=float).ravel()
    x_m = np.asarray(x_m, dtype=float).ravel()
    K_mm = Cov(x_m)
    K_nm = Cov2(x_data, x_m)
    sig2 = sigma ** 2
    A = sig2 * K_mm + K_nm.T @ K_nm + nu * np.eye(len(x_m))
    b = K_nm.T @ y_data
    alpha_star = np.linalg.solve(A, b)
    return alpha_star, K_mm, K_nm


def predict_on_grid(alpha, x_m, x_grid):
    K_t = Cov2(x_grid, x_m)
    return K_t @ alpha
