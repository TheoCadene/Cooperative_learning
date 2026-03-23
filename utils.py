"""
Utility functions: kernel matrices, data loading, matplotlib style.
Notation aligned with course 5OD14 (K_mm, K_nm, sigma, nu).
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib setup (sample_figure_file convention)
matplotlib.rc("font", family="sans-serif", size=12)


def Cov(x):
    """Kernel matrix K_mm for Nyström centers x (list or 1d array)."""
    m = len(x)
    x = np.asarray(x, dtype=float).ravel()
    Kmm = np.eye(m)
    for ii in range(m):
        for jj in range(ii + 1, m):
            Kmm[ii, jj] = np.exp(-((x[ii] - x[jj]) ** 2))
            Kmm[jj, ii] = Kmm[ii, jj]
    return Kmm


def Cov2(x1, x2):
    """Cross-kernel K_nm: rows = data points x1, columns = Nyström centers x2."""
    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    d = x1[:, np.newaxis] - x2[np.newaxis, :]
    return np.exp(-(d ** 2))


def load_first_database(path=None):
    """Load (x, y) from first_database.pkl or generate synthetic data if missing."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "first_database.pkl")
    if os.path.isfile(path):
        import pickle

        with open(path, "rb") as f:
            x, y = pickle.load(f)
        return np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    # Synthetic fallback (smooth function + noise, sigma=0.5)
    rng = np.random.default_rng(42)
    n = 1_000_000
    x = rng.uniform(-1.0, 1.0, size=n)
    y = np.sin(2 * np.pi * x) + 0.5 * rng.standard_normal(size=n)
    return x, y


def load_second_database(path=None):
    """Load (X, Y) for federated setting: X[i], Y[i] for agent i."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "second_database.pkl")
    if os.path.isfile(path):
        import pickle

        with open(path, "rb") as f:
            X, Y = pickle.load(f)
        return X, Y
    return None, None


def make_synthetic_second_database(n_agents=5, points_per_agent=20, m=10, seed=0):
    """Build second_database-like structure from first_database or synthetic."""
    rng = np.random.default_rng(seed)
    x_all = rng.uniform(-1.0, 1.0, size=n_agents * points_per_agent)
    y_all = np.sin(2 * np.pi * x_all) + 0.5 * rng.standard_normal(size=n_agents * points_per_agent)
    X = []
    Y = []
    for a in range(n_agents):
        sl = slice(a * points_per_agent, (a + 1) * points_per_agent)
        X.append(x_all[sl])
        Y.append(y_all[sl])
    return X, Y


def savefig_pdf(name, fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "figures", name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
