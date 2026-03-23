"""
Distributed kernel ridge regression problem (Part I formulation).
Each agent a has f_a(alpha) =
  (sigma^2/num_agents) * (1/2) alpha^T K_mm alpha
  + (1/2) sum_{i in A_a} (y_i - K^(i)_m alpha)^2
  + (nu/(2*num_agents)) ||alpha||^2

Sum over agents equals centralized objective with (sigma^2/2) alpha^T K_mm alpha + ...
"""
import numpy as np
from utils import Cov, Cov2


class KernelDistributedProblem:
    def __init__(
        self,
        x_data,
        y_data,
        x_m,
        num_agents=5,
        sigma=0.5,
        nu=1.0,
        agent_indices=None,
        seed=42,
    ):
        self.x_data = np.asarray(x_data, dtype=float).ravel()
        self.y_data = np.asarray(y_data, dtype=float).ravel()
        self.n = len(self.x_data)
        self.x_m = np.asarray(x_m, dtype=float).ravel()
        self.m = len(self.x_m)
        self.num_agents = num_agents
        self.sigma = sigma
        self.nu = nu
        self.sig2 = sigma ** 2

        self.K_mm = Cov(self.x_m)
        self.K_nm = Cov2(self.x_data, self.x_m)

        if agent_indices is None:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(self.n)
            chunk = self.n // num_agents
            self.agent_indices = [
                perm[a * chunk : (a + 1) * chunk] for a in range(num_agents)
            ]
        else:
            self.agent_indices = [np.asarray(idx, dtype=int) for idx in agent_indices]

        # Local Hessian pieces (constant for quadratic f_a)
        self._H_base = []
        self._b_local = []
        na = float(num_agents)
        for a in range(num_agents):
            idx = self.agent_indices[a]
            K_A = self.K_nm[idx, :]  # n_a x m
            H_a = (self.sig2 / na) * self.K_mm + K_A.T @ K_A + (nu / na) * np.eye(self.m)
            b_a = K_A.T @ self.y_data[idx]
            self._H_base.append(H_a)
            self._b_local.append(b_a)

    def grad_f(self, a, alpha):
        """Gradient of f_a at alpha (column vector convention)."""
        return self._H_base[a] @ alpha - self._b_local[a]

    def f_a(self, a, alpha):
        """Scalar f_a(alpha)."""
        na = float(self.num_agents)
        term1 = 0.5 * (self.sig2 / na) * (alpha @ self.K_mm @ alpha)
        idx = self.agent_indices[a]
        r = self.y_data[idx] - self.K_nm[idx, :] @ alpha
        term2 = 0.5 * np.sum(r ** 2)
        term3 = 0.5 * (self.nu / na) * (alpha @ alpha)
        return term1 + term2 + term3

    def F_total(self, alpha):
        return sum(self.f_a(a, alpha) for a in range(self.num_agents))

    def F_global_centralized_form(self, alpha):
        """Same as sum_a f_a(alpha); equals centralized ridge objective."""
        return self.F_total(alpha)

    def grad_f_minibatch(self, a, alpha, idx_global):
        """Stochastic-style gradient using subset idx_global of agent a's data."""
        na = float(self.num_agents)
        K_A = self.K_nm[idx_global, :]
        y_A = self.y_data[idx_global]
        return (self.sig2 / na) * (self.K_mm @ alpha) - K_A.T @ (
            y_A - K_A @ alpha
        ) + (self.nu / na) * alpha

    def hessian_f(self, a):
        return self._H_base[a]

    def lipschitz_local_max(self):
        """Max eigenvalue of local Hessians (Lipschitz constant of grad f_a)."""
        return max(np.linalg.eigvalsh(self._H_base[a]).max() for a in range(self.num_agents))

    def consensus_error(self, alpha_stack):
        """
        max_i ||alpha_i - mean(alpha)|| — measures lack of consensus.
        At optimum of the global problem, all alpha_i should coincide (unique minimizer).
        """
        mu = alpha_stack.mean(axis=0)
        return float(np.linalg.norm(alpha_stack - mu, axis=1).max())

    def grad_sum_norm(self, alpha_stack):
        """
        ||sum_a grad f_a(alpha_a)|| — vanishes at a stationary point of the sum objective
        when all alpha_a are equal (global optimum in the consensus subspace).
        """
        g = np.zeros(self.m)
        for a in range(self.num_agents):
            g += self.grad_f(a, alpha_stack[a])
        return float(np.linalg.norm(g))


def build_from_second_database(X, Y, m=10, num_agents=None, sigma=0.5, nu=1.0, seed=0):
    """Part II: X[i], Y[i] local to agent i; Nyström centers on linspace(-1,1,m)."""
    X = [np.asarray(x, dtype=float).ravel() for x in X]
    Y = [np.asarray(y, dtype=float).ravel() for y in Y]
    if num_agents is None:
        num_agents = len(X)
    x_all = np.concatenate(X)
    y_all = np.concatenate(Y)
    rng = np.random.default_rng(seed)
    x_m = np.linspace(-1.0, 1.0, m)
    agent_indices = []
    off = 0
    for a in range(num_agents):
        l = len(X[a])
        agent_indices.append(np.arange(off, off + l, dtype=int))
        off += l
    prob = KernelDistributedProblem(
        x_all,
        y_all,
        x_m,
        num_agents=num_agents,
        sigma=sigma,
        nu=nu,
        agent_indices=agent_indices,
        seed=seed,
    )
    return prob


def build_problem(n=100, m=None, num_agents=5, seed=42):
    """
    Standard Part I setup: first n points, random Nyström subset of size m.
    If m is None, use m = ceil(sqrt(n)) as in the project scaling experiment.
    """
    rng = np.random.default_rng(seed)
    from utils import load_first_database

    if m is None:
        m = int(np.ceil(np.sqrt(n)))
    x_full, y_full = load_first_database()
    x_data = x_full[:n].copy()
    y_data = y_full[:n].copy()
    sel = np.arange(n)
    ind = rng.choice(sel, m, replace=False)
    x_m = x_data[ind].copy()
    return KernelDistributedProblem(x_data, y_data, x_m, num_agents=num_agents, seed=seed)
