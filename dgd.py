"""Decentralized Gradient Descent (course §2.2.1)."""
import numpy as np


def run_dgd(prob, W, alpha0=None, max_iter=5000, step=None, track_gap=None):
    """
    alpha_i^{k+1} = sum_j W_ij alpha_j^k - step * grad f_i(alpha_i^k)
    """
    n_agents = prob.num_agents
    m = prob.m
    if alpha0 is None:
        alpha = np.zeros((n_agents, m))
    else:
        alpha = np.asarray(alpha0, dtype=float).copy()
    if step is None:
        L = prob.lipschitz_local_max()
        # Th. 2.1: step O(1/L); DGD often needs smaller constant than 2/L
        step = 0.5 / L
    gaps = []
    for k in range(max_iter):
        grad = np.stack([prob.grad_f(i, alpha[i]) for i in range(n_agents)])
        alpha = W @ alpha - step * grad
        if track_gap is not None:
            gaps.append(
                np.linalg.norm(alpha - track_gap[np.newaxis, :], axis=1).max()
            )
    return alpha, np.array(gaps), step
