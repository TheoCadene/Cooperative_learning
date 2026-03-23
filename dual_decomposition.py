"""Peer-to-peer dual decomposition (course §4.3.2)."""
import numpy as np
from graphs import edge_list


def run_dual_decomposition(prob, adjacency, max_iter=5000, step=None, track_gap=None):
    """
    Primal: alpha_i = H_i^{-1} (b_i - c_i), c_i = sum_{j<i} lambda_ij - sum_{j>i} lambda_ji
    Dual: lambda_ij += step * (alpha_i - alpha_j)
    """
    n_agents = prob.num_agents
    m = prob.m
    edges = edge_list(adjacency)
    # lambda[(i,j)] for j < i
    lam = {e: np.zeros(m) for e in edges}
    neigh = [np.where(adjacency[i] > 0)[0] for i in range(n_agents)]

    def c_vec(i):
        s = np.zeros(m)
        for j in neigh[i]:
            if j < i:
                s += lam[(i, j)]
            elif j > i:
                s -= lam[(j, i)]
        return s

    # step from Th. 4.2: alpha < 2m / sigma_max^2(A); use conservative bound
    if step is None:
        # Build incidence-like A for edges: rough sigma_max ~ sqrt(2 * max degree) for m-dim blocks
        # Use a safe small step
        step = 0.05

    gaps = []
    zero = np.zeros(m)
    b_list = [-prob.grad_f(i, zero) for i in range(n_agents)]

    for k in range(max_iter):
        alpha = np.zeros((n_agents, m))
        for i in range(n_agents):
            Hi = prob.hessian_f(i)
            bi = b_list[i]
            ci = c_vec(i)
            alpha[i] = np.linalg.solve(Hi, bi - ci)
        for e in edges:
            i, j = e
            lam[e] = lam[e] + step * (alpha[i] - alpha[j])
        if track_gap is not None:
            gaps.append(
                np.linalg.norm(alpha - track_gap[np.newaxis, :], axis=1).max()
            )
    return alpha, np.array(gaps), step
