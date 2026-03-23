"""
Peer-to-peer ADMM for consensus: minimize sum_i f_i(alpha_i) s.t. alpha_i = alpha_j on edges.

Augmented Lagrangian on edge e = (a,b) with a > b (single multiplier lambda_e):
  lambda_e^T (alpha_a - alpha_b) + (beta/2) ||alpha_a - alpha_b||^2

Jacobi primal update (fixed neighbors' iterates from previous round):
  grad f_i(alpha_i) + sum_{j in N_i} [ lambda_{e(i,j)} + beta (alpha_i - alpha_j^{old}) ] = 0
  => (H_i + beta * deg(i) I) alpha_i = b_i - c_i(lambda) + beta * sum_{j in N_i} alpha_j^{old}

where c_i = sum_{e=(a,b)} (+lambda_e if i==a else -lambda_e if i==b).

Dual (after all primals at k+1):
  lambda_e^{k+1} = lambda_e^k + beta * (alpha_a^{k+1} - alpha_b^{k+1})

This matches the consensus ADMM splitting (see Boyd et al., BPC11; course §4.5 discussion).
Use a modest beta (e.g. 1e-3 ... 5e-2): large beta can make the Jacobi map unstable.
"""
import numpy as np
from graphs import edge_list


def run_admm(prob, adjacency, beta=1.0, max_iter=5000, track_gap=None):
    n_agents = prob.num_agents
    m = prob.m
    edges = edge_list(adjacency)
    neigh = [np.where(adjacency[i] > 0)[0] for i in range(n_agents)]
    deg = np.array([len(neigh[i]) for i in range(n_agents)], dtype=float)

    lam = {e: np.zeros(m) for e in edges}
    zero = np.zeros(m)
    b_list = [-prob.grad_f(i, zero) for i in range(n_agents)]

    gaps = []
    alpha = np.zeros((n_agents, m))

    def c_from_lambda(i):
        """Linear term from lambda^T (x_a - x_b): +lambda on max index, -lambda on min."""
        s = np.zeros(m)
        for e in edges:
            a, b = e  # a > b
            if i == a:
                s += lam[e]
            elif i == b:
                s -= lam[e]
        return s

    for _k in range(max_iter):
        alpha_old = np.array(alpha, copy=True)
        alpha_new = np.zeros((n_agents, m))
        for i in range(n_agents):
            Hi = prob.hessian_f(i)
            rhs = b_list[i] - c_from_lambda(i)
            for j in neigh[i]:
                rhs += beta * alpha_old[j]
            M = Hi + beta * deg[i] * np.eye(m)
            alpha_new[i] = np.linalg.solve(M, rhs)
        alpha = alpha_new
        for e in edges:
            a, b = e
            lam[e] = lam[e] + beta * (alpha[a] - alpha[b])
        if track_gap is not None:
            gaps.append(
                np.linalg.norm(alpha - track_gap[np.newaxis, :], axis=1).max()
            )
    return alpha, np.array(gaps), beta
