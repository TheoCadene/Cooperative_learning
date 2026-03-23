"""
Communication graphs and Metropolis-Hastings weights (course §1.4.1).
"""
import numpy as np


def adjacency_line(num_nodes):
    """Path graph 1-2-...-N (0-indexed)."""
    A = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def adjacency_full(num_nodes):
    A = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    return A


def adjacency_small_world(num_nodes, rng=None, p_extra=0.4, seed=42):
    """
    Ring plus random extra edges (Erdős–Rényi–style small augmentation).
    Ensures connectivity.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    A = adjacency_line(num_nodes)
    # add ring wrap for robustness
    A[0, num_nodes - 1] = 1.0
    A[num_nodes - 1, 0] = 1.0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if A[i, j] < 0.5 and rng.random() < p_extra:
                A[i, j] = 1.0
                A[j, i] = 1.0
    return A


def metropolis_weights(adjacency):
    """
    Metropolis-Hastings weights on undirected graph (course Example 1.5).
    W_ij = 1/(1+max(d_i,d_j)) for (i,j) in E, W_ii = 1 - sum_j W_ij.
    """
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1).astype(float)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and adjacency[i, j] > 0:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
    for i in range(n):
        W[i, i] = 1.0 - W[i, :].sum() + W[i, i]
    return W


def spectral_gamma(W):
    """Second largest eigenvalue modulus (course §1.4.1)."""
    eig = np.linalg.eigvalsh(0.5 * (W + W.T))
    eig = np.sort(eig)[::-1]
    if len(eig) < 2:
        return 0.0
    return max(abs(eig[1]), abs(eig[-1]))


def edge_list(adjacency):
    """Edges (i, j) with j < i for undirected graph without double count."""
    n = adjacency.shape[0]
    edges = []
    for i in range(n):
        for j in range(i):
            if adjacency[i, j] > 0:
                edges.append((i, j))
    return edges


def neighbors(adjacency):
    """List of neighbor indices per node."""
    n = adjacency.shape[0]
    return [np.where(adjacency[i] > 0)[0].tolist() for i in range(n)]


# --- Directed graphs (Part I robustness / push-sum) ----------------------------

def row_stochastic_directed_ring(num_nodes):
    """
    Strongly connected directed cycle: each node i mixes with itself and predecessor (i-1).
    Rows sum to 1 (standard DGD mixing matrix on a directed graph).
    """
    n = num_nodes
    W = np.zeros((n, n))
    for i in range(n):
        W[i, i] = 0.5
        W[i, (i - 1) % n] = 0.5
    return W


def column_stochastic_directed_ring(num_nodes):
    """
    Column-stochastic matrix for push-sum: column j has mass split between j and j+1.
    Columns sum to 1 => preserves total mass 1^T s under s <- A @ s.
    """
    n = num_nodes
    A = np.zeros((n, n))
    for j in range(n):
        A[j, j] = 0.5
        A[(j + 1) % n, j] = 0.5
    return A


def row_stochastic_from_directed_adjacency(adj, self_weight=True):
    """
    adj[i,j] > 0 means i receives from j (or j sends to i), row i sums incoming.
    Normalize each row to sum to 1 (add self-loop if isolated).
    """
    n = adj.shape[0]
    A = np.asarray(adj, dtype=float).copy()
    if self_weight:
        A += np.eye(n)
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    return A / row_sums
