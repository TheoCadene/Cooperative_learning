"""
DGD with Laplacian noise (DGD-DP, course §9.5, arXiv:2202.01113).
chi_i = alpha_i + zeta_i; alpha_new = W @ chi - step * grad
Noise: Laplace with scale nu_k per dimension (course Thm 9.2).

Important: the step must respect local Lipschitz constants (same order as DGD, ~1/L).
The previous placeholder step ~1/(1+0.1k) was O(1) while L~10^2 here → divergence.
Noise scale must not grow with k (that amplified Laplace variance to absurd levels).
"""
import numpy as np


def run_dgd_dp(
    prob,
    W,
    alpha0=None,
    max_iter=5000,
    epsilon_dp=1.0,
    seed=42,
    track_gap=None,
):
    """
    Laplace noise scale ν_k ∝ 1/ε (Remark 5, arXiv:2202.01113): smaller ε (stronger DP) → larger ν_k.
    We use ν_k = (ν0/ε) / sqrt(1+k) so variance vanishes as k→∞ (stable) while keeping ε-ordering.

    Steps: diminishing 0.5/L / sqrt(1+k) (same spirit as Thm 2.1 constant step bound).
    epsilon_dp is the target ε in ε-DP (e.g. 0.1, 1, 10).
    """
    rng = np.random.default_rng(seed)
    n_agents = prob.num_agents
    m = prob.m
    if alpha0 is None:
        alpha = np.zeros((n_agents, m))
    else:
        alpha = np.asarray(alpha0, dtype=float).copy()

    L = prob.lipschitz_local_max()
    # Base step O(1/L); mild decay for noisy gradient flow
    step0 = 0.5 / L

    gaps = []
    for k in range(max_iter):
        step_k = step0 / np.sqrt(1.0 + k)
        # Stronger privacy (smaller ε) → larger noise; variance decays with k
        nu_k = (0.15 / max(epsilon_dp, 1e-6)) / np.sqrt(1.0 + k)
        noise = rng.laplace(0.0, nu_k, size=(n_agents, m))
        chi = alpha + noise
        grad = np.stack([prob.grad_f(i, alpha[i]) for i in range(n_agents)])
        alpha = W @ chi - step_k * grad
        if track_gap is not None:
            gaps.append(
                np.linalg.norm(alpha - track_gap[np.newaxis, :], axis=1).max()
            )
    return alpha, np.array(gaps)
