"""Gradient Tracking (course §2.3)."""
import numpy as np


def run_gradient_tracking(
    prob, W, alpha0=None, max_iter=5000, step=None, track_gap=None, snapshot_iters=None
):
    n_agents = prob.num_agents
    m = prob.m
    if alpha0 is None:
        alpha = np.zeros((n_agents, m))
    else:
        alpha = np.asarray(alpha0, dtype=float).copy()
    grad = np.stack([prob.grad_f(i, alpha[i]) for i in range(n_agents)])
    g = grad.copy()
    if step is None:
        L = prob.lipschitz_local_max()
        # Th. 2.2: small enough alpha; often smaller than DGD constant
        step = 0.1 / L
    gaps = []
    want_snap = set(snapshot_iters) if snapshot_iters is not None else set()
    snapshots = {} if snapshot_iters is not None else None
    for k in range(max_iter):
        alpha_new = W @ alpha - step * g
        grad_new = np.stack(
            [prob.grad_f(i, alpha_new[i]) for i in range(n_agents)]
        )
        g = W @ g + (grad_new - grad)
        alpha = alpha_new
        grad = grad_new
        if track_gap is not None:
            gaps.append(
                np.linalg.norm(alpha - track_gap[np.newaxis, :], axis=1).max()
            )
        if want_snap and (k + 1) in want_snap:
            snapshots[k + 1] = alpha.copy()
    return alpha, np.array(gaps), step, snapshots
