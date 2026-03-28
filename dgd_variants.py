"""
DGD variants: directed mixing, packet loss, asynchrony, push-sum (directed graphs).

Theory (course):
- Undirected + doubly stochastic W: consensus toward average; DGD converges to optimum
  under standard assumptions.
- Row-stochastic W (directed): consensus point is NOT the average in general; DGD incurs
  bias vs. the centralized minimizer (unless weights are doubly stochastic).
- Push-sum with column-stochastic A: tracks sum-weights w and uses z_i = s_i / w_i to
  recover the average consensus on directed strongly connected graphs.
- Random packet drops: time-varying stochastic weights; consensus and descent degrade.
- Asynchronous updates: partial updates; with small step and bounded asynchrony, behavior
  approximates delayed gradient methods (can still converge if assumptions hold).
"""
import numpy as np


def _default_step(prob):
    L = prob.lipschitz_local_max()
    return 0.5 / L


def run_dgd_directed(prob, W_row, max_iter=5000, step=None, track_gap=None):
    """
    Same as standard DGD but W is row-stochastic (typically from a directed graph).
    alpha^{k+1} = W @ alpha^k - step * grad.
    """
    n_agents = prob.num_agents
    m = prob.m
    alpha = np.zeros((n_agents, m))
    if step is None:
        step = _default_step(prob)
    gaps = []
    for k in range(max_iter):
        grad = np.stack([prob.grad_f(i, alpha[i]) for i in range(n_agents)])
        alpha = W_row @ alpha - step * grad
        if track_gap is not None:
            gaps.append(
                np.linalg.norm(alpha - track_gap[np.newaxis, :], axis=1).max()
            )
    return alpha, np.array(gaps), step


def run_dgd_packet_loss(
    prob,
    W_base,
    rng,
    loss_prob=0.4,
    max_iter=5000,
    step=None,
    track_gap=None,
    drop_self=False,
):
    """
    Each iteration: incoming weights W_ij are masked with Bernoulli(1-loss_prob),
    then rows renormalized. Simulates random packet loss on links.
    """
    n_agents = prob.num_agents
    m = prob.m
    alpha = np.zeros((n_agents, m))
    if step is None:
        step = _default_step(prob)
    gaps = []
    for k in range(max_iter):
        M = rng.random((n_agents, n_agents)) > loss_prob
        if not drop_self:
            np.fill_diagonal(M, True)
        W = W_base * M
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        W = W / row_sums
        grad = np.stack([prob.grad_f(i, alpha[i]) for i in range(n_agents)])
        alpha = W @ alpha - step * grad
        if track_gap is not None:
            gaps.append(
                np.linalg.norm(alpha - track_gap[np.newaxis, :], axis=1).max()
            )
    return alpha, np.array(gaps), step


def run_dgd_async(
    prob,
    W,
    rng,
    update_prob=0.4,
    max_iter=5000,
    step=None,
    track_gap=None,
):
    """
    Each iteration: compute full DGD update, but only a random subset of agents
    applies it; others keep previous alpha (asynchronous / gossip-style).
    """
    n_agents = prob.num_agents
    m = prob.m
    alpha = np.zeros((n_agents, m))
    if step is None:
        step = _default_step(prob)
    gaps = []
    for k in range(max_iter):
        grad = np.stack([prob.grad_f(i, alpha[i]) for i in range(n_agents)])
        alpha_new = W @ alpha - step * grad
        upd = rng.random(n_agents) < update_prob
        if not np.any(upd):
            upd[rng.integers(0, n_agents)] = True
        alpha[upd] = alpha_new[upd]
        if track_gap is not None:
            gaps.append(
                np.linalg.norm(alpha - track_gap[np.newaxis, :], axis=1).max()
            )
    return alpha, np.array(gaps), step


def run_push_sum_dgd(
    prob,
    A_col,
    max_iter=5000,
    step=None,
    track_gap=None,
):
    """
    Push-sum / ratio consensus style DGD on directed graphs (column-stochastic A).

    s^{k+1} = A @ s^k - step * grad(z^k),  w^{k+1} = A @ w^k,
    z^k = s^k / w^k (row-wise), gradients at z.

    Recovers average consensus of s/w under strong connectivity + column-stochastic A,
    mitigating bias from plain row-stochastic mixing on directed graphs.
    """
    n_agents = prob.num_agents
    m = prob.m
    s = np.zeros((n_agents, m))
    w = np.ones(n_agents)
    if step is None:
        step = _default_step(prob)
    gaps = []
    eps = 1e-12
    for k in range(max_iter):
        z = s / np.maximum(w[:, np.newaxis], eps)
        grad = np.stack([prob.grad_f(i, z[i]) for i in range(n_agents)])
        s = A_col @ s - step * grad
        w = A_col @ w
        if track_gap is not None:
            z_track = s / np.maximum(w[:, np.newaxis], eps)
            gaps.append(
                np.linalg.norm(z_track - track_gap[np.newaxis, :], axis=1).max()
            )
    z_final = s / np.maximum(w[:, np.newaxis], eps)
    return z_final, np.array(gaps), step


def track_metrics_loop(prob, W_or_A, mode, max_iter, step, rng=None, **extra):
    """
    mode: 'dgd_row' | 'packet_loss' | 'async' | 'push_sum'
    Returns dict with histories consensus_err, grad_sum_norm, gap_to_star (if extra has alpha_star).
    """
    n_agents = prob.num_agents
    m = prob.m
    alpha_star = extra.get("alpha_star")
    loss_prob = extra.get("loss_prob", 0.4)
    update_prob = extra.get("update_prob", 0.4)
    if rng is None:
        rng = np.random.default_rng(0)

    if mode == "push_sum":
        A = W_or_A
        s = np.zeros((n_agents, m))
        w = np.ones(n_agents)
        eps = 1e-12
    else:
        alpha = np.zeros((n_agents, m))

    cons, gsum, gaps = [], [], []
    for k in range(max_iter):
        if mode == "dgd_row":
            grad = np.stack([prob.grad_f(i, alpha[i]) for i in range(n_agents)])
            alpha = W_or_A @ alpha - step * grad
            state = alpha
        elif mode == "packet_loss":
            M = rng.random((n_agents, n_agents)) > loss_prob
            np.fill_diagonal(M, True)
            W = W_or_A * M
            W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
            grad = np.stack([prob.grad_f(i, alpha[i]) for i in range(n_agents)])
            alpha = W @ alpha - step * grad
            state = alpha
        elif mode == "async":
            grad = np.stack([prob.grad_f(i, alpha[i]) for i in range(n_agents)])
            alpha_new = W_or_A @ alpha - step * grad
            upd = rng.random(n_agents) < update_prob
            if not np.any(upd):
                upd[rng.integers(0, n_agents)] = True
            alpha[upd] = alpha_new[upd]
            state = alpha
        elif mode == "push_sum":
            z = s / np.maximum(w[:, np.newaxis], eps)
            grad = np.stack([prob.grad_f(i, z[i]) for i in range(n_agents)])
            s = A @ s - step * grad
            w = A @ w
            state = z
        else:
            raise ValueError(mode)

        ce = prob.consensus_error(state)
        # Push-sum: at t=0 all z_i = 0 / w_i = 0 — consensus is trivially exact but
        # not meaningful; plotting ~1e-16 on a log axis then a jump looks broken.
        if mode == "push_sum" and k == 0:
            ce = np.nan
        cons.append(ce)
        gsum.append(prob.grad_sum_norm(state))
        if alpha_star is not None:
            gaps.append(
                np.linalg.norm(state - alpha_star[np.newaxis, :], axis=1).max()
            )

    out = {
        "consensus_error": np.array(cons),
        "grad_sum_norm": np.array(gsum),
    }
    if alpha_star is not None:
        out["gap_to_star"] = np.array(gaps)
    return out
