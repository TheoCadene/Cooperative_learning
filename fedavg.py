"""Federated Averaging (course §7.2)."""
import numpy as np


def run_fedavg(
    prob,
    rounds,
    E,
    B,
    lr,
    seed=42,
    track_star=None,
    clients_participate=None,
):
    """
    Server averages client models each round.
    clients_participate: number of clients per round (default: all).
    """
    rng = np.random.default_rng(seed)
    m = prob.m
    C = prob.num_agents
    alpha = np.zeros(m)
    obj_gap = []

    if clients_participate is None:
        clients_participate = C

    for t in range(rounds):
        updates = []
        chosen = rng.choice(C, size=min(clients_participate, C), replace=False)
        for a in chosen:
            local = alpha.copy()
            local_idx = prob.agent_indices[a]
            n_loc = len(local_idx)
            for _epoch in range(E):
                perm = rng.permutation(n_loc)
                for s in range(0, n_loc, B):
                    batch = perm[s : s + B]
                    idx_g = local_idx[batch]
                    g = prob.grad_f_minibatch(a, local, idx_g)
                    local = local - lr * g
            updates.append(local)
        alpha = np.mean(np.stack(updates, axis=0), axis=0)
        if track_star is not None:
            obj_gap.append(
                prob.F_global_centralized_form(alpha)
                - prob.F_global_centralized_form(track_star)
            )
    return alpha, np.array(obj_gap)
