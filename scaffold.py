"""
SCAFFOLD (Stochastic Controlled Averaging) — Karimireddy et al., 2020.

Mitigates client drift under non-IID local data by maintaining control variates c (server)
and c_i (each client). Local SGD uses corrected gradients:

    y <- y - lr * ( g_i(y) - c_i + c )

After K local steps on client i (starting from server model x^t):

    c_i <- c_i - c + (1/(lr*K)) * (x^t - y_i^K)

Server aggregates models and updates:

    x^{t+1} <- mean_i y_i^K   (on participating clients, then average)
    c^{t+1} <- c + (1/N) * sum_i (c_i^{new} - c_i^{old})

Non-participating clients keep c_i unchanged (their contribution to the sum is 0).

Same API as fedavg.run_fedavg for drop-in comparison.
"""
import numpy as np


def run_scaffold(
    prob,
    rounds,
    E,
    B,
    lr,
    seed=42,
    track_star=None,
    clients_participate=None,
):
    rng = np.random.default_rng(seed)
    m = prob.m
    C = prob.num_agents
    alpha = np.zeros(m)
    c = np.zeros(m)
    c_clients = np.zeros((C, m))
    obj_gap = []

    if clients_participate is None:
        clients_participate = C

    for _t in range(rounds):
        c_snapshot = c_clients.copy()
        chosen = rng.choice(C, size=min(clients_participate, C), replace=False)
        x_t = alpha.copy()
        locals_out = []

        for a in chosen:
            local = x_t.copy()
            local_idx = prob.agent_indices[a]
            n_loc = len(local_idx)
            step_count = 0
            for _epoch in range(E):
                perm = rng.permutation(n_loc)
                for s in range(0, n_loc, B):
                    batch = perm[s : s + B]
                    idx_g = local_idx[batch]
                    g = prob.grad_f_minibatch(a, local, idx_g)
                    local = local - lr * (g - c_clients[a] + c)
                    step_count += 1

            K = max(step_count, 1)
            c_clients[a] = c_clients[a] - c + (1.0 / (lr * K)) * (x_t - local)
            locals_out.append(local)

        alpha = np.mean(np.stack(locals_out, axis=0), axis=0)
        # c^{t+1} = c^t + (1/N) sum_i (c_i^{new} - c_i^{old}); non-participating: delta 0
        c = c + np.mean(c_clients - c_snapshot, axis=0)

        if track_star is not None:
            obj_gap.append(
                prob.F_global_centralized_form(alpha)
                - prob.F_global_centralized_form(track_star)
            )

    return alpha, np.array(obj_gap)
