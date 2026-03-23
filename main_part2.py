"""Part II: FedAvg — objective gap vs rounds."""
import numpy as np
import matplotlib.pyplot as plt

from kernel_problem import build_from_second_database
from centralized import solve_centralized
from fedavg import run_fedavg
from scaffold import run_scaffold
from utils import load_second_database, make_synthetic_second_database, savefig_pdf


def main():
    np.random.seed(42)
    X, Y = load_second_database()
    if X is None:
        X, Y = make_synthetic_second_database(n_agents=5, points_per_agent=20, m=10, seed=0)
    prob = build_from_second_database(X, Y, m=10, sigma=0.5, nu=1.0, seed=0)
    x_all = np.concatenate([np.asarray(x).ravel() for x in X])
    y_all = np.concatenate([np.asarray(y).ravel() for y in Y])
    alpha_star, _, _ = solve_centralized(x_all, y_all, prob.x_m)

    rounds = 10000
    lr = 0.002
    fig, ax = plt.subplots(figsize=(7, 5))
    for E in [1, 5, 50]:
        _, gap = run_fedavg(
            prob,
            rounds=rounds,
            E=E,
            B=20,
            lr=lr,
            seed=42,
            track_star=alpha_star,
        )
        it = np.arange(1, len(gap) + 1)
        ax.loglog(it, np.maximum(gap, 1e-16), label=f"$E={E}$")
    ax.set_xlabel("Server round $t$")
    ax.set_ylabel(r"$F(\alpha^t) - F(\alpha^\star)$")
    ax.set_title(r"FedAvg: $B=20$, $C=5$, constant lr$=0.002$")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    savefig_pdf("part2_fedavg_E.pdf", fig)

    # diminishing lr
    lr_f = lambda t: 0.002 / (0.01 * t + 1.0)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for E in [1, 5, 50]:
        alpha = np.zeros(prob.m)
        gaps = []
        for t in range(rounds):
            lr_t = lr_f(t + 1)
            updates = []
            for a in range(prob.num_agents):
                local = alpha.copy()
                for _ in range(E):
                    idx = prob.agent_indices[a]
                    g = prob.grad_f_minibatch(a, local, idx)
                    local = local - lr_t * g
                updates.append(local)
            alpha = np.mean(np.stack(updates, axis=0), axis=0)
            gaps.append(
                prob.F_global_centralized_form(alpha)
                - prob.F_global_centralized_form(alpha_star)
            )
        it = np.arange(1, len(gaps) + 1)
        ax2.loglog(it, np.maximum(np.array(gaps), 1e-16), label=f"$E={E}$")
    ax2.set_xlabel("Server round $t$")
    ax2.set_ylabel(r"$F(\alpha^t) - F(\alpha^\star)$")
    ax2.set_title(r"FedAvg: diminishing lr $=0.002/(0.01 t+1)$")
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    savefig_pdf("part2_fedavg_diminishing.pdf", fig2)

    # partial participation
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    _, gap = run_fedavg(
        prob,
        rounds=rounds,
        E=5,
        B=15,
        lr=lr,
        seed=42,
        track_star=alpha_star,
        clients_participate=5,
    )
    ax3.loglog(np.arange(1, len(gap) + 1), np.maximum(gap, 1e-16), label="$B=15$, $C=5$")
    _, gap2 = run_fedavg(
        prob,
        rounds=rounds,
        E=5,
        B=15,
        lr=lr,
        seed=42,
        track_star=alpha_star,
        clients_participate=3,
    )
    ax3.loglog(np.arange(1, len(gap2) + 1), np.maximum(gap2, 1e-16), label="$B=15$, $C=3$")
    ax3.set_xlabel("Server round $t$")
    ax3.set_ylabel(r"$F(\alpha^t) - F(\alpha^\star)$")
    ax3.set_title("FedAvg: mini-batch / partial participation")
    ax3.legend()
    ax3.grid(True, which="both", ls="--", alpha=0.5)
    savefig_pdf("part2_fedavg_partial.pdf", fig3)

    # partial participation: effect of local epochs E (same B, C as above partial case)
    B_pp, C_partial = 15, 3
    fig3e, ax3e = plt.subplots(figsize=(7, 5))
    for E in [1, 5, 50]:
        _, gap_e = run_fedavg(
            prob,
            rounds=rounds,
            E=E,
            B=B_pp,
            lr=lr,
            seed=42,
            track_star=alpha_star,
            clients_participate=C_partial,
        )
        it_e = np.arange(1, len(gap_e) + 1)
        ax3e.loglog(
            it_e,
            np.maximum(gap_e, 1e-16),
            label=rf"$E={E}$, $B={B_pp}$, $C={C_partial}$",
        )
    ax3e.set_xlabel("Server round $t$")
    ax3e.set_ylabel(r"$F(\alpha^t) - F(\alpha^\star)$")
    ax3e.set_title(
        r"FedAvg: partial participation — $C=3$ clients/round, varying $E$"
    )
    ax3e.legend()
    ax3e.grid(True, which="both", ls="--", alpha=0.5)
    savefig_pdf("part2_fedavg_partial_E.pdf", fig3e)

    # SCAFFOLD vs FedAvg (non-IID across 20-point groups in second_database)
    rounds_sc = 2000
    lr_sc = 0.002
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    for E in [1, 5, 20]:
        _, gap_f = run_fedavg(
            prob,
            rounds=rounds_sc,
            E=E,
            B=20,
            lr=lr_sc,
            seed=42,
            track_star=alpha_star,
        )
        _, gap_s = run_scaffold(
            prob,
            rounds=rounds_sc,
            E=E,
            B=20,
            lr=lr_sc,
            seed=42,
            track_star=alpha_star,
        )
        it = np.arange(1, len(gap_f) + 1)
        ax4.loglog(it, np.maximum(gap_f, 1e-16), label=rf"FedAvg $E={E}$")
        ax4.loglog(
            it,
            np.maximum(gap_s, 1e-16),
            linestyle="--",
            label=rf"SCAFFOLD $E={E}$",
        )
    ax4.set_xlabel("Server round $t$")
    ax4.set_ylabel(r"$F(\alpha^t) - F(\alpha^\star)$")
    ax4.set_title(
        r"FedAvg vs.\ SCAFFOLD ($B=20$, lr$=0.002$); control variates reduce client drift"
    )
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, which="both", ls="--", alpha=0.5)
    savefig_pdf("part2_scaffold.pdf", fig4)

    print("Part II figures saved.")


if __name__ == "__main__":
    main()
