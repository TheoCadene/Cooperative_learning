"""Part III: DGD-DP optimality gap vs iterations for epsilon scaling."""
import numpy as np
import matplotlib.pyplot as plt

from kernel_problem import build_problem
from centralized import solve_centralized
from graphs import adjacency_line, metropolis_weights
from dgd_dp import run_dgd_dp
from utils import savefig_pdf


def _plot_dgd_dp_panel(prob, alpha_star, W, max_iter, fname, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    for epsilon_dp in [0.1, 1.0, 10.0]:
        g = run_dgd_dp(
            prob,
            W,
            max_iter=max_iter,
            epsilon_dp=epsilon_dp,
            seed=42,
            track_gap=alpha_star,
        )[1]
        it = np.arange(1, len(g) + 1)
        ax.loglog(
            it,
            np.maximum(g, 1e-16),
            label=f"$\\epsilon$-DP, $\\epsilon={epsilon_dp}$",
        )
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel(r"$\max_i \|\alpha_i^t - \alpha^\star\|$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    savefig_pdf(fname, fig)


def main():
    np.random.seed(42)
    prob = build_problem(n=100, m=10, num_agents=5, seed=42)
    alpha_star, _, _ = solve_centralized(prob.x_data, prob.y_data, prob.x_m)
    W = metropolis_weights(adjacency_line(5))
    _plot_dgd_dp_panel(
        prob,
        alpha_star,
        W,
        max_iter=10000,
        fname="part3_dgd_dp.pdf",
        title=r"DGD with Laplacian noise ($n=100$, $m=10$, $a=5$)",
    )

    # Optional (project PDF): larger problem — more agents, higher dimension m.
    prob_l = build_problem(n=1000, m=33, num_agents=100, seed=42)
    alpha_l, _, _ = solve_centralized(
        prob_l.x_data, prob_l.y_data, prob_l.x_m
    )
    W_l = metropolis_weights(adjacency_line(100))
    _plot_dgd_dp_panel(
        prob_l,
        alpha_l,
        W_l,
        max_iter=8000,
        fname="part3_dgd_dp_large.pdf",
        title=r"DGD-DP (optional): $n=1000$, $m=33$, $a=100$, line + Metropolis",
    )
    print("Part III figures saved (base + optional large).")


if __name__ == "__main__":
    main()
