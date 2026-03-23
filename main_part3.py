"""Part III: DGD-DP optimality gap vs iterations for epsilon scaling."""
import numpy as np
import matplotlib.pyplot as plt

from kernel_problem import build_problem
from centralized import solve_centralized
from graphs import adjacency_line, metropolis_weights
from dgd_dp import run_dgd_dp
from utils import savefig_pdf


def main():
    np.random.seed(42)
    prob = build_problem(n=100, m=10, num_agents=5, seed=42)
    alpha_star, _, _ = solve_centralized(prob.x_data, prob.y_data, prob.x_m)
    W = metropolis_weights(adjacency_line(5))

    fig, ax = plt.subplots(figsize=(7, 5))
    for epsilon_dp in [0.1, 1.0, 10.0]:
        g = run_dgd_dp(
            prob,
            W,
            max_iter=4000,
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
    ax.set_title("DGD with Laplacian noise (noise scale $\\propto 1/\\epsilon$)")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    savefig_pdf("part3_dgd_dp.pdf", fig)
    print("Part III figure saved.")


if __name__ == "__main__":
    main()
