"""
Part I: DGD, Gradient Tracking, Dual Decomposition, ADMM — figures (PDF).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from kernel_problem import build_problem
from centralized import solve_centralized, predict_on_grid
from graphs import (
    adjacency_line,
    adjacency_full,
    adjacency_small_world,
    metropolis_weights,
    spectral_gamma,
)
from dgd import run_dgd
from gradient_tracking import run_gradient_tracking
from dual_decomposition import run_dual_decomposition
from admm import run_admm
from utils import savefig_pdf, load_first_database

FIG = os.path.join(os.path.dirname(__file__), "figures")


def plot_first_database_sample(fname, n_show=12_000, seed=42):
    """
    Project 'Visualize the data': scatter of a random subset of first_database.pkl
    (full file has 1e6 points; plotting all is unreadable).
    """
    x, y = load_first_database()
    rng = np.random.default_rng(seed)
    n_show = min(n_show, len(x))
    idx = rng.choice(len(x), size=n_show, replace=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        x[idx],
        y[idx],
        s=2,
        alpha=0.25,
        c="C0",
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlabel(r"Feature $x$")
    ax.set_ylabel(r"Label $y$")
    ax.set_title(
        rf"First database: {n_show:,} / {len(x):,} points, "
        r"$\sigma=0.5$"
    )
    ax.set_xlim(-1.0, 1.0)
    ax.grid(True, alpha=0.4)
    savefig_pdf(fname, fig)


def run_all_algorithms(prob, W, alpha_star, max_iter=4000, adjacency=None):
    out = {}
    if adjacency is None:
        adjacency = adjacency_line(prob.num_agents)
    a, g, s = run_dgd(prob, W, max_iter=max_iter, track_gap=alpha_star)
    out["DGD"] = (g, s)
    a, g, s, _ = run_gradient_tracking(
        prob, W, max_iter=max_iter, track_gap=alpha_star
    )
    out["GT"] = (g, s)
    a, g, s = run_dual_decomposition(
        prob, adjacency, max_iter=max_iter, step=0.1, track_gap=alpha_star
    )
    out["DualDec"] = (g, s)
    a, g, b = run_admm(
        prob, adjacency, beta=0.05, max_iter=max_iter, track_gap=alpha_star
    )
    out["ADMM"] = (g, b)
    return out


def plot_convergence_curves(results, title, fname):
    fig, ax = plt.subplots(figsize=(7, 5))
    it = np.arange(1, len(next(iter(results.values()))[0]) + 1)
    for name, (gaps, _) in results.items():
        ax.loglog(it, np.maximum(gaps, 1e-16), label=name)
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel(r"$\max_i \|\alpha_i^t - \alpha^\star\|$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    savefig_pdf(fname, fig)


def plot_topology_comparison(prob, alpha_star, fname):
    fig, ax = plt.subplots(figsize=(7, 5))
    configs = [
        ("Line", adjacency_line(5)),
        ("Small-world", adjacency_small_world(5, seed=1)),
        ("Full", adjacency_full(5)),
    ]
    max_iter = 3000
    for label, Adj in configs:
        W = metropolis_weights(Adj)
        g = run_dgd(prob, W, max_iter=max_iter, track_gap=alpha_star)[1]
        it = np.arange(1, len(g) + 1)
        ax.loglog(it, np.maximum(g, 1e-16), label=f"DGD ({label}), $\\gamma={spectral_gamma(W):.3f}$")
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel(r"$\max_i \|\alpha_i^t - \alpha^\star\|$")
    ax.set_title("DGD: effect of topology (Metropolis weights)")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    savefig_pdf(fname, fig)


def plot_reconstruction(prob, alpha_star, fname):
    nt = 250
    xg = np.linspace(-1, 1, nt)
    y_star = predict_on_grid(alpha_star, prob.x_m, xg)
    W = metropolis_weights(adjacency_line(5))
    _, _, _, snaps = run_gradient_tracking(
        prob,
        W,
        max_iter=4000,
        track_gap=None,
        snapshot_iters=[100, 4000],
    )
    alpha1_100 = snaps[100][0]
    alpha1_4000 = snaps[4000][0]
    y_100 = predict_on_grid(alpha1_100, prob.x_m, xg)
    y_4000 = predict_on_grid(alpha1_4000, prob.x_m, xg)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        prob.x_data,
        prob.y_data,
        "s",
        ms=3,
        color="C0",
        alpha=0.85,
        label="Data",
        zorder=1,
    )
    ax.plot(
        xg,
        y_star,
        "-",
        lw=2,
        color="C1",
        label="Reconstruction",
        zorder=3,
    )
    ax.plot(
        xg,
        y_100,
        "--",
        lw=2,
        color="C2",
        label="Reconstruction Agent 1 after 100 iterations",
        zorder=2,
    )
    ax.plot(
        xg,
        y_4000,
        "--",
        lw=2,
        color="C3",
        label="Reconstruction Agent 1 after 4000 iterations",
        zorder=4,
    )
    ax.set_xlabel("Features")
    ax.set_ylabel("Labels")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=9,
        frameon=True,
    )
    ax.grid(True, alpha=0.5)
    fig.subplots_adjust(top=0.82, bottom=0.14)
    fig.text(
        0.5,
        0.02,
        "Figure 2.3. Decentralized reconstruction by using Gradient Tracking "
        "on the Kernel ridge regression example.",
        ha="center",
        fontsize=10,
        style="italic",
    )
    savefig_pdf(fname, fig, tight_layout=False)


def main():
    np.random.seed(42)
    plot_first_database_sample("part0_data_visualization.pdf")
    prob = build_problem(n=100, m=10, num_agents=5, seed=42)
    alpha_star, _, _ = solve_centralized(prob.x_data, prob.y_data, prob.x_m)

    W_line = metropolis_weights(adjacency_line(5))
    res = run_all_algorithms(prob, W_line, alpha_star, adjacency=adjacency_line(5))
    plot_convergence_curves(
        res,
        "Part I: algorithms (line graph, Metropolis)",
        "part1_algorithms_line.pdf",
    )

    plot_topology_comparison(prob, alpha_star, "part1_dgd_topologies.pdf")
    plot_reconstruction(prob, alpha_star, "part1_reconstruction.pdf")

    # ADMM beta comparison (Jacobi consensus ADMM; very large beta can diverge)
    A = adjacency_line(5)
    fig, ax = plt.subplots(figsize=(7, 5))
    for beta in [0.02, 0.05, 0.1]:
        g = run_admm(
            prob, A, beta=beta, max_iter=3000, track_gap=alpha_star
        )[1]
        it = np.arange(1, len(g) + 1)
        ax.loglog(it, np.maximum(g, 1e-16), label=f"$\\beta={beta}$")
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel(r"$\max_i \|\alpha_i^t - \alpha^\star\|$")
    ax.set_title(r"ADMM vs.\ $\beta$ (stable range; $\beta\gtrsim 0.15$ may diverge)")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    savefig_pdf("part1_admm_beta.pdf", fig)

    print("Part I figures saved to", FIG)


if __name__ == "__main__":
    main()
