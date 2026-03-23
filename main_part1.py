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
from utils import savefig_pdf

FIG = os.path.join(os.path.dirname(__file__), "figures")


def run_all_algorithms(prob, W, alpha_star, max_iter=4000, adjacency=None):
    out = {}
    if adjacency is None:
        adjacency = adjacency_line(prob.num_agents)
    a, g, s = run_dgd(prob, W, max_iter=max_iter, track_gap=alpha_star)
    out["DGD"] = (g, s)
    a, g, s = run_gradient_tracking(
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
    alpha_ag, _, _ = run_gradient_tracking(
        prob, W, max_iter=5000, track_gap=None
    )
    alpha1 = alpha_ag[0]
    y1 = predict_on_grid(alpha1, prob.x_m, xg)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(prob.x_data, prob.y_data, "o", ms=3, alpha=0.5, label="Data")
    ax.plot(xg, y_star, "-", lw=2, label=r"Centralized $\alpha^\star$")
    ax.plot(xg, y1, "--", lw=2, label="Agent 1 (GT)")
    ax.set_xlabel(r"Feature $x$")
    ax.set_ylabel(r"Label $y$")
    ax.legend()
    ax.grid(True, alpha=0.5)
    savefig_pdf(fname, fig)


def main():
    np.random.seed(42)
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
