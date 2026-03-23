"""
Part I extensions: directed graphs, packet loss, asynchrony, push-sum; scaling n with m=ceil(sqrt(n)).

Theory (short):
- DGD with doubly stochastic W (undirected Metropolis): consensus toward average; convergence
  to the global minimizer under smoothness/coercivity assumptions.
- Row-stochastic W (directed): consensus limit is NOT the average in general → bias vs alpha*.
  Optimality gap and consensus error need not vanish at the same rate.
- Push-sum with column-stochastic A on a strongly connected digraph: s/w converges to
  average consensus, restoring a correct consensus target for the ratio states.
- Random link failures: time-varying stochastic weights → degraded consensus / descent.
- Asynchronous updates: related to bounded-delay models; small step + sufficient mixing helps.

Figures: part1_robustness.pdf, part1_scaling_n.pdf
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from kernel_problem import build_problem
from centralized import solve_centralized
from graphs import (
    adjacency_line,
    metropolis_weights,
    row_stochastic_directed_ring,
    column_stochastic_directed_ring,
)
from dgd_variants import track_metrics_loop
from utils import savefig_pdf, load_first_database

FIG = os.path.join(os.path.dirname(__file__), "figures")


def plot_robustness_panel(prob, alpha_star, fname):
    """Single figure: undirected DGD vs directed DGD vs packet loss vs async vs push-sum."""
    n_ag = prob.num_agents
    W_undir = metropolis_weights(adjacency_line(n_ag))
    W_dir = row_stochastic_directed_ring(n_ag)
    A_col = column_stochastic_directed_ring(n_ag)
    step = 0.5 / prob.lipschitz_local_max()
    T = 3500
    rng = np.random.default_rng(7)

    # Baseline: undirected Metropolis
    hist_undir = track_metrics_loop(
        prob, W_undir, "dgd_row", T, step, rng=rng, alpha_star=alpha_star
    )
    # Directed row-stochastic (breaks average consensus)
    hist_dir = track_metrics_loop(
        prob, W_dir, "dgd_row", T, step, rng=rng, alpha_star=alpha_star
    )
    # Packet loss on undirected weights
    hist_loss = track_metrics_loop(
        prob,
        W_undir,
        "packet_loss",
        T,
        step,
        rng=rng,
        alpha_star=alpha_star,
        loss_prob=0.45,
    )
    # Async on undirected
    hist_async = track_metrics_loop(
        prob,
        W_undir,
        "async",
        T,
        step,
        rng=rng,
        alpha_star=alpha_star,
        update_prob=0.35,
    )
    # Push-sum on directed column-stochastic A
    hist_ps = track_metrics_loop(
        prob, A_col, "push_sum", T, step, rng=rng, alpha_star=alpha_star
    )

    it = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.loglog(it, np.maximum(hist_undir["gap_to_star"], 1e-16), label="DGD (Metropolis)")
    ax.loglog(it, np.maximum(hist_dir["gap_to_star"], 1e-16), label="DGD (directed ring)")
    ax.loglog(it, np.maximum(hist_loss["gap_to_star"], 1e-16), label="DGD + packet loss")
    ax.loglog(it, np.maximum(hist_async["gap_to_star"], 1e-16), label="DGD async")
    ax.loglog(it, np.maximum(hist_ps["gap_to_star"], 1e-16), label="Push-sum (directed)")
    ax.set_xlabel(r"Iteration $t$")
    ax.set_ylabel(r"$\max_i \|z_i^t - \alpha^\star\|$ (or $\alpha_i$)")
    ax.set_title("Distance to centralized minimizer")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    ax = axes[1]
    ax.loglog(it, np.maximum(hist_undir["consensus_error"], 1e-16), label="DGD (Metropolis)")
    ax.loglog(it, np.maximum(hist_dir["consensus_error"], 1e-16), label="DGD (directed)")
    ax.loglog(it, np.maximum(hist_loss["consensus_error"], 1e-16), label="+ packet loss")
    ax.loglog(it, np.maximum(hist_async["consensus_error"], 1e-16), label="async")
    ax.loglog(it, np.maximum(hist_ps["consensus_error"], 1e-16), label="Push-sum")
    ax.set_xlabel(r"Iteration $t$")
    ax.set_ylabel(r"$\max_i \|\alpha_i - \bar\alpha\|$ (consensus error)")
    ax.set_title("Consensus error")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    fig.suptitle(
        "Robustness: directed mixing / loss / async (theory: need doubly stochastic or push-sum)"
    )
    plt.tight_layout()
    savefig_pdf(fname, fig)


def scaling_experiment(n_list, num_agents_list, max_iter, seed=123):
    """
    For each n, m = ceil(sqrt(n)), optional num_agents per entry.
    Records final consensus error and ||sum grad|| after max_iter DGD (Metropolis).
    Centralized solve kept when affordable (same m, moderate n).
    """
    rows = []
    for k, n in enumerate(n_list):
        na = num_agents_list[k] if num_agents_list is not None else min(24, max(5, n // 400))
        prob = build_problem(n=n, m=None, num_agents=na, seed=seed + k)
        alpha_star, _, _ = solve_centralized(prob.x_data, prob.y_data, prob.x_m)
        W = metropolis_weights(adjacency_line(prob.num_agents))
        step = 0.5 / prob.lipschitz_local_max()
        hist = track_metrics_loop(
            prob, W, "dgd_row", max_iter, step, rng=np.random.default_rng(seed + k), alpha_star=alpha_star
        )
        rows.append(
            {
                "n": n,
                "m": prob.m,
                "num_agents": prob.num_agents,
                "final_gap": hist["gap_to_star"][-1],
                "final_consensus": hist["consensus_error"][-1],
                "final_grad_sum": hist["grad_sum_norm"][-1],
            }
        )
    return rows


def plot_scaling(rows, fname, max_iter):
    ns = np.array([r["n"] for r in rows])
    fc = np.array([r["final_consensus"] for r in rows])
    fg = np.array([r["final_gap"] for r in rows])
    gs = np.array([r["final_grad_sum"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ns, np.maximum(fc, 1e-16), "o-", label=f"Consensus err. @ t={max_iter}")
    ax.loglog(ns, np.maximum(fg, 1e-16), "s--", label="Gap to $\\alpha^\\star$")
    ax.loglog(ns, np.maximum(gs, 1e-16), "^-", label=r"$\|\sum_a \nabla f_a\|$")
    ax.set_xlabel(r"Data size $n$ ($m=\lceil\sqrt{n}\rceil$)")
    ax.set_ylabel("Metric value (log scale)")
    ax.set_title(f"DGD scaling: larger $n$ increases local problem size $m$; fixed iterations={max_iter}")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    savefig_pdf(fname, fig)


def main():
    np.random.seed(42)
    # --- Panel (small n, m=10 for comparability with Part I default) ---
    prob = build_problem(n=100, m=10, num_agents=5, seed=42)
    alpha_star, _, _ = solve_centralized(prob.x_data, prob.y_data, prob.x_m)
    plot_robustness_panel(prob, alpha_star, "part1_robustness.pdf")

    # --- Scaling: push n as far as reasonable (cap by dataset size and RAM) ---
    x_all, _ = load_first_database()
    # Synthetic fallback has 1e6 points; full K_nm for n~1e6, m~1e3 is not practical here.
    n_cap = min(len(x_all), 100_000)
    candidates = [100, 400, 900, 2500, 4900, 10000]
    agents_cand = [10, 12, 14, 16, 20, 24]
    n_list = []
    num_agents_list = []
    for n, a in zip(candidates, agents_cand):
        if n <= n_cap:
            n_list.append(n)
            num_agents_list.append(a)
    if n_list and n_cap > n_list[-1]:
        n_list.append(n_cap)
        num_agents_list.append(num_agents_list[-1])
    max_iter = 2500
    rows = scaling_experiment(n_list, num_agents_list, max_iter=max_iter, seed=123)
    plot_scaling(rows, "part1_scaling_n.pdf", max_iter=max_iter)

    print("Robustness & scaling figures saved to", FIG)
    for r in rows:
        print(
            f"  n={r['n']}, m={r['m']}, agents={r['num_agents']}: "
            f"consensus={r['final_consensus']:.2e}, gap={r['final_gap']:.2e}, grad_sum={r['final_grad_sum']:.2e}"
        )


if __name__ == "__main__":
    main()
