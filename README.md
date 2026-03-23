# Cooperative Kernel Regression (5OD14 project)

## Run everything

```bash
cd Cooperative_learning
python3 LastName1_LastName2_LastName3_Final.py
```

PDF figures are written to `figures/`.

## Dependencies

- `numpy`, `matplotlib`, `pickle` (stdlib)

Place `first_database.pkl` and `second_database.pkl` next to the scripts if available; otherwise synthetic data is generated automatically.

## Report

Compile `report.tex` with `pdflatex` after filling in names and discussion. Rename the PDF per submission instructions.

Part II also writes `figures/part2_fedavg_partial_E.pdf` (partial participation $C=3$, $E\in\{1,5,50\}$, $B=15$).

## ADMM (Part I)

Implementation: **Jacobi consensus ADMM** on `alpha_a - alpha_b = 0` per edge (one multiplier `lambda_e` per edge), not the `y_ij` splitting from the notes (that formulation needs consistent duals on both endpoints; the consensus form is standard and stable).

Penalty `beta` must stay **moderate** (e.g. `0.02`–`0.1` here): larger values make the Jacobi primal–dual map unstable (gaps can blow up to `10^100+`).

## Module map

| File | Role |
|------|------|
| `utils.py` | `Cov`, `Cov2`, data loaders, `savefig_pdf` |
| `centralized.py` | Reference `α*` |
| `kernel_problem.py` | Local `f_a`, gradients, `KernelDistributedProblem` |
| `graphs.py` | Line / full / small-world, Metropolis `W` |
| `dgd.py`, `gradient_tracking.py`, `dual_decomposition.py`, `admm.py` | Part I algorithms |
| `fedavg.py` | Part II FedAvg |
| `scaffold.py` | Part II SCAFFOLD (non-IID / client drift) |
| `dgd_dp.py` | Part III |
| `main_part*.py` | Plot scripts |
| `dgd_variants.py`, `main_part1_robustness.py` | Directed DGD, packet loss, async, push-sum; scaling \(n\), \(m=\lceil\sqrt n\rceil\) |

## Part I — robustness (directed / loss / async / push-sum)

`main_part1_robustness.py` generates `figures/part1_robustness.pdf` and `part1_scaling_n.pdf`.

- **Directed row-stochastic \(W\)**: consensus does not converge to the arithmetic mean in general (unlike doubly stochastic Metropolis on undirected graphs), so DGD can exhibit **bias** vs. \(\alpha^\star\).
- **Push-sum** (`column_stochastic_directed_ring` + ratio \(z=s/w\)): restores **average consensus** on strongly connected digraphs (course push-sum / ratio consensus).
- **Packet loss**: time-varying, random sparsification of \(W_t\) → **no fixed** doubly stochastic limit in expectation; consensus and descent degrade.
- **Asynchrony**: only a random subset updates each iteration; related to **bounded-delay** asynchronous models (small step helps).
- **Scaling \(n\)**: with \(m=\lceil\sqrt n\rceil\), local problems grow; for fixed iteration budget \(T\), **optimality** and **consensus** metrics typically **worsen as \(n\) increases** (larger dimension \(m\), same graph size in agent count). The script caps `n` at `100_000` by default for memory; increase if your machine allows.

`LastName1_LastName2_LastName3_Final.py` runs `main_part1_robustness` after `main_part1`.

## Part III — DGD-DP (Laplace noise)

Use a step **\(O(1/L)\)** with \(L=\max_a \lambda_{\max}(\nabla^2 f_a)\). A constant step \(\sim 1\) was unstable here (\(L\sim 10^2\)). Laplace scale should **not grow with \(k\)** (the old `nu_k ∝ k^{0.3}` blew up the state). Implemented: \(\eta_k = \frac{0.5/L}{\sqrt{1+k}}\), \(\nu_k = \frac{0.15/\varepsilon}{\sqrt{1+k}}\).
