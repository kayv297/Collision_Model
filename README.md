# Collision Model Quantum Simulation

This project contains Python code for simulating and analyzing quantum collision models, preparing collective states (e.g., Dicke/W), optimizing gate parameters (partial-swap), and evaluating/visualizing results. It bundles small scripts for exploration and batch runs alongside utilities for metrics and visualization.

## Repository layout

- `auto_find_gammas.py` — automated search for collision rates/angles ("gammas").
- `collision_dynamics.py` — core collision model dynamics utilities.
- `find_gammas_bruteforce.py` — brute-force sweep for suitable gamma parameters.
- `find_gammas_graddesc.py` — gradient-based search for gammas.
- `gen_dicke_sim.py` — Dicke-state generation/simulation driver.
- `metrics.py` — helper metrics (fidelity, overlaps, etc.).
- `quantum_gates.py` — basic gates and operators used across simulations.
- `state_analysis.py` — analysis helpers for produced states.
- `visualize.py` — plotting/visualization helpers or CLI.
- `pswap_circuit/` — partial-swap (p-swap) related utilities and optimizers.
- `archive/` — older experiments and notebooks kept for reference.
- `data/` — outputs, checkpoints, and large artifacts (ignored by Git).

> Note: All `*.txt` files and the entire `data/` folder are ignored by version control to keep the repo lean.

## Getting started

- Python 3.9+ is recommended.
- Create a virtual environment and install your preferred scientific stack (e.g., NumPy/SciPy/Matplotlib) if required by your workflow.

Example (PowerShell):

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
# Install packages you need for your experiments, for example:
# pip install numpy scipy matplotlib
```

## Usage

Each script is intended to be runnable on its own. Common entry points include:

```powershell
python gen_dicke_sim.py --help
python auto_find_gammas.py --help
python visualize.py --help
```

Most scripts write intermediate and final outputs under `data/` by convention (which is ignored by Git).

## Data management

- Place large outputs, logs, or checkpoints in `data/`.
- Text artifacts (`*.txt`) are intentionally ignored by Git to avoid committing large logs and sweep results.

## Contributing

- Keep computational outputs and logs out of version control (use `data/`).
- Prefer small, testable utilities and document CLI flags via `--help`.

## License

TBD — add a license if/when you publish the project.
