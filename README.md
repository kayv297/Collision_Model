# Dicke State Preparation via Collision Model

This repository implements a quantum collision model for preparing arbitrary Dicke states $|D^k_N\rangle$. The approach uses a two-stage optimization workflow to achieve high-fidelity state preparation with magnitude and phase alignment.

## Two-Stage Workflow

### Stage 1: Magnitude Alignment

**Script:** [`find_gammas_graddesc.py`](find_gammas_graddesc.py)

**Purpose:** Optimize the interaction strengths (gamma parameters) to align the magnitudes of state amplitudes with the ideal Dicke state distribution.

**What it does:**
- Uses L-BFGS-B gradient descent optimization with multiple initial guesses
- Optimizes two key parameters:
  - `gamma_sh` (γ_shuttle): Shuttle-register partial-swap angle
  - `gamma_in` (γ_intra): Intra-register interaction strength
- Maximizes adaptive fidelity between the evolved state and ideal Dicke state magnitudes

**How to run:**
```python
python find_gammas_graddesc.py
```

**Configuration:** Edit the parameters directly in the script:
- `N`: Total number of qubits in Dicke state (default: 5)
- `K`: Number of excitations (default: 2)
- `MAX_ROUNDS`: Maximum collision rounds to simulate (default: 200)
- `NOISE_CONFIG`: Noise model to apply (default: `NO_NOISE`)

**Output:**
- Optimized gamma values saved to checkpoint files
- Best-performing parameters and corresponding loss values
- Information about the optimal round number for state preparation

### Stage 2: Phase Alignment

**Script:** [`rz_optimize.py`](rz_optimize.py)

**Purpose:** After magnitude alignment, optimize phase corrections using single-qubit Rz rotations to achieve the correct relative phases for the ideal Dicke state.

**What it does:**
- Loads the state from Stage 1 (magnitude-aligned state)
- Optimizes individual Rz rotation angles for each qubit
- Uses L-BFGS-B optimization to maximize true fidelity including phase information
- Supports both pure states and mixed states (density matrices)

**How to run:**
```python
python rz_optimize.py
```

**Configuration:** Specify in the `main()` function:
- `N`: Number of qubits (must match Stage 1)
- `K`: Number of excitations (must match Stage 1)
- `best_round`: The optimal round from Stage 1 optimization
- `evol_path` or `evol_file`: Path to the state evolution data from Stage 1

**Output:**
- Optimized Rz rotation angles for each qubit
- Final fidelity after phase correction
- Comparison of true fidelity before and after phase alignment

## Files Description

### Core Simulation

#### [`gen_dicke_sim.py`](gen_dicke_sim.py)
Main simulation driver for the collision model. Configures system parameters and orchestrates the Dicke state preparation process.

**Key features:**
- Configures system: number of qubits, ancilla positions, register sizes
- Defines interaction strengths (gamma parameters)
- Sets noise models and collision schemes
- Runs the collision dynamics evolution
- Handles both pure state vectors and density matrices
- Decide whether to use post-selection on shuttles or not

**Main function:** `dicke_evolution()` - Simulates the full evolution process across multiple collision rounds.

#### [`collision_dynamics.py`](collision_dynamics.py)
Implements the physics of quantum collisions: intra-register and inter-register (shuttle-mediated) interactions.

### Optimization Algorithms

#### [`find_gammas_graddesc.py`](find_gammas_graddesc.py)
**Stage 1 script.** Gradient-based optimization using L-BFGS-B to find optimal interaction parameters (gamma values) for magnitude alignment.

#### [`rz_optimize.py`](rz_optimize.py)
**Stage 2 script.** Phase correction optimization using single-qubit Rz gates to align relative phases.

### State Analysis and Metrics

#### [`state_analysis.py`](state_analysis.py)
Quantum state initialization, manipulation, and analysis utilities.

### Quantum Gates and Operations

#### [`quantum_gates.py`](quantum_gates.py)
Quantum gate definitions and operations for multi-qubit systems.

**Key features:**
- `partial_swap()`: Creates partial SWAP gate with tunable angle γ
- `apply_two_qubit()`: Applies 2-qubit gates to state vectors efficiently
- `expand_two_qubit_gate()`: Expands 2-qubit gates to full Hilbert space
- Basic quantum states: `zero`, `one`
- SWAP gate definition

### Noise Modeling

#### [`noise_config.py`](noise_config.py)
Centralized configuration system for quantum noise models.

**Noise models available:**
- `NO_NOISE`: Ideal unitary evolution
- `FAILED_INTERACTIONS`: Classical control failures (probabilistic gate failures)
- `DEPHASING`: Phase damping (T₂ decoherence)
- `AMPLITUDE_DAMPING`: Energy relaxation (T₁ decay, $|1\rangle \to |0\rangle$)
- `DEPOLARIZING`: Uniform random Pauli noise

**Class:** `NoiseConfig` - Container for noise parameters and model selection

#### [`noise_channels.py`](noise_channels.py)
Quantum noise channel implementations using Kraus operator formalism.

**Channels implemented:**
- `dephasing_channel()`: Phase damping via Kraus operators $\{K_0, K_1\}$
- `amplitude_damping_channel()`: Energy relaxation $|1\rangle \to |0\rangle$
- `depolarizing_channel()`: Uniform Pauli noise mixture
- `apply_single_qubit_channel()`: Generic single-qubit channel application

### Visualization and Analysis

#### [`visualize.py`](visualize.py)
Plotting and visualization functions for quantum state evolution.

#### [`collision_analysis.py`](collision_analysis.py)
Advanced analysis tools for parameter sweeps and visualizations.

## Dependencies

```python
numpy              # Numerical computation
scipy              # Optimization algorithms
matplotlib         # Visualization
tqdm               # Progress bars
multiprocessing    # Parallel optimization
```

## Usage Example

### Complete Two-Stage Workflow

**Step 1: Magnitude Alignment**

Edit [`find_gammas_graddesc.py`](find_gammas_graddesc.py) to set your target state:
```python
N = 5  # Total qubits
K = 2  # Number of excitations
MAX_ROUNDS = 200
NOISE_CONFIG = AMPLITUDE_DAMPING
```

Run the optimization:
```bash
python find_gammas_graddesc.py
```

The script will output the optimized parameters:
- `gamma_sh_opt`: Optimized shuttle interaction strength
- `gamma_in_opt`: Optimized intra-register strength  
- `best_round`: Optimal round number for state extraction

**Step 2: Phase Alignment**

Using the results from Step 1, edit [`rz_optimize.py`](rz_optimize.py):
```python
N = 5
K = 2
best_round = 128  # From Step 1 output
evol_file = "path/to/amp_evolution.npy"  # From Step 1 output
```

Run the phase optimization:
```bash
python rz_optimize.py
```

This will compute the optimal Rz angles and report the final fidelity.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{Tam_collision_model_2025,
  author = {Nguyen, Minh Tam and Vu, Duc-Kha and Fatih, Ozaydin},
  title = {{Dicke State Preparation via Collision Model}},
  year = {2025},
  url = {https://github.com/kayv297/Collision_Model},
  version = {1.0.0},
  note = {Python implementation of gradient-based optimization for quantum collision model parameters using L-BFGS-B}
}
```

and

```bibtex
our published paper
```

## Authors

- **Minh Tam Nguyen**
- **Duc-Kha Vu**
- **Fatih Ozaydin**