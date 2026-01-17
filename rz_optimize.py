from typing import Tuple
from scipy.optimize import minimize
from math import comb
from state_analysis import calc_fidelity, calc_uhlmann_fidelity
import numpy as np
from itertools import combinations


def state_to_density_matrix(state_vector):
    """Convert state vector |ψ⟩ to density matrix |ψ⟩⟨ψ|."""
    return np.outer(state_vector, state_vector.conj())


def print_density_matrix(rho, label=""):
    """Print density matrix in a readable format."""
    if label:
        print(f"\n{label}")
    
    # Set numpy print options for better complex number display
    np.set_printoptions(precision=6, suppress=True, linewidth=100)
    print(rho)
    
    # Also print useful properties
    trace = np.trace(rho)
    purity = np.trace(rho @ rho)
    print(f"Trace: {trace.real:.6f}")
    print(f"Purity: {purity.real:.6f}")
    return trace


def apply_rz_gates(state, indices: list, phi: list, N: int, is_density_matrix: bool = False) -> np.ndarray:
    """
    Apply different Rz gates to the state at specified indices.
    
    Args:
        state: Either amplitude vector (pure state) or density matrix
        indices: Indices of non-zero basis states (only used for pure state)
        phi: Rotation angles for each qubit
        N: Number of qubits
        is_density_matrix: If True, state is a density matrix; if False, state is amplitude vector
    
    Returns:
        Rotated state (either density matrix or amplitude vector)
    """
    if is_density_matrix:
        # Density matrix case: Apply unitary transformation U ρ U†
        # where U is the diagonal phase gate
        dim = 2**N
        
        # Create diagonal phase operator
        phase_operator = np.zeros(dim, dtype=complex)
        for basis_idx in range(dim):
            total_phase = 0.0
            for qubit_pos in range(N):
                # Check if this qubit is excited (bit is 1)
                if (basis_idx >> (N - 1 - qubit_pos)) & 1:
                    total_phase += phi[qubit_pos]
            phase_operator[basis_idx] = np.exp(1j * total_phase)
        
        # Create diagonal unitary matrix
        U = np.diag(phase_operator)
        
        # Apply: ρ' = U ρ U†
        rotated_state = U @ state @ U.conj().T
        
    else:
        # Pure state case: Apply phase to amplitudes
        rotated_state = np.array(state, dtype=complex)
        
        for i, idx in enumerate(indices):
            # Calculate total phase for this basis state
            total_phase = 0.0
            for qubit_pos in range(N):
                # Check if this qubit is excited (bit is 1)
                if (idx >> (N - 1 - qubit_pos)) & 1:
                    total_phase += phi[qubit_pos]
            
            # Apply phase rotation
            rotated_state[i] *= np.exp(1j * total_phase)
    
    return rotated_state


def loss_function(
    phi: list, state, indices: list, N: int, ideal_state: np.ndarray, is_density_matrix: bool = False
) -> float:
    """
    Maximize fidelity by optimizing Rz gate angles.
    
    Args:
        phi: Rotation angles for each qubit
        state: Either amplitude vector (pure state) or density matrix
        indices: Indices of non-zero basis states (only used for pure state)
        N: Number of qubits
        ideal_state: Target ideal state (amplitude vector for both cases)
        is_density_matrix: If True, state is a density matrix
    
    Returns:
        Negative fidelity (for minimization)
    """
    rotated_state = apply_rz_gates(state, indices, phi, N, is_density_matrix)
    
    if is_density_matrix:
        fidelity = calc_uhlmann_fidelity(rotated_state, ideal_state, use_abs=False)
    else:
        fidelity = calc_fidelity(ideal_state, rotated_state, use_abs=False)
    
    return -float(np.real(fidelity))  # Negative because we minimize


def main(
    N=None, K=None, best_round=None, evol_path=None, evol_file=None, verbose=False
) -> Tuple[float, float, float, float, np.ndarray]:

    if N is None:
        N = 10

    if K is None:
        K = 4

    if best_round is None:
        best_round = 784

    # Load data - support both explicit file path and folder path
    if evol_file is not None:
        # Use explicit file path
        pass
    elif evol_path is not None:
        # Construct file path from folder
        evol_file = f"{evol_path}/D{N}_{K}/amp_evolution.npy"
    else:
        raise ValueError("Must provide either evol_file or evol_path")

    if verbose:
        print(f"Loading data from {evol_file}...")
    amp_evols = np.load(evol_file)
    if verbose:
        print(f"Loaded shape: {amp_evols.shape}")

    # Extract state
    raw_state = amp_evols[best_round]
    indices = []
    amps = []
    is_density_matrix = len(raw_state.shape) == 2
    
    if verbose:
        print(f"State type: {'Density Matrix' if is_density_matrix else 'Pure State'}")
        print(f"Raw state shape: {raw_state.shape}")

    # Extract non-zero values
    if is_density_matrix:
        for i in range(raw_state.shape[0]):
            for j in range(raw_state.shape[1]):
                if abs(raw_state[i, j]) > 1e-8:
                    indices.append((i, j))
                    amps.append(raw_state[i, j])
    else:
        # Pure state case
        for i, amp in enumerate(raw_state):
            if abs(amp) <= 1e-6:
                continue
            indices.append(i)
            amps.append(amp)

    if verbose:
        print(f"Number of non-zero terms: {len(amps)}")

    # Create ideal state
    if is_density_matrix:
        num_ancillas = K
        total_qubits = N
        
        num_terms = comb(total_qubits, num_ancillas)
        ideal_amp = 1 / np.sqrt(num_terms)
        
        # Create ideal Dicke state vector
        ideal_state = np.zeros(2**total_qubits, dtype=complex)
        for excitation_positions in combinations(range(total_qubits), num_ancillas):
            basis_idx = 0
            for pos in excitation_positions:
                basis_idx |= (1 << (total_qubits - 1 - pos))
            ideal_state[basis_idx] = ideal_amp
    else:
        num_term = comb(N, K)
        ideal_term = 1 / np.sqrt(num_term)
        ideal_state = np.array([ideal_term] * len(amps))

    # Calculate raw fidelity
    if is_density_matrix:
        raw_fidelity = calc_uhlmann_fidelity(raw_state, ideal_state, use_abs=False)
        abs_fidelity = calc_uhlmann_fidelity(raw_state, ideal_state, use_abs=True)
    else:
        raw_fidelity = calc_fidelity(ideal_state, np.array(amps), use_abs=False)
        abs_fidelity = calc_fidelity(ideal_state, np.array(amps), use_abs=True)
    
    if verbose:
        print(f"Raw Fidelity: {raw_fidelity:.6f}")
        print(f"Absolute Fidelity: {abs_fidelity:.6f}")

    # Optimize RZ rotations
    if verbose:
        print("\nOptimizing RZ rotation angles...")
    phi_init = np.zeros(N)

    # Select the appropriate state to optimize
    state_to_optimize = raw_state if is_density_matrix else amps

    result = minimize(
        loss_function,
        phi_init,
        args=(state_to_optimize, indices, N, ideal_state, is_density_matrix),
        method="L-BFGS-B",
        options={
            "maxiter": 10000,
            "ftol": 1e-10,
            "gtol": 1e-8,
            "maxfun": 20000 * N,
        },
    )

    optimal_phi = result.x

    if verbose:
        print(f"\nOptimal rotation angles (radians):")
        for i, angle in enumerate(optimal_phi):
            print(f"  Qubit {i}: {angle:.4f} rad = {np.degrees(angle):.2f}°")

    # Apply optimal rotations
    if is_density_matrix:
        final_state = apply_rz_gates(raw_state, indices, optimal_phi, N, is_density_matrix=True)
        final_fidelity = calc_uhlmann_fidelity(final_state, ideal_state, use_abs=False)
    else:
        final_amps = apply_rz_gates(amps, indices, optimal_phi, N, is_density_matrix=False)
        final_fidelity = calc_fidelity(ideal_state, final_amps, use_abs=False)
    
    improvement = (final_fidelity - raw_fidelity) * 100

    if verbose:
        print(f"\nFinal Fidelity: {final_fidelity:.6f}")
        print(f"Improvement: {improvement:.2f}%")

    return raw_fidelity, abs_fidelity, final_fidelity, improvement, optimal_phi


if __name__ == "__main__":
    main(verbose=True)