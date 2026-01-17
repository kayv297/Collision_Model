"""
Quantum state analysis: initialization, post-selection, and coefficient extraction.
"""
import numpy as np
from itertools import combinations
from quantum_gates import zero, one
from typing import List


def gen_init_state(ancillas_pos, n_registers):
    """
    Generate initial state |A1A2...Am⟩|r1r2...rn⟩|s1s2...sk⟩.

    Args:
        ancillas_pos: List of positions where ancillas are in state |1⟩
        n_registers: List with number of qubits in each register

    Returns:
        Initial state vector as numpy array
    """
    num_qbits_A = len(ancillas_pos)
    num_qbits_r = n_registers[0]
    num_qbits_s = n_registers[1]

    # Generate A_reg with all |1⟩
    A_reg = one
    for i in range(1, num_qbits_A):
        A_reg = np.kron(A_reg, one)

    # Generate r_reg with all |0⟩
    r_reg = zero
    for i in range(1, num_qbits_r):
        r_reg = np.kron(r_reg, zero)

    # Generate s_reg with all |0⟩
    s_reg = zero
    for i in range(1, num_qbits_s):
        s_reg = np.kron(s_reg, zero)

    return np.kron(A_reg, np.kron(r_reg, s_reg))


def postselection_step(cur_state, A_pos, n_registers):
    """
    Perform projective measurement onto |0⟩⊗m state for all ancillas.
    Implements equation (7) from paper.

    Args:
        cur_state: Current quantum state vector
        A_pos: List of ancilla positions
        n_registers: List with number of qubits in each register

    Returns:
        rho_registers: Density matrix of registers after post-selection
        prob: Post-selection probability
    """
    # Convert to density matrix
    rho = np.outer(cur_state, np.conj(cur_state))

    num_ancillas = len(A_pos)
    total_register_qubits = sum(n_registers)

    # Create projector |0⟩⊗m⟨0|⊗m for all ancillas
    proj_0_single = np.array([[1, 0], [0, 0]], dtype=complex)
    proj_0_ancillas = proj_0_single
    for i in range(1, num_ancillas):
        proj_0_ancillas = np.kron(proj_0_ancillas, proj_0_single)

    # Extend to full system
    I_registers = np.eye(2**total_register_qubits, dtype=complex)
    full_projector = np.kron(proj_0_ancillas, I_registers)

    # Apply projection
    projected_rho = full_projector @ rho @ full_projector

    # Calculate probability
    prob = np.trace(projected_rho).real

    if prob > 1e-12:
        # Normalize
        normalized_projected_rho = projected_rho / prob

        # Partial trace over ancillas
        dim_ancillas = 2**num_ancillas
        dim_registers = 2**total_register_qubits

        reshaped = normalized_projected_rho.reshape(
            dim_ancillas, dim_registers, dim_ancillas, dim_registers)

        # Trace over ancilla indices
        rho_registers = np.trace(reshaped, axis1=0, axis2=2)

        return rho_registers, prob
    else:
        return np.zeros((2**total_register_qubits, 2**total_register_qubits), dtype=complex), 0.0


def extract_w_coefficients(state, A_pos, n_registers, project_shuttle=True):
    """
    Extract coefficients b_i from state after projecting/tracing ancillas.

    The generalized state is: |ψ⟩ = Σ b_i |basis_i⟩ 
    where each basis state has len(A_pos) excitations across register qubits.

    Args:
        state: Quantum state vector
        A_pos: List of ancilla positions
        n_registers: List with number of qubits in each register
        project_shuttle: Whether to project ancillas onto |0⟩

    Returns:
        Array of coefficients for W-state basis
    """
    num_ancillas = len(A_pos)
    total_register_qubits = sum(n_registers)
    total_qubits = num_ancillas + total_register_qubits

    if len(state) == 2**total_qubits:  # Full state
        if project_shuttle:
            # Project ancillas onto |0⟩
            reg_state = np.zeros(2**total_register_qubits, dtype=complex)

            for i in range(len(state)):
                # Check if all ancillas are in |0⟩
                ancilla_bits = (i >> total_register_qubits) & (
                    (1 << num_ancillas) - 1)
                if ancilla_bits == 0:
                    i_red = i & ((1 << total_register_qubits) - 1)
                    reg_state[i_red] = state[i]

            # Normalize
            norm = np.linalg.norm(reg_state)
            if norm > 0:
                reg_state /= norm
    else:
        reg_state = state

    # Extract coefficients for states with num_ancillas excitations
    coeffs = []
    for excitation_positions in combinations(range(total_register_qubits), num_ancillas):
        basis_idx = 0
        for pos in excitation_positions:
            basis_idx |= (1 << (total_register_qubits - 1 - pos))
        coeffs.append(reg_state[basis_idx])

    return np.array(coeffs)


def get_amps(amps_evol: np.ndarray, round_num: int) -> List[float]:
    '''
    Get the non-zero amplitudes of all basis states at a specific round number.

    Args:
        amps_evol: Array of shape (num_rounds, 2^total_qubits) containing amplitude evolution
        round_num: The round number to extract amplitudes for

    Returns:
        Array of amplitudes at the specified round number
    '''
    if round_num < 0 or round_num >= amps_evol.shape[0]:
        raise ValueError("round_num must be between 0 and num_rounds-1")

    amps = amps_evol[round_num]
    non_zero_amps = [amp for amp in amps if abs(amp) > 1e-6]

    return non_zero_amps


def calc_fidelity(ideal_state: np.ndarray, actual_state: np.ndarray, use_abs: bool = True) -> float:
    '''
    Calculate fidelity between ideal and actual quantum states.

    Args:
        ideal_state: Ideal quantum state vector
        actual_state: Actual quantum state vector
    Returns:
        Fidelity value between 0 and 1
    '''

    # ignore the local phase (complex part)
    if use_abs:
        return np.abs(np.vdot(ideal_state, np.abs(actual_state))) ** 2

    return np.abs(np.vdot(ideal_state, actual_state)) ** 2

def calc_uhlmann_fidelity(rho: np.ndarray, psi: np.ndarray, use_abs: bool = True) -> float:
    '''
    Calculate Uhlmann fidelity between a density matrix and a pure state.
    F = ⟨ψ|ρ|ψ⟩
    
    Args:
        rho: Density matrix (can be mixed state)
        psi: Pure state vector
        
    Returns:
        Fidelity value between 0 and 1
    '''
    # For pure state |ψ⟩, fidelity is F = ⟨ψ|ρ|ψ⟩
    if use_abs:
        return np.vdot(psi, np.abs(rho) @ psi)
    return np.vdot(psi, rho @ psi)