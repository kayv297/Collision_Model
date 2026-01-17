"""
Quantum noise channel implementations using Kraus operators.
All channels work on density matrices.
"""
import numpy as np


def dephasing_channel(rho, p):
    """
    Phase damping: lose phase information without energy loss.
    
    Kraus operators:
    K0 = sqrt(1-p) * I
    K1 = sqrt(p) * Z
    
    Args:
        rho: 2x2 single-qubit density matrix
        p: Dephasing probability per interaction
    
    Returns:
        Evolved 2x2 density matrix
    """
    if p == 0:
        return rho
    
    K0 = np.sqrt(1 - p) * np.eye(2, dtype=complex)
    K1 = np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=complex)
    
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def amplitude_damping_channel(rho, gamma):
    """
    Energy relaxation: |1⟩ → |0⟩ decay.
    
    Kraus operators:
    K0 = [[1, 0], [0, sqrt(1-gamma)]]
    K1 = [[0, sqrt(gamma)], [0, 0]]
    
    Args:
        rho: 2x2 single-qubit density matrix
        gamma: Decay rate per interaction
    
    Returns:
        Evolved 2x2 density matrix
    """
    if gamma == 0:
        return rho
    
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def depolarizing_channel(rho, p):
    """
    Depolarizing channel: applies random Pauli errors.
    
    Channel: ρ → (1-p)ρ + p(I/2)
    
    Implemented using Kraus operators:
    K0 = sqrt(1-p) * I
    K1 = sqrt(p/3) * X
    K2 = sqrt(p/3) * Y
    K3 = sqrt(p/3) * Z
    
    Args:
        rho: 2x2 single-qubit density matrix
        p: Depolarizing probability (0 ≤ p ≤ 1)
           p=0: no noise
           p=1: complete depolarization to I/2
    
    Returns:
        Evolved 2x2 density matrix
    """
    if p == 0:
        return rho
    
    if not (0 <= p <= 1):
        raise ValueError(f"Depolarizing probability must be in [0,1], got {p}")
    
    # Define Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Kraus operators
    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p / 3) * X
    K2 = np.sqrt(p / 3) * Y
    K3 = np.sqrt(p / 3) * Z
    
    # Apply: ρ' = Σ_i K_i ρ K_i†
    rho_out = (K0 @ rho @ K0.conj().T + 
               K1 @ rho @ K1.conj().T + 
               K2 @ rho @ K2.conj().T + 
               K3 @ rho @ K3.conj().T)
    
    return rho_out


def apply_single_qubit_channel(rho, channel_func, qubit_idx, n_qubits, **channel_params):
    """
    Apply a single-qubit quantum channel to a specific qubit in full system.
    
    This properly handles multi-qubit density matrices by:
    1. Reshaping to isolate target qubit
    2. Applying channel to ALL blocks (including off-diagonal coherences)
    3. Reconstructing full density matrix
    
    Args:
        rho: Full system density matrix (2^n × 2^n)
        channel_func: Single-qubit channel function (e.g., dephasing_channel)
        qubit_idx: Index of target qubit (0-indexed)
        n_qubits: Total number of qubits in system
        **channel_params: Parameters for the channel (e.g., p=0.1)
    
    Returns:
        Evolved full system density matrix
    """
    dim = 2**n_qubits
    
    # Split system: [before target] ⊗ [target] ⊗ [after target]
    n_before = qubit_idx
    n_after = n_qubits - qubit_idx - 1
    
    dim_before = 2**n_before if n_before > 0 else 1
    dim_target = 2
    dim_after = 2**n_after if n_after > 0 else 1
    
    # Reshape to (dim_before, dim_target, dim_after, dim_before, dim_target, dim_after)
    rho_reshaped = rho.reshape(dim_before, dim_target, dim_after, 
                                dim_before, dim_target, dim_after)
    
    # Initialize output
    rho_out = np.zeros_like(rho, dtype=complex)
    rho_out_reshaped = rho_out.reshape(dim_before, dim_target, dim_after,
                                        dim_before, dim_target, dim_after)
    
    # Apply channel to ALL blocks (including off-diagonal coherences)
    for i_before in range(dim_before):
        for i_after in range(dim_after):
            for j_before in range(dim_before):
                for j_after in range(dim_after):
                    # Extract 2x2 block for target qubit
                    # This includes both diagonal and off-diagonal blocks
                    rho_target = rho_reshaped[i_before, :, i_after, j_before, :, j_after]
                    
                    # Apply channel
                    rho_target_evolved = channel_func(rho_target, **channel_params)
                    
                    # Write back
                    rho_out_reshaped[i_before, :, i_after, j_before, :, j_after] = rho_target_evolved
    
    return rho_out.reshape(dim, dim)