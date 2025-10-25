"""
Quantum gate definitions and operations for collision model simulation.
"""
import numpy as np

# Basic states
zero = np.array([1, 0], dtype=complex)
one = np.array([0, 1], dtype=complex)

# SWAP gate (2-qubit)
SWAP = np.zeros((4, 4), dtype=complex)
SWAP[0, 0] = 1
SWAP[1, 2] = 1
SWAP[2, 1] = 1
SWAP[3, 3] = 1


def partial_swap(gamma):
    """
    Create partial SWAP gate with angle gamma.
    
    Args:
        gamma: Partial-swap angle
        
    Returns:
        4x4 partial SWAP unitary matrix
    """
    return np.cos(gamma) * np.eye(4, dtype=complex) + 1j * np.sin(gamma) * SWAP


def apply_two_qubit(state, U, i, j, n_qbits):
    """
    Apply two-qubit unitary U to qubits i and j of an n_qbits-qubit statevector.
    
    Args:
        state: Quantum state vector of size 2^n_qbits
        U: 4x4 unitary matrix to apply
        i, j: Qubit indices to apply the gate to
        n_qbits: Total number of qubits in the system
        
    Returns:
        Updated state vector after applying U to qubits i and j
    """
    # Reshape state to n-dimensional tensor: [2, 2, ..., 2] (n_qbits times)
    tensor = state.reshape([2] * n_qbits)
    
    # Create permutation that moves qubits i and j to the front
    perm = [i, j] + [k for k in range(n_qbits) if k not in (i, j)]
    
    # Apply permutation
    tensor_permuted = np.transpose(tensor, perm)
    
    # Reshape to group the first two dimensions (qubits i,j) together
    tensor_reshaped = tensor_permuted.reshape(4, -1)
    
    # Apply the 2-qubit unitary
    new_tensor = U @ tensor_reshaped
    
    # Reshape back to separate the two qubits
    new_tensor_shaped = new_tensor.reshape([2, 2] + [2] * (n_qbits - 2))
    
    # Create inverse permutation to restore original qubit order
    inv_perm = np.argsort(perm)
    
    # Apply inverse permutation
    result_tensor = np.transpose(new_tensor_shaped, inv_perm)
    
    # Flatten back to state vector
    return result_tensor.reshape(2**n_qbits)
