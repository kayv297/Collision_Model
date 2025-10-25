"""
Symbolic Partial SWAP Gate for Collision Models

Implementation of the partial SWAP gate from Çakmak et al. (2019):
    U(γ) = cos(γ)·I + i·sin(γ)·SWAP

Matrix form:
    U(γ) = [e^(iγ)    0           0        0     ]
           [0         cos(γ)      i·sin(γ)  0     ]
           [0         i·sin(γ)    cos(γ)    0     ]
           [0         0           0         e^(iγ)]

This gate is useful for collision models in open quantum systems.

Realize the symbolic partial SWAP gate using:
- Two-qubit gates: RXX, RYY, RZZ
- Single-qubit gates: P, CP


Author: Derived through systematic gate analysis
Date: October 2025
Status: Verified to machine precision (< 10^-15 error)
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator


def partial_swap(gamma):
    """
    Create a symbolic partial SWAP gate circuit.
    
    Parameters:
        gamma (float or Parameter): Rotation angle parameter
            - γ = 0: Identity gate
            - γ = π/2: Full iSWAP gate
            - γ ∈ (0, π/2): Partial swap with tunable strength
    
    Returns:
        QuantumCircuit: 2-qubit circuit implementing U(γ)
        
    Circuit structure:
        1. RXX(-γ) + RYY(-γ): Creates cos/sin structure in middle block
        2. RZZ(-2γ): Adds parity-dependent phases
        3. P(γ) gates: Phase corrections for |01⟩ and |10⟩
        4. CP(-2γ): Final phase correction for |11⟩
        
    Example:
        >>> from qiskit.circuit import Parameter
        >>> gamma = Parameter('γ')
        >>> qc = partial_swap(gamma)
        >>> print(qc)
        
        Or with specific value:
        >>> qc = partial_swap(0.05)
        >>> from qiskit.quantum_info import Operator
        >>> print(Operator(qc))
    """
    qc = QuantumCircuit(2, name=f'PartialSWAP({gamma})')
    
    # Step 1: Create middle block structure
    # RXX(-γ) + RYY(-γ) produces [cos(γ), i·sin(γ); i·sin(γ), cos(γ)] 
    # in the {|01⟩, |10⟩} subspace
    qc.rxx(-gamma, 0, 1)
    qc.ryy(-gamma, 0, 1)
    
    # Step 2: Add phase corrections
    # RZZ adds e^(iγ) to even parity states (|00⟩, |11⟩)
    # and e^(-iγ) to odd parity states (|01⟩, |10⟩)
    qc.rzz(-2*gamma, 0, 1)
    
    # Step 3: Correct middle block phases using single-qubit gates
    # P(γ) adds e^(iγ) to |1⟩ state of that qubit
    qc.p(gamma, 0)
    qc.p(gamma, 1)
    
    # Step 4: Remove excess phase from |11⟩
    # After step 3, |11⟩ has e^(3iγ) but we want e^(iγ)
    qc.cp(-2*gamma, 0, 1)
    
    return qc


def verify_partial_swap(gamma_value, verbose=True):
    """
    Verify that the partial_swap circuit produces the correct unitary.
    
    Parameters:
        gamma_value (float): Specific value of γ to test
        verbose (bool): Whether to print detailed results
        
    Returns:
        tuple: (success, error) where success is bool and error is float
        
    Example:
        >>> success, error = verify_partial_swap(0.05)
        >>> print(f"Error: {error:.2e}")
    """
    # Build circuit
    qc = partial_swap(gamma_value)
    U_circuit = Operator(qc).data
    
    # Build target matrix
    cos_g = np.cos(gamma_value)
    sin_g = np.sin(gamma_value)
    exp_g = np.exp(1j * gamma_value)
    
    U_target = np.array([
        [exp_g, 0, 0, 0],
        [0, cos_g, 1j*sin_g, 0],
        [0, 1j*sin_g, cos_g, 0],
        [0, 0, 0, exp_g]
    ], dtype=complex)
    
    # Compare (account for global phase)
    product = U_circuit.conj().T @ U_target
    global_phase = product[0, 0]
    normalized = product / global_phase
    error = np.linalg.norm(normalized - np.eye(4))
    
    success = error < 1e-10
    
    if verbose:
        print(f"Verification for γ = {gamma_value:.4f}:")
        print(f"  Error: {error:.2e}")
        print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")
        
        if success:
            print(f"  Global phase: {np.angle(global_phase):.6f} rad")
            print(f"\n  Circuit:")
            print(qc)
    
    return success, error


def get_target_matrix(gamma):
    """
    Get the target partial SWAP matrix for a given γ.
    
    Parameters:
        gamma (float): Rotation angle
        
    Returns:
        np.ndarray: 4×4 complex matrix representing U(γ)
    """
    cos_g = np.cos(gamma)
    sin_g = np.sin(gamma)
    exp_g = np.exp(1j * gamma)
    
    return np.array([
        [exp_g, 0, 0, 0],
        [0, cos_g, 1j*sin_g, 0],
        [0, 1j*sin_g, cos_g, 0],
        [0, 0, 0, exp_g]
    ], dtype=complex)


def example_usage():
    """Demonstrate usage of the partial SWAP gate."""
    print("="*70)
    print("Symbolic Partial SWAP Gate - Example Usage")
    print("="*70)
    
    # Example 1: Symbolic parameter
    print("\n1. Creating circuit with symbolic parameter:")
    gamma_param = Parameter('γ')
    qc_symbolic = partial_swap(gamma_param)
    print(qc_symbolic)
    
    # Example 2: Specific value
    print("\n2. Creating circuit with γ = 0.05:")
    qc_numeric = partial_swap(0.05)
    print(qc_numeric)
    
    # Example 3: Verification
    print("\n3. Verification:")
    verify_partial_swap(0.05, verbose=True)
    
    # Example 4: Test multiple values
    print("\n4. Testing multiple γ values:")
    test_values = [0.01, 0.1, 0.5, np.pi/4, np.pi/2]
    
    for gamma in test_values:
        success, error = verify_partial_swap(gamma, verbose=False)
        status = "✓" if success else "✗"
        print(f"  γ = {gamma:6.4f}: {status} (error = {error:.2e})")
    
    # Example 5: Matrix representation
    print("\n5. Matrix representation for γ = π/4:")
    U = get_target_matrix(np.pi/4)
    print(U)
    
    print("\n" + "="*70)


if __name__ == "__main__":
    example_usage()
