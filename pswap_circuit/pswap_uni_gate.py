"""
Partial SWAP Gate Using Universal Gates (CX, RX, RY, RZ)

Decomposition of the partial SWAP gate into fundamental gates that can be
visualized in Quirk and implemented on universal quantum hardware.

This module converts:
    RXX, RYY, RZZ, P, CP gates → CX, RX, RY, RZ gates
    
The realization has 8 CNOTs and ~21 total gates.

Author: Derived from symbolic partial_swap_gate.py
Date: October 2025
Status: Verified to machine precision
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator


def rxx_to_universal(qc, theta, q0, q1):
    """
    Decompose RXX(θ) into universal gates.

    RXX(θ) = exp(-i·θ/2·X⊗X)

    Decomposition:
        RXX(θ) = H₀·H₁·RZZ(θ)·H₀·H₁
               = H₀·H₁·(CX·RZ(θ)·CX)·H₀·H₁
    """
    # Change basis: X → Z using Hadamard
    qc.h(q0)
    qc.h(q1)

    # Apply RZZ(θ) = CX·RZ(θ)·CX
    qc.cx(q0, q1)
    qc.rz(theta, q1)
    qc.cx(q0, q1)

    # Change back
    qc.h(q0)
    qc.h(q1)


def ryy_to_universal(qc, theta, q0, q1):
    """
    Decompose RYY(θ) into universal gates.

    RYY(θ) = exp(-i·θ/2·Y⊗Y)

    Decomposition:
        RYY(θ) = (RX(π/2)₀·RX(π/2)₁)·RZZ(θ)·(RX(-π/2)₀·RX(-π/2)₁)
               = (RX(π/2)₀·RX(π/2)₁)·(CX·RZ(θ)·CX)·(RX(-π/2)₀·RX(-π/2)₁)
    """
    # Change basis: Y → Z using RX(π/2)
    qc.rx(np.pi/2, q0)
    qc.rx(np.pi/2, q1)

    # Apply RZZ(θ) = CX·RZ(θ)·CX
    qc.cx(q0, q1)
    qc.rz(theta, q1)
    qc.cx(q0, q1)

    # Change back
    qc.rx(-np.pi/2, q0)
    qc.rx(-np.pi/2, q1)


def rzz_to_universal(qc, theta, q0, q1):
    """
    Decompose RZZ(θ) into universal gates.

    RZZ(θ) = exp(-i·θ/2·Z⊗Z)

    Decomposition:
        RZZ(θ) = CX₀₁·RZ(θ)₁·CX₀₁
    """
    qc.cx(q0, q1)
    qc.rz(theta, q1)
    qc.cx(q0, q1)


def p_to_universal(qc, theta, q):
    """
    Decompose P(θ) phase gate into universal gates.

    P(θ) = diag(1, e^(iθ)) = RZ(θ)

    Note: P(θ) is exactly RZ(θ) in Qiskit convention
    """
    qc.rz(theta, q)


def cp_to_universal(qc, theta, q0, q1):
    """
    Decompose CP(θ) controlled-phase gate into universal gates.

    CP(θ) = diag(1, 1, 1, e^(iθ))

    Decomposition:
        CP(θ) = RZ(θ/2)₀·RZ(θ/2)₁·CX₀₁·RZ(-θ/2)₁·CX₀₁
    """
    qc.rz(theta/2, q0)
    qc.rz(theta/2, q1)
    qc.cx(q0, q1)
    qc.rz(-theta/2, q1)
    qc.cx(q0, q1)


def partial_swap_universal(gamma):
    """
    Create partial SWAP gate using only universal gates {CX, RX, RY, RZ}.

    This decomposition is compatible with Quirk circuit simulator and
    universal quantum hardware.

    Parameters:
        gamma (float or Parameter): Rotation angle parameter
            - γ = 0: Identity gate
            - γ = π/2: Full iSWAP gate
            - γ ∈ (0, π/2): Partial swap with tunable strength

    Returns:
        QuantumCircuit: 2-qubit circuit using only {CX, RX, RY, RZ}

    Gate count:
        - Original: 6 gates (RXX, RYY, RZZ, P, P, CP)
        - Universal: ~21 gates (mostly CX and single-qubit rotations)

    Example:
        >>> qc = partial_swap_universal(0.05)
        >>> print(qc.count_ops())
    """
    qc = QuantumCircuit(2, name=f'PartialSWAP_Universal({gamma})')

    # Step 1: RXX(-γ)
    rxx_to_universal(qc, -gamma, 0, 1)

    # Step 2: RYY(-γ)
    ryy_to_universal(qc, -gamma, 0, 1)

    # Step 3: RZZ(-2γ)
    rzz_to_universal(qc, -2*gamma, 0, 1)

    # Step 4: P(γ) on qubit 0
    p_to_universal(qc, gamma, 0)

    # Step 5: P(γ) on qubit 1
    p_to_universal(qc, gamma, 1)

    # Step 6: CP(-2γ)
    cp_to_universal(qc, -2*gamma, 0, 1)

    return qc


def verify_universal_decomposition(gamma_value, verbose=True):
    """
    Verify that the universal decomposition produces the correct unitary.

    Parameters:
        gamma_value (float): Specific value of γ to test
        verbose (bool): Whether to print detailed results

    Returns:
        tuple: (success, error) where success is bool and error is float
    """
    # Build circuit
    qc = partial_swap_universal(gamma_value)
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
        print(
            f"Universal Decomposition Verification for γ = {gamma_value:.4f}:")
        print(f"  Error: {error:.2e}")
        print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")

        if success:
            print(f"  Global phase: {np.angle(global_phase):.6f} rad")

        # Print gate statistics
        gate_counts = qc.count_ops()
        print(f"\n  Gate counts:")
        for gate, count in sorted(gate_counts.items()):
            print(f"    {gate}: {count}")
        print(f"  Total gates: {sum(gate_counts.values())}")
        print(f"  Circuit depth: {qc.depth()}")

    return success, error


def export_to_quirk_format(gamma_value):
    """
    Generate information for manually creating the circuit in Quirk.

    Quirk URL: https://algassert.com/quirk

    Parameters:
        gamma_value (float): Specific value of γ

    Returns:
        str: Instructions for Quirk circuit construction
    """
    qc = partial_swap_universal(gamma_value)

    output = []
    output.append(f"Quirk Circuit for Partial SWAP (γ = {gamma_value:.4f})")
    output.append("="*70)
    output.append("\nGate sequence (apply left to right in Quirk):\n")

    for idx, (gate, qubits, params) in enumerate(qc.data):
        gate_name = gate.name
        qubit_indices = [qc.qubits.index(q) for q in qubits]

        if gate_name == 'cx':
            output.append(
                f"{idx+1}. CNOT: control=q{qubit_indices[0]}, target=q{qubit_indices[1]}")
        elif gate_name == 'h':
            output.append(f"{idx+1}. H on q{qubit_indices[0]}")
        elif gate_name == 'rx':
            angle = params[0]
            if isinstance(angle, Parameter):
                output.append(f"{idx+1}. Rx({angle}) on q{qubit_indices[0]}")
            else:
                degrees = np.degrees(angle)
                output.append(
                    f"{idx+1}. Rx({angle:.4f} rad = {degrees:.2f}°) on q{qubit_indices[0]}")
        elif gate_name == 'ry':
            angle = params[0]
            degrees = np.degrees(angle)
            output.append(
                f"{idx+1}. Ry({angle:.4f} rad = {degrees:.2f}°) on q{qubit_indices[0]}")
        elif gate_name == 'rz':
            angle = params[0]
            if isinstance(angle, Parameter):
                output.append(f"{idx+1}. Rz({angle}) on q{qubit_indices[0]}")
            else:
                degrees = np.degrees(angle)
                output.append(
                    f"{idx+1}. Rz({angle:.4f} rad = {degrees:.2f}°) on q{qubit_indices[0]}")

    output.append(f"\nTotal gates: {len(qc.data)}")
    output.append(f"CNOT count: {qc.count_ops().get('cx', 0)}")
    output.append(
        "\nNote: In Quirk, use the rotation gates with the angles shown above.")
    output.append("      Quirk uses the same convention: Rx(θ), Ry(θ), Rz(θ)")

    return "\n".join(output)


def compare_decompositions(gamma_value):
    """
    Compare the original and universal decompositions.

    Parameters:
        gamma_value (float): Test value
    """
    from partial_swap.original_gate import partial_swap

    print("="*70)
    print(f"Decomposition Comparison (γ = {gamma_value:.4f})")
    print("="*70)

    # Original (high-level gates)
    qc_original = partial_swap(gamma_value)
    print("\nOriginal (high-level gates):")
    print(f"  Gates: {qc_original.count_ops()}")
    print(f"  Total: {len(qc_original.data)}")
    print(f"  Depth: {qc_original.depth()}")

    # Universal decomposition
    qc_universal = partial_swap_universal(gamma_value)
    print("\nUniversal decomposition:")
    print(f"  Gates: {qc_universal.count_ops()}")
    print(f"  Total: {len(qc_universal.data)}")
    print(f"  Depth: {qc_universal.depth()}")

    # Verify both produce same unitary (up to global phase)
    U_original = Operator(qc_original).data
    U_universal = Operator(qc_universal).data

    # Account for global phase difference
    product = U_universal.conj().T @ U_original
    global_phase = product[0, 0]
    U_universal_normalized = U_universal * global_phase / np.abs(global_phase)

    diff = np.linalg.norm(U_original - U_universal_normalized)
    phase_diff = np.angle(global_phase)

    print(
        f"\nDirect matrix difference: {np.linalg.norm(U_original - U_universal):.2e}")
    print(
        f"Global phase difference: {phase_diff:.6f} rad ({np.degrees(phase_diff):.2f}°)")
    print(f"Difference (accounting for global phase): {diff:.2e}")
    print(f"Match (up to global phase): {'✓ YES' if diff < 1e-10 else '✗ NO'}")

    print("="*70)


def example_usage():
    """Demonstrate usage of the universal decomposition."""
    print("="*70)
    print("Partial SWAP Gate - Universal Decomposition")
    print("="*70)

    # Example 1: Create circuit
    print("\n1. Creating universal decomposition for γ = 0.05:")
    gamma = 0.05
    qc = partial_swap_universal(gamma)
    print(f"\nCircuit depth: {qc.depth()}")
    print(f"Gate counts: {qc.count_ops()}")
    print(f"\nCircuit:")
    print(qc)

    # Example 2: Verify
    print("\n" + "="*70)
    print("2. Verification:")
    verify_universal_decomposition(gamma, verbose=True)

    # Example 3: Test multiple values
    print("\n" + "="*70)
    print("3. Testing multiple γ values:")
    test_values = [0.01, 0.05, 0.1, 0.5, np.pi/4]

    for g in test_values:
        success, error = verify_universal_decomposition(g, verbose=False)
        status = "✓" if success else "✗"
        print(f"  γ = {g:6.4f}: {status} (error = {error:.2e})")

    # Example 4: Quirk export
    print("\n" + "="*70)
    # print("4. Export for Quirk:")
    # quirk_instructions = export_to_quirk_format(0.05)
    # print(quirk_instructions)

    # Example 5: Comparison
    print("\n" + "="*70)
    print("5. Comparison with original:")
    compare_decompositions(0.05)


if __name__ == "__main__":
    example_usage()
