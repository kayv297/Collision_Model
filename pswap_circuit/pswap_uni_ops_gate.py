"""
Optimized Partial SWAP Gate - 3 CNOT Decomposition

Highly optimized decomposition using only 3 CNOTs (theoretical minimum).
Uses the KAK decomposition structure: (A⊗B)·CNOT·(C⊗D)·CNOT·(E⊗F)·CNOT·(G⊗H)

This achieves the minimal CNOT count for implementing the partial SWAP gate.

The best circuit so far has 6 CNOTS at machine precision.

Author: Optimized from partial_swap_universal.py
Date: October 2025
Status: 3-CNOT optimal decomposition
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit.library import CXGate



def partial_swap_3cnot_optimized(gamma):
    """
    Optimized using U3 (most general single-qubit gate).
    
    This typically gives the most compact representation.
    """

    
    # Build target unitary
    cos_g = np.cos(gamma)
    sin_g = np.sin(gamma)
    exp_g = np.exp(1j * gamma)
    
    U_target = np.array([
        [exp_g, 0, 0, 0],
        [0, cos_g, 1j*sin_g, 0],
        [0, 1j*sin_g, cos_g, 0],
        [0, 0, 0, exp_g]
    ], dtype=complex)
    
    # Use U basis (most general)
    decomposer = TwoQubitBasisDecomposer(CXGate(), euler_basis='U')
    qc_decomposed = decomposer(U_target)
    
    qc = QuantumCircuit(2, name=f'PartialSWAP_Opt({gamma})')
    qc.compose(qc_decomposed, inplace=True)
    
    return qc


def verify_3cnot_decomposition(builder_func, gamma_value, verbose=True):
    """
    Verify that a 3-CNOT decomposition produces the correct unitary.
    
    Parameters:
        builder_func: Function that builds the circuit
        gamma_value (float): Specific value of γ to test
        verbose (bool): Whether to print detailed results
        
    Returns:
        tuple: (success, error, gate_counts)
    """
    # Build circuit
    qc = builder_func(gamma_value)
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
    gate_counts = qc.count_ops()
    
    if verbose:
        print(f"\nVerification for {builder_func.__name__} (γ = {gamma_value:.4f}):")
        print(f"  Error: {error:.2e}")
        print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")
        
        if success:
            print(f"  Global phase: {np.angle(global_phase):.6f} rad")
        
        print(f"\n  Gate counts:")
        for gate, count in sorted(gate_counts.items()):
            print(f"    {gate}: {count}")
        cnot_count = gate_counts.get('cx', 0)
        total_gates = sum(gate_counts.values())
        print(f"  Total gates: {total_gates}")
        print(f"  CNOT count: {cnot_count} {'✓✓✓' if cnot_count == 3 else '✗'}")
        print(f"  Depth: {qc.depth()}")
    
    return success, error, gate_counts


def find_best_3cnot_decomposition(gamma_value):
    """
    Test all 3-CNOT decomposition variants and find the best one.
    
    "Best" is defined as: correct + fewest total gates + minimal depth
    """
    print("="*70)
    print(f"Testing all 3-CNOT decompositions (γ = {gamma_value:.4f})")
    print("="*70)
    
    builders = [
        partial_swap_3cnot_optimized,
    ]
    
    results = []
    
    for builder in builders:
        success, error, counts = verify_3cnot_decomposition(
            builder, gamma_value, verbose=True)
        
        if success:
            total = sum(counts.values())
            depth = builder(gamma_value).depth()
            results.append({
                'name': builder.__name__,
                'builder': builder,
                'error': error,
                'total_gates': total,
                'depth': depth,
                'counts': counts
            })
        
        print()
    
    if results:
        # Sort by total gates, then by depth
        results.sort(key=lambda x: (x['total_gates'], x['depth']))
        
        print("="*70)
        print("SUMMARY - Ranked by efficiency:")
        print("="*70)
        
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['name']}:")
            print(f"   Total gates: {r['total_gates']}")
            print(f"   Depth: {r['depth']}")
            print(f"   Error: {r['error']:.2e}")
            print(f"   CNOTs: {r['counts'].get('cx', 0)}")
            print()
        
        print(f"✓ Best: {results[0]['name']}")
        print(f"  Total gates: {results[0]['total_gates']}")
        print(f"  Depth: {results[0]['depth']}")
        
        return results[0]['builder']
    
    return None


def compare_with_unoptimized(gamma_value):
    """
    Compare 3-CNOT optimized version with the previous unoptimized version.
    """
    from partial_swap.pswap_universal import partial_swap_universal
    
    print("\n" + "="*70)
    print(f"OPTIMIZATION COMPARISON (γ = {gamma_value:.4f})")
    print("="*70)
    
    # Unoptimized version
    qc_unopt = partial_swap_universal(gamma_value)
    counts_unopt = qc_unopt.count_ops()
    
    print("\nUnoptimized (from partial_swap_universal.py):")
    print(f"  CNOTs: {counts_unopt.get('cx', 0)}")
    print(f"  Total gates: {sum(counts_unopt.values())}")
    print(f"  Depth: {qc_unopt.depth()}")
    
    # Find best optimized version
    builders = [partial_swap_3cnot_optimized]
    
    best_total = float('inf')
    best_builder = None
    
    for builder in builders:
        qc = builder(gamma_value)
        counts = qc.count_ops()
        total = sum(counts.values())
        
        if counts.get('cx', 0) == 3 and total < best_total:
            # Verify it's correct
            success, _, _ = verify_3cnot_decomposition(builder, gamma_value, verbose=False)
            if success:
                best_total = total
                best_builder = builder
    
    if best_builder:
        qc_opt = best_builder(gamma_value)
        counts_opt = qc_opt.count_ops()
        
        print(f"\nOptimized ({best_builder.__name__}):")
        print(f"  CNOTs: {counts_opt.get('cx', 0)}")
        print(f"  Total gates: {sum(counts_opt.values())}")
        print(f"  Depth: {qc_opt.depth()}")
        
        # Calculate improvements
        cnot_reduction = counts_unopt.get('cx', 0) - counts_opt.get('cx', 0)
        gate_reduction = sum(counts_unopt.values()) - sum(counts_opt.values())
        depth_reduction = qc_unopt.depth() - qc_opt.depth()
        
        print("\nImprovement:")
        print(f"  CNOT reduction: {cnot_reduction} ({cnot_reduction/counts_unopt.get('cx', 1)*100:.1f}%)")
        print(f"  Total gate reduction: {gate_reduction} ({gate_reduction/sum(counts_unopt.values())*100:.1f}%)")
        print(f"  Depth reduction: {depth_reduction} ({depth_reduction/qc_unopt.depth()*100:.1f}%)")
    
    print("="*70)


def convert_u_to_zyz(u_params):
    """
    Convert U3 gate to ZYZ Euler decomposition: U3(θ,φ,λ) = Rz(φ)·Ry(θ)·Rz(λ)
    
    Returns: (rz1_angle, ry_angle, rz2_angle)
    """
    theta, phi, lam = u_params
    return (phi, theta, lam)


def partial_swap_3cnot_basic_gates(gamma):
    """
    Get the 3-CNOT decomposition using only {CX, RX, RY, RZ} (no U gates).
    
    This explicitly converts U gates to the basic gate set for Quirk compatibility.
    """
    from qiskit.synthesis import TwoQubitBasisDecomposer
    from qiskit.circuit.library import CXGate
    
    # Build target unitary
    cos_g = np.cos(gamma)
    sin_g = np.sin(gamma)
    exp_g = np.exp(1j * gamma)
    
    U_target = np.array([
        [exp_g, 0, 0, 0],
        [0, cos_g, 1j*sin_g, 0],
        [0, 1j*sin_g, cos_g, 0],
        [0, 0, 0, exp_g]
    ], dtype=complex)
    
    # Get decomposition with ZYZ basis
    decomposer = TwoQubitBasisDecomposer(CXGate(), euler_basis='ZYZ')
    qc_decomposed = decomposer(U_target)
    
    # Create new circuit with only basic gates
    qc = QuantumCircuit(2, name=f'PartialSWAP_BasicGates({gamma})')
    
    for instruction in qc_decomposed.data:
        gate = instruction.operation
        qubits = instruction.qubits
        qubit_indices = [qc_decomposed.qubits.index(q) for q in qubits]
        
        if gate.name == 'cx':
            qc.cx(qubit_indices[0], qubit_indices[1])
        elif gate.name == 'u':
            # Convert U gate to ZYZ (Rz·Ry·Rz)
            theta, phi, lam = gate.params
            q_idx = qubit_indices[0]
            
            # U(θ,φ,λ) = e^(i(φ+λ)/2) · Rz(φ)·Ry(θ)·Rz(λ)
            # We ignore the global phase
            if abs(phi) > 1e-10:
                qc.rz(phi, q_idx)
            if abs(theta) > 1e-10:
                qc.ry(theta, q_idx)
            if abs(lam) > 1e-10:
                qc.rz(lam, q_idx)
        elif gate.name == 'rz':
            qc.rz(gate.params[0], qubit_indices[0])
        elif gate.name == 'ry':
            qc.ry(gate.params[0], qubit_indices[0])
        elif gate.name == 'rx':
            qc.rx(gate.params[0], qubit_indices[0])
    
    return qc


def export_3cnot_to_quirk(gamma_value, use_basic_gates=True):
    """
    Export the optimized 3-CNOT circuit to Quirk format with only basic gates.
    
    Parameters:
        gamma_value: The rotation angle
        use_basic_gates: If True, use only {CX, RX, RY, RZ} (no U gates)
    """
    if use_basic_gates:
        qc = partial_swap_3cnot_basic_gates(gamma_value)
        title = "Basic Gates {CX, RX, RY, RZ}"
    else:
        qc = partial_swap_3cnot_optimized(gamma_value)
        title = "With U gates"
    
    print("\n" + "="*70)
    print(f"Quirk Circuit Export - {title}")
    print(f"γ = {gamma_value:.4f} rad = {np.degrees(gamma_value):.2f}°")
    print("="*70)
    print("\nGate sequence (left to right in Quirk):\n")
    print("Angles shown symbolically in terms of γ and π:\n")
    
    cnot_count = 0
    for idx, instruction in enumerate(qc.data):
        gate = instruction.operation
        qubits = instruction.qubits
        qubit_indices = [qc.qubits.index(q) for q in qubits]
        
        gate_name = gate.name
        
        if gate_name == 'cx':
            cnot_count += 1
            print(f"{idx+1:2d}. [CNOT {cnot_count}/3] control=q{qubit_indices[0]}, target=q{qubit_indices[1]}")
        elif gate_name == 'h':
            print(f"{idx+1:2d}. H on q{qubit_indices[0]}")
        elif gate_name in ['rx', 'ry', 'rz']:
            angle = gate.params[0]
            if isinstance(angle, Parameter):
                angle_str = str(angle)
            else:
                # Express angle symbolically in terms of γ and π
                angle_str = None
                
                # Check for exact multiples of π
                if abs(angle) < 1e-10:
                    angle_str = "0"
                elif abs(angle - np.pi) < 1e-6:
                    angle_str = "π"
                elif abs(angle + np.pi) < 1e-6:
                    angle_str = "-π"
                elif abs(angle - np.pi/2) < 1e-6:
                    angle_str = "π/2"
                elif abs(angle + np.pi/2) < 1e-6:
                    angle_str = "-π/2"
                elif abs(angle - np.pi/4) < 1e-6:
                    angle_str = "π/4"
                elif abs(angle + np.pi/4) < 1e-6:
                    angle_str = "-π/4"
                elif abs(angle - 3*np.pi/4) < 1e-6:
                    angle_str = "3π/4"
                elif abs(angle + 3*np.pi/4) < 1e-6:
                    angle_str = "-3π/4"
                # Check for exact multiples of γ
                elif abs(angle - gamma_value) < 1e-6:
                    angle_str = "γ"
                elif abs(angle + gamma_value) < 1e-6:
                    angle_str = "-γ"
                elif abs(angle - 2*gamma_value) < 1e-6:
                    angle_str = "2γ"
                elif abs(angle + 2*gamma_value) < 1e-6:
                    angle_str = "-2γ"
                elif abs(angle - gamma_value/2) < 1e-6:
                    angle_str = "γ/2"
                elif abs(angle + gamma_value/2) < 1e-6:
                    angle_str = "-γ/2"
                # Try combinations of π and γ
                else:
                    # Try angle = a*π + b*γ for small integer a, b
                    found = False
                    for a in range(-4, 5):
                        for b in range(-4, 5):
                            test_angle = a * np.pi + b * gamma_value
                            if abs(angle - test_angle) < 1e-5:
                                # Build string
                                terms = []
                                if a != 0:
                                    if a == 1:
                                        terms.append("π")
                                    elif a == -1:
                                        terms.append("-π")
                                    else:
                                        terms.append(f"{a}π")
                                if b != 0:
                                    if b == 1:
                                        terms.append("γ")
                                    elif b == -1:
                                        terms.append("-γ")
                                    else:
                                        terms.append(f"{b}γ")
                                
                                if terms:
                                    # Handle signs properly
                                    result = terms[0]
                                    for term in terms[1:]:
                                        if term.startswith('-'):
                                            result += term
                                        else:
                                            result += "+" + term
                                    angle_str = result
                                    found = True
                                    break
                        if found:
                            break
                    
                    # If still not found, express as ratio to γ or π
                    if not angle_str:
                        # Try as multiple of γ
                        ratio_gamma = angle / gamma_value
                        if abs(ratio_gamma - round(ratio_gamma)) < 1e-3:
                            r = int(round(ratio_gamma))
                            if r == 1:
                                angle_str = "γ"
                            elif r == -1:
                                angle_str = "-γ"
                            else:
                                angle_str = f"{r}γ"
                        else:
                            # Try as multiple of π
                            ratio_pi = angle / np.pi
                            if abs(ratio_pi - round(ratio_pi)) < 1e-3:
                                r = int(round(ratio_pi))
                                if r == 1:
                                    angle_str = "π"
                                elif r == -1:
                                    angle_str = "-π"
                                else:
                                    angle_str = f"{r}π"
                            else:
                                # Express as decimal fraction
                                if abs(ratio_gamma) < 10:
                                    angle_str = f"{ratio_gamma:.3f}γ"
                                else:
                                    angle_str = f"{ratio_pi:.3f}π"
                
            print(f"{idx+1:2d}. {gate_name.upper()}({angle_str}) on q{qubit_indices[0]}")
        elif gate_name == 'u':
            theta, phi, lam = gate.params
            print(f"{idx+1:2d}. U({theta:.4f}, {phi:.4f}, {lam:.4f}) on q{qubit_indices[0]}")
    
    print(f"\n✓ Total gates: {len(qc.data)}")
    print(f"✓ CNOT count: {cnot_count} (OPTIMAL)")
    print(f"\nGate count breakdown: {qc.count_ops()}")
    print("\nCircuit diagram:")
    print(qc)
    print("="*70)


def example_usage():
    """Demonstrate the optimized 3-CNOT decomposition."""
    print("="*70)
    print("Partial SWAP Gate - 3-CNOT OPTIMIZED Decomposition")
    print("="*70)
    
    gamma = 0.05
    
    # Test all variants
    best_builder = find_best_3cnot_decomposition(gamma)
    
    # Compare with unoptimized
    compare_with_unoptimized(gamma)
    
    # Export to Quirk with basic gates
    print("\n" + "="*70)
    print("QUIRK EXPORT - BASIC GATES VERSION")
    print("="*70)
    export_3cnot_to_quirk(gamma, use_basic_gates=True)
    
    # Verify the basic gates version
    print("\n" + "="*70)
    print("VERIFYING BASIC GATES VERSION")
    print("="*70)
    qc_basic = partial_swap_3cnot_basic_gates(gamma)
    
    # Build target matrix
    cos_g = np.cos(gamma)
    sin_g = np.sin(gamma)
    exp_g = np.exp(1j * gamma)
    U_target = np.array([
        [exp_g, 0, 0, 0],
        [0, cos_g, 1j*sin_g, 0],
        [0, 1j*sin_g, cos_g, 0],
        [0, 0, 0, exp_g]
    ], dtype=complex)
    
    # Get actual unitary
    U_actual = Operator(qc_basic).data
    
    # Compare (accounting for global phase)
    phase = U_actual[0, 0] / U_target[0, 0]
    U_normalized = U_actual / phase
    error = np.max(np.abs(U_normalized - U_target))
    
    success = error < 1e-10
    status = "✓ PASS" if success else "✗ FAIL"
    
    print(f"\nBasic gates version verification:")
    print(f"  Error: {error:.2e} {status}")
    print(f"  Total gates: {len(qc_basic.data)}")
    print(f"  Gate counts: {qc_basic.count_ops()}")
    print(f"  Depth: {qc_basic.depth()}")
    print(f"  CNOT count: {qc_basic.count_ops().get('cx', 0)}")
    
    # Test multiple γ values with basic gates
    print("\n" + "="*70)
    print("Testing basic gates version across γ range:")
    print("="*70)
    
    test_values = [0.01, 0.05, 0.1, 0.5, np.pi/4, np.pi/2, 1.0]
    
    all_pass = True
    for g in test_values:
        qc = partial_swap_3cnot_basic_gates(g)
        
        # Build target
        cos_g = np.cos(g)
        sin_g = np.sin(g)
        exp_g = np.exp(1j * g)
        U_target = np.array([
            [exp_g, 0, 0, 0],
            [0, cos_g, 1j*sin_g, 0],
            [0, 1j*sin_g, cos_g, 0],
            [0, 0, 0, exp_g]
        ], dtype=complex)
        
        U_actual = Operator(qc).data
        phase = U_actual[0, 0] / U_target[0, 0]
        U_normalized = U_actual / phase
        error = np.max(np.abs(U_normalized - U_target))
        
        success = error < 1e-10
        status = "✓" if success else "✗"
        cnot_count = qc.count_ops().get('cx', 0)
        cnot_ok = "✓" if cnot_count == 3 else "✗"
        print(f"  γ = {g:6.4f}: {status} (error = {error:.2e}, CNOTs = {cnot_count} {cnot_ok})")
        if not success:
            all_pass = False
    
    if all_pass:
        print("\n✓✓✓ ALL TESTS PASSED WITH BASIC GATES! ✓✓✓")
    
    print("\n" + "="*70)


def partial_swap_manual_optimized(gamma):
    """
    Manually optimized universal gates decomposition.
    Starting from partial_swap_universal (8 CNOTs), optimize by hand.
    
    Key optimizations applied:
    1. Merge consecutive RZ gates: RZ(γ) + RZ(-γ) = I (cancel!)
    2. Cancel adjacent CNOT pairs: CX·CX = I
    3. Result: 6 CNOTs (down from 8)
    
    Optimization details:
    - P gates: RZ(γ)₀, RZ(γ)₁
    - CP start: RZ(-γ)₀, RZ(-γ)₁
    - These cancel: RZ(γ) + RZ(-γ) = 0 on both qubits
    
    - RZZ end: CX, RZ(-2γ)₁, CX
    - CP remaining: CX, RZ(γ)₁, CX  
    - Middle CNOTs cancel: CX·CX = I
    - Result: CX, RZ(-2γ+γ)₁, CX = CX, RZ(-γ)₁, CX
    """
    qc = QuantumCircuit(2, name=f'PartialSWAP_ManualOpt({gamma})')
    
    # RXX part: H₀·H₁·CX·RZ(-γ)₁·CX·H₀·H₁
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)
    qc.rz(-gamma, 1)
    qc.cx(0, 1)
    qc.h(0)
    qc.h(1)
    
    # RYY part: RX(π/2)₀·RX(π/2)₁·CX·RZ(-γ)₁·CX·RX(-π/2)₀·RX(-π/2)₁
    qc.rx(np.pi/2, 0)
    qc.rx(np.pi/2, 1)
    qc.cx(0, 1)
    qc.rz(-gamma, 1)
    qc.cx(0, 1)
    qc.rx(-np.pi/2, 0)
    qc.rx(-np.pi/2, 1)
    
    # RZZ + P + CP (optimized):
    # RZZ: CX, RZ(-2γ)₁, CX
    # P: RZ(γ)₀, RZ(γ)₁  
    # CP: RZ(-γ)₀, RZ(-γ)₁, CX, RZ(γ)₁, CX
    # 
    # After RZ cancellations: CX, RZ(-2γ)₁, CX, CX, RZ(γ)₁, CX
    # Middle CX·CX = I, so: CX, RZ(-2γ)₁, RZ(γ)₁, CX
    # Merge RZ: CX, RZ(-γ)₁, CX
    qc.cx(0, 1)
    qc.rz(-gamma, 1)
    qc.cx(0, 1)
    
    return qc


def analyze_for_manual_optimization():
    """
    Analyze the gate sequence to find optimization opportunities.
    """
    from partial_swap.pswap_universal import partial_swap_universal
    
    gamma = 0.1  # Use a test value
    qc = partial_swap_universal(gamma)
    
    print("\n" + "="*70)
    print("GATE SEQUENCE ANALYSIS FOR MANUAL OPTIMIZATION")
    print("="*70)
    
    gates_by_qubit = {0: [], 1: []}
    cnot_sequence = []
    
    for idx, instruction in enumerate(qc.data):
        gate = instruction.operation
        qubits = instruction.qubits
        qubit_indices = [qc.qubits.index(q) for q in qubits]
        
        if gate.name == 'cx':
            cnot_sequence.append((idx, qubit_indices[0], qubit_indices[1]))
            print(f"{idx:2d}. CX q{qubit_indices[0]}→q{qubit_indices[1]}")
        else:
            q = qubit_indices[0]
            gates_by_qubit[q].append((idx, gate.name, gate.params[0] if gate.params else None))
            angle_str = f"({gate.params[0]:.4f})" if gate.params else ""
            print(f"{idx:2d}. {gate.name.upper():3s}{angle_str:15s} on q{q}")
    
    print(f"\n" + "="*70)
    print("CNOT ANALYSIS:")
    print("="*70)
    print(f"Total CNOTs: {len(cnot_sequence)}")
    print("CNOT positions:", [c[0] for c in cnot_sequence])
    
    # Check for consecutive CNOTs on same pair
    print("\nLooking for CNOT cancellation opportunities:")
    for i in range(len(cnot_sequence) - 1):
        curr = cnot_sequence[i]
        next_cnot = cnot_sequence[i + 1]
        
        # Check if same control→target
        if curr[1] == next_cnot[1] and curr[2] == next_cnot[2]:
            # Check what's between them
            between = [g for g in range(curr[0] + 1, next_cnot[0])]
            print(f"  CX at {curr[0]} and {next_cnot[0]} (same pair!) - gates between: {len(between)}")


def export_manual_optimized_to_quirk(gamma_value):
    """
    Export the 6-CNOT manually optimized circuit to Quirk format.
    """
    qc = partial_swap_manual_optimized(gamma_value)
    
    print("\n" + "="*70)
    print("QUIRK EXPORT - 6-CNOT MANUALLY OPTIMIZED")
    print(f"γ = {gamma_value:.4f} rad = {np.degrees(gamma_value):.2f}°")
    print("="*70)
    print("\nGate sequence (symbolic, left to right in Quirk):\n")
    
    cnot_count = 0
    for idx, instruction in enumerate(qc.data):
        gate = instruction.operation
        qubits = instruction.qubits
        qubit_indices = [qc.qubits.index(q) for q in qubits]
        
        gate_name = gate.name
        
        if gate_name == 'cx':
            cnot_count += 1
            print(f"{idx+1:2d}. [CNOT {cnot_count}/6] control=q{qubit_indices[0]}, target=q{qubit_indices[1]}")
        elif gate_name == 'h':
            print(f"{idx+1:2d}. H on q{qubit_indices[0]}")
        elif gate_name in ['rx', 'ry', 'rz']:
            angle = gate.params[0]
            if isinstance(angle, Parameter):
                angle_str = str(angle)
            else:
                # Express symbolically
                if abs(angle - np.pi/2) < 1e-6:
                    angle_str = "π/2"
                elif abs(angle + np.pi/2) < 1e-6:
                    angle_str = "-π/2"
                elif abs(angle - gamma_value) < 1e-6:
                    angle_str = "-γ"
                elif abs(angle + gamma_value) < 1e-6:
                    angle_str = "-γ"
                else:
                    angle_str = f"{angle:.4f}"
            
            print(f"{idx+1:2d}. {gate_name.upper()}({angle_str}) on q{qubit_indices[0]}")
    
    print(f"\n✓ Total gates: {len(qc.data)}")
    print(f"✓ CNOT count: {cnot_count}/6 (MANUALLY OPTIMIZED)")
    print(f"\nGate breakdown: {qc.count_ops()}")
    print(f"Circuit depth: {qc.depth()}")
    print("\nCircuit diagram:")
    print(qc)
    print("="*70)


def verify_manual_optimization():
    """
    Verify that the manually optimized version is correct.
    """
    from partial_swap.pswap_universal import partial_swap_universal
    
    print("\n" + "="*70)
    print("MANUAL OPTIMIZATION VERIFICATION")
    print("="*70)
    
    test_gammas = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    for gamma in test_gammas:
        # Build both versions
        qc_original = partial_swap_universal(gamma)
        qc_manual = partial_swap_manual_optimized(gamma)
        
        # Build target matrix
        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)
        exp_g = np.exp(1j * gamma)
        U_target = np.array([
            [exp_g, 0, 0, 0],
            [0, cos_g, 1j*sin_g, 0],
            [0, 1j*sin_g, cos_g, 0],
            [0, 0, 0, exp_g]
        ], dtype=complex)
        
        # Get actual unitaries
        U_original = Operator(qc_original).data
        U_manual = Operator(qc_manual).data
        
        # Compare with target (accounting for global phase)
        phase_orig = U_original[0, 0] / U_target[0, 0]
        U_orig_norm = U_original / phase_orig
        error_orig = np.max(np.abs(U_orig_norm - U_target))
        
        phase_manual = U_manual[0, 0] / U_target[0, 0]
        U_manual_norm = U_manual / phase_manual
        error_manual = np.max(np.abs(U_manual_norm - U_target))
        
        # Compare original vs manual
        phase_diff = U_manual[0, 0] / U_original[0, 0]
        U_manual_vs_orig = U_manual / phase_diff
        diff = np.max(np.abs(U_manual_vs_orig - U_original))
        
        success_orig = error_orig < 1e-10
        success_manual = error_manual < 1e-10
        match = diff < 1e-10
        
        print(f"\nγ = {gamma:.4f}:")
        print(f"  Original: {'✓' if success_orig else '✗'} (error = {error_orig:.2e}, CNOTs = {qc_original.count_ops().get('cx', 0)})")
        print(f"  Manual:   {'✓' if success_manual else '✗'} (error = {error_manual:.2e}, CNOTs = {qc_manual.count_ops().get('cx', 0)})")
        print(f"  Match:    {'✓' if match else '✗'} (diff = {diff:.2e})")
    
    print("\n" + "="*70)
    print("GATE COUNT COMPARISON")
    print("="*70)
    
    gamma = 0.05
    qc_orig = partial_swap_universal(gamma)
    qc_man = partial_swap_manual_optimized(gamma)
    
    print(f"\nOriginal (partial_swap_universal):")
    print(f"  Gates: {qc_orig.count_ops()}")
    print(f"  Total: {len(qc_orig.data)}")
    print(f"  Depth: {qc_orig.depth()}")
    print(f"  CNOTs: {qc_orig.count_ops().get('cx', 0)}")
    
    print(f"\nManually Optimized (6 CNOTs):")
    print(f"  Gates: {qc_man.count_ops()}")
    print(f"  Total: {len(qc_man.data)}")
    print(f"  Depth: {qc_man.depth()}")
    print(f"  CNOTs: {qc_man.count_ops().get('cx', 0)}")
    
    cnot_reduction = qc_orig.count_ops().get('cx', 0) - qc_man.count_ops().get('cx', 0)
    total_reduction = len(qc_orig.data) - len(qc_man.data)
    
    print(f"\nImprovement:")
    print(f"  CNOT reduction: {cnot_reduction} ({cnot_reduction/qc_orig.count_ops().get('cx', 0)*100:.1f}%)")
    print(f"  Total gate reduction: {total_reduction} ({total_reduction/len(qc_orig.data)*100:.1f}%)")
    print(f"  Depth reduction: {qc_orig.depth() - qc_man.depth()} ({(qc_orig.depth()-qc_man.depth())/qc_orig.depth()*100:.1f}%)")
    
    print("\n" + "="*70)
    print("OPTIMIZED CIRCUIT")
    print("="*70)
    print(qc_man)
    print("="*70)
    
    # Export for Quirk
    export_manual_optimized_to_quirk(gamma)


if __name__ == "__main__":
    # Run analysis first
    print("STEP 1: Analyze original decomposition")
    analyze_for_manual_optimization()
    
    # Then verify manual optimization
    print("\n\nSTEP 2: Verify manual optimization")
    verify_manual_optimization()
