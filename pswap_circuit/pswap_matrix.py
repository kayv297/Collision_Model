import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
from qiskit import transpile

'''
This code constructs the partial swap unitary matrix U(gamma) and decomposes it into basic gates with Qiskit optimization.
'''


# parameter gamma
gamma = np.pi / 2

# construct the 4x4 matrix for U(gamma)
cos = np.cos(gamma)
sin = np.sin(gamma)
# partial-swap unitary
U = np.array([
    [np.exp(1j*gamma), 0, 0, 0],
    [0, cos, 1j*sin, 0],
    [0, 1j*sin, cos, 0],
    [0, 0, 0, np.exp(1j*gamma)]
], dtype=complex)

qc = QuantumCircuit(2)
qc.append(UnitaryGate(U), [0,1])

# Ask Qiskit to decompose into CNOT + 1Q gates
decomposed = qc.decompose(reps=10)  # reps=10 = keep decomposing recursively
print("Decomposed circuit:")
print(decomposed)

# get the 2nd qbit, unitary matrix (V2) applied after first CNOT
# Find the first CNOT, then get the next gate on qubit 1
found_first_cnot = False
v2 = None
a, b, c = None, None, None

for instruction in decomposed.data:
    if instruction.operation.name == 'cx' and not found_first_cnot:
        found_first_cnot = True
    elif found_first_cnot and 1 in [qubit._index for qubit in instruction.qubits]:
        # This is the gate on qubit 1 after the first CNOT
        if instruction.operation.name == 'u':
            params = instruction.operation.params
            a, b, c = params[0], params[1], params[2]  # theta, phi, lambda
            # Construct the V2 matrix
            v2 = np.array([
                [np.cos(a/2), -np.exp(1j*c) * np.sin(a/2)],
                [np.exp(1j*b) * np.sin(a/2), np.exp(1j*(b+c)) * np.cos(a/2)]
            ], dtype=complex)
        break

print(f"V2 parameters: a={a}, b={b}, c={c}")
print(f"V2 matrix:\n{v2}")


basis = ['ry', 'rz', 'cx']   # only allow these gates
transpiled = transpile(qc, basis_gates=basis, optimization_level=1)
print(transpiled)


# test the unitary
print("Testing the partial swap unitary on computational basis states:")
print("=" * 60)

# Define the computational basis states
basis_states = {
    "|00⟩": np.array([1, 0, 0, 0]),
    "|01⟩": np.array([0, 1, 0, 0]),
    "|10⟩": np.array([0, 0, 1, 0]),
    "|11⟩": np.array([0, 0, 0, 1])
}

print(f"Partial swap parameter γ = {gamma}")
print(f"cos(γ) = {cos:.6f}, sin(γ) = {sin:.6f}")
print(f"e^(iγ) = {np.exp(1j*gamma):.6f}")
print()

for state_name, state_vector in basis_states.items():
    # Apply the unitary to the state
    result = U @ state_vector
    
    print(f"U|{state_name[1:-1]}> = ", end="")
    
    # Format the output nicely
    terms = []
    for i, amplitude in enumerate(result):
        if abs(amplitude) > 1e-10:  # Only show non-zero terms
            if i == 0:
                basis = "|00>"
            elif i == 1:
                basis = "|01>"
            elif i == 2:
                basis = "|10>"
            else:
                basis = "|11>"
            
            # Format complex numbers nicely
            if np.isreal(amplitude):
                terms.append(f"{amplitude.real:.6f}{basis}")
            else:
                real_part = amplitude.real
                imag_part = amplitude.imag
                if abs(real_part) < 1e-10:
                    terms.append(f"{imag_part:.6f}i{basis}")
                elif abs(imag_part) < 1e-10:
                    terms.append(f"{real_part:.6f}{basis}")
                else:
                    sign = "+" if imag_part >= 0 else ""
                    terms.append(f"({real_part:.6f}{sign}{imag_part:.6f}i){basis}")
    
    print(" + ".join(terms))
