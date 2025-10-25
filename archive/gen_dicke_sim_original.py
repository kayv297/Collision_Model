from typing import List
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

# global parameters/config
max_rounds = 200               # simulate rounds = 1..max_rounds

gamma_sh = 0.05               # shuttle-register partial-swap angle
gamma_in = 0  # intra-register partial-swap (weak interaction)
# gamma_in = 0.95*np.pi/2  # intra-register partial-swap (strong interaction)

scheme = 'sequential'  # 'sequential' or 'interleaved'
# scheme = 'interleaved'  # 'sequential' or 'interleaved'

# Initial state |00>|11>|00> in ordering [r1, r2, A1, A2, s1, s2] for D4,2 preparation
A_pos = [0]  # positions of ancillas in the full state
n_registers = [2, 2]  # number of qubits in each register
# step = 1 # 1: theoretical round (KT style), 2^len(A_pos): operation round  (Cakmak style, 4 ops per operation round)
step = 2**len(A_pos)


# helper functions & variables

zero = np.array([1, 0], dtype=complex)
one = np.array([0, 1], dtype=complex)

# Partial SWAP gate (2-qubit)
SWAP = np.zeros((4, 4), dtype=complex)
SWAP[0, 0] = 1
SWAP[1, 2] = 1
SWAP[2, 1] = 1
SWAP[3, 3] = 1


def partial_swap(gamma):
    # gamma is the partial-swap angle
    return np.cos(gamma)*np.eye(4, dtype=complex) + 1j*np.sin(gamma)*SWAP


U_shutter = partial_swap(gamma_sh)  # This is the shuttle-register partial-swap
U_intra = partial_swap(gamma_in)  # This is the intra-register partial-swap

'''
This blocks contain main logic functions

Collision Model Logic:
Flow: First, giving 2 register R (r1 & r2) and S (s1 & s2) and 1 shuttle A (ancilla)
The collision model perform (i), (ii), (iii), (iv) and (i'), (iii') as below:
(i) A interacts with r1 then (iii) A interacts with s1
(i') A interacts with r2 then (iii') A interacts with s2
Intra register interaction (ii) r1 interacts with r2 and (iv) s1 interacts with s2
'''


def apply_two_qubit(state, U, i, j, n_qbits):
    """
    Apply two-qubit unitary U to qubits i and j of an n_qbits-qubit statevector.
    Generalized version that works for any number of qubits.

    Args:
        state: quantum state vector of size 2^n_qbits
        U: 4x4 unitary matrix to apply
        i, j: qubit indices to apply the gate to
        n_qbits: total number of qubits in the system

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
    # Shape becomes [4, 2^(n_qbits-2)]
    tensor_reshaped = tensor_permuted.reshape(4, -1)

    # Apply the 2-qubit unitary
    new_tensor = U @ tensor_reshaped

    # Reshape back to separate the two qubits
    # Shape becomes [2, 2, 2^(n_qbits-2)]
    new_tensor_shaped = new_tensor.reshape([2, 2] + [2] * (n_qbits - 2))

    # Create inverse permutation to restore original qubit order
    inv_perm = np.argsort(perm)

    # Apply inverse permutation
    result_tensor = np.transpose(new_tensor_shaped, inv_perm)

    # Flatten back to state vector
    return result_tensor.reshape(2**n_qbits)


def perform_intra_register_collisions(state, index, reg_qubits, n_qbits):
    """
    Perform intra-register collisions for a list of qubits in a register for a specific qbit[idx].
    Only nearest-neighbor interactions are considered.

    Args:
        state: quantum state vector
        reg_qubits: List of qubit indices in the register
        n_qbits: Total number of qubits in the system

    Returns:
        Updated state vector after intra-register collisions
    """
    if len(reg_qubits) < 2 or gamma_in == 0:
        return state  # No intra-register collisions needed

    cur_bit = reg_qubits[index]
    # Perform nearest-neighbor interactions left-ward
    for left_index in range(index - 1, -1, -1):
        left_bit = reg_qubits[left_index]
        state = apply_two_qubit(state, U_intra, cur_bit, left_bit, n_qbits)
        break

    # Perform nearest-neighbor interactions right-ward
    for right_index in range(index + 1, len(reg_qubits)):
        right_bit = reg_qubits[right_index]
        state = apply_two_qubit(state, U_intra, cur_bit, right_bit, n_qbits)
        break

    return state


def perform_collision_round(state, n_registers=[2, 2], A=0, A_pos=[0], n_qbits=5, reg_collide_idx=0):
    '''
    Peform 1 full collision round (i), (ii), (iii), (iv) for a specific ancilla A and specific register qubit index

    # When run this function 1 times, meaning you doing 4 interactions

    A: position of ancilla in the full state (or index)
    A_pos: list of ancilla positions in the full state
    n_registers: list of number of qubits in each register
    n_qbits: total number of qubits in the full state
    reg_collide_idx: collide index in A-r or A-s position. e.g: reg_collide_idx=0 -> A-r1 and A-s1; reg_collide_idx=1 -> A-r2 and A-s2
    '''
    # positions of r1, r2 in the full state
    # r1, r2, ... (1, 2, ...)
    r_reg = list(range(len(A_pos), len(A_pos) + n_registers[0]))
    # s1, s2, ... (3, 4, ...)
    s_reg = [r_reg[-1] + 1 + i for i in range(n_registers[1])]

    r = r_reg[reg_collide_idx]
    s = s_reg[reg_collide_idx]

    # FIRST, perform A-r
    # inter-register collision (r, A)
    state = apply_two_qubit(state, U_shutter, A, r, n_qbits)

    # SECOND, intra-collide in r_reg (ONLY consider nearest-neighbor interaction):
    # e.g r2-r1, then r2-r3 (if exists), loop until meet 2 ends of r_reg
    if gamma_in != 0:
        # find current r's index in r_reg
        r_index = r_reg.index(r)
        state = perform_intra_register_collisions(
            state, r_index, r_reg, n_qbits)

    # THIRD, perform A-s
    # inter-register collision (s, A)
    state = apply_two_qubit(state, U_shutter, A, s, n_qbits)

    # FOURTH, intra-collide in s_reg (ONLY consider nearest-neighbor interaction):
    # e.g s2-s1, then s2-s3 (if exists), loop until meet 2 ends of s_reg
    if gamma_in != 0:

        # find current s's index in s_reg
        s_index = s_reg.index(s)
        state = perform_intra_register_collisions(
            state, s_index, s_reg, n_qbits)

    # END: this is called 1 collision round (OR FIRST-half of a theoretical round)

    '''
    Below is the old version before generalization, where only 2 registers with 2 qubits each and 1 ancilla to create W4-state with initial state |00>|1>|00>
    
    mode = normal mean the A[0] interacts with r1 first, then s1
    '''

    # if mode == 'normal':
    #     # (i) A-r1
    #     state = apply_two_qubit(state, U_shutter, A, r1, n_qbits)
    # else:  # prime mode
    #     # (i') A-r2
    #     state = apply_two_qubit(state, U_shutter, A, r2, n_qbits)

    # # (ii) r1-r2 (if gamma_in != 0) - intra-register
    # if gamma_in != 0:
    #     state = apply_two_qubit(state, U_intra, r1, r2, n_qbits)

    # if mode == 'normal':
    #     # (iii) A-s1
    #     state = apply_two_qubit(state, U_shutter, A, s1, n_qbits)
    # else:  # prime mode
    #     # (iii') A-s2
    #     state = apply_two_qubit(state, U_shutter, A, s2, n_qbits)

    # # (iv) s1-s2 (if gamma_in != 0) - intra-register
    # if gamma_in != 0:
    #     state = apply_two_qubit(state, U_intra, s1, s2, n_qbits)

    return state


def perform_theoretical_round(state, n_registers=[2, 2], A_pos=[0], round_num=1, scheme='sequential'):
    '''
    Perform 1 full theoretical collision round (i), (ii), (iii), (iv), (i'), (ii'), (iii'), (iv')
    '''
    n_qbits = sum(n_registers) + len(A_pos)

    states = []
    k_iters = []

    if scheme not in ['sequential', 'interleaved']:
        raise ValueError("Scheme must be 'sequential' or 'interleaved'")

    if scheme == 'sequential':
        for i, A in enumerate(A_pos):
            # assuming both registers have the same number of qubits
            for reg_collide_idx in range(n_registers[0]):
                state = perform_collision_round(
                    state, n_registers, A, A_pos, n_qbits, reg_collide_idx)
                states.append(state.copy())
                k_iters.append(round_num + i)

            # state = perform_collision_round(
            #     state, n_registers, A, A_pos, n_qbits)
            # states.append(state.copy())
            # k_iters.append(round_num)

            # state = perform_collision_round(
            #     state, n_registers, A, A_pos, n_qbits)
            # states.append(state.copy())
            # k_iters.append(round_num + 1)

        return states, k_iters

    # Interleaved scheme
    # Flow A1-normal, A2-normal, A1-prime, A2-prime
    # assuming both registers have the same number of qubits
    for i, reg_collide_idx in enumerate(range(n_registers[0])):
        for A in A_pos:
            state = perform_collision_round(
                state, n_registers, A, A_pos, n_qbits, reg_collide_idx)
            states.append(state.copy())
            k_iters.append(round_num + i)

    return states, k_iters


def postselection_step(cur_state, A_pos, n_registers):
    """
    Perform proper projective measurement according to paper equation (7)
    Generalized for arbitrary ancilla positions and register sizes.

    Args:
        cur_state: Current quantum state vector
        A_pos: List of ancilla positions (0-indexed)
        n_registers: List with number of qubits in each register

    Returns:
        rho_registers: Density matrix of registers after post-selection
        prob: Post-selection probability
    """
    # Convert state vector to density matrix
    rho = np.outer(cur_state, np.conj(cur_state))

    # Calculate dimensions
    num_ancillas = len(A_pos)
    total_register_qubits = sum(n_registers)

    # Create projector |0⟩^⊗m_A⟨0|^⊗m for all ancillas (project to |00...0⟩)
    proj_0_single = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|
    proj_0_ancillas = proj_0_single
    for i in range(1, num_ancillas):
        proj_0_ancillas = np.kron(proj_0_ancillas, proj_0_single)

    # Extend to full system: |0⟩^⊗m_A⟨0|^⊗m ⊗ I_registers
    I_registers = np.eye(2**total_register_qubits, dtype=complex)
    full_projector = np.kron(proj_0_ancillas, I_registers)

    # Apply projection: |0⟩^⊗m_A⟨0|^⊗m ρ |0⟩^⊗m_A⟨0|^⊗m
    projected_rho = full_projector @ rho @ full_projector

    # Calculate probability: Tr[|0⟩^⊗m_A⟨0|^⊗m ρ |0⟩^⊗m_A⟨0|^⊗m]
    prob = np.trace(projected_rho).real

    if prob > 1e-12:
        # Normalize: divide by probability
        normalized_projected_rho = projected_rho / prob

        # Partial trace over ancillas to get register density matrix
        # Reshape for partial trace: (dim_ancillas, dim_registers, dim_ancillas, dim_registers)
        dim_ancillas = 2**num_ancillas
        dim_registers = 2**total_register_qubits

        reshaped = normalized_projected_rho.reshape(
            dim_ancillas, dim_registers, dim_ancillas, dim_registers)

        # Trace over ancilla indices (0 and 2)
        rho_registers = np.trace(reshaped, axis1=0, axis2=2)

        return rho_registers, prob
    else:
        return np.zeros((2**total_register_qubits, 2**total_register_qubits), dtype=complex), 0.0


'''
Analysis and visualization functions
'''


def print_state(state):
    """Print the state in a readable format."""
    n_qbits = int(np.log2(len(state))
                  )  # Calculate number of qubits from state vector length
    for i, amp in enumerate(state):
        if np.abs(amp) > 1e-6:  # Print only significant amplitudes
            print(f"|{i:0{n_qbits}b}>: {amp:.4f}, magnitude: {np.abs(amp)**2:.4f}")
    print()


def plot_amplitudes(state):
    """Plot the amplitudes of a single quantum state."""
    labels = [f"{i:05b}" for i in range(32)]
    probabilities = np.abs(state)**2  # Probability amplitudes

    plt.figure(figsize=(15, 6))
    bars = plt.bar(range(32), probabilities)

    # Color bars with significant probability differently
    for i, bar in enumerate(bars):
        if probabilities[i] > 1e-6:
            bar.set_color('red')
        else:
            bar.set_color('lightgray')

    plt.xlabel('Basis States')
    plt.ylabel('Probability |ψ|²')
    plt.title('Quantum State Probabilities')
    plt.xticks(range(32), labels, rotation=90, fontsize=8)
    plt.ylim(0, 1.1 * np.max(probabilities))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    # plt.show()
    plt.savefig('state_probabilities.png', dpi=300)


def plot_amplitude_evolution(density_matrices, rounds_data, n_registers=4):
    """
    Plot populations of each register state vs rounds using density matrices
    """
    plt.figure(figsize=(12, 8))

    # Extract populations from density matrices
    n_terms = 2**n_registers  # 16 states for 4 registers

    # Plot each register state population
    for term_idx in range(n_terms):
        populations = []
        for rho in density_matrices:
            # Extract diagonal element (population) from density matrix
            pop = np.abs(rho[term_idx, term_idx].real)
            # Take sqrt to get amplitude magnitude
            populations.append(np.sqrt(pop))

        # Only plot if the amplitude is non-negligible at some point
        if max(populations) > 1e-6:
            plt.plot(rounds_data, populations,
                     label=f'|{term_idx:0{n_registers}b}⟩',
                     markersize=2, alpha=0.7)

    plt.xlabel('Round Number')
    plt.ylabel('Conditional Amplitude |ψ|')
    plt.title('Evolution of Post-selected State Amplitudes from Density Matrix')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('density_matrix_amplitude_evolution.png', dpi=300)


def plot_register_populations(states):
    """Plot total population in registers R and S over time."""
    rounds = range(len(states))

    register_R_pop = []  # Population in r1,r2
    register_S_pop = []  # Population in s1,s2
    shuttle_pop = []     # Population in shuttle A

    for state in states:
        probs = np.abs(state)**2

        # For ordering [A, r1, r2, s1, s2]:
        # A is bit 4 (value 16), r1 is bit 3 (value 8), r2 is bit 2 (value 4)
        # s1 is bit 1 (value 2), s2 is bit 0 (value 1)

        a_pop = sum(probs[i] for i in range(32) if (i >> 4) & 1)  # bit 4 (A)
        r_pop = sum(probs[i] for i in range(32) if (
            (i >> 2) & 3) >= 1)  # bits 2,3 (r2,r1)
        s_pop = sum(probs[i]
                    for i in range(32) if (i & 3) >= 1)  # bits 0,1 (s2,s1)

        register_R_pop.append(r_pop)
        register_S_pop.append(s_pop)
        shuttle_pop.append(a_pop)

    plt.figure(figsize=(12, 6))
    plt.plot(rounds, shuttle_pop, 'r-', label='Shuttle A', linewidth=2)
    plt.plot(rounds, register_R_pop, 'b-',
             label='Register R (r1,r2)', linewidth=2)
    plt.plot(rounds, register_S_pop, 'g-',
             label='Register S (s1,s2)', linewidth=2)

    plt.xlabel('Round Number')
    plt.ylabel('Total Population')
    plt.title('Population Transfer Between Shuttle and Registers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    # plt.show()
    plt.savefig('population_transfer.png', dpi=300)


def extract_w_coefficients(state, A_pos, n_registers, project_shuttle=True):
    """
    Extract the coefficients b_i from the state after projecting/tracing the shuttle

    The generalized state is: |ψ⟩ = Σ b_i |basis_i⟩ 
    where each basis state has len(A_pos) excitations distributed across register qubits

    For the [A1A2...Am, r1r2...rn, s1s2...sk] ordering:
    - Ancillas A are at the beginning
    - Register qubits follow

    Args:
        state: quantum state vector
        A_pos: List of ancilla positions (0-indexed)
        n_registers: List with number of qubits in each register
        project_shuttle: Whether to project ancillas onto |0⟩
    """
    num_ancillas = len(A_pos)
    total_register_qubits = sum(n_registers)
    total_qubits = num_ancillas + total_register_qubits

    if len(state) == 2**total_qubits:  # Full state
        if project_shuttle:
            # Project ancillas onto |0⟩ and extract register state
            reg_state = np.zeros(2**total_register_qubits, dtype=complex)

            for i in range(len(state)):
                # Check if all ancillas are in |0⟩ state
                ancilla_bits = (i >> total_register_qubits) & (
                    (1 << num_ancillas) - 1)
                if ancilla_bits == 0:  # All ancillas in |0⟩
                    # Extract the register bits (lower bits)
                    i_red = i & ((1 << total_register_qubits) - 1)
                    reg_state[i_red] = state[i]

            # Normalize
            norm = np.linalg.norm(reg_state)
            if norm > 0:
                reg_state /= norm
    else:
        reg_state = state

    # Extract coefficients for states with len(A_pos) excitations
    # Generate all possible combinations of len(A_pos) excitations across register qubits

    coeffs = []

    # Get all ways to place num_ancillas excitations among total_register_qubits positions
    for excitation_positions in combinations(range(total_register_qubits), num_ancillas):
        # Create basis state with excitations at the specified positions
        basis_idx = 0
        for pos in excitation_positions:
            # Big-endian bit ordering
            basis_idx |= (1 << (total_register_qubits - 1 - pos))

        coeffs.append(reg_state[basis_idx])

    return np.array(coeffs)


def plot_state_evolution(states, A_pos, n_registers, gamma_sh, gamma_in, max_rounds, x_vals=None):
    """Analyze and plot W-state coefficient evolution"""

    # Extract coefficients
    coeffs_evolution = []
    for state in states:
        coeffs = extract_w_coefficients(
            state, A_pos, n_registers, project_shuttle=True)
        coeffs_evolution.append(coeffs)
    coeffs_evolution = np.array(coeffs_evolution)

    # Plot
    plt.figure(figsize=(12, 8))

    # Generate labels based on register structure
    if len(coeffs_evolution) == 0:
        raise ValueError(
            "No coefficient data to plot, coeffs_evolution is empty.")
    num_terms = len(coeffs_evolution[0])

    # Create detailed legend mapping and save to file
    import json
    num_ancillas = len(A_pos)
    total_register_qubits = sum(n_registers)

    legend_data = {
        "parameters": {
            "num_ancillas": num_ancillas,
            "n_registers": n_registers,
            "total_register_qubits": total_register_qubits,
            "num_terms": num_terms,
            "gamma_shuttle": gamma_sh,
            "gamma_intra": gamma_in
        },
        "coefficients": {}
    }

    # Generate detailed labels showing excitation patterns
    for i, excitation_positions in enumerate(combinations(range(total_register_qubits), num_ancillas)):
        # Create binary representation
        binary_str = ['0'] * total_register_qubits
        for pos in excitation_positions:
            binary_str[pos] = '1'

        # Add register separators for clarity
        if len(n_registers) == 2:  # Two registers
            r_part = ''.join(binary_str[:n_registers[0]])
            s_part = ''.join(binary_str[n_registers[0]:])
            detailed_label = f"|{r_part}⟩⊗|{s_part}⟩"
        else:
            # General case - just show positions
            positions_str = ','.join(map(str, excitation_positions))
            detailed_label = f"|exc@{positions_str}⟩"

        legend_data["coefficients"][f"b{i+1}"] = {
            "excitation_positions": list(excitation_positions),
            "binary_representation": ''.join(binary_str),
            "state_notation": detailed_label
        }

    # Save legend to JSON file
    with open('coefficient_legend.json', 'w') as f:
        json.dump(legend_data, f, indent=2)

    print(f"Legend saved to 'coefficient_legend.json'")

    labels = [f'b{i}' for i in range(1, num_terms + 1)]

    # Use provided x_vals or create default iteration numbers
    if x_vals is None:
        x_vals = np.arange(len(coeffs_evolution))

    for i in range(num_terms):
        # Y data comes from the coefficients extracted from states
        y_data = np.abs(coeffs_evolution[:, i])

        # Plot scatter points with connecting lines (NO LEGEND)
        plt.plot(x_vals, y_data, linewidth=2, alpha=0.7)
        plt.scatter(x_vals, y_data, s=30, alpha=0.8)

    # Perfect State line depends on number of qubits
    perfect_w_amplitude = 1.0 / np.sqrt(num_terms)
    plt.axhline(y=perfect_w_amplitude, color='gray', linestyle='--',
                alpha=0.5, label=f'Perfect State (1/√{num_terms})')

    print(f'Number of terms: {num_terms}')
    print(f'Perfect amplitude: {perfect_w_amplitude:.4f}')

    plt.xlabel('Round Number')
    plt.ylabel('|bᵢ|')
    plt.title(
        f'State Coefficient Evolution (γ_shuttle={gamma_sh}, γ_intra={gamma_in:.3f})')

    # Only show perfect state line in legend
    plt.legend([f'Perfect State (1/√{num_terms})'])

    plt.grid(True, alpha=0.3)
    plt.xlim([0, max_rounds])
    plt.ylim([0, 0.8])
    # plt.show()
    plt.savefig('amplitude_evolution.png', dpi=300)

    return coeffs_evolution


def plot_state_no_measure(states, A_pos, n_registers, gamma_sh, gamma_in, max_rounds, x_vals=None, save_fig: bool = True):
    """
    Plot amplitude evolution without post-selection (full state analysis)
    Shows all basis states with significant population.
    """
    num_ancillas = len(A_pos)
    total_register_qubits = sum(n_registers)
    total_qubits = num_ancillas + total_register_qubits

    # Extract amplitudes for all basis states
    amplitudes_evolution = []
    for state in states:
        amplitudes = np.abs(state)  # Get magnitude of all amplitudes
        amplitudes_evolution.append(amplitudes)
    amplitudes_evolution = np.array(amplitudes_evolution)

    # Find basis states with non-negligible population at any point
    max_amps = np.max(amplitudes_evolution, axis=0)
    significant_indices = np.where(
        max_amps > 1e-3)[0]  # Threshold for plotting

    plt.figure(figsize=(14, 8))

    # Use provided x_vals or create default iteration numbers
    if x_vals is None:
        x_vals = np.arange(len(amplitudes_evolution))

    # Plot each significant basis state
    for idx in significant_indices:
        # Create label showing full state in binary
        binary_full = f"{idx:0{total_qubits}b}"

        # Split into ancilla and register parts for clarity
        ancilla_part = binary_full[:num_ancillas]
        register_part = binary_full[num_ancillas:]

        # Further split registers if needed
        if len(n_registers) == 2:
            r_part = register_part[:n_registers[0]]
            s_part = register_part[n_registers[0]:]
            label = f"|A:{ancilla_part}⟩|R:{r_part}⟩|S:{s_part}⟩"
        else:
            label = f"|A:{ancilla_part}⟩|Reg:{register_part}⟩"

        y_data = amplitudes_evolution[:, idx]
        plt.plot(x_vals, y_data, linewidth=2, alpha=0.7, label=label)
        plt.scatter(x_vals, y_data, s=20, alpha=0.6)

    plt.xlabel('Round Number')
    plt.ylabel('Amplitude |ψᵢ|')
    plt.title(
        f'State Evolution (No Post-selection) (γ_shuttle={gamma_sh}, γ_intra={gamma_in:.3f})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, max_rounds])
    plt.tight_layout()

    if save_fig:
        plt.savefig('amp_evol_no_postsel.png', dpi=300, bbox_inches='tight')

    return amplitudes_evolution


def gen_init_state(ancillas_pos: List[int], n_registers: List[int]) -> np.ndarray:
    """
    Generate the initial state vector given ancilla positions and number of registers.

    Assume output state always looks like this: |A1A2...Am>|r1r2...rn>|s1s2...sk>

    Args:
        ancillas_pos: List of positions (0-indexed) where ancillas are in state |1⟩.
        n_registers: List with number of qubits in each register. There are only 2 registers for now r_reg and s_reg.
    Returns:
        state: The initial state vector as a numpy array.
    """
    # build n_reg first
    num_qbits_r = n_registers[0]
    num_qbits_s = n_registers[1]
    num_qbits_A = len(ancillas_pos)
    total_qbits = num_qbits_r + num_qbits_A + num_qbits_s

    # gen A_reg with m numbers of 1 (moved to beginning)
    A_reg = one
    for i in range(1, num_qbits_A):
        A_reg = np.kron(A_reg, one)

    # generate r_reg with n numbers of zero
    r_reg = zero
    for i in range(1, num_qbits_r):
        r_reg = np.kron(r_reg, zero)

    # generate s_reg with n numbers of zero
    s_reg = zero
    for i in range(1, num_qbits_s):
        s_reg = np.kron(s_reg, zero)

    return np.kron(A_reg, np.kron(r_reg, s_reg))


def dicke_evolution(save_fig: bool = True) -> np.ndarray:
    '''
    Main Simulation function:

    This function sets up the initial state and runs the collision model simulation.

    It aims to create any Dicke state across two registers using a number of ancillas.
    '''
    initial_state = gen_init_state(A_pos, n_registers)

    print("Initial state:")
    print_state(initial_state)

    states = [initial_state.copy()]  # Store all states
    n_iters = [0]  # Track iteration numbers

    # Store register density matrices after post-selection
    register_density_matrices = []
    postsel_probs = []  # Track post-selection probabilities
    cur_state = initial_state.copy()

    # * Run simulation with operation-level tracking
    # 1 theoretical round = 2 operation rounds (every (i), (ii), (iii), (iv) is one operation round)
    for round_num in range(1, max_rounds + 1, step):
        # cur_state = perform_collision_round(
        #     cur_state, n_registers, 0, A_pos, n_qbits=5, mode='normal')

        # states.append(cur_state.copy())
        # n_iters.append(round_num)  # First half of "theoretical" round

        # cur_state = perform_collision_round(
        #     cur_state, n_registers, 0, A_pos, n_qbits=5, mode='prime')

        # states.append(cur_state.copy())
        # n_iters.append(round_num + 1)  # Second half of "theoretical" round

        new_states, k_iters = perform_theoretical_round(
            cur_state, n_registers, A_pos, round_num, scheme)
        # states.append(cur_state.copy())
        # n_iters.append(round_num + 1)  # End of theoretical round
        cur_state = new_states[-1]
        states.extend(new_states)
        n_iters.extend(k_iters)

        # * Keep Post-selection step for reference to Cakmak et al. paper
        # rho_registers, prob = postselection_step(cur_state, A_pos, n_registers)

        # print(f'rho_registers (round {round_num}):')
        # print(rho_registers)
        # print(rho_registers.shape)
        # print(f'Post-selection probability: {prob:.6f}\n')

        # register_density_matrices.append(rho_registers.copy())
        # postsel_probs.append(prob)

        # Progress tracking
        if round_num % 10 == 0:
            print(f"Completed round {round_num}")

    print("Simulation complete.")

    # Print first few states
    # for idx, state in enumerate(states[:55]):
    #     print(f"State after operation {idx}:")
    #     print_state(state)

    # plot_amplitude_evolution(register_density_matrices, list(range(1, max_rounds + 1)))

    # coeffs_evolution = plot_state_evolution(
    #     states, A_pos, n_registers, gamma_sh, gamma_in, max_rounds, x_vals=n_iters)

    amps_no_ps = plot_state_no_measure(
        states, A_pos, n_registers, gamma_sh, gamma_in, max_rounds, x_vals=n_iters, save_fig=save_fig)

    print(amps_no_ps.shape)
    print('Amplitudes (no post-selection)', amps_no_ps)
    return amps_no_ps


if __name__ == "__main__":
    dicke_evolution()
