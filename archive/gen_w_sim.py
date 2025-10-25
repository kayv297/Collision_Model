import matplotlib.pyplot as plt
import numpy as np

# global parameters
gamma_sh = 0.05               # shuttle-register partial-swap angle
gamma_in = 0  # intra-register partial-swap (e.g. near pi/2)
# gamma_in = 0.95*np.pi/2  # intra-register partial-swap (e.g. near pi/2)
max_rounds = 100               # simulate rounds = 1..max_rounds


# helper functions

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


def apply_two_qubit(state, U, i, j):
    """
    Apply two-qubit unitary U to qubits i and j of a 5-qubit statevector.
    Specialized for the collision model: A(0), r1(1), r2(2), s1(3), s2(4)
    """
    # Reshape state to 5D tensor: [2,2,2,2,2]
    tensor = state.reshape([2, 2, 2, 2, 2])

    if i == 0 and j == 1:  # A-r1 interaction
        # Move A(0) and r1(1) to front: already there
        tensor_reshaped = tensor.reshape(4, 8)  # [A,r1] x [r2,s1,s2]
        new_tensor = U @ tensor_reshaped
        result = new_tensor.reshape([2, 2, 2, 2, 2])

    elif i == 0 and j == 2:  # A-r2 interaction
        # Swap r1(1) and r2(2): (A, r2, r1, s1, s2)
        tensor_perm = np.transpose(tensor, [0, 2, 1, 3, 4])
        tensor_reshaped = tensor_perm.reshape(4, 8)
        new_tensor = U @ tensor_reshaped
        new_tensor_shaped = new_tensor.reshape([2, 2, 2, 2, 2])
        result = np.transpose(new_tensor_shaped, [0, 2, 1, 3, 4])  # Swap back

    elif i == 0 and j == 3:  # A-s1 interaction
        # Move s1(3) to position 1: (A, s1, r1, r2, s2)
        tensor_perm = np.transpose(tensor, [0, 3, 1, 2, 4])
        tensor_reshaped = tensor_perm.reshape(4, 8)
        new_tensor = U @ tensor_reshaped
        new_tensor_shaped = new_tensor.reshape([2, 2, 2, 2, 2])
        result = np.transpose(new_tensor_shaped, [
                              0, 2, 3, 1, 4])  # Restore order

    elif i == 0 and j == 4:  # A-s2 interaction
        # Move s2(4) to position 1: (A, s2, r1, r2, s1)
        tensor_perm = np.transpose(tensor, [0, 4, 1, 2, 3])
        tensor_reshaped = tensor_perm.reshape(4, 8)
        new_tensor = U @ tensor_reshaped
        new_tensor_shaped = new_tensor.reshape([2, 2, 2, 2, 2])
        result = np.transpose(new_tensor_shaped, [
                              0, 2, 3, 4, 1])  # Restore order

    elif i == 1 and j == 2:  # r1-r2 interaction (intra-register R)
        # Move r1(1), r2(2) to front: (r1, r2, A, s1, s2)
        tensor_perm = np.transpose(tensor, [1, 2, 0, 3, 4])
        tensor_reshaped = tensor_perm.reshape(4, 8)
        new_tensor = U @ tensor_reshaped
        new_tensor_shaped = new_tensor.reshape([2, 2, 2, 2, 2])
        result = np.transpose(new_tensor_shaped, [
                              2, 0, 1, 3, 4])  # Restore order

    elif i == 3 and j == 4:  # s1-s2 interaction (intra-register S)
        # Move s1(3), s2(4) to front: (s1, s2, A, r1, r2)
        tensor_perm = np.transpose(tensor, [3, 4, 0, 1, 2])
        tensor_reshaped = tensor_perm.reshape(4, 8)
        new_tensor = U @ tensor_reshaped
        new_tensor_shaped = new_tensor.reshape([2, 2, 2, 2, 2])
        result = np.transpose(new_tensor_shaped, [
                              2, 3, 4, 0, 1])  # Restore order

    else:
        raise ValueError(
            f"Unsupported qubit pair ({i},{j}) for this collision model")

    return result.reshape(32)  # Back to 5-qubit state vector


def perform_collision_round(state):
    # (i) A-r1
    state = apply_two_qubit(state, U_shutter, 0, 1)
    # (ii) r1-r2 (if gamma_in != 0)
    if gamma_in != 0:
        state = apply_two_qubit(state, U_intra, 1, 2)
    # (iii) A-s1
    state = apply_two_qubit(state, U_shutter, 0, 3)
    # (iv) s1-s2 (if gamma_in != 0)
    if gamma_in != 0:
        state = apply_two_qubit(state, U_intra, 3, 4)

    return state


def perform_collision_round_2(state):
    # (i') A-r2
    state = apply_two_qubit(state, U_shutter, 0, 2)
    # (ii) r1-r2 (if gamma_in != 0)
    if gamma_in != 0:
        state = apply_two_qubit(state, U_intra, 1, 2)
    # (iii') A-s2
    state = apply_two_qubit(state, U_shutter, 0, 4)
    # (iv) s1-s2 (if gamma_in != 0)
    if gamma_in != 0:
        state = apply_two_qubit(state, U_intra, 3, 4)

    return state


def postselection_step(current_state, n_registers=4):
    """
    Perform proper projective measurement according to paper equation (7)
    """
    # Convert state vector to density matrix
    rho = np.outer(current_state, np.conj(current_state))

    # Create projector |0⟩_A⟨0| for shuttle
    proj_0_shuttle = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|

    # Extend to full system: |0⟩_A⟨0| ⊗ I_registers
    I_registers = np.eye(2**n_registers, dtype=complex)
    full_projector = np.kron(proj_0_shuttle, I_registers)

    # Apply projection: |0⟩_A⟨0| ρ |0⟩_A⟨0|
    projected_rho = full_projector @ rho @ full_projector

    # Calculate probability: Tr[|0⟩_A⟨0| ρ |0⟩_A⟨0|]
    prob = np.trace(projected_rho).real

    if prob > 1e-12:
        # Normalize: divide by probability
        normalized_projected_rho = projected_rho / prob

        # Partial trace over shuttle to get register density matrix
        # Reshape for partial trace: (dim_shuttle, dim_registers, dim_shuttle, dim_registers)
        reshaped = normalized_projected_rho.reshape(
            2, 2**n_registers, 2, 2**n_registers)

        # Trace over shuttle indices (0 and 2)
        rho_registers = np.trace(reshaped, axis1=0, axis2=2)

        return rho_registers, prob
    else:
        return np.zeros((2**n_registers, 2**n_registers), dtype=complex), 0.0


'''
Analysis and visualization functions
'''


def print_state(state):
    """Print the state in a readable format."""
    for i, amp in enumerate(state):
        if np.abs(amp) > 1e-6:  # Print only significant amplitudes
            print(f"|{i:05b}>: {amp:.4f}, magnitude: {np.abs(amp)**2:.4f}")
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
    plt.show()


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
    plt.show()


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
    plt.show()


def extract_w_coefficients(state, project_shuttle=True):
    """
    Extract the coefficients b_i from the state after projecting/tracing the shuttle

    The W-state is: |ψ_W⟩ = b_1|0001⟩ + b_2|0010⟩ + b_3|0100⟩ + b_4|1000⟩
    where the ordering is {r_1, r_2, s_1, s_2}

    For the [A, r1, r2, s1, s2] ordering:
    - Shuttle A is at bit position 4
    - Register qubits are at positions 3,2,1,0
    """
    if len(state) == 32:  # Full 5-qubit state
        if project_shuttle:
            # Project shuttle onto |0⟩ and extract register state
            reg_state = np.zeros(16, dtype=complex)
            for i in range(32):
                if ((i >> 4) & 1) == 0:  # Shuttle A in |0⟩ (bit 4)
                    # Extract the 4 register bits
                    i_red = i & 0b01111  # Get lower 4 bits (r1, r2, s1, s2)
                    reg_state[i_red] = state[i]
            # Normalize
            norm = np.linalg.norm(reg_state)
            if norm > 0:
                reg_state /= norm
    else:
        reg_state = state

    # Extract coefficients for single-excitation states
    # For register ordering [r1, r2, s1, s2]:
    # |0001⟩ = index 1 (s2 excited) -> b1
    # |0010⟩ = index 2 (s1 excited) -> b2
    # |0100⟩ = index 4 (r2 excited) -> b3
    # |1000⟩ = index 8 (r1 excited) -> b4

    b1 = reg_state[0b0001]  # s2
    b2 = reg_state[0b0010]  # s1
    b3 = reg_state[0b0100]  # r2
    b4 = reg_state[0b1000]  # r1

    return np.array([b1, b2, b3, b4])


def analyze_w_state_evolution(states, gamma_sh, gamma_in, max_rounds, x_vals=None):
    """Analyze and plot W-state coefficient evolution"""

    # Extract coefficients
    coeffs_evolution = []
    for state in states:
        coeffs = extract_w_coefficients(state, project_shuttle=True)
        coeffs_evolution.append(coeffs)
    coeffs_evolution = np.array(coeffs_evolution)

    # Plot
    plt.figure(figsize=(12, 8))
    colors = ['red', 'green', 'blue', 'orange']
    labels = ['|b₁| (s₂)', '|b₂| (s₁)', '|b₃| (r₂)', '|b₄| (r₁)']

    # Use provided x_vals or create default iteration numbers
    if x_vals is None:
        x_vals = np.arange(len(coeffs_evolution))

    for i in range(4):
        # Y data comes from the coefficients extracted from states
        y_data = np.abs(coeffs_evolution[:, i])

        # Plot scatter points with connecting lines
        plt.plot(x_vals, y_data, color=colors[i], linewidth=2, alpha=0.7)
        plt.scatter(x_vals, y_data,
                    color=colors[i], label=labels[i], s=30, alpha=0.8)

    plt.axhline(y=0.5, color='gray', linestyle='--',
                alpha=0.5, label='Perfect W-state')
    plt.xlabel('Round Number')
    plt.ylabel('|bᵢ|')
    plt.title(
        f'W-state Coefficient Evolution (γ_shuttle={gamma_sh}, γ_intra={gamma_in:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, max_rounds])
    plt.ylim([0, 0.8])
    plt.show()

    # Find and print when coefficients are close to 0.5
    # print("\nAnalyzing W-state formation:")
    # for idx in range(len(coeffs_evolution)):
    #     coeffs_mag = np.abs(coeffs_evolution[idx])
    #     if np.all(np.abs(coeffs_mag - 0.5) < 0.05):  # All close to 0.5
    #         print(
    #             f"Round {idx}: Near-perfect W-state with coefficients {coeffs_mag}")

    return coeffs_evolution


def main() -> None:
    # Initialize state |1>|0000> = |A=1, r1=0, r2=0, s1=0, s2=0>
    initial_state = np.kron(one, np.kron(
        zero, np.kron(zero, np.kron(zero, zero))))

    states = [initial_state.copy()]  # Store all states
    n_iters = [0]  # Track iteration numbers

    # Store register density matrices after post-selection
    register_density_matrices = []
    postsel_probs = []  # Track post-selection probabilities
    current_state = initial_state.copy()

    # Run simulation with operation-level tracking
    for round_num in range(1, max_rounds + 1, 2):
        current_state = perform_collision_round(
            current_state)  # Use the round function directly

        states.append(current_state.copy())
        n_iters.append(round_num)  # Half-step for first operation

        current_state = perform_collision_round_2(current_state)
        states.append(current_state.copy())
        n_iters.append(round_num + 1)

        rho_registers, prob = postselection_step(current_state)
        register_density_matrices.append(rho_registers.copy())
        postsel_probs.append(prob)

        # Progress tracking
        if round_num % 10 == 0:
            print(f"Completed round {round_num}")

    print("Simulation complete.")

    # Print first few states
    # for idx, state in enumerate(states[:55]):
    #     print(f"State after operation {idx}:")
    #     print_state(state)

    # plot_amplitude_evolution(register_density_matrices, list(range(1, max_rounds + 1)))

    coeffs_evolution = analyze_w_state_evolution(
        states, gamma_sh, gamma_in, max_rounds, x_vals=n_iters)


if __name__ == "__main__":
    main()
