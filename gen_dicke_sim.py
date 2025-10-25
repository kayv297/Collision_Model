"""
Collision Model Simulation for Dicke State Preparation
Main driver script with configuration.
"""
from visualize import print_state, plot_state_evolution, plot_state_no_measure
from state_analysis import gen_init_state, postselection_step, get_amps, calc_fidelity
from collision_dynamics import perform_theoretical_round
from quantum_gates import partial_swap


# ============================================================================
# CONFIGURATION
# ============================================================================
MAX_ROUNDS = 600                # Simulate rounds = 1..max_rounds
# Step size: 1 for theoretical round, 2^len(A_pos) for operation round
# 1: theoretical (KT round)
# 2^len(A_pos): operation (Cakmak round)
STEP = 1

# Interaction strengths
GAMMA_SHUTTLE = 6.25972892            # Shuttle-register partial-swap angle
# Intra-register partial-swap (0 = no interaction)
GAMMA_INTRA = 5.47100077
# GAMMA_INTRA = 0.95 * np.pi/2  # Strong intra-register interaction

# System configuration
A_POS = [0, 1]                     # Ancilla positions (0-indexed)
N_REGISTERS = [4, 4]            # Number of qubits in each register [R, S]

# Collision scheme
SCHEME = 'sequential'           # 'sequential' or 'interleaved'


# Config for post-simluation analysis
c_round = 187  # either c_round or c_round -1 fix the problem, double check with plot


# ============================================================================
# MAIN SIMULATION
# ============================================================================
def dicke_evolution(
    max_rounds: int = MAX_ROUNDS,
    step: int = STEP,
    gamma_sh: float = GAMMA_SHUTTLE,
    gamma_in: float = GAMMA_INTRA,
    A_pos: list = A_POS,
    n_registers: list = N_REGISTERS,
    scheme: str = SCHEME,
    save_fig=False,
    verbose=False
):
    """
    Main simulation function for Dicke state preparation via collision model.

    This function:
    1. Initializes the system with ancillas in |1⟩ and registers in |0⟩
    2. Runs collision rounds with shuttle-register and intra-register interactions
    3. Tracks state evolution before post-selection
    4. Visualizes amplitude evolution

    Args:
        save_fig: Whether to save figures to files
        verbose: Whether to print detailed progress

    Returns:
        Amplitude evolution array (no post-selection)
    """
    # Create interaction unitaries
    U_shuttle = partial_swap(gamma_sh)
    U_intra = partial_swap(gamma_in)

    # Initialize state
    initial_state = gen_init_state(A_pos, n_registers)
    if verbose:
        print("Initial state:")
        print_state(initial_state)

    # Storage for simulation
    states = [initial_state.copy()]
    n_iters = [0]
    cur_state = initial_state.copy()

    # Run collision model simulation
    if verbose:
        print(f"Running simulation: {max_rounds} rounds, scheme={scheme}")
    for round_num in range(1, max_rounds + 1, step):
        new_states, k_iters = perform_theoretical_round(
            cur_state, n_registers, A_pos, round_num, scheme,
            U_shuttle, U_intra, gamma_in)

        cur_state = new_states[-1]
        states.extend(new_states)
        n_iters.extend(k_iters)

        # Progress tracking
        if verbose and round_num % 10 == 0:
            print(f"Completed round {round_num}")

    if verbose:
        print("Simulation complete.")

    # Visualize evolution (no post-selection)
    amps_no_ps = plot_state_no_measure(
        states, A_pos, n_registers, gamma_sh, gamma_in,
        max_rounds, x_vals=n_iters, save_fig=save_fig)

    if verbose:
        print(f"\nAmplitudes shape (no post-selection): {amps_no_ps.shape}")
        print(f"  - Rows: {amps_no_ps.shape[0]} time steps")
        print(
            f"  - Columns: {amps_no_ps.shape[1]} basis states (2^{len(A_pos) + sum(n_registers)} total)")

    return amps_no_ps


def dicke_evolution_postsel(save_fig=True):
    """
    Simulation with post-selection analysis (following paper equation 7).

    This function additionally performs post-selection on ancillas and
    visualizes the W-state coefficient evolution.

    Args:
        save_fig: Whether to save figures to files

    Returns:
        Tuple of (coefficients_evolution, amplitudes_no_postselection)
    """
    # Create interaction unitaries
    U_shuttle = partial_swap(GAMMA_SHUTTLE)
    U_intra = partial_swap(GAMMA_INTRA)

    # Initialize state
    initial_state = gen_init_state(A_POS, N_REGISTERS)

    # Storage
    states = [initial_state.copy()]
    n_iters = [0]
    register_density_matrices = []
    postsel_probs = []
    cur_state = initial_state.copy()

    # Run simulation
    for round_num in range(1, MAX_ROUNDS + 1, STEP):
        new_states, k_iters = perform_theoretical_round(
            cur_state, N_REGISTERS, A_POS, round_num, SCHEME,
            U_shuttle, U_intra, GAMMA_INTRA)

        cur_state = new_states[-1]
        states.extend(new_states)
        n_iters.extend(k_iters)

        # Post-selection
        rho_registers, prob = postselection_step(cur_state, A_POS, N_REGISTERS)
        register_density_matrices.append(rho_registers.copy())
        postsel_probs.append(prob)

        if round_num % 10 == 0:
            print(f"Round {round_num}, post-selection prob: {prob:.6f}")

    print("Simulation complete.")

    # Visualize both with and without post-selection
    coeffs_evolution = plot_state_evolution(
        states, A_POS, N_REGISTERS, GAMMA_SHUTTLE, GAMMA_INTRA,
        MAX_ROUNDS, x_vals=n_iters)

    amps_no_ps = plot_state_no_measure(
        states, A_POS, N_REGISTERS, GAMMA_SHUTTLE, GAMMA_INTRA,
        MAX_ROUNDS, x_vals=n_iters, save_fig=save_fig)

    return coeffs_evolution, amps_no_ps


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import numpy as np
    from collections import Counter

    # Run basic simulation (no post-selection analysis)
    amp_evols = dicke_evolution(verbose=False, save_fig=True)

    round_amps = get_amps(amp_evols, c_round - 1)

    num_term = len(round_amps)
    ideal_term = 1/np.sqrt(num_term)
    ideal_state = np.array([ideal_term]*num_term)

    fid = calc_fidelity(ideal_state, round_amps)

    # ========================================================================
    # IMPROVED POST-PROCESSING ANALYSIS
    # ========================================================================
    print(f'\n{"="*70}')
    print(f'ANALYSIS AT ROUND {c_round}')
    print(f'{"="*70}')
    print(f'Fidelity: {fid * 100:.6f}%')
    print(f'Number of terms: {num_term}')
    print(f'Perfect Amplitude: {ideal_term:.6f}')
    print(f'--- Amplitude ---')
    print(f'(Min, Max) = ({np.min(round_amps):.6f}, {np.max(round_amps):.6f})')
    print(
        f'(Mean, Std deviation): ({np.mean(round_amps):.6f}, {np.std(round_amps):.6f})')

    # Histogram analysis with 0.01 buckets
    print(f'\n{"-"*70}')
    print(f'AMPLITUDE DISTRIBUTION (bucket size = 0.01)')
    print(f'{"-"*70}')

    bucket_size = 0.01
    min_amp = np.floor(min(round_amps) / bucket_size) * bucket_size
    max_amp = np.ceil(max(round_amps) / bucket_size) * bucket_size

    # Create buckets
    buckets = {}
    for amp in round_amps:
        bucket_key = np.floor(amp / bucket_size) * bucket_size
        buckets[bucket_key] = buckets.get(bucket_key, 0) + 1

    # Sort and display buckets with non-zero counts
    for bucket_start in sorted(buckets.keys()):
        bucket_end = bucket_start + bucket_size
        count = buckets[bucket_start]
        percentage = (count / num_term) * 100

        # Create bar visualization
        bar_length = int(percentage * 0.5)  # Scale factor for display
        bar = '█' * bar_length

        print(
            f'[{bucket_start:.2f}, {bucket_end:.2f}): {count:3d} ({percentage:5.1f}%) {bar}')

    # Optional: Print all individual amplitudes if needed
    # print(f'\n{"-"*70}')
    # print(f'INDIVIDUAL AMPLITUDES (sorted)')
    # print(f'{"-"*70}')
    # sorted_amps = sorted(enumerate(round_amps), key=lambda x: x[1], reverse=True)
    # for i, amp in sorted_amps:
    #     deviation = ((amp - ideal_term) / ideal_term) * 100
    #     print(f'  [{i:2d}] {amp:.6f}  (ideal {deviation:+6.2f}%)')

    # Uncomment to run with post-selection analysis:
    # dicke_evolution_postsel()
