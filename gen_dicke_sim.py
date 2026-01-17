"""
Collision Model Simulation for Dicke State Preparation
Main driver script with configuration.
"""
import os
from visualize import print_state, plot_state_evolution, plot_state_no_measure
from state_analysis import gen_init_state, postselection_step, get_amps, calc_fidelity, calc_uhlmann_fidelity
from collision_dynamics import perform_theoretical_round
from quantum_gates import partial_swap
from noise_config import NoiseConfig, NoiseModel, NO_NOISE, FAILED, DEPHASING, AMPLITUDE_DAMPING, DEPOLARIZING
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================
MAX_ROUNDS = 200                # Simulate rounds = 1..max_rounds
# Step size: 1 for theoretical round, 2^len(A_pos) for operation round
# 1: theoretical (KT round)
# 2^len(A_pos): operation (Cakmak round)
STEP = 1

# Interaction strengths
GAMMA_SHUTTLE = 2.61771574          # Shuttle-register partial-swap angle
# Intra-register partial-swap (0 = no interaction)
GAMMA_INTRA = 2.83478767
# GAMMA_INTRA = 0.95 * np.pi/2  # Strong intra-register interaction

# System configuration
N = 5 # Dicke N param
K = 2 # Dicke K param

A_POS = list(range(K))                     # Ancilla positions (0-indexed)
# Number of qubits in each register [R, S]
N_REGISTERS = [(N - K) // 2, (N - K) - (N - K) // 2]

# Collision scheme
SCHEME = 'sequential'           # 'sequential' or 'interleaved'

# Noise configuration
NOISE_CONFIG = AMPLITUDE_DAMPING  # Options: NO_NOISE, FAILED, DEPHASING, AMPLITUDE_DAMPING, DEPOLARIZING

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def generate_failed_interactions(max_rounds, n_registers, A_pos, shuttle_fail_prob, seed=42):
    '''
    Pre-generate which shuttle-register interactions will fail.
    
    Args:
        max_rounds: Total number of rounds
        n_registers: List of number of qubits in each register
        A_pos: List of ancilla positions
        shuttle_fail_prob: Fraction of interactions that should fail (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Set of (round_num, A, reg_collide_idx, register) tuples representing failed interactions
    '''
    np.random.seed(seed)
    max_reg_size = max(n_registers[0], n_registers[1])
    
    # Generate all possible interactions with round numbers
    all_interactions = []
    for round_num in range(1, max_rounds + 1):
        for A in A_pos:
            for reg_collide_idx in range(max_reg_size):
                # Check which registers exist for this collision index
                if reg_collide_idx < n_registers[0]:
                    all_interactions.append((round_num, A, reg_collide_idx, 'r'))
                if reg_collide_idx < n_registers[1]:
                    all_interactions.append((round_num, A, reg_collide_idx, 's'))
    
    # Calculate exact number of failures
    total_interactions = len(all_interactions)
    num_failures = int(total_interactions * shuttle_fail_prob)
    
    # Randomly select which interactions will fail
    failed_indices = np.random.choice(total_interactions, size=num_failures, replace=False)
    failed_interactions = set(all_interactions[i] for i in failed_indices)
    
    # uncomment for debugging when running gen_dicke_sim alone
    # print(f'Total shuttle-register interactions: {total_interactions}')
    # print(f'Failed interactions: {num_failures} ({100*num_failures/total_interactions:.2f}%)')
    
    return failed_interactions

def dicke_evolution(
    max_rounds: int = MAX_ROUNDS,
    step: int = STEP,
    gamma_sh: float = GAMMA_SHUTTLE,
    gamma_in: float = GAMMA_INTRA,
    A_pos: list = A_POS,
    n_registers: list = N_REGISTERS,
    scheme: str = SCHEME,
    noise_config: NoiseConfig = NO_NOISE,
    debug=False,
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
        noise_config: Noise configuration to apply during simulation

    Returns:
        Amplitude evolution array (no post-selection) for pure states
        OR list of density matrices for mixed states (with noise)
    """
    # Pre-generate failed interactions
    if noise_config.shuttle_fail_prob > 0:
        failed_interactions = generate_failed_interactions(
            max_rounds, n_registers, A_pos, 
            noise_config.shuttle_fail_prob, noise_config.seed)
    else:
        failed_interactions = set()
    
    # Create interaction unitaries
    U_shuttle = partial_swap(gamma_sh)
    U_intra = partial_swap(gamma_in)

    # Initialize state
    initial_state = gen_init_state(A_pos, n_registers)
    if verbose:
        print("Initial state:")
        print_state(initial_state)

    # Convert to density matrix if noise model requires it
    if noise_config.requires_density_matrix():
        initial_state = np.outer(initial_state, np.conj(initial_state))
        if verbose:
            print("Initial state: DENSITY MATRIX (quantum noise enabled)")
    elif verbose:
        print("Initial state: PURE STATE VECTOR")
        print_state(initial_state)
    
    if verbose:
        print(noise_config)
    
    # Storage for simulation
    states = [initial_state.copy()]
    n_iters = [0]
    cur_state = initial_state.copy()

    # Get channel parameters from noise config
    noise_channel_params = noise_config.get_channel_params()

    # Run collision model simulation
    if verbose:
        print(f"Running simulation: {max_rounds} rounds, scheme={scheme}")
    for round_num in range(1, max_rounds + 1, step):
        new_states, k_iters = perform_theoretical_round(
            cur_state, n_registers, A_pos, round_num, scheme,
            U_shuttle, U_intra, gamma_in, failed_interactions, noise_channel_params)

        cur_state = new_states[-1]
        states.extend(new_states)
        n_iters.extend(k_iters)

        # Progress tracking
        if verbose and round_num % 10 == 0:
            print(f"Completed round {round_num}")

    if verbose:
        print("Simulation complete.")

    amps_no_ps = np.array([state for state in states])
    # Save debug files and print info
    if debug or verbose:
        excitations = len(A_pos)
        qbits = excitations + sum(n_registers)
        
        if debug:
            os.makedirs('debug', exist_ok=True)
            suffix = '_dm_evol' if noise_config in [DEPHASING, AMPLITUDE_DAMPING, DEPOLARIZING] else '_evol'
            np.save(f'debug/D{qbits}_{excitations}{suffix}.npy', amps_no_ps)
            print(f"Saved amplitude evolution to debug/D{qbits}_{excitations}{suffix}.npy")
            
            if noise_config == NO_NOISE:
                plot_state_no_measure(states, A_pos, n_registers, gamma_sh,
                                    gamma_in, max_rounds, x_vals=n_iters, save_fig=save_fig)
        
        if verbose and noise_config == NO_NOISE:
            print(f"\nAmplitudes shape (no post-selection): {amps_no_ps.shape}")
            print(f"  - Rows: {amps_no_ps.shape[0]} time steps")
            print(f"  - Columns: {amps_no_ps.shape[1]} basis states (2^{qbits} total)")

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


def find_min_loss(amp_evols, ideal_amp):
    '''
    Find minimum loss across all rounds from pre-computed amplitude evolution.

    Args:
        amp_evols: Pre-computed amplitude evolution array from dicke_evolution()
        ideal_amp: ideal amplitude for Dicke state
        n_registers: List of number of qubits in each register
        A_pos: List of ancilla positions
        is_density_matrix: Whether the states are density matrices (with noise)

    Returns:
        tuple: (min_loss, best_round, losses)
            - min_loss: minimum loss value found
            - best_round: round number with minimum loss
            - losses: array of loss values at each round
    '''
    # Calculate loss at each round
    min_loss = float('inf')
    best_round = 0
    losses = []

    for r in range(amp_evols.shape[0]):
        amps = get_amps(amp_evols, r)
        ideal_state = np.array([ideal_amp] * len(amps))
        loss = 1 - calc_fidelity(ideal_state, amps, use_abs=True)

        if loss < min_loss:
            min_loss = loss
            best_round = r

        losses.append(loss)

    return min_loss, best_round, losses


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from math import comb

    # Run basic simulation (no post-selection analysis) - ONLY ONCE
    amp_evols = dicke_evolution(debug=True, verbose=False, save_fig=True, noise_config=NOISE_CONFIG)

    # Check if result is density matrices
    is_density_matrix = len(amp_evols[0].shape) == 2

    # Calculate losses for current configuration
    num_ex = len(A_POS)
    total_qbits = sum(N_REGISTERS) + num_ex
    num_term = comb(total_qbits, num_ex)
    ideal_term = 1/np.sqrt(num_term)
    
    # Only continue if pure state evolution
    if is_density_matrix:
        print("[ERROR] Current analysis only supports pure state evolution (no quantum noise).")
        exit(1)

    # Analyze the already-computed amp_evols
    min_loss, best_round, loss_vals = find_min_loss(
        amp_evols=amp_evols,
        ideal_amp=ideal_term,
    )

    # Get amplitudes at analysis round
    round_amps = get_amps(amp_evols, best_round)
    # round_amps = get_amps(amp_evols, best_round)
    ideal_state = np.array([ideal_term]*len(round_amps))
    fid = calc_fidelity(ideal_state, round_amps, use_abs=True)

    # ========================================================================
    # LOSS EVOLUTION PLOT
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Full loss evolution
    ax1.plot(loss_vals, linewidth=1.5, color='#2E86AB')
    ax1.axhline(y=min_loss, color='red', linestyle='--',
                label=f'Min loss: {min_loss:.6f} at round {best_round}')
    ax1.axvline(x=best_round, color='green', linestyle='--', alpha=0.7,
                label=f'Analysis round: {best_round}')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Loss (1 - Fidelity)', fontsize=12)
    ax1.set_title(f'Loss Evolution - γ_sh={GAMMA_SHUTTLE:.4f}, γ_in={GAMMA_INTRA:.4f}',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Zoomed around minimum
    window = 50
    start_idx = max(0, best_round - window)
    end_idx = min(len(loss_vals), best_round + window)

    ax2.plot(range(start_idx, end_idx), loss_vals[start_idx:end_idx],
             linewidth=2, color='#A23B72', marker='o', markersize=3)
    ax2.axhline(y=min_loss, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(x=best_round, color='orange', linestyle='--', alpha=0.7,
                label=f'Best round: {best_round}')
    if start_idx <= best_round <= end_idx:
        ax2.axvline(x=best_round, color='green', linestyle='--', alpha=0.7,
                    label=f'Analysis round: {best_round}')
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Loss (1 - Fidelity)', fontsize=12)
    ax2.set_title(f'Zoomed View: ±{window} rounds around minimum', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'./figures/loss_evolution_gamma_sh_{GAMMA_SHUTTLE:.4f}_gamma_in_{GAMMA_INTRA:.4f}.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    # Print loss statistics
    print(f'\n{"="*70}')
    print(f'LOSS EVOLUTION STATISTICS')
    print(f'{"="*70}')
    print(f'Min loss:          {min_loss:.8f}')
    print(f'Max loss:          {max(loss_vals):.8f}')
    print(f'Loss at round {best_round}:   {loss_vals[best_round-1]:.8f}')
    print(f'Loss at round {best_round}:   {loss_vals[best_round]:.8f}')
    print(f'Best round:        {best_round}')
    print(f'Mean loss:         {np.mean(loss_vals):.8f}')
    print(f'Std loss:          {np.std(loss_vals):.8f}')

    # Check for local minima
    local_minima = []
    for i in range(1, len(loss_vals) - 1):
        if loss_vals[i] < loss_vals[i-1] and loss_vals[i] < loss_vals[i+1]:
            local_minima.append((i, loss_vals[i]))

    if len(local_minima) > 1:
        print(f'\n[WARNING] Found {len(local_minima)} local minima:')
        for idx, loss in sorted(local_minima, key=lambda x: x[1])[:5]:
            print(f'  Round {idx}: loss = {loss:.8f}')
    else:
        print(f'\n[INFO] Single minimum detected - smooth convergence')

    # ========================================================================
    # IMPROVED POST-PROCESSING ANALYSIS
    # ========================================================================
    print(f'\n{"="*70}')
    print(f'ANALYSIS AT ROUND {best_round}')
    print(f'{"="*70}')
    print(f'Fidelity: {fid * 100:.6f}%')
    print(f'Number of terms: {len(round_amps)}')
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
    amp_abs = np.abs(round_amps)  # Take absolute value for complex amplitudes
    min_amp = np.floor(min(amp_abs) / bucket_size) * bucket_size
    max_amp = np.ceil(max(amp_abs) / bucket_size) * bucket_size

    # Create buckets
    buckets = {}
    for amp in amp_abs:
        bucket_key = np.floor(amp / bucket_size) * bucket_size
        buckets[bucket_key] = buckets.get(bucket_key, 0) + 1

    # Sort and display buckets with non-zero counts
    for bucket_start in sorted(buckets.keys()):
        bucket_end = bucket_start + bucket_size
        count = buckets[bucket_start]
        percentage = (count / len(amp_abs)) * 100

        # Create bar visualization
        bar_length = int(percentage * 0.5)
        bar = '█' * bar_length

        print(
            f'[{bucket_start:.2f}, {bucket_end:.2f}): {count:3d} ({percentage:5.1f}%) {bar}')
