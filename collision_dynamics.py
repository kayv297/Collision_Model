'''
Collision model dynamics: intra-register and inter-register interactions.
'''
from quantum_gates import apply_two_qubit
import numpy as np
from noise_channels import apply_single_qubit_channel, dephasing_channel, amplitude_damping_channel, depolarizing_channel


def apply_unitary_to_density_matrix(rho, U, qubit_indices, n_qbits):
    '''
    Apply a two-qubit unitary to a density matrix.
    
    Args:
        rho: Density matrix
        U: 4x4 unitary matrix
        qubit_indices: Tuple of (qubit1, qubit2) indices
        n_qbits: Total number of qubits
        
    Returns:
        Updated density matrix
    '''
    from quantum_gates import expand_two_qubit_gate
    
    # Expand U to full Hilbert space
    U_full = expand_two_qubit_gate(U, qubit_indices[0], qubit_indices[1], n_qbits)
    
    # Apply: ρ' = U ρ U†
    return U_full @ rho @ U_full.conj().T

def perform_intra_register_collisions(state, index, reg_qubits, n_qbits, U_intra):
    '''
    Perform intra-register nearest-neighbor collisions for a specific qubit.

    Args:
        state: Quantum state vector
        index: Index of current qubit within reg_qubits list
        reg_qubits: List of qubit indices in the register
        n_qbits: Total number of qubits in the system
        U_intra: Intra-register interaction unitary

    Returns:
        Updated state vector after intra-register collisions
    '''
    if len(reg_qubits) < 2:
        return state

    is_density_matrix = len(state.shape) == 2
    cur_bit = reg_qubits[index]

    # Left-ward interaction
    if index > 0:
        left_bit = reg_qubits[index - 1]
        if is_density_matrix:
            state = apply_unitary_to_density_matrix(state, U_intra, (cur_bit, left_bit), n_qbits)
        else:
            state = apply_two_qubit(state, U_intra, cur_bit, left_bit, n_qbits)

    # Right-ward interaction
    if index < len(reg_qubits) - 1:
        right_bit = reg_qubits[index + 1]
        if is_density_matrix:
            state = apply_unitary_to_density_matrix(state, U_intra, (cur_bit, right_bit), n_qbits)
        else:
            state = apply_two_qubit(state, U_intra, cur_bit, right_bit, n_qbits)

    return state


def perform_collision_round(state, n_registers, A, A_pos, n_qbits,
                            reg_collide_idx, U_shutter, U_intra, gamma_in,
                            failed_interactions, round_num, noise_channel_params=None):
    '''
    Perform one collision round: (i) A-r interaction, (ii) r intra-register,
    (iii) A-s interaction (if s exists), (iv) s intra-register (if s exists).

    Args:
        state: Current quantum state
        n_registers: List of number of qubits in each register
        A: Ancilla qubit index
        A_pos: List of all ancilla positions
        n_qbits: Total number of qubits
        reg_collide_idx: Index of register qubit to interact with
        U_shutter: Shuttle-register interaction unitary
        U_intra: Intra-register interaction unitary
        gamma_in: Intra-register interaction strength

    Returns:
        Updated state after collision round
    '''
    is_density_matrix = len(state.shape) == 2

    # Register qubit positions
    r_reg = list(range(len(A_pos), len(A_pos) + n_registers[0]))
    s_reg = [r_reg[-1] + 1 + i for i in range(n_registers[1])]

    # Check if we have a paired collision or unpaired collision
    has_r = reg_collide_idx < n_registers[0]
    has_s = reg_collide_idx < n_registers[1]

    if has_r:
        r = r_reg[reg_collide_idx]

        # (i) A-r interaction
        if (round_num, A, reg_collide_idx, 'r') not in failed_interactions:
            if is_density_matrix:
                state = apply_unitary_to_density_matrix(state, U_shutter, (A, r), n_qbits)
            else:
                state = apply_two_qubit(state, U_shutter, A, r, n_qbits)

        # (ii) Noise channel to r (if specified)
        if is_density_matrix:
            # Dephasing channel
            if noise_channel_params['dephasing_prob'] > 0:
                state = apply_single_qubit_channel(
                    state, dephasing_channel, r, n_qbits, 
                    p=noise_channel_params['dephasing_prob'])
            
            # Amplitude damping channel
            if noise_channel_params['amplitude_damping_gamma'] > 0:
                state = apply_single_qubit_channel(
                    state, amplitude_damping_channel, r, n_qbits,
                    gamma=noise_channel_params['amplitude_damping_gamma'])
            
            # Depolarizing channel
            if noise_channel_params['depolarizing_prob'] > 0:
                state = apply_single_qubit_channel(
                    state, depolarizing_channel, r, n_qbits,
                    p=noise_channel_params['depolarizing_prob'])

        # (iii) r intra-register collision
        if gamma_in != 0:
            r_index = reg_collide_idx
            state = perform_intra_register_collisions(
                state, r_index, r_reg, n_qbits, U_intra)

    if has_s:
        s = s_reg[reg_collide_idx]

        # (iv) A-s interaction
        if (round_num, A, reg_collide_idx, 's') not in failed_interactions:
            if is_density_matrix:
                state = apply_unitary_to_density_matrix(state, U_shutter, (A, s), n_qbits)
            else:
                state = apply_two_qubit(state, U_shutter, A, s, n_qbits)
        
        # (v) Noise channel to s (if specified)
        if is_density_matrix:
            # Dephasing channel
            if noise_channel_params['dephasing_prob'] > 0:
                state = apply_single_qubit_channel(
                    state, dephasing_channel, s, n_qbits, 
                    p=noise_channel_params['dephasing_prob'])
            
            # Amplitude damping channel
            if noise_channel_params['amplitude_damping_gamma'] > 0:
                state = apply_single_qubit_channel(
                    state, amplitude_damping_channel, s, n_qbits,
                    gamma=noise_channel_params['amplitude_damping_gamma'])
            
            # Depolarizing channel
            if noise_channel_params['depolarizing_prob'] > 0:
                state = apply_single_qubit_channel(
                    state, depolarizing_channel, s, n_qbits,
                    p=noise_channel_params['depolarizing_prob'])

        # (vi) s intra-register collision
        if gamma_in != 0:
            s_index = reg_collide_idx
            state = perform_intra_register_collisions(
                state, s_index, s_reg, n_qbits, U_intra)

    return state


def perform_theoretical_round(state, n_registers, A_pos, round_num, scheme,
                              U_shutter, U_intra, gamma_in, failed_interactions,
                              noise_channel_params=None):
    '''
    Perform one full theoretical collision round.

    Args:
        state: Current quantum state
        n_registers: List of number of qubits in each register
        A_pos: List of ancilla positions
        round_num: Current round number
        scheme: 'sequential' or 'interleaved'
        U_shutter: Shuttle-register interaction unitary
        U_intra: Intra-register interaction unitary
        gamma_in: Intra-register interaction strength

    Returns:
        states: List of intermediate states
        k_iters: List of iteration numbers
    '''
    n_qbits = sum(n_registers) + len(A_pos)
    states = []
    k_iters = []

    if scheme not in ['sequential', 'interleaved']:
        raise ValueError('Scheme must be \'sequential\' or \'interleaved\'')

    # Use the maximum register size to handle unpaired qubits
    max_reg_size = max(n_registers[0], n_registers[1])

    if scheme == 'sequential':
        for i, A in enumerate(A_pos):
            for reg_collide_idx in range(max_reg_size):
                state = perform_collision_round(
                    state, n_registers, A, A_pos, n_qbits,
                    reg_collide_idx, U_shutter, U_intra, gamma_in,
                    failed_interactions, round_num, noise_channel_params)
                
                #* Level 3: Collision Round
                # states.append(state.copy())
                # k_iters.append((round_num - 1) * len(A_pos) * max_reg_size + i * max_reg_size + reg_collide_idx + 1)
                
            #* Level 2: Ancilla Round
            states.append(state.copy())
            k_iters.append((round_num - 1) * len(A_pos) + i + 1)

        #* Level 1: Theoretical Round
        # states.append(state.copy())
        # k_iters.append(round_num + 1)
    else:  # Interleaved
        for i, reg_collide_idx in enumerate(range(max_reg_size)):
            for A in A_pos:
                state = perform_collision_round(
                    state, n_registers, A, A_pos, n_qbits,
                    reg_collide_idx, U_shutter, U_intra, gamma_in,
                    failed_interactions, round_num, noise_channel_params)
                states.append(state.copy())
                k_iters.append(round_num + i)

    return states, k_iters

