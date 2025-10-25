'''
Collision model dynamics: intra-register and inter-register interactions.
'''
from quantum_gates import apply_two_qubit


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

    cur_bit = reg_qubits[index]

    # Left-ward interaction
    if index > 0:
        left_bit = reg_qubits[index - 1]
        state = apply_two_qubit(state, U_intra, cur_bit, left_bit, n_qbits)

    # Right-ward interaction
    if index < len(reg_qubits) - 1:
        right_bit = reg_qubits[index + 1]
        state = apply_two_qubit(state, U_intra, cur_bit, right_bit, n_qbits)

    return state


def perform_collision_round(state, n_registers, A, A_pos, n_qbits,
                            reg_collide_idx, U_shutter, U_intra, gamma_in):
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
    # Register qubit positions
    r_reg = list(range(len(A_pos), len(A_pos) + n_registers[0]))
    s_reg = [r_reg[-1] + 1 + i for i in range(n_registers[1])]

    # Check if we have a paired collision or unpaired collision
    has_r = reg_collide_idx < n_registers[0]
    has_s = reg_collide_idx < n_registers[1]

    if has_r:
        r = r_reg[reg_collide_idx]

        # (i) A-r interaction
        state = apply_two_qubit(state, U_shutter, A, r, n_qbits)

        # (ii) r intra-register collision
        if gamma_in != 0:
            r_index = reg_collide_idx
            state = perform_intra_register_collisions(
                state, r_index, r_reg, n_qbits, U_intra)

    if has_s:
        s = s_reg[reg_collide_idx]

        # (iii) A-s interaction
        state = apply_two_qubit(state, U_shutter, A, s, n_qbits)

        # (iv) s intra-register collision
        if gamma_in != 0:
            s_index = reg_collide_idx
            state = perform_intra_register_collisions(
                state, s_index, s_reg, n_qbits, U_intra)

    return state


def perform_theoretical_round(state, n_registers, A_pos, round_num, scheme,
                              U_shutter, U_intra, gamma_in):
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
                    reg_collide_idx, U_shutter, U_intra, gamma_in)
                
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
                    reg_collide_idx, U_shutter, U_intra, gamma_in)
                states.append(state.copy())
                k_iters.append(round_num + i)

    return states, k_iters

