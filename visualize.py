"""
Visualization and analysis functions for quantum state evolution.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from itertools import combinations
from state_analysis import extract_w_coefficients


def print_state(state):
    """Print quantum state in readable format."""
    n_qbits = int(np.log2(len(state)))
    for i, amp in enumerate(state):
        if np.abs(amp) > 1e-6:
            print(f"|{i:0{n_qbits}b}>: {amp:.4f}, magnitude: {np.abs(amp)**2:.4f}")
    print()


def plot_state_evolution(states, A_pos, n_registers, gamma_sh, gamma_in, 
                         max_rounds, x_vals=None, save_fig: bool = True):
    """
    Analyze and plot W-state coefficient evolution (with post-selection).
    
    Args:
        states: List of quantum states over time
        A_pos: List of ancilla positions
        n_registers: List of register qubit counts
        gamma_sh: Shuttle interaction strength
        gamma_in: Intra-register interaction strength
        max_rounds: Maximum number of rounds
        x_vals: X-axis values (defaults to iteration numbers)
        
    Returns:
        Array of coefficient evolution over time
    """
    # Extract coefficients
    coeffs_evolution = []
    for state in states:
        coeffs = extract_w_coefficients(state, A_pos, n_registers, project_shuttle=True)
        coeffs_evolution.append(coeffs)
    coeffs_evolution = np.array(coeffs_evolution)
    
    if len(coeffs_evolution) == 0:
        raise ValueError("No coefficient data to plot")
    
    num_terms = len(coeffs_evolution[0])
    num_ancillas = len(A_pos)
    total_register_qubits = sum(n_registers)
    
    # Create and save legend mapping
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
    
    for i, excitation_positions in enumerate(combinations(range(total_register_qubits), num_ancillas)):
        binary_str = ['0'] * total_register_qubits
        for pos in excitation_positions:
            binary_str[pos] = '1'
        
        if len(n_registers) == 2:
            r_part = ''.join(binary_str[:n_registers[0]])
            s_part = ''.join(binary_str[n_registers[0]:])
            detailed_label = f"|{r_part}⟩⊗|{s_part}⟩"
        else:
            positions_str = ','.join(map(str, excitation_positions))
            detailed_label = f"|exc@{positions_str}⟩"
        
        legend_data["coefficients"][f"b{i+1}"] = {
            "excitation_positions": list(excitation_positions),
            "binary_representation": ''.join(binary_str),
            "state_notation": detailed_label
        }
    
    if save_fig:
        with open('coefficient_legend.json', 'w') as f:
            json.dump(legend_data, f, indent=2)
        
        print(f"Legend saved to 'coefficient_legend.json'")
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    if x_vals is None:
        x_vals = np.arange(len(coeffs_evolution))
    
    for i in range(num_terms):
        y_data = np.abs(coeffs_evolution[:, i])
        plt.plot(x_vals, y_data, linewidth=2, alpha=0.7)
        plt.scatter(x_vals, y_data, s=30, alpha=0.8)
    
    # Perfect W-state amplitude
    perfect_w_amplitude = 1.0 / np.sqrt(num_terms)
    plt.axhline(y=perfect_w_amplitude, color='gray', linestyle='--',
                alpha=0.5, label=f'Perfect State (1/√{num_terms})')
    
    print(f'Number of terms: {num_terms}')
    print(f'Perfect amplitude: {perfect_w_amplitude:.4f}')
    
    plt.xlabel('Round Number')
    plt.ylabel('|bᵢ|')
    plt.title(f'State Coefficient Evolution (γ_shuttle={gamma_sh}, γ_intra={gamma_in:.3f})')
    plt.legend([f'Perfect State (1/√{num_terms})'])
    plt.grid(True, alpha=0.3)
    plt.xlim([0, max_rounds])
    plt.ylim([0, 0.8])
    
    if save_fig:
        plt.savefig('amplitude_evolution.png', dpi=300)
    plt.close()
    
    return coeffs_evolution


def plot_state_no_measure(states, A_pos, n_registers, gamma_sh, gamma_in, 
                          max_rounds, x_vals=None, save_fig=True):
    """
    Plot amplitude evolution without post-selection (full state analysis).
    Shows all basis states with significant population.
    
    Args:
        states: List of quantum states over time
        A_pos: List of ancilla positions
        n_registers: List of register qubit counts
        gamma_sh: Shuttle interaction strength
        gamma_in: Intra-register interaction strength
        max_rounds: Maximum number of rounds
        x_vals: X-axis values (defaults to iteration numbers)
        save_fig: Whether to save figure to file
        
    Returns:
        Array of amplitude evolution (shape: [num_rounds, 2^total_qubits])
    """
    num_ancillas = len(A_pos)
    total_register_qubits = sum(n_registers)
    total_qubits = num_ancillas + total_register_qubits
    
    # Extract amplitudes for all basis states
    amplitudes_evolution = np.array([np.abs(state) for state in states])
    
    # Find basis states with non-negligible population
    max_amps = np.max(amplitudes_evolution, axis=0)
    significant_indices = np.where(max_amps > 1e-3)[0]
    
    plt.figure(figsize=(14, 8))
    
    if x_vals is None:
        x_vals = np.arange(len(amplitudes_evolution))
    
    # Plot each significant basis state
    for idx in significant_indices:
        y_data = amplitudes_evolution[:, idx]
        plt.plot(x_vals, y_data, linewidth=2, alpha=0.7)
        plt.scatter(x_vals, y_data, s=20, alpha=0.6)
    
    plt.xlabel('Round Number')
    plt.ylabel('Amplitude |ψᵢ|')
    plt.title(f'State Evolution (No Post-selection) (γ_shuttle={gamma_sh}, γ_intra={gamma_in:.3f})')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, max_rounds])
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('amp_evol_no_postsel.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    plt.close()
    return amplitudes_evolution