"""
Animated analysis of collision model across multiple gamma configurations.
Visualizes loss evolution as parameters change.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from gen_dicke_sim import dicke_evolution, find_min_loss, A_POS, N_REGISTERS, MAX_ROUNDS, STEP, SCHEME
from math import comb
from tqdm import tqdm


def run_config_sweep(gamma_in_list, gamma_sh_list, verbose=True):
    """
    Run simulation for multiple gamma configurations.

    Args:
        gamma_in_list: List of gamma_in values to test
        gamma_sh_list: List of gamma_sh values to test
        verbose: Show progress bar

    Returns:
        list of dicts: Each dict contains {gamma_in, gamma_sh, loss_vals, min_loss, best_round}
    """
    results = []

    # Calculate ideal amplitude
    num_ex = len(A_POS)
    total_qbits = sum(N_REGISTERS) + num_ex
    num_term = comb(total_qbits, num_ex)
    ideal_term = 1/np.sqrt(num_term)

    iterator = zip(gamma_in_list, gamma_sh_list)
    if verbose:
        iterator = tqdm(list(iterator), desc="Running configs")

    for gamma_in, gamma_sh in iterator:
        # Run simulation
        amp_evols = dicke_evolution(
            verbose=False,
            save_fig=False,
            gamma_in=gamma_in,
            gamma_sh=gamma_sh,
            A_pos=A_POS,
            n_registers=N_REGISTERS,
            max_rounds=MAX_ROUNDS,
            step=STEP,
            scheme=SCHEME
        )

        # Analyze loss
        min_loss, best_round, loss_vals = find_min_loss(
            amp_evols=amp_evols,
            ideal_amp=ideal_term,
        )

        results.append({
            'gamma_in': gamma_in,
            'gamma_sh': gamma_sh,
            'loss_vals': loss_vals,
            'min_loss': min_loss,
            'best_round': best_round
        })

    return results


def create_static_comparison(results, save_path='./figures/gamma_comparison.png'):
    """
    Create static comparison plot of all configurations.
    """
    n_configs = len(results)
    
    # Calculate grid dimensions dynamically
    n_cols = 5
    n_rows = (n_configs + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    # Flatten axes for easier indexing (handle both 1D and 2D arrays)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, res in enumerate(results):
        ax = axes[idx]
        loss_vals = res['loss_vals']
        best_round = res['best_round']
        min_loss = res['min_loss']

        # Plot loss evolution
        ax.plot(loss_vals, linewidth=1.5, color='#2E86AB', alpha=0.8)
        ax.axhline(y=min_loss, color='red',
                   linestyle='--', alpha=0.6, linewidth=1)
        ax.axvline(x=best_round, color='green',
                   linestyle='--', alpha=0.6, linewidth=1)

        ax.set_title(f"γ_in={res['gamma_in']:.3f}\nγ_sh={res['gamma_sh']:.3f}",
                     fontsize=10)
        ax.set_xlabel('Round', fontsize=9)
        ax.set_ylabel('Loss', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Add min loss text
        ax.text(0.95, 0.95, f'Min: {min_loss:.4f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved static comparison to {save_path}")
    plt.close()


def create_trajectory_plot(results, save_path='./figures/gamma_trajectory.png'):
    """
    Plot trajectory of min_loss in gamma parameter space.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    gamma_ins = [r['gamma_in'] for r in results]
    gamma_shs = [r['gamma_sh'] for r in results]
    min_losses = [r['min_loss'] for r in results]

    # Left: Parameter trajectory
    scatter = ax1.scatter(gamma_ins, gamma_shs, c=min_losses,
                          s=150, cmap='viridis', edgecolors='black', linewidth=1.5)

    # Connect points with lines
    for i in range(len(results)-1):
        ax1.plot([gamma_ins[i], gamma_ins[i+1]],
                 [gamma_shs[i], gamma_shs[i+1]],
                 'k--', alpha=0.3, linewidth=1)

    # Mark start and end
    ax1.plot(gamma_ins[0], gamma_shs[0], 'go', markersize=12,
             label='Start', markeredgecolor='black', markeredgewidth=2)
    ax1.plot(gamma_ins[-1], gamma_shs[-1], 'r*', markersize=15,
             label='End', markeredgecolor='black', markeredgewidth=2)

    ax1.set_xlabel('γ_intra', fontsize=12)
    ax1.set_ylabel('γ_shuttle', fontsize=12)
    ax1.set_title('Parameter Space Trajectory', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Min Loss', fontsize=11)

    # Right: Loss progression
    ax2.plot(range(len(min_losses)), min_losses, 'o-',
             linewidth=2, markersize=8, color='#2E86AB')
    ax2.set_xlabel('Configuration Index', fontsize=12)
    ax2.set_ylabel('Minimum Loss', fontsize=12)
    ax2.set_title('Loss Progression Across Configs',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Mark best config
    best_idx = np.argmin(min_losses)
    ax2.plot(best_idx, min_losses[best_idx], 'r*', markersize=20,
             label=f'Best (config {best_idx+1})')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved trajectory plot to {save_path}")
    plt.close()


def create_individual_plots(results, output_dir='./figures/configs'):
    """
    Create individual plot for each configuration and save separately.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Find global min/max for consistent y-axis across all plots
    all_losses = [res['loss_vals'] for res in results]
    global_min = min(min(losses) for losses in all_losses)
    global_max = max(max(losses) for losses in all_losses)
    y_margin = (global_max - global_min) * 0.1

    print(f"Creating individual plots in {output_dir}/")
    for idx, res in enumerate(tqdm(results, desc="Generating plots")):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        loss_vals = res['loss_vals']
        best_round = res['best_round']
        min_loss = res['min_loss']

        # Left plot: Full loss evolution
        ax1.plot(loss_vals, linewidth=2, color='#2E86AB', label='Loss')
        ax1.axhline(y=min_loss, color='red', linestyle='--', alpha=0.7,
                    label=f'Min: {min_loss:.6f}')
        ax1.axvline(x=best_round, color='green', linestyle='--', alpha=0.7,
                    label=f'Best round: {best_round}')

        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Loss (1 - Fidelity)', fontsize=12)
        ax1.set_title(f"Config {idx+1}/{len(results)}\n" +
                      f"γ_in = {res['gamma_in']:.4f}, γ_sh = {res['gamma_sh']:.4f}",
                      fontsize=14, fontweight='bold')
        ax1.set_ylim(global_min - y_margin, global_max + y_margin)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # Right plot: Zoomed around minimum
        window = 50
        start_idx = max(0, best_round - window)
        end_idx = min(len(loss_vals), best_round + window)

        rounds_zoom = range(start_idx, end_idx)
        losses_zoom = loss_vals[start_idx:end_idx]

        ax2.plot(rounds_zoom, losses_zoom, linewidth=2, color='#A23B72',
                 marker='o', markersize=4, alpha=0.8)
        ax2.axhline(y=min_loss, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(x=best_round, color='orange', linestyle='--', alpha=0.7)

        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Loss (1 - Fidelity)', fontsize=12)
        ax2.set_title(f'Zoomed: ±{window} rounds around minimum', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with descriptive filename
        filename = f'config_{idx+1:02d}_gin_{res["gamma_in"]:.4f}_gsh_{res["gamma_sh"]:.4f}.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(results)} individual plots to {output_dir}/")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import os

    # Create output directory
    os.makedirs('./figures', exist_ok=True)

    # ========================================================================
    # DEFINE CONFIGURATIONS
    # ========================================================================
    gamma_in_list = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10]
    gamma_sh_list = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

    print(f"Testing {len(gamma_in_list)} configurations...")
    print(
        f"γ_intra range: [{min(gamma_in_list):.3f}, {max(gamma_in_list):.3f}]")
    print(
        f"γ_shuttle range: [{min(gamma_sh_list):.3f}, {max(gamma_sh_list):.3f}]")

    # ========================================================================
    # RUN SIMULATIONS
    # ========================================================================
    results = run_config_sweep(gamma_in_list, gamma_sh_list, verbose=True)

    # ========================================================================
    # CREATE VISUALIZATIONS
    # ========================================================================
    print("\nGenerating visualizations...")

    # 1. Static comparison grid
    create_static_comparison(results)

    # 2. Individual config plots (REPLACED ANIMATION)
    create_individual_plots(results)

    # 3. Parameter trajectory
    create_trajectory_plot(results)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("CONFIGURATION SWEEP SUMMARY")
    print("="*70)

    for idx, res in enumerate(results):
        print(f"Config {idx+1}: γ_in={res['gamma_in']:.4f}, "
              f"γ_sh={res['gamma_sh']:.4f} → "
              f"Min Loss={res['min_loss']:.6f} (round {res['best_round']})")

    # Find best configuration
    best_idx = np.argmin([r['min_loss'] for r in results])
    best_res = results[best_idx]
    print(f"\n{'='*70}")
    print(f"BEST CONFIGURATION: Config {best_idx+1}")
    print(f"  γ_intra   = {best_res['gamma_in']:.6f}")
    print(f"  γ_shuttle = {best_res['gamma_sh']:.6f}")
    print(f"  Min Loss  = {best_res['min_loss']:.8f}")
    print(f"  Best Round = {best_res['best_round']}")
    print(f"{'='*70}")
