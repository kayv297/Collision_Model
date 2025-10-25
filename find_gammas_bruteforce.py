'''
This file run the Dicke Simulation to find the optimal gamma_intra and gamma_shuttle that maximizes the fidelity of the generated Dicke state with the ideal Dicke state (brute-force)

For now, not consider fidelity yet, but find the difference in the ideal amplitudes vs the means of the generated amplitudes of all terms every round.
'''
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from gen_dicke_sim import dicke_evolution
from state_analysis import get_amps
from math import comb
import numpy as np
import time


def calc_sum_dist(ideal_amp: float, amps: List[float]) -> float:
    '''
    Find sum of abs distance between ideal amplitude and generated amplitudes.
    Return distance. The lower the distance, the better the similarity (0 = perfect, higher = worse).
    '''
    # Find the distance between the ideal amplitude to each of the generated amplitudes
    distances = [abs(ideal_amp - amp) for amp in amps]

    # Calculate the sum distance (precision error)
    sum_dist = np.sum(distances)

    return sum_dist


def calc_cluster(ideal_amp: float, amps: List[float]) -> float:
    '''
    Find the cluster centroid of generated amplitudes. Then find distance between ideal amplitude and cluster centroid. This distance is also a measure of similarity (lower = better).

    Step 1: Normalize all amplitudes: Find dist between amps and ideal_amp. For all negative distances, flip symetrically through the ideap_amp line by adding 2*dist to the amp. This way, all amplitudes are on one side of the ideal_amp line.

    Step 2: Find the centroid of the normalized amplitudes. Achieve by finding the mean of the normalized amplitudes.

    Step 3: Find the distance between the ideal_amp and the centroid. Return this distance and the centroid.
    '''
    # Step 1: Normalize all amplitudes
    norm_amps = []
    for amp in amps:
        dist = amp - ideal_amp
        norm_amp = amp
        if dist < 0:
            norm_amp = amp + 2 * abs(dist)

        norm_amps.append(norm_amp)

    # Step 2: Find the centroid of the normalized amplitudes
    centroid = np.mean(norm_amps)

    # Step 3: Find the distance between the ideal_amp and the centroid
    centroid_dist = abs(ideal_amp - centroid)

    return centroid_dist


def process_gamma_in(args: Tuple) -> Tuple[float, float, float, int, List[float]]:
    '''
    Process a single gamma_in value by searching over all gamma_sh values.
    Returns: (gamma_in, gamma_sh, min_dist, best_round, best_amps)
    '''
    gamma_in, gamma_sh_values, perfect_w_amplitude, a_pos, n_registers, max_rounds, step = args

    overall_min_dist = float('inf')  # Start with infinity, we want to minimize
    overall_best_round = 0
    overall_best_gamma_sh = 0
    overall_best_amps = []

    for gamma_sh in gamma_sh_values:
        amps_evol = dicke_evolution(
            save_fig=False,
            verbose=False,
            gamma_in=float(gamma_in),
            gamma_sh=float(gamma_sh),
            A_pos=a_pos,
            n_registers=n_registers,
            max_rounds=max_rounds,
            step=step
        )

        min_dist = float('inf')  # Start with infinity for this gamma_sh
        best_round = 0
        best_amps = []
        for round_num in range(max_rounds):
            amps = get_amps(amps_evol, round_num)
            dist = calc_sum_dist(perfect_w_amplitude, amps)
            if dist < min_dist:  # Lower distance is better
                min_dist = dist
                best_round = round_num
                best_amps = amps

        if min_dist < overall_min_dist:  # Lower distance is better
            overall_min_dist = min_dist
            overall_best_round = best_round
            overall_best_gamma_sh = gamma_sh
            overall_best_amps = best_amps

    return (gamma_in, overall_best_gamma_sh, overall_min_dist, overall_best_round, overall_best_amps)


def main() -> None:
    '''
    In non-postel simulation, predefine config with some properties:
    - Total output qbits = sum(n_registers) + num_ancillas
    - Total excitations = num_ancillas
    '''
    n_registers = [3, 3]  # Two registers with 2 qubits each
    a_pos = [0, 1]  # Ancilla at position 0

    num_ex = len(a_pos)  # Number of excitations = number of ancillas
    # Total qubits = register qubits + ancillas
    total_qbits = sum(n_registers) + num_ex

    # calculate number of terms for ideal Dicke state
    num_terms = comb(total_qbits, num_ex)
    perfect_w_amplitude = 1 / np.sqrt(num_terms)

    print(
        f'Perfect amplitude: {perfect_w_amplitude:.4f}, num_terms: {num_terms}')

    # define gamma_in, step, and max_rounds
    max_rounds = 500  # Total number of collision rounds
    step = 1  # theoretical round

    # Search over gamma_sh values
    gamma_in_values = np.arange(0.0, np.pi / 2, 0.01)  # step = 0.01
    gamma_sh_values = np.arange(0.01, np.pi / 2, 0.01)  # step = 0.01

    print(f"Total gamma_in values to process: {len(gamma_in_values)}")
    print(f"Gamma_sh values per gamma_in: {len(gamma_sh_values)}")
    print(f"Total simulations: {len(gamma_in_values) * len(gamma_sh_values)}")

    # Prepare arguments for parallel processing
    args_list = [
        (gamma_in, gamma_sh_values, perfect_w_amplitude,
         a_pos, n_registers, max_rounds, step)
        for gamma_in in gamma_in_values
    ]

    # Use all available CPUs minus 1 (to keep system responsive)
    num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} processes")
    print("Starting computation...\n")

    # Process gamma_in values in parallel
    overall_min_dist = float('inf')  # Start with infinity, we want to minimize
    overall_best_round = 0
    overall_best_gamma_in = 0
    overall_best_gamma_sh = 0
    overall_best_amps = []

    start_time = time.time()
    completed = 0
    total = len(args_list)

    # Use imap_unordered for better performance
    with Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(process_gamma_in, args_list, chunksize=1):
            completed += 1
            gamma_in, gamma_sh, min_dist, best_round, best_amps = result

            # Update overall best (lower distance is better)
            if min_dist < overall_min_dist:
                overall_min_dist = min_dist
                overall_best_round = best_round
                overall_best_gamma_in = gamma_in
                overall_best_gamma_sh = gamma_sh
                overall_best_amps = best_amps

            # Print progress every 5% or every 5 completions for small totals
            print_interval = max(5, total // 20)
            if completed % print_interval == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                progress_pct = 100 * completed / total

                print(f"[{completed:3d}/{total}] {progress_pct:5.1f}% | "
                      f"Elapsed: {elapsed:6.1f}s | "
                      f"ETA: {eta:6.1f}s | "
                      f"Rate: {rate:.2f} it/s | "
                      f"Best dist: {overall_min_dist:.4f}")

    elapsed_total = time.time() - start_time
    print(f"\nCompleted in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

    print("\n" + "="*60)
    print("Overall best result (lowest distance):")
    print(f"  Distance:   {overall_min_dist:.4f}")
    print(f"  gamma_in:   {overall_best_gamma_in:.4f}")
    print(f"  gamma_sh:   {overall_best_gamma_sh:.4f}")
    print(f"  Round:      {overall_best_round}")
    print("="*60)

    # Save best rounds data to a file
    with open("best_rounds.txt", "a") as f:
        f.write(f"{'='*60}\n")
        f.write(f"gamma_in:    {overall_best_gamma_in:.4f}\n")
        f.write(f"gamma_sh:    {overall_best_gamma_sh:.4f}\n")
        f.write(f"best_round:  {overall_best_round}\n")
        f.write(f"distance:    {overall_min_dist:.4f} (lower is better)\n")
        f.write(f"{'-'*60}\n")
        f.write(f"Best amplitudes (n={len(overall_best_amps)}):\n")
        for i, amp in enumerate(overall_best_amps):
            f.write(f"  [{i:2d}] {amp:.6f}\n")
        f.write(f"{'='*60}\n\n")


if __name__ == "__main__":
    main()
