'''
Gradient-based optimization for quantum gate parameters using L-BFGS-B.
Parallelized across multiple initial guesses to find global optimum.
WITH CHECKPOINTING AND ERROR RECOVERY
'''
from scipy.optimize import minimize
from gen_dicke_sim import dicke_evolution
from state_analysis import get_amps, calc_fidelity
from math import comb
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import time
import os
import signal
import sys


class OptResult:
    '''Container for optimization results'''

    def __init__(self, init_guess, opt_params, loss, best_round, success):
        self.init_guess = init_guess
        self.opt_params = opt_params
        self.loss = loss
        self.best_round = best_round
        self.success = success


def eval_loss(params, round_num, ideal_amp, a_pos, n_reg, step):
    '''
    Evaluate L1 loss at specific round.

    Returns:
        float: L1 distance from ideal Dicke state
    '''
    gamma_in, gamma_sh = params

    amps_evol = dicke_evolution(
        save_fig=False,
        verbose=False,
        gamma_in=float(gamma_in),
        gamma_sh=float(gamma_sh),
        A_pos=a_pos,
        n_registers=n_reg,
        max_rounds=round_num + 1,
        step=step
    )

    amps = get_amps(amps_evol, round_num)

    ideal_state = np.array([ideal_amp] * len(amps))
    return 1 - calc_fidelity(ideal_state, amps)
    # return calc_l1(ideal_amp, amps)


def find_min_loss(params, ideal_amp, a_pos, n_reg, max_rounds, step):
    '''
    Find minimum L1 loss across all rounds.

    Returns:
        tuple: (min_loss, best_round)
    '''
    gamma_in, gamma_sh = params

    # Bounds check
    if not (0 <= gamma_in <= 2*np.pi) or not (0.01 <= gamma_sh <= 2*np.pi):
        return 1e10, 0

    # Run simulation
    amps_evol = dicke_evolution(
        save_fig=False,
        verbose=False,
        gamma_in=float(gamma_in),
        gamma_sh=float(gamma_sh),
        A_pos=a_pos,
        n_registers=n_reg,
        max_rounds=max_rounds,
        step=step
    )

    # Find best round
    min_loss = float('inf')
    best_round = 0

    for r in range(max_rounds):
        amps = get_amps(amps_evol, r)

        ideal_state = np.array([ideal_amp] * len(amps))
        loss = 1 - calc_fidelity(ideal_state, amps)
        # loss = calc_l1(ideal_amp, amps)

        if loss < min_loss:
            min_loss = loss
            best_round = r

    return min_loss, best_round


def opt_worker(args):
    '''
    Worker: optimize from single initial guess via L-BFGS-B.
    WITH ERROR HANDLING

    Args:
        args: (init_guess, bounds, config)

    Returns:
        OptResult: optimization result
    '''
    init_guess, bounds, cfg = args

    try:
        # Unpack config
        ideal_amp = cfg['ideal_amp']
        a_pos = cfg['a_pos']
        n_reg = cfg['n_reg']
        max_rounds = cfg['max_rounds']
        step = cfg['step']
        max_iter = cfg['max_iter']
        tol = cfg['tol']

        # Objective function
        def obj(params):
            loss, _ = find_min_loss(params, ideal_amp, a_pos,
                                    n_reg, max_rounds, step)
            return loss

        # Run L-BFGS-B
        result = minimize(
            obj,
            x0=init_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol}
        )

        # Get final stats
        final_loss, best_round = find_min_loss(
            result.x, ideal_amp, a_pos, n_reg, max_rounds, step
        )

        return OptResult(
            init_guess=init_guess,
            opt_params=result.x,
            loss=final_loss,
            best_round=best_round + 1,  # Normalize to 1-based indexing
            success=result.success
        )

    except KeyboardInterrupt:
        # Propagate interrupt
        raise
    except Exception as e:
        # Log error but continue
        print(f"\n[ERROR] Worker failed on init {init_guess}: {e}")
        return OptResult(
            init_guess=init_guess,
            opt_params=init_guess,
            loss=float('inf'),
            best_round=0,
            success=False
        )


def gen_grid(gamma_in_range, gamma_sh_range):
    '''Generate grid of initial guesses.'''
    return [[gi, gs] for gi in gamma_in_range for gs in gamma_sh_range]


def load_checkpoint(checkpoint_file):
    '''Load processed initial guesses from checkpoint - FIXED VERSION'''
    processed = set()
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            header = next(f)  # Read header

            # Check format
            if 'init_gamma_in' not in header:
                raise ValueError("Checkpoint file format not recognized")

            for line in f:
                try:
                    parts = line.strip().split(',')
                    init_gi, init_gs = float(parts[0]), float(parts[1])
                    processed.add((round(init_gi, 8), round(init_gs, 8)))
                except:
                    continue
    return processed


def main(n_reg = None, a_pos = None):
    # ========== Config ==========
    # Quantum system parameters
    
    if n_reg is None:
        n_reg = [2, 2]
        
    if a_pos is None:
        a_pos = [0, 1, 2]

    max_rounds = 200 * len(a_pos) # normalize by number of ancillas. 200 is num_theoretical round
    step = 1

    # Optimizer
    max_iter = 150  # Maximum number of iterations for the optimizer
    tol = 1e-6  # Tolerance for optimizer convergence
    chunk_size = 15

    # Search space
    bounds = [(0.0, np.pi), (0.01, np.pi)]
    gamma_in_range = np.arange(0.0, np.pi, 0.2)
    gamma_sh_range = np.arange(0.01, np.pi, 0.2)

    # Checkpoint
    check_interval = 1000

    # ========== Setup ==========
    num_ex = len(a_pos)
    total_qbits = sum(n_reg) + num_ex
    num_terms = comb(total_qbits, num_ex)
    ideal_amp = 1 / np.sqrt(num_terms)

    # Config dict
    cfg = {
        'ideal_amp': ideal_amp,
        'a_pos': a_pos,
        'n_reg': n_reg,
        'max_rounds': max_rounds,
        'step': step,
        'max_iter': max_iter,
        'tol': tol
    }

    # Create data directory
    os.makedirs('./data', exist_ok=True)
    checkpoint_file = './data/checkpoint_progress.txt'
    output_file = './data/best_rounds.txt'

    # Load checkpoint
    processed = load_checkpoint(checkpoint_file)

    # Generate initial guesses
    all_grid = gen_grid(gamma_in_range, gamma_sh_range)

    # Filter out already processed
    init_grid = [x0 for x0 in all_grid
                 if (round(x0[0], 8), round(x0[1], 8)) not in processed]

    # ========== Info ==========
    print(f'Ideal amplitude: {ideal_amp:.4f}, terms: {num_terms}')
    print("\n" + "="*70)
    print("PARALLEL L-BFGS-B OPT WITH CHECKPOINTING")
    print("="*70)
    print(f"Total guesses:       {len(all_grid)}")
    print(f"Already done:        {len(processed)}")
    print(f"Remaining:           {len(init_grid)}")
    print(f"Bounds:              γ_in ∈ [0, 2π], γ_sh ∈ [0.01, 2π]")
    print(f"Max iter:            {max_iter}")
    print(f"Tol:                 {tol}")

    n_proc = max(1, cpu_count() - 2)  # Leave 2 cores free
    print(f"CPUs:                {n_proc}")
    print("="*70 + "\n")

    if len(init_grid) == 0:
        print("All optimizations already completed!")
        return None

    # Initialize output file if needed
    if not os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'w') as f:
            f.write(
                'init_gamma_in,init_gamma_sh,opt_gamma_in,opt_gamma_sh,round,loss\n')

    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            f.write('gamma_in,gamma_sh,round,loss\n')

    # Prep args
    args_list = [(x0, bounds, cfg) for x0 in init_grid]

    # ========== Signal Handling ==========
    pool_ref = [None]  # Mutable container for pool reference

    def signal_handler(sig, frame):
        print('\n\n[INTERRUPT] Gracefully shutting down...')
        if pool_ref[0]:
            pool_ref[0].terminate()
            pool_ref[0].join()
        print('[INTERRUPT] Progress saved to checkpoint. Run again to resume.')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ========== Run Parallel Opt ==========
    t0 = time.perf_counter()
    best = None
    best_loss = float('inf')
    completed = 0
    checkpoint_buffer = []

    try:
        with Pool(processes=n_proc) as pool:
            pool_ref[0] = pool

            with tqdm(total=len(args_list), desc="Opt", unit="opt",
                      miniters=max(1, len(args_list)//100), smoothing=0.1) as pbar:

                for res in pool.imap_unordered(opt_worker, args_list, chunksize=chunk_size):
                    # Skip failed optimizations
                    if res.loss == float('inf'):
                        pbar.update(1)
                        continue

                    completed += 1

                    # Add to buffer
                    checkpoint_buffer.append(
                        f'{res.init_guess[0]:.8f},{res.init_guess[1]:.8f},'
                        f'{res.opt_params[0]:.8f},{res.opt_params[1]:.8f},'
                        f'{res.best_round},{res.loss:.8f}\n'
                    )

                    # Write buffer when full
                    if len(checkpoint_buffer) >= check_interval:
                        with open(checkpoint_file, 'a') as f:
                            f.writelines(checkpoint_buffer)
                        checkpoint_buffer.clear()

                    # Update best
                    if res.loss < best_loss:
                        best_loss = res.loss
                        best = res
                        with open(output_file, 'a') as f:
                            f.write(
                                f'{best.opt_params[0]:.8f},{best.opt_params[1]:.8f},'
                                f'{best.best_round},{best.loss:.8f}\n')

                    pbar.update(1)
                    pbar.set_postfix({
                        'best_loss': f'{best_loss:.6f}',
                        'round': best.best_round if best else 0,
                        'buffered': len(checkpoint_buffer)
                    })

        # Write remaining buffer
        if checkpoint_buffer:
            with open(checkpoint_file, 'a') as f:
                f.writelines(checkpoint_buffer)
            checkpoint_buffer.clear()

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Caught keyboard interrupt")
        # Flush buffer before exit
        if checkpoint_buffer:
            print(
                f"[INTERRUPT] Flushing {len(checkpoint_buffer)} buffered results...")
            with open(checkpoint_file, 'a') as f:
                f.writelines(checkpoint_buffer)
        raise

    elapsed = time.perf_counter() - t0

    # ========== Results ==========
    if best is None:
        print("\n[WARNING] No successful optimizations completed")
        return None

    print("\n" + "="*70)
    print("OPT COMPLETE")
    print(f'Ideal amplitude: {ideal_amp:.4f}, terms: {num_terms}')
    print("="*70)
    print(f"Best init:           γ_in={best.init_guess[0]:.4f}, "
          f"γ_sh={best.init_guess[1]:.4f}")
    print(f"\nOpt params:")
    print(f"  γ_in:              {best.opt_params[0]:.8f}")
    print(f"  γ_sh:              {best.opt_params[1]:.8f}")
    print(f"\nPerf:")
    print(f"  Loss:              {best.loss:.8f}")
    print(f"  Best round:        {best.best_round}")
    print(f"  Success:           {best.success}")
    print(f"\nStats:")
    print(f"  Time:              {elapsed:.1f}s")
    print(f"  Completed:         {completed}/{len(args_list)}")
    print(f"  Time/opt:          {elapsed/max(1, completed):.2f}s")
    print(f"  Throughput:        {completed/elapsed:.2f} opt/s")
    print("="*70 + "\n")

    return best


if __name__ == "__main__":
    best = main()
