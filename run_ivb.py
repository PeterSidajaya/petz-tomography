"""
Section IV B -- random sweep over dimension and purity.

For each (d, eps) cell we draw N_TRIALS independent instances: a random IC
POVM with d^2 + 2 rank-1 effects, a true state
    rho = (1 - eps) |psi><psi| + eps I/d,   psi Haar,
and exact evidence q = E[rho].  Each of the three batch maps is iterated from
the maximally mixed state.

Capture criterion: trace distance from rho above TAU at the end of the run.
Since q is exact and the POVM is IC, rho is the unique MLE, so distance from
rho and distance from the MLE coincide and the single criterion suffices.

lambda_max(R) - 1 and lambda_min(sigma) are recorded per trial but not used
in the criterion, so that thresholds can be revisited from the raw records
without re-running anything.

Each trial runs until it comes within TAU of rho, or until N_ITER iterations
have elapsed; capture is failure to reach TAU within the budget.  The step
size max|sigma_{i+1} - sigma_i| at exit is recorded as `move`, which is what
distinguishes a captured trial sitting at a fixed point from one still
descending at O(1/i) when the budget ended.
"""

import csv
import time
import numpy as np
import petz_lib as L

# ------------------------- parameters -------------------------------------
D_LIST    = [2, 3, 4, 5]
EPS_GRID  = [0.5, 0.35, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
N_TRIALS  = 200          # per (d, eps) cell, per map
N_ITER    = 50000       # iteration budget
TAU       = 1e-2         # capture threshold, trace distance
MAPS      = ['petz', 'mirror', 'rrr']
SEED      = 20260820
OUT       = 'ivb_trials.csv'
# --------------------------------------------------------------------------


def make_cell(d, eps, n_trials, rng):
    M = np.stack([L.random_ic_povm(d, rng) for _ in range(n_trials)])
    rho = np.stack([L.state_from_eps(d, eps, rng) for _ in range(n_trials)])
    q = np.clip(np.einsum('tjab,tba->tj', M, rho).real, 0.0, None)
    sig0 = np.broadcast_to(np.eye(d) / d, (n_trials, d, d)).astype(complex).copy()
    return M, rho, q, sig0


def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    with open(OUT, 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['d', 'eps', 'purity', 'map', 'trial', 'dist',
                     'kkt', 'lmin', 'iters', 'move', 'converged',
                     'tau', 'n_iter'])
        for d in D_LIST:
            for eps in EPS_GRID:
                M, rho, q, sig0 = make_cell(d, eps, N_TRIALS, rng)
                pur = (1 - eps) ** 2 + 2 * (1 - eps) * eps / d + eps ** 2 / d
                for name in MAPS:
                    step = L.BATCH_MAPS_STACK[name]
                    dist, kkt, lmin, iters, move, conv = L.run_batch_stack(
                        M, q, sig0, step, N_ITER, rho, tau=TAU)
                    for t in range(N_TRIALS):
                        wr.writerow([d, eps, pur, name, t,
                                     f'{dist[t]:.6e}', f'{kkt[t]:.6e}',
                                     f'{lmin[t]:.6e}', iters[t],
                                     f'{move[t]:.6e}', int(conv[t]),
                                     TAU, N_ITER])
                    cap = ~conv
                    med_move = np.median(move[cap]) if cap.any() else np.nan
                    print(f'd={d} eps={eps:<6g} purity={pur:.4f} '
                          f'{name:6s} capture={cap.mean():.3f} '
                          f'median_iters={np.median(iters):.0f} '
                          f'median_move_if_captured={med_move:.1e} '
                          f'[{time.time() - t0:.0f}s]', flush=True)
                fh.flush()
    print(f'wrote {OUT} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
