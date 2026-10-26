"""
Figure 5 (Appendix D in the manuscript) -- failure rate as a function of eigenbasis misalignment, at fixed
distance from the true state.

Every starting state sits at the same trace distance d0 from rho and carries
rho's exact spectrum: sigma_0 = W rho W^dag, so purity and distance to the
boundary are held fixed and positivity is automatic.  What varies is where
the rotation acts.  The generator is split into a part that acts inside the
strong and weak blocks separately and a part that couples them,

    H(s) = sqrt(1 - s^2) H_block + s H_coupling,

and the amplitude is fixed by bisection so that D(sigma_0, rho) = d0 for
every s.  At s = 0 the weak eigenspace is exactly invariant and the principal
angle phi vanishes; at s = 1 the whole displacement is strong-weak.

Misalignment is reported as the largest principal angle between the two
weak eigenspaces. Instances are selected to reach the entire requested
angle grid within the rotation family used here.

The spectrum is non-degenerate,

    lambda = (0.70, 0.30 - 3 eps, 2 eps, eps),

so that block-internal rotations move the state a finite distance too.  With
a degenerate weak block, every displacement is strong-weak and phi becomes a
function of d0 rather than an independent variable.

Failure criterion, as in Sec. IV B: a run fails if it has not come within TAU
of rho after N_ITER iterations.  lmin and move at exit are recorded but not
used in the criterion.

Output: appendixC_trials.csv, one row per trial.
"""

import csv
import time
import numpy as np
import petz_lib as L

# ------------------------- parameters -------------------------------------
D         = 4
EPS       = 1e-4
D0        = 0.05                              # fixed trace distance from rho
PHI_GRID  = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # degrees
N_TRIALS  = 200
N_ITER    = 50000
TAU       = 1e-2
N_WEAK    = 2                                 # size of the weak block
MAPS      = ['petz', 'mirror']
SEED      = 20260821
OUT       = 'appendixC_trials.csv'
BISECT    = 40                                # amplitude bisection steps
SCAN      = 150                               # amplitude grid points
S_BISECT  = 30                                # mixing bisection steps
PHI_TOL   = 1e-3                              # degrees
# --------------------------------------------------------------------------


def spectrum(eps):
    lam = np.array([0.70, 0.30 - 3 * eps, 2 * eps, eps])
    return lam / lam.sum()


def haar_unitary(d, rng):
    """QR of a Ginibre matrix is Haar only after fixing the phases of R."""
    g = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(g)
    return Q * (np.diagonal(R) / np.abs(np.diagonal(R)))


def blocked_H(rng, kind, d=D, k=N_WEAK):
    g = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    H = 0.5 * (g + g.conj().T)
    mask = np.zeros((d, d), bool)
    if kind == 'block':
        mask[:d - k, :d - k] = True
        mask[d - k:, d - k:] = True
    else:
        mask[:d - k, d - k:] = True
        mask[d - k:, :d - k] = True
    H = H * mask
    H = H - np.trace(H).real / d * np.eye(d)
    n = np.linalg.norm(H)
    return H / n if n > 0 else H


def to_distance(rho, U, H, d0, n_bisect=BISECT, n_scan=SCAN, t_max=None):
    """
    W = U exp(i t H) U^dag with t set so that D(W rho W^dag, rho) = d0.

    D(t) is oscillatory in t, so a doubling search can skip past the first
    branch and land the bisection on a non-monotone interval.  We instead
    scan t on a grid, take the FIRST crossing of d0, and bisect inside that
    bracket.  Returns (None, nan) if d0 is out of reach for this generator,
    which happens when the eigenvalue gaps it connects are too small.
    """
    def f(t):
        w, v = np.linalg.eigh(t * H)
        W = U @ ((v * np.exp(1j * w)) @ v.conj().T) @ U.conj().T
        sig = W @ rho @ W.conj().T
        sig = 0.5 * (sig + sig.conj().T)
        return L.trace_distance(sig, rho), sig

    t_max = np.pi if t_max is None else t_max
    ts = np.linspace(0.0, t_max, n_scan)
    ds = np.array([f(t)[0] for t in ts])
    idx = np.argmax(ds >= d0)
    if ds[idx] < d0:
        return None, np.nan
    lo, hi = ts[idx - 1], ts[idx]
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if f(mid)[0] < d0:
            lo = mid
        else:
            hi = mid
    dist, sig = f(0.5 * (lo + hi))
    return sig, dist


def principal_angle(sig, rho, k=N_WEAK):
    """Largest principal angle between the two weak eigenspaces, in degrees."""
    def weak(x):
        w, v = np.linalg.eigh(0.5 * (x + x.conj().T))
        return v[:, np.argsort(w)[:k]]
    s = np.linalg.svd(weak(sig).conj().T @ weak(rho), compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0))))


def state_at_mixing(rho, U, Hb, Hc, s, d0):
    """sigma_0 at mixing s, with the amplitude reset so D(sigma_0, rho) = d0."""
    H = np.sqrt(max(1 - s * s, 0.0)) * Hb + s * Hc
    return to_distance(rho, U, H, d0)


def solve_for_phi(rho, U, Hb, Hc, phi_target, d0):
    """
    Bisect the mixing s so that the principal angle hits phi_target.  The
    amplitude is re-fitted inside state_at_mixing at every s, so d0 is held
    exactly throughout and phi is the only thing being solved for.
    Returns (sigma_0, phi, achieved_d0), or None if phi_target is out of
    reach for this draw.
    """
    if phi_target == 0.0:
        sg, dd = state_at_mixing(rho, U, Hb, Hc, 0.0, d0)
        return (sg, principal_angle(sg, rho), dd) if sg is not None else None

    sg_hi, _ = state_at_mixing(rho, U, Hb, Hc, 1.0, d0)
    if sg_hi is None or principal_angle(sg_hi, rho) < phi_target:
        return None

    lo, hi = 0.0, 1.0
    for _ in range(S_BISECT):
        mid = 0.5 * (lo + hi)
        sg, dd = state_at_mixing(rho, U, Hb, Hc, mid, d0)
        if sg is None:
            lo = mid
            continue
        if principal_angle(sg, rho) < phi_target:
            lo = mid
        else:
            hi = mid
    sg, dd = state_at_mixing(rho, U, Hb, Hc, 0.5 * (lo + hi), d0)
    if sg is None:
        return None
    ph = principal_angle(sg, rho)
    return (sg, ph, dd) if abs(ph - phi_target) < PHI_TOL else None


def draw_instances(n_trials, rng, max_redraw=40):
    """
    One paired ensemble, reused across the whole phi grid.

    Each instance carries its own rho, POVM and pair of generators, and is
    accepted only if it can reach EVERY phi in PHI_GRID at this d0.  Points
    on the curve are then a within-instance comparison: the state, the
    measurement and the rotation planes are identical from phi = 0 to
    phi = max, and only the mixing s changes.  Accepting per-point instead
    would leave the ensemble different at each phi, so instance-to-instance
    variation would sit inside the phi dependence rather than beside it.

    Screening on the whole grid does select instances that reach the largest
    phi, so `n_rejected` is reported: a large value means PHI_GRID is
    reaching too far for this d0 and the ensemble is no longer generic.
    """
    lam = spectrum(EPS)
    inst, rejected = [], 0
    while len(inst) < n_trials:
        ok = False
        for _ in range(max_redraw):
            U = haar_unitary(D, rng)
            r = (U * lam) @ U.conj().T
            rho = 0.5 * (r + r.conj().T)
            Hb, Hc = blocked_H(rng, 'block'), blocked_H(rng, 'coupling')
            sols = [solve_for_phi(rho, U, Hb, Hc, p, D0) for p in PHI_GRID]
            if all(x is not None for x in sols):
                inst.append({'rho': rho, 'M': L.random_ic_povm(D, rng),
                             'sols': sols})
                ok = True
                break
            rejected += 1
        if not ok:
            raise RuntimeError(f'PHI_GRID unreachable at d0 = {D0}')
    return inst, rejected


def cell_from_instances(inst, j):
    """Stack the j-th phi target across the paired ensemble."""
    n = len(inst)
    M = np.stack([x['M'] for x in inst])
    rho = np.stack([x['rho'] for x in inst])
    sig0 = np.stack([x['sols'][j][0] for x in inst])
    phi = np.array([x['sols'][j][1] for x in inst])
    ach = np.array([x['sols'][j][2] for x in inst])
    q = np.clip(np.einsum('tjab,tba->tj', M, rho).real, 0.0, None)
    return M, rho, q, sig0, phi, ach


def main():
    rng = np.random.default_rng(SEED)
    lam = spectrum(EPS)
    t0 = time.time()
    print(f'd = {D}, spectrum ({", ".join("%.5g" % x for x in lam)}), '
          f'purity {np.sum(lam ** 2):.6f}')
    print(f'fixed trace distance d0 = {D0:g}, tau = {TAU:g}, '
          f'budget = {N_ITER}, {N_TRIALS} paired instances')


    inst, rejected = draw_instances(N_TRIALS, rng)
    print(f'drew {N_TRIALS} instances reaching the whole grid '
          f'({rejected} rejected) [{time.time() - t0:.0f}s]\n')

    fail_by_phi = {m: np.zeros((len(PHI_GRID), N_TRIALS), dtype=bool)
                   for m in MAPS}
    with open(OUT, 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['d', 'eps', 'purity', 'd0', 'phi_target', 'phi_deg',
                     'achieved_d0', 'map', 'instance', 'dist', 'kkt', 'lmin',
                     'iters', 'move', 'converged', 'tau', 'n_iter'])
        for j, phi_t in enumerate(PHI_GRID):
            M, rho, q, sig0, phi, ach = cell_from_instances(inst, j)
            for name in MAPS:
                step = L.BATCH_MAPS_STACK[name]
                dist, kkt, lmin, iters, move, conv = L.run_batch_stack(
                    M, q, sig0, step, N_ITER, rho, tau=TAU)
                fail_by_phi[name][j] = ~conv
                for t in range(N_TRIALS):
                    wr.writerow([D, EPS, float(np.sum(lam ** 2)), D0,
                                 f'{phi_t:.6e}', f'{phi[t]:.6e}',
                                 f'{ach[t]:.6e}', name, t, f'{dist[t]:.6e}',
                                 f'{kkt[t]:.6e}', f'{lmin[t]:.6e}', iters[t],
                                 f'{move[t]:.6e}', int(conv[t]), TAU, N_ITER])
                fail = ~conv
                print(f'phi={phi_t:5.2f} deg (spread {phi.max() - phi.min():.1e})'
                      f'  {name:6s} fail={fail.mean():.3f}  '
                      f'median_lmin_if_failed='
                      f'{np.median(lmin[fail]) if fail.any() else np.nan:.1e}  '
                      f'[{time.time() - t0:.0f}s]', flush=True)
            fh.flush()

    # Pairing makes a per-instance critical angle meaningful.
    print('\ncritical angle per instance (first phi at which it fails)')
    grid = np.array(PHI_GRID)
    for name in MAPS:
        F = fail_by_phi[name]
        never = ~F.any(axis=0)
        first = np.where(never, -1, np.argmax(F, axis=0))
        nonmono = int(np.sum([(not never[t]) and
                              (not F[first[t]:, t].all())
                              for t in range(N_TRIALS)]))
        got = grid[first[~never]] if (~never).any() else np.array([])
        print(f'  {name:6s} never failed {never.sum()}/{N_TRIALS}; '
              f'median phi_c '
              f'{np.median(got) if got.size else np.nan:.2f} deg; '
              f'non-monotone instances {nonmono}')
    print(f'\nwrote {OUT} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()