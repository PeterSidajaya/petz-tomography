"""
Section IV A -- worked examples at d = 4.

  batch example : Sec. II C Petz, Sec. III C extended Petz (CSE),
                  mirror map, R-rho-R.
  event example : Sec. II B Petz, Sec. III B Bayesian tomography (SMC).

The event-wise example uses a full-rank POVM obtained by adding
uniform measurement noise to the original rank-one POVM.
The batch example retains the original POVM.

Also emits the R-in-eigenbasis table at the stalling point.
"""

import json
import numpy as np
import petz_lib as L
from qinfer_bayes import run_smc_bayes

# ------------------------- parameters -------------------------------------
D           = 4
EPS         = 0.01        # rho = (1-eps)|psi><psi| + eps I/d
N_ITER      = 50000       # batch iterations
N_EVENTS    = 50000       # event-wise samples
EVENT_NOISE = 0.1         # M_event[j] = (1-eta) M[j] + eta I/m
N_CAND      = 2000        # CSE candidates (protocol 4)
N_PART      = 10000       # Hilbert-Schmidt SMC particles (protocol 3)
MOVE_STEPS  = 60          # pCN Metropolis steps after each resampling
TAU         = 1e-2
SEED_SCAN   = range(200)
OUT         = 'iva_data.npz'
# --------------------------------------------------------------------------


def find_capture_instance(d, eps, tau, seeds, probe_iter=20000,
                          lmin_tol=1e-10, kkt_tol=1e-3):
    """
    Find the first seed for which ordinary batch Petz has:
      - trace-distance error above tau,
      - smallest eigenvalue below lmin_tol,
      - optimality gap lambda_max(R) - 1 above kkt_tol.

    These are numerical selection criteria, not an asymptotic proof.
    """
    for s in seeds:
        rng = np.random.default_rng(s)
        M = L.random_ic_povm(d, rng)
        rho = L.state_from_eps(d, eps, rng)
        q = L.born(M, rho)

        sig, h = L.run_batch(
            M, q, np.eye(d) / d, L.petz_step,
            probe_iter, rho, track=True
        )

        if (
            h['dist'][-1] > tau
            and h['kkt'][-1] > kkt_tol
            and h['lmin'][-1] < lmin_tol
        ):
            return s, M, rho, q

    raise RuntimeError('no capturing instance found in the scanned seeds')


def main():
    if not 0.0 < EVENT_NOISE < 1.0:
        raise ValueError('EVENT_NOISE must be strictly between 0 and 1')

    seed, M, rho, q = find_capture_instance(
        D, EPS, TAU, SEED_SCAN
    )

    print(
        f'instance seed = {seed}, purity = {L.purity(rho):.6f}, '
        f'm = {M.shape[0]} effects, IC = {L.is_ic(M)}'
    )

    sig0 = np.eye(D) / D
    out = {
        'seed': seed,
        'eps': EPS,
        'd': D,
        'purity': L.purity(rho),
        'rho': rho,
        'M': M,
        'q': q,
    }

    # ---------------- batch example ---------------------------------------
    for name, step in L.BATCH_MAPS.items():
        sig, h = L.run_batch(
            M, q, sig0, step, N_ITER, rho, track=True
        )
        out[f'batch_{name}_dist'] = h['dist']
        out[f'batch_{name}_kkt'] = h['kkt']
        out[f'batch_{name}_lmin'] = h['lmin']

        print(
            f'  {name:7s}: dist {h["dist"][-1]:.3e}  '
            f'kkt {h["kkt"][-1]:.3e}  '
            f'lmin {h["lmin"][-1]:.2e}'
        )

        if name == 'petz':
            sig_star = sig

    rng = np.random.default_rng(seed + 10_000)
    cands = L.make_cse(
        D, N_CAND, rng, rho_true=rho, include_true=True
    )
    res = L.cse_hull_certificate(cands, rho)
    p0 = np.full(N_CAND, 1.0 / N_CAND)

    est, p, idx, dist, kkt = L.run_cse_batch(
        M, q, cands, p0, N_ITER, rho
    )
    out['batch_cse_dist'] = dist
    out['batch_cse_kkt'] = kkt
    out['cse_hull_residual'] = res

    print(
        f'  cse    : dist {dist[-1]:.3e}  '
        f'kkt {kkt[-1]:.3e}  '
        f'hull residual {res:.2e}  n_cand {N_CAND}'
    )

    # ---------------- event-wise example ----------------------------------
    # Full-rank POVM for Fig. 2; the batch POVM M is unchanged.
    m = len(M)
    M_event = (
        (1.0 - EVENT_NOISE) * M
        + EVENT_NOISE * np.eye(D)[None, :, :] / m
    )

    if not np.allclose(M_event.sum(axis=0), np.eye(D)):
        raise RuntimeError('event POVM does not sum to the identity')
    if np.linalg.eigvalsh(M_event).min() <= 0:
        raise RuntimeError('event POVM effects are not full rank')
    if not L.is_ic(M_event):
        raise RuntimeError('event POVM is not informationally complete')

    q_event = L.born(M_event, rho)
    out['event_M'] = M_event
    out['event_q'] = q_event
    out['event_noise'] = EVENT_NOISE

    print(
        f'  event POVM: eta = {EVENT_NOISE:g}, '
        f'minimum effect eigenvalue = '
        f'{np.linalg.eigvalsh(M_event).min():.3e}'
    )

    # Both event-wise methods receive the same measurement record.
    rng_e = np.random.default_rng(seed + 20_000)
    outcomes = rng_e.choice(
        m, size=N_EVENTS, p=q_event / q_event.sum()
    )
    out['event_outcomes'] = outcomes

    event_final, idx1, d1 = L.run_event_petz(
        M_event, q_event, sig0, outcomes, rho,
        record_every=10
    )
    out['event_petz_final'] = event_final
    out['event_petz_idx'] = idx1
    out['event_petz_dist'] = d1

    print(
        f'  event Petz (Sec. II B): final dist {d1[-1]:.3e}'
    )

    rng_s = np.random.default_rng(seed + 30_000)
    smc_diagnostics = {}
    est_s, idx3, d3, ess = run_smc_bayes(
        M_event, q_event, outcomes, rho, D, rng_s,
        n_particles=N_PART, record_every=10,
        move_steps=MOVE_STEPS, diagnostics=smc_diagnostics
    )
    out['event_smc_final'] = est_s
    out['event_smc_idx'] = idx3
    out['event_smc_dist'] = d3
    out['event_smc_ess'] = ess
    out['event_smc_method'] = smc_diagnostics['method']
    out['event_smc_prior'] = smc_diagnostics['prior']
    out['event_smc_n_particles'] = N_PART
    out['event_smc_move_steps'] = MOVE_STEPS
    out['event_smc_seed'] = seed + 30_000
    out['event_smc_resample_events'] = smc_diagnostics['resample_events']
    out['event_smc_move_acceptance'] = smc_diagnostics['move_acceptance']

    print(
        f'  SMC Bayes (Sec. III B): final dist {d3[-1]:.3e}, '
        f'n_particles {N_PART}, pCN move steps {MOVE_STEPS}'
    )

    # ---------------- eigenbasis diagnostic table --------------------------
    w, v = np.linalg.eigh(
        0.5 * (sig_star + sig_star.conj().T)
    )
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]

    R = L.R_operator(M, q, sig_star)
    Rb = v.conj().T @ R @ v
    offdiag = Rb - np.diag(np.diag(Rb))

    table = {
        'sigma_eigenvalues': w.tolist(),
        'R_diagonals': np.diag(Rb).real.tolist(),
        'R_eigenvalues': np.sort(
            np.linalg.eigvalsh(R)
        )[::-1].tolist(),
        'largest_offdiagonal_modulus': float(
            np.abs(offdiag).max()
        ),
        'kkt_gap': float(
            np.linalg.eigvalsh(R).max() - 1.0
        ),
        'top_eigvec_overlaps': (
            np.abs(
                v.conj().T @ np.linalg.eigh(R)[1][:, -1]
            ) ** 2
        ).tolist(),
    }

    # Retain the existing key for compatibility with plot_iva.py.
    out['appendixC'] = json.dumps(table, indent=2)

    print('\nR in the eigenbasis of the stalling point:')
    print(out['appendixC'])

    np.savez_compressed(OUT, **out)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
