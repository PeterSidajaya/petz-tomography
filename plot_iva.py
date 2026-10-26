"""Figures for Section IV A."""

import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 150, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
})

D = np.load('iva_data.npz', allow_pickle=True)
COL = {'petz': '#D55E00', 'mirror': '#0072B2', 'rrr': '#009E73',
       'cse': '#CC79A7', 'event_petz': '#D55E00', 'event_smc': '#CC79A7'}


def thin(y, n=400):
    """Log-spaced subsample so markers stay legible on a log x-axis."""
    idx = np.unique(np.round(np.geomspace(1, len(y), n)).astype(int) - 1)
    return idx, y[idx]


# ---------------------------------------------------------------- batch ----
fig, ax = plt.subplots(figsize=(3.6, 2.9))
labels = {'petz': 'batch Petz',
          'cse': 'batch extended Petz',
          'mirror': 'mirror map',
          'rrr': r'$R\rho R$'}
for key in ['petz', 'cse', 'mirror', 'rrr']:
    y = D[f'batch_{key}_dist']
    i, yy = thin(y)
    ax.plot(i + 1, np.maximum(yy, 1e-17), color=COL[key], lw=1.3,
            label=labels[key],
            ls='-' if key in ('petz', 'cse') else '--')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('iteration $i$')
ax.set_ylabel(r'$\frac{1}{2}\|\hat\rho_i-\rho\|_1$')
ax.legend(loc='lower left', frameon=True)
# ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2),ncol=2) 

# axi = ax.inset_axes([0.14, 0.1, 0.42, 0.36])
# for key in ['petz', 'cse', 'mirror', 'rrr']:
#     y = D[f'batch_{key}_kkt']
#     i, yy = thin(y, 200)
#     axi.plot(i + 1, np.maximum(yy, 1e-17), color=COL[key], lw=1.0,
#              ls='-' if key in ('petz', 'cse') else '--')
# axi.set_xscale('log'); axi.set_yscale('log')
# axi.set_title(r'$\lambda_{\max}(R(\hat\rho_i))-1$', fontsize=7, pad=2)
# axi.tick_params(labelsize=6)
fig.savefig('fig_iva_batch.pdf'); fig.savefig('fig_iva_batch.png', dpi=200)

# ---------------------------------------------------------------- event ----
fig, ax = plt.subplots(figsize=(3.6, 2.9))
ax.plot(D['event_petz_idx'][1:], D['event_petz_dist'][1:],
        color=COL['event_petz'], lw=1.3,
        label='event-wise Petz')
ax.plot(D['event_smc_idx'][1:], D['event_smc_dist'][1:],
        color=COL['event_smc'], lw=1.3,
        label='event-wise extended Petz')
# n = D['event_smc_idx'][1:]
# ax.plot(n, 1.2 * D['event_smc_dist'][1] * np.sqrt(n[0] / n), 'k:', lw=0.9,
#         label=r'$\propto N^{-1/2}$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('number of events $N$')
ax.set_ylabel(r'$\frac{1}{2}\|\hat\rho_N-\rho\|_1$')
ax.legend(loc='lower left', frameon=True)
fig.savefig('fig_iva_event.pdf'); fig.savefig('fig_iva_event.png', dpi=200)

# ---------------------------------------------------------------- table ----
tab = json.loads(str(D['appendixC']))
rows = []
rows.append(r'\begin{tabular}{cccc}')
rows.append(r'\hline\hline')
rows.append(r'$k$ & $\lambda_k(\hat\rho_\star)$ & $R_{kk}$ & $\lambda_k(R)$ \\')
rows.append(r'\hline')
for k, (s, dg, ev) in enumerate(zip(tab['sigma_eigenvalues'],
                                    tab['R_diagonals'],
                                    tab['R_eigenvalues']), start=1):
    rows.append(f'{k} & {s:.6f} & {dg:.9f} & {ev:.9f} ' + r'\\')
rows.append(r'\hline\hline')
rows.append(r'\end{tabular}')
open('appendixC_table.tex', 'w').write('\n'.join(rows) + '\n')

print(json.dumps({
    'instance_seed': int(D['seed']), 'd': int(D['d']),
    'purity': float(D['purity']), 'n_effects': int(D['M'].shape[0]),
    'cse_hull_residual': float(D['cse_hull_residual']),
    'final_dist': {k: float(D[f'batch_{k}_dist'][-1])
                   for k in ['petz', 'cse', 'mirror', 'rrr']},
    'final_kkt': {k: float(D[f'batch_{k}_kkt'][-1])
                  for k in ['petz', 'cse', 'mirror', 'rrr']},
    'petz_lmin': float(D['batch_petz_lmin'][-1]),
    'event_petz_final': float(D['event_petz_dist'][-1]),
    'event_smc_final': float(D['event_smc_dist'][-1]),
    'largest_offdiagonal_modulus': tab['largest_offdiagonal_modulus'],
    'kkt_gap': tab['kkt_gap'],
}, indent=2))
