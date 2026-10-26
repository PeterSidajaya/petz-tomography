"""Figure and audit for Appendix C."""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 150, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
})

COL = {'petz': '#D55E00', 'mirror': '#0072B2'}
LAB = {'petz': 'Petz', 'mirror': 'mirror map'}


def binom_sd(k, n):
    """Standard error of a binomial proportion, sqrt(p(1-p)/n)."""
    if n == 0:
        return 0.0
    p = k / n
    return float(np.sqrt(p * (1 - p) / n))


df = pd.read_csv('appendixC_trials.csv')

# tau comes from the records, not from this script, so the criterion plotted
# is always the criterion that was run.
assert df['tau'].nunique() == 1, f"mixed tau: {df['tau'].unique()}"
TAU = float(df['tau'].iloc[0])
n_iter = int(df['n_iter'].iloc[0])
eps = float(df['eps'].iloc[0])
pur = float(df['purity'].iloc[0])
d0 = float(df['d0'].iloc[0])
df['failed'] = ~df['converged'].astype(bool)

fig, ax = plt.subplots(figsize=(3.6, 2.9))
for name in ['petz', 'mirror']:
    sub = df[df['map'] == name]
    g = sub.groupby('phi_target')
    phi = np.array(sorted(g.groups.keys()))
    k = np.array([g.get_group(x)['failed'].sum() for x in phi], dtype=float)
    n = np.array([g.get_group(x)['failed'].size for x in phi], dtype=float)
    rate = k / n
    err = np.array([binom_sd(a, b) for a, b in zip(k, n)])
    ax.errorbar(phi, rate, yerr=err,
                color=COL[name], marker='o', ms=3, lw=1.2, capsize=1.5,
                elinewidth=0.8, label=LAB[name])
ax.set_xlabel(r'principal angle $\phi$ (deg)')
ax.set_ylabel('failure rate')
ax.set_ylim(-0.04, 1.04)
ax.legend(loc='upper left', frameon=False)
fig.savefig('fig_appendixC.pdf'); fig.savefig('fig_appendixC.png', dpi=200)

# ------------------------------- audit ------------------------------------
print(f'purity {pur:.6f} (eps = {eps:g}), fixed d0 = {d0:g}, '
      f'tau = {TAU:g}, budget = {n_iter}')
print(f'paired instances: {df.groupby(["phi_target", "map"]).size().unique()}')
ach = df['achieved_d0']
err = (df['phi_deg'] - df['phi_target']).abs()
print(f'achieved d0 spans {ach.min():.6e} to {ach.max():.6e}')
print(f'worst |phi - target|: {err.max():.2e} deg\n')

print('failure rate by misalignment (phi held to target, d0 held fixed)')
piv = df.pivot_table(index='phi_target', columns='map', values='failed')
print(piv.round(4).to_string())

# Pairing: the same instances appear at every phi, so a per-instance
# critical angle is well defined.
print('\ncritical angle per instance (first phi at which it fails)')
grid = np.array(sorted(df['phi_target'].unique()))
for name in ['petz', 'mirror']:
    sub = df[df['map'] == name]
    F = sub.pivot_table(index='phi_target', columns='instance',
                        values='failed').reindex(grid).to_numpy().astype(bool)
    never = ~F.any(axis=0)
    first = np.where(never, -1, np.argmax(F, axis=0))
    nonmono = int(sum((not never[t]) and (not F[first[t]:, t].all())
                      for t in range(F.shape[1])))
    got = grid[first[~never]] if (~never).any() else np.array([])
    print(f'  {name:6s} never failed {never.sum()}/{F.shape[1]}; '
          f'median phi_c {np.median(got) if got.size else np.nan:.2f} deg; '
          f'non-monotone {nonmono}')

fail = df[df['failed']]
if len(fail):
    print('\nfailed trials: is the limit collapsed, or just slow?')
    print(fail.groupby('map')[['lmin', 'move']]
          .agg(['size', 'median', 'max']).to_string())
    print('\na failed trial with lmin ~ 1e-16 and move ~ 0 sits at a rank-'
          'deficient\nfixed point; one with healthy lmin and large move was '
          'still descending\nwhen the budget ran out, and says more about '
          'N_ITER than about the map')
else:
    print('\nno failures at any angle')