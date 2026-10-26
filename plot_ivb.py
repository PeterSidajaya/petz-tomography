"""Figure and audit for Section IV B."""

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

COL = {'petz': '#D55E00', 'mirror': '#0072B2', 'rrr': '#009E73'}
LAB = {'petz': 'batch Petz', 'mirror': 'mirror map', 'rrr': r'$R\rho R$'}


def binom_sd(k, n):
    """Standard error of a binomial proportion, sqrt(p(1-p)/n)."""
    if n == 0:
        return 0.0
    p = k / n
    return float(np.sqrt(p * (1 - p) / n))


df = pd.read_csv('ivb_trials.csv')

# tau is read from the records rather than set here.  Setting it independently
# in this script and in run_ivb.py is a way to silently plot a different
# criterion from the one that was run.
assert df['tau'].nunique() == 1, f"mixed tau in the records: {df['tau'].unique()}"
TAU = float(df['tau'].iloc[0])
df['captured'] = ~df['converged'].astype(bool)
n_iter = int(df['n_iter'].iloc[0])
dims = sorted(df['d'].unique())

fig, axes = plt.subplots(1, len(dims), figsize=(2.2 * len(dims), 2.5),
                         sharey=True)
for ax, d in zip(np.atleast_1d(axes), dims):
    sub = df[df['d'] == d]
    for name in ['petz', 'mirror', 'rrr']:
        g = sub[sub['map'] == name].groupby('purity')['captured']
        pur = np.array(sorted(g.groups.keys()))
        k = np.array([g.get_group(p).sum() for p in pur], dtype=float)
        n = np.array([g.get_group(p).size for p in pur], dtype=float)
        rate = k / n
        err = np.array([binom_sd(kk, nn) for kk, nn in zip(k, n)])
        ax.errorbar(pur, rate, yerr=err,
                    color=COL[name], marker='o', ms=2.5, lw=1.1,
                    capsize=1.5, elinewidth=0.8,
                    label=LAB[name] if d == dims[0] else None)
    ax.set_title(f'$d={d}$', fontsize=9)
    ax.set_xlabel(r'purity $\mathrm{Tr}\,\rho^2$')
    ax.set_ylim(-0.04, 1.04)
np.atleast_1d(axes)[0].set_ylabel('failure rate')
np.atleast_1d(axes)[0].legend(loc='upper left', frameon=True)
fig.savefig('fig_ivb.pdf'); fig.savefig('fig_ivb.png', dpi=200)

# ------------------------------- audit ------------------------------------
print(f'trials per cell: {df.groupby(["d", "eps", "map"]).size().unique()}')
print(f'iteration budget: {n_iter}, tau = {TAU}')

cap = df[df['captured']]
print('\ncaptured trials: step size at the budget, by map')
print(cap.groupby('map')['move'].agg(['size', 'median', 'max']).to_string())
print('a captured trial with a large `move` was still descending when the '
      'budget ran out, and is a statement about N_ITER, not about the map\n')

print('captured trials by map:')
for name in ['petz', 'mirror', 'rrr']:
    c = cap[cap['map'] == name]
    tot = (df['map'] == name).sum()
    unresolved = int((c['move'] > 1e-12).sum())
    print(f'  {name:6s} {len(c):5d}/{tot:5d}  '
          f'median lmin {c["lmin"].median():.2e}  '
          f'median kkt {c["kkt"].median():.2e}  '
          f'still moving at budget: {unresolved}')

print('\ncapture rate table (rows = purity, cols = d), batch Petz:')
p = df[df['map'] == 'petz'].pivot_table(index='purity', columns='d',
                                        values='captured')
print(p.round(3).to_string())

print('\ncaptures split by whether the limit is rank-deficient '
      '(genuine) or not:')
cc = cap.copy()
cc['rank_deficient'] = cc['lmin'] < 1e-12
print(cc.groupby(['map', 'rank_deficient'])['lmin']
      .agg(['size', 'median']).to_string())