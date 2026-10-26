# Petz tomography

Code and numerical data accompanying **Tomographic Limits of the Petz Recovery Map**, by Peter Sidajaya, Clive Cenxin Aw, Mingxuan Liu, and Valerio Scarani.

## Figures and files

| Manuscript figure | Simulation | Plotting | Saved data |
| --- | --- | --- | --- |
| Fig. 2: event-wise comparison | `run_iva.py` | `plot_iva.py` | `iva_data.npz` |
| Fig. 3: batch comparison | `run_iva.py` | `plot_iva.py` | `iva_data.npz` |
| Fig. 4: dimension and purity sweep | `run_ivb.py` | `plot_ivb.py` | `ivb_trials.csv` |
| Fig. 5: principal-angle sweep | `run_appendixC.py` | `plot_appendixC.py` | `appendixC_trials.csv` |

The `appendixC` filenames are historical: the principal-angle experiment appears in Appendix D of the current manuscript. The analytic qubit proof in Appendix C does not require a simulation. `petz_lib.py` contains shared numerical routines; `qinfer_bayes.py` implements the Bayesian tomography adapter.

## Setup

The scripts were checked using Python 3.9 and the dependency versions in `requirements.txt`. QInfer is an older package; compatibility with newer Python and dependency versions is not established.

```sh
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Regenerate the figures from saved data

Run from this repository directory:

```sh
python plot_iva.py
python plot_ivb.py
python plot_appendixC.py
```

These commands write PDF and PNG figures. `plot_iva.py` also writes an eigenbasis diagnostic table. The saved data let readers reproduce the plots without running the simulations.

## Rerun the simulations

```sh
python run_iva.py
python run_ivb.py
python run_appendixC.py
```

These commands overwrite the corresponding saved data files. Parameters and random seeds are set near the top of each script. The dimension and angle sweeps are substantially more expensive than plotting; each allows up to 50,000 iterations per trial. Numerical roundoff and library differences can affect trajectories near rank-deficient states.

## Numerical construction

- **Figs. 2–3:** dimension 4, target state `(1-eps)|psi><psi| + eps I/4` with `eps=0.01`, and 18 measurement effects. The script selects an instance in which ordinary batch Petz stalls using the explicit numerical criteria in `find_capture_instance` (the saved instance has seed 1). Batch methods start at `I/4` and use exact Born probabilities for 50,000 iterations.
- **Fig. 2:** full-rank effects `M_event[j] = 0.9 M[j] + 0.1 I/18`. Both event-wise methods receive the same 50,000 outcomes. Bayesian tomography uses a Hilbert–Schmidt prior, 10,000 particles, an effective-sample-size threshold of half the particle count, and multinomial resampling followed by 60 pCN Metropolis moves. QInfer supplies weight updates and effective-sample-size checks; the resampler is custom. Moves target the likelihood of all outcomes observed so far. The true state is used for error diagnostics, not Bayesian inference.
- **Fig. 3:** the batch CSE has 2,000 candidates with uniform initial weights and explicitly includes the true state. This ensures convex-hull coverage; the experiment does not test whether an independently sampled finite CSE covers an unknown target.
- **Fig. 4:** dimensions 2–5; 200 random instances per dimension/purity setting, using rank-one informationally complete POVMs with `d*d+2` effects. The three maps use the same instances. Failure means not reaching trace distance at most 0.01 within the 50,000-iteration budget, checked every 100 iterations. Successful trials stop early. Error bars are binomial standard errors `sqrt(p*(1-p)/n)`. These finite-budget failures alone do not establish asymptotic nonconvergence.
- **Fig. 5:** 200 paired instances at each of 12 angles from 0 to 4 degrees. Targets and initial states share spectrum `(0.7, 0.2997, 0.0002, 0.0001)` and are separated by trace distance 0.05. The angle is the largest principal angle between their two-dimensional weak eigenspaces. Instances are selected to reach the complete angle grid under the chosen rotation construction. The failure criterion and error bars follow Fig. 4.

## References

- Granade, Combes, and Cory, *Practical Bayesian tomography*, New Journal of Physics **18**, 033024 (2016): https://arxiv.org/abs/1509.03770
- Cotter, Roberts, Stuart, and White, *MCMC methods for functions: modifying old algorithms to make them faster*, Statistical Science **28**, 424–446 (2013): https://arxiv.org/abs/1202.0709

## AI assistance

The code was developed and revised with assistance from generative AI tools. Numerical checks included comparison of scalar and batched updates, checks of state and measurement validity, and comparison of the Bayesian estimate with an independent full-data MCMC calculation. AI assistance does not replace author responsibility for the implementation and reported results.
