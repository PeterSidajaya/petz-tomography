"""QInfer Bayesian tomography with posterior-preserving resample-move SMC.

Uses the Hilbert-Schmidt prior and exact multi-outcome Born likelihood.
QInfer performs sequential weight updates and effective sample size checks.
The custom resampler uses multinomial resampling followed by pCN Metropolis
moves targeting the posterior for ALL observations collected so far.
This replaces Liu-West resampling; it is not unchanged Granade et al. code.

References: Granade et al., NJP 18, 033024 (2016), arXiv:1509.03770;
Cotter et al., Statistical Science 28, 424-446 (2013), arXiv:1202.0709.
Tested with Python 3.9, NumPy 1.26.4 and QInfer 1.0.
"""
import numpy as np
import qinfer as qi

class FactorModel(qi.FiniteOutcomeModel):
    """Particles are real/imaginary coordinates of square Ginibre factors."""
    def __init__(self, effects):
        self.effects = np.asarray(effects, dtype=complex)
        if (self.effects.ndim != 3 or
                self.effects.shape[1] != self.effects.shape[2]):
            raise ValueError('M must have shape (m, d, d)')
        self.m, self.d, _ = self.effects.shape
        if self.m < 1 or self.d < 2 or not np.isfinite(self.effects).all():
            raise ValueError('Need a finite POVM with d >= 2')
        if not np.allclose(self.effects, self.effects.conj().swapaxes(-1, -2),
                           atol=1e-12, rtol=0):
            raise ValueError('POVM effects must be Hermitian')
        if np.linalg.eigvalsh(self.effects).min() < -1e-12:
            raise ValueError('POVM effects must be positive semidefinite')
        if not np.allclose(self.effects.sum(axis=0), np.eye(self.d),
                           atol=1e-10, rtol=0):
            raise ValueError('POVM effects must sum to I')
        super().__init__()

    @property
    def n_modelparams(self):
        return 2*self.d**2

    @property
    def is_n_outcomes_constant(self):
        return True

    @property
    def expparams_dtype(self):
        return [('unused', 'int64')]

    def n_outcomes(self, expparams):
        return self.m

    def states(self, x):
        x = np.atleast_2d(x)
        k = self.d**2
        if x.shape[1] != 2*k or not np.isfinite(x).all():
            raise ValueError('Invalid Ginibre coordinates')
        g = (x[:, :k]+1j*x[:, k:]).reshape(-1,self.d,self.d)
        rho = g @ g.conj().swapaxes(-1,-2)
        norm = np.trace(rho,axis1=1,axis2=2).real
        if np.any(norm <= 0) or not np.isfinite(norm).all():
            raise ValueError('Ginibre factors must be finite and nonzero')
        return rho/norm[:,None,None]

    def are_models_valid(self, x):
        x = np.atleast_2d(x)
        return np.isfinite(x).all(axis=1) & np.any(x != 0, axis=1)

    def probabilities(self,x,labels=None):
        effects = self.effects if labels is None else self.effects[labels]
        p = np.einsum('jab,pba->pj', effects,self.states(x)).real
        if not np.isfinite(p).all() or np.any(p < -1e-10):
            raise ValueError('Invalid Born probabilities')
        return np.clip(p,0,1)

    def likelihood(self,outcomes,modelparams,expparams):
        super().likelihood(outcomes,modelparams,expparams)
        raw = np.asarray(outcomes).reshape(-1)
        if (not np.isfinite(raw).all() or np.any(raw != np.floor(raw)) or
                np.any(raw < 0) or np.any(raw >= self.m)):
            raise ValueError('Invalid outcome labels')
        # QInfer axes: outcomes, particles, experiments.
        p = self.probabilities(modelparams,raw.astype(int))
        return np.repeat(p.T[:,:,None],len(expparams),axis=2)

class NormalPrior(qi.Distribution):
    def __init__(self, k, rng): self.k,self.rng=k,rng
    @property
    def n_rvs(self): return self.k
    def sample(self,n=1): return self.rng.normal(size=(n,self.k))

class MoveResampler:
    """pCN is reversible for the Gaussian prior on the Ginibre factors.

    Its prior and proposal terms cancel in the Metropolis ratio, leaving
    only the full-data likelihood ratio. The proposal scale depends on
    event count alone; neither truth nor an MLE enters the inference.
    Finite particle count and finite mixing still need numerical checks.
    """
    def __init__(self,rng,steps):
        self.rng,self.steps=rng,steps
        self.counts=None
        self.acceptance=[]
        self.events=[]

    def log_likelihood(self,model,x):
        used=np.flatnonzero(self.counts)
        if not len(used):
            return np.zeros(len(x))
        # Zero probability for an observed effect means log L = -infinity.
        with np.errstate(divide='ignore'):
            return np.log(model.probabilities(x,used))@self.counts[used]

    def __call__(self,model,w,x):
        # Multinomial resampling, followed by fixed-kernel pCN moves.
        x=x[self.rng.choice(len(w),len(w),p=w)].copy()
        logl=self.log_likelihood(model,x)
        if not np.isfinite(logl).all():
            raise RuntimeError('Resampled particles have zero likelihood')
        n=int(self.counts.sum())
        beta=.5/np.sqrt(1+n/100)
        accepted=0
        for _ in range(self.steps):
            proposal=np.sqrt(1-beta**2)*x+beta*self.rng.normal(size=x.shape)
            prop_ll=self.log_likelihood(model,proposal)
            take=np.log(self.rng.random(len(x)))<prop_ll-logl
            x[take]=proposal[take]
            logl[take]=prop_ll[take]
            accepted+=take.sum()
        self.acceptance.append(accepted/(len(w)*self.steps))
        self.events.append(n)
        return np.full(len(w),1/len(w)),x


def run_smc_bayes(M,q,outcomes,rho_true,d,rng,n_particles=3000,
                  ess_frac=.5,record_every=10,move_steps=60,diagnostics=None):
    """Return (posterior mean state, event indices, distances, ESS fractions).

    Drop-in replacement for run_iva.py's existing call. q is unused and
    retained for compatibility. rho_true is used ONLY for error diagnostics.
    All randomness uses rng, without altering NumPy's global RNG.
    Optional diagnostics is a dict populated with resampling information.
    """
    for value,minimum,name in [(n_particles,2,'n_particles'),
                               (record_every,1,'record_every'),
                               (move_steps,1,'move_steps')]:
        if not isinstance(value,(int,np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not 0 < ess_frac <= 1:
        raise ValueError('ess_frac must be in (0,1]')
    model=FactorModel(M)
    if model.d != d or np.asarray(rho_true).shape != (d,d):
        raise ValueError('State dimension does not match POVM')
    obs=np.asarray(outcomes)
    if (obs.ndim != 1 or not np.isfinite(obs).all() or
            np.any(obs != np.floor(obs)) or np.any(obs < 0) or
            np.any(obs >= model.m)):
        raise ValueError('outcomes must be a one-dimensional array of valid labels')
    obs=obs.astype(int)
    mover=MoveResampler(rng,move_steps); mover.counts=np.zeros(model.m,int)
    updater=qi.SMCUpdater(model,n_particles,NormalPrior(model.n_modelparams,rng),
        resampler=mover,resample_thresh=ess_frac,canonicalize=False,zero_weight_policy='error')
    exp=np.zeros(1,dtype=model.expparams_dtype)
    indices,distances,ess_history=[],[],[]

    def record(n):
        # Nonlinear map: average density matrices, NEVER average factors first.
        est=np.einsum('p,pab->ab',updater.particle_weights,
                      model.states(updater.particle_locations))
        delta=est-rho_true
        dist=.5*np.abs(np.linalg.eigvalsh(.5*(delta+delta.conj().T))).sum()
        indices.append(n); distances.append(float(dist))
        ess_history.append(float(updater.n_ess/n_particles))
        return est

    est=record(0)
    for n,y in enumerate(obs,1):
        # Include the new event BEFORE QInfer can trigger resampling.
        mover.counts[y]+=1
        updater.update(int(y),exp)
        if n%record_every==0 or n==len(obs):
            est=record(n)
    if diagnostics is not None:
        diagnostics.update(
            method='QInfer SMC + multinomial resampling + pCN MH',
            prior='Hilbert-Schmidt (square complex Ginibre)',
            n_particles=int(n_particles),move_steps=int(move_steps),
            ess_threshold=float(ess_frac),
            resample_events=np.asarray(mover.events,dtype=int),
            move_acceptance=np.asarray(mover.acceptance),
            qinfer_version=str(qi.__version__))
    return est,np.asarray(indices),np.asarray(distances),np.asarray(ess_history)
