"""
petz_lib.py -- core objects for the Section IV numerics.

Conventions
-----------
POVM      : array M of shape (m, d, d), Hermitian, sum_j M[j] = I.
Evidence  : q[j] = Tr[M_j rho], exact (asymptotic / population regime).
R operator: R(sigma) = sum_j (q_j / Tr[M_j sigma]) M_j.

Batch maps (all trace preserving on the nose when q_j = Tr[M_j rho]):
    Petz    : sigma -> sqrt(sigma) R sqrt(sigma)
    mirror  : sigma -> sqrt(R) sigma sqrt(R)
    R-rho-R : sigma -> R sigma R / Tr[R sigma R]

Note Tr[R sigma] = sum_j q_j = 1 identically, which is why Petz and mirror
need no renormalisation while R-rho-R does.

KKT / optimality diagnostic: lambda_max(R(sigma)) - 1.  Zero at a global MLE,
strictly positive at a spurious fixed point.

Distance: trace distance D(a,b) = 0.5 * ||a - b||_1.
"""

import numpy as np

# --------------------------------------------------------------------------
# linear algebra helpers
# --------------------------------------------------------------------------

def _herm(a):
    return 0.5 * (a + a.conj().T)


def msqrt(a, floor=0.0):
    """PSD matrix square root via eigendecomposition, negatives clipped."""
    w, v = np.linalg.eigh(_herm(a))
    w = np.clip(w, floor, None)
    return (v * np.sqrt(w)) @ v.conj().T


def project_state(a, floor=0.0):
    """Nearest-ish valid density matrix: hermitise, clip, renormalise trace."""
    w, v = np.linalg.eigh(_herm(a))
    w = np.clip(w, floor, None)
    s = w.sum()
    if s <= 0:
        return np.eye(a.shape[0], dtype=complex) / a.shape[0]
    w = w / s
    return (v * w) @ v.conj().T


def trace_distance(a, b):
    """D = 0.5 * ||a - b||_1."""
    w = np.linalg.eigvalsh(_herm(a - b))
    return 0.5 * np.abs(w).sum()


def purity(rho):
    return float(np.real(np.trace(rho @ rho)))


# --------------------------------------------------------------------------
# random states and POVMs
# --------------------------------------------------------------------------

def random_pure(d, rng):
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    v /= np.linalg.norm(v)
    return np.outer(v, v.conj())


def random_hs_state(d, rng, k=None):
    """Ginibre / Hilbert-Schmidt random density matrix (k = d gives HS measure)."""
    k = d if k is None else k
    g = rng.normal(size=(d, k)) + 1j * rng.normal(size=(d, k))
    a = g @ g.conj().T
    return a / np.trace(a).real


def state_from_eps(d, eps, rng, psi=None):
    """rho_eps = (1-eps)|psi><psi| + eps I/d.  One monotone purity knob."""
    p = random_pure(d, rng) if psi is None else psi
    return (1.0 - eps) * p + eps * np.eye(d) / d


def random_ic_povm(d, rng, m=None, max_tries=50):
    """
    m rank-1 effects, normalised to resolve the identity via the
    S^{-1/2} A S^{-1/2} construction.  Default m = d^2 + 2 (overcomplete IC).
    Informational completeness is verified, not assumed.
    """
    m = d * d + 2 if m is None else m
    for _ in range(max_tries):
        A = np.stack([random_pure(d, rng) for _ in range(m)])
        S = A.sum(axis=0)
        w, v = np.linalg.eigh(_herm(S))
        if w.min() <= 1e-10:
            continue
        Sm = (v * (1.0 / np.sqrt(w))) @ v.conj().T
        M = np.einsum('ab,jbc,cd->jad', Sm, A, Sm)
        M = 0.5 * (M + np.conj(np.transpose(M, (0, 2, 1))))
        if is_ic(M):
            return M
    raise RuntimeError("failed to build an IC POVM")


def is_ic(M, tol=1e-9):
    """IC iff the effects span the full d^2-dimensional Hermitian space."""
    m, d, _ = M.shape
    V = M.reshape(m, d * d)
    V = np.concatenate([V.real, V.imag], axis=1)
    return np.linalg.matrix_rank(V, tol=tol) >= d * d


# --------------------------------------------------------------------------
# evidence and the R operator
# --------------------------------------------------------------------------

def born(M, rho):
    """q_j = Tr[M_j rho], real, clipped to be safe."""
    q = np.einsum('jab,ba->j', M, rho).real
    return np.clip(q, 0.0, None)


def R_operator(M, q, sigma, floor=1e-300):
    p = np.clip(born(M, sigma), floor, None)
    return _herm(np.einsum('j,jab->ab', q / p, M))


def kkt_gap(M, q, sigma):
    """lambda_max(R(sigma)) - 1.  Optimality of sigma iff this is <= 0."""
    return float(np.linalg.eigvalsh(R_operator(M, q, sigma)).max() - 1.0)


# --------------------------------------------------------------------------
# the three batch maps
# --------------------------------------------------------------------------

def petz_step(M, q, sigma):
    R = R_operator(M, q, sigma)
    s = msqrt(sigma)
    out = _herm(s @ R @ s)
    return out / np.trace(out).real


def mirror_step(M, q, sigma):
    R = R_operator(M, q, sigma)
    r = msqrt(R)
    out = _herm(r @ sigma @ r)
    return out / np.trace(out).real


def rrr_step(M, q, sigma):
    R = R_operator(M, q, sigma)
    out = _herm(R @ sigma @ R)
    return out / np.trace(out).real


BATCH_MAPS = {'petz': petz_step, 'mirror': mirror_step, 'rrr': rrr_step}


def run_batch(M, q, sigma0, step, n_iter, rho_true=None, track=False,
              stall_tol=1e-14):
    """
    Iterate a batch map.  Returns (sigma_final, history) where history is a dict
    of arrays if track=True, else empty.  Early exit on numerical stalling.
    """
    sigma = sigma0.copy()
    hist = {'dist': [], 'kkt': [], 'lmin': []} if track else None
    for i in range(n_iter):
        if track:
            hist['dist'].append(trace_distance(sigma, rho_true))
            hist['kkt'].append(kkt_gap(M, q, sigma))
            hist['lmin'].append(float(np.linalg.eigvalsh(sigma).min()))
        nxt = step(M, q, sigma)
        move = np.abs(nxt - sigma).max()
        sigma = nxt
        if move < stall_tol:
            if track:
                hist['dist'].append(trace_distance(sigma, rho_true))
                hist['kkt'].append(kkt_gap(M, q, sigma))
                hist['lmin'].append(float(np.linalg.eigvalsh(sigma).min()))
            break
    if track:
        return sigma, {k: np.asarray(v) for k, v in hist.items()}
    return sigma, {}


# --------------------------------------------------------------------------
# protocol 1: event-wise unextended Petz  (Sec. II B)
# --------------------------------------------------------------------------

def run_event_petz(M, q, sigma0, outcomes, rho_true, record_every=1):
    sigma = sigma0.copy()
    idx, dist = [0], [trace_distance(sigma, rho_true)]
    for n, y in enumerate(outcomes, start=1):
        s = msqrt(sigma)
        num = _herm(s @ M[y] @ s)
        tr = np.trace(num).real
        if tr <= 0:
            break
        sigma = num / tr
        if n % record_every == 0:
            idx.append(n)
            dist.append(trace_distance(sigma, rho_true))
    return sigma, np.asarray(idx), np.asarray(dist)


# --------------------------------------------------------------------------
# protocol 3: event-wise extended Petz == Bayesian tomography, SMC
#             (Granade, Combes & Cory, NJP 18, 033024 (2016))
# --------------------------------------------------------------------------

def gellmann_basis(d):
    """Orthonormal traceless Hermitian basis: Tr[B_j B_k] = delta_jk."""
    B = []
    for j in range(d):
        for k in range(j + 1, d):
            x = np.zeros((d, d), dtype=complex); x[j, k] = 1; x[k, j] = 1
            B.append(x / np.sqrt(2))
            y = np.zeros((d, d), dtype=complex); y[j, k] = -1j; y[k, j] = 1j
            B.append(y / np.sqrt(2))
    for l in range(1, d):
        z = np.zeros((d, d), dtype=complex)
        for j in range(l):
            z[j, j] = 1
        z[l, l] = -l
        B.append(z / np.sqrt(l * (l + 1)))
    return np.stack(B)


def states_to_vecs(states, B):
    return np.einsum('kab,xba->xk', B, states).real


def vecs_to_states(vecs, B, d):
    rho = np.eye(d, dtype=complex)[None, :, :] / d + np.einsum('xk,kab->xab', vecs, B)
    return np.stack([project_state(r) for r in rho])


def make_cse(d, n_cand, rng, rho_true=None, include_true=True):
    """
    Candidate state ensemble.  The true state is included by construction so
    that rho lies in the convex hull, which Sec. III C requires; hull
    membership is separately certified by cse_hull_certificate.
    """
    cands = [random_hs_state(d, rng, k=rng.integers(1, d + 1)) for _ in range(n_cand)]
    if include_true and rho_true is not None:
        cands[0] = rho_true.copy()
    return np.stack(cands)


def cse_hull_certificate(cands, rho, B=None):
    """
    Least-squares residual of rho against the convex hull of the candidates,
    solved as a non-negative least squares on the Hermitian coordinates.
    Residual ~ 0 certifies rho in conv{rho_x}.
    """
    from scipy.optimize import nnls
    d = rho.shape[0]
    B = gellmann_basis(d) if B is None else B
    A = np.concatenate([states_to_vecs(cands, B).T,
                        np.ones((1, len(cands)))], axis=0)
    b = np.concatenate([states_to_vecs(rho[None], B)[0], [1.0]])
    _, res = nnls(A, b)
    return float(res)


def run_cse_batch(M, q, cands, p0, n_iter, rho_true, track=True,
                  stall_tol=1e-15):
    """
    p_{i+1}(x) = p_i(x) sum_j q_j A[j,x] / (sum_x' p_i(x') A[j,x'])
    with A[j,x] = Tr[M_j rho_x], and rho_hat_i = sum_x p_i(x) rho_x.
    """
    A = np.einsum('jab,xba->jx', M, cands).real
    A = np.clip(A, 0.0, None)
    p = p0.copy()
    idx, dist, kkt = [], [], []
    for i in range(n_iter):
        est = np.einsum('x,xab->ab', p, cands)
        if track:
            idx.append(i)
            dist.append(trace_distance(est, rho_true))
            kkt.append(kkt_gap(M, q, est))
        denom = np.clip(A @ p, 1e-300, None)
        p_new = p * (A.T @ (q / denom))
        p_new /= p_new.sum()
        move = np.abs(p_new - p).max()
        p = p_new
        if move < stall_tol:
            break
    est = np.einsum('x,xab->ab', p, cands)
    if track:
        idx.append(len(idx)); dist.append(trace_distance(est, rho_true))
        kkt.append(kkt_gap(M, q, est))
    return est, p, np.asarray(idx), np.asarray(dist), np.asarray(kkt)


# --------------------------------------------------------------------------
# stacked (vectorised-over-trials) batch iteration for the Sec. IV B sweep
# --------------------------------------------------------------------------

def _msqrt_stack(A, floor=0.0):
    w, v = np.linalg.eigh(0.5 * (A + np.conj(np.swapaxes(A, -1, -2))))
    w = np.clip(w, floor, None)
    return (v * np.sqrt(w)[..., None, :]) @ np.conj(np.swapaxes(v, -1, -2))


def _R_stack(M, q, sig, floor=1e-300):
    p = np.clip(np.einsum('tjab,tba->tj', M, sig).real, floor, None)
    R = np.einsum('tj,tjab->tab', q / p, M)
    return 0.5 * (R + np.conj(np.swapaxes(R, -1, -2)))


def _norm_stack(X):
    X = 0.5 * (X + np.conj(np.swapaxes(X, -1, -2)))
    tr = np.einsum('taa->t', X).real
    return X / tr[:, None, None]


def petz_step_stack(M, q, sig):
    R = _R_stack(M, q, sig)
    s = _msqrt_stack(sig)
    return _norm_stack(s @ R @ s)


def mirror_step_stack(M, q, sig):
    R = _R_stack(M, q, sig)
    r = _msqrt_stack(R)
    return _norm_stack(r @ sig @ r)


def rrr_step_stack(M, q, sig):
    R = _R_stack(M, q, sig)
    return _norm_stack(R @ sig @ R)


BATCH_MAPS_STACK = {'petz': petz_step_stack,
                    'mirror': mirror_step_stack,
                    'rrr': rrr_step_stack}


def trace_distance_stack(A, B):
    w = np.linalg.eigvalsh(0.5 * ((A - B) + np.conj(np.swapaxes(A - B, -1, -2))))
    return 0.5 * np.abs(w).sum(axis=-1)


def diagnostics_stack(M, q, sig, rho):
    """Per-trial (trace distance, lambda_max(R)-1, lambda_min(sigma))."""
    dist = trace_distance_stack(sig, rho)
    kkt = np.linalg.eigvalsh(_R_stack(M, q, sig)).max(axis=-1) - 1.0
    lmin = np.linalg.eigvalsh(0.5 * (sig + np.conj(np.swapaxes(sig, -1, -2)))).min(axis=-1)
    return dist, kkt, lmin


def run_batch_stack(M, q, sig0, step, n_iter, rho, tau=1e-3, check_every=100):
    """
    Iterate a batch map over a stack of trials.  A trial is dropped from the
    stack as soon as it comes within tau of rho -- that is the criterion
    itself, so once met the verdict cannot change.  Everything else runs to
    n_iter and is classified by its final distance.

    Returns (dist, kkt, lmin, iters, move, converged), all length T:

      dist       trace distance at exit
      kkt        lambda_max(R) - 1 at exit
      lmin       lambda_min(sigma) at exit
      iters      iteration at exit
      move       max|sigma_{i+1} - sigma_i| at exit
      converged  True if the trial reached tau before the budget ran out

    `move` is recorded but not used in the criterion.  It is what separates a
    trial sitting at a fixed point from one still descending at O(1/i) when
    the budget ended, so if the capture rate is to be quoted, check that the
    captured trials have small `move`; otherwise n_iter is the binding
    constraint and the rate is a statement about the budget.
    """
    T = sig0.shape[0]
    alive = np.arange(T)
    sig = sig0.copy()
    Mi, qi, rhoi = M, q, rho
    dist = np.full(T, np.nan); kkt = np.full(T, np.nan)
    lmin = np.full(T, np.nan); move = np.full(T, np.nan)
    iters = np.full(T, n_iter, dtype=int)
    converged = np.zeros(T, dtype=bool)

    for i in range(1, n_iter + 1):
        nxt = step(Mi, qi, sig)
        mv = np.abs(nxt - sig).reshape(len(alive), -1).max(axis=1)
        sig = nxt
        if i % check_every == 0 or i == n_iter:
            d_, k_, l_ = diagnostics_stack(Mi, qi, sig, rhoi)
            done = d_ < tau
            conv = done.copy()
            if i == n_iter:
                done = np.ones_like(done, dtype=bool)
            if done.any():
                idx = alive[done]
                dist[idx], kkt[idx], lmin[idx] = d_[done], k_[done], l_[done]
                move[idx], iters[idx] = mv[done], i
                converged[idx] = conv[done]
                keep = ~done
                alive = alive[keep]
                sig, Mi, qi, rhoi = sig[keep], Mi[keep], qi[keep], rhoi[keep]
            if len(alive) == 0:
                break
    return dist, kkt, lmin, iters, move, converged
