import os, sys, time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'vendor'))   # parameter_bounds.py, plague_double_sigmoid_model.py, simulation.py
from staged_model import *
from simulation import simulate_vetlyanka
from parameter_bounds import BOUNDS, PARAM_NAMES

def sse(a, b): return float(np.sum((a - b) ** 2))

def aic(n, k, s): return n * np.log(s / n) + 2 * k
def bic(n, k, s): return n * np.log(s / n) + k * np.log(n)
def aicc(n, k, s): return aic(n, k, s) + 2 * k * (k + 1) / (n - k - 1)

# ---------- PNAS double-sigmoid fit (same recipe as the paper: LHS scan + multistart L-BFGS-B, x1-x0 >= 1) ----------
def dbl_obj(x, obs):
    x = np.asarray(x)
    if x[3] - x[2] < 1.0: return 1e9
    sim = simulate_vetlyanka(x)
    return 1e9 if sim is None else sse(sim, obs)

def fit_double(obs, n_lhs=2000, top_k=4, seed=0):
    lo = np.array([b[0] for b in BOUNDS]); hi = np.array([b[1] for b in BOUNDS])
    X = lo + qmc.LatinHypercube(d=13, seed=seed).random(n_lhs) * (hi - lo)
    sc = np.array([dbl_obj(x, obs) for x in X])
    idx = np.argsort(sc)[:top_k]
    best = (np.inf, None)
    for i in idx:
        r = minimize(dbl_obj, X[i], args=(obs,), method='L-BFGS-B', bounds=BOUNDS, options=dict(maxiter=200, maxfun=2000))
        if r.fun < best[0]: best = (r.fun, r.x)
    return best

# ---------- staged fits ----------
STAGED_BOUNDS = dict(lam=(1, 2000), beta=(0.05, 4.0), tau_E=(0.2, 6.0), e0=(-1.5, 1.5), tau_F=(0.2, 6.0), sens=(0.0, 0.2), bH=(-1.5, 1.5))
def staged_pred(theta, names, clim, base=None):
    p = dict(TRUTH if base is None else base)
    for n, v in zip(names, theta): p[n] = v
    return weekly_DR(p, clim)

def staged_obj(theta, names, obs, clim, base=None):
    for nm, v in zip(names, theta):
        lo, hi = STAGED_BOUNDS[nm]
        if not (lo <= v <= hi):
            return 1e9
    pred = staged_pred(theta, names, clim, base)
    return 1e9 if pred is None else sse(pred, obs)

def fit_staged(obs, names, clim, n_starts=40, seed=1, base=None, n_polish=5, extra_starts=None):
    rng = np.random.default_rng(seed)
    bnds = [STAGED_BOUNDS[n] for n in names]
    best = (np.inf, None)
    starts = []
    for _ in range(n_starts):
        starts.append([rng.uniform(*b) if n not in ('lam',) else np.exp(rng.uniform(np.log(5), np.log(1500))) for n, b in zip(names, bnds)])
    scores = [staged_obj(s, names, obs, clim, base) for s in starts]
    chosen = list(np.argsort(scores)[:n_polish])
    for e in (extra_starts or []):          # e.g. the solution of a simpler, nested model: guarantees the richer model fits at least as well
        starts.append(list(e)); scores.append(staged_obj(e, names, obs, clim, base)); chosen.append(len(starts) - 1)
    for i in chosen:
        r = minimize(staged_obj, starts[i], args=(names, obs, clim, base), method='L-BFGS-B', bounds=bnds,
                     options=dict(maxiter=150, maxfun=1500))
        r2 = minimize(staged_obj, r.x, args=(names, obs, clim, base), method='Nelder-Mead',
                      bounds=bnds, options=dict(maxiter=600, maxfev=900, xatol=1e-4, fatol=1e-4))
        for rr in (r, r2):
            rr_x = np.clip(rr.x, [b[0] for b in bnds], [b[1] for b in bnds])
            f = staged_obj(rr_x, names, obs, clim, base)
            if f < best[0]: best = (f, rr_x)
    return best

def synth_data(seed, p=None):
    p = dict(TRUTH if p is None else p)
    dr = weekly_DR(p, climate)
    inc = np.diff(np.r_[0.0, dr])
    rng = np.random.default_rng(seed)
    noisy = rng.poisson(inc)
    return np.cumsum(noisy).astype(float), dr