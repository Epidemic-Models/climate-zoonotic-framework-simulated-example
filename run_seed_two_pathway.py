"""
Example 2 (two pathways). Truth = TRUTH2: climate raises spillover through the ecological stage AND lowers
transmission among people (bH > 0). One dataset per seed (Poisson noise). Fits:
  one_pathway : climate acts only through the ecological stage (bH fixed at 0), 6 free parameters
  two_pathway : climate also acts on human transmission (bH free), 7 free parameters
  double      : the double-sigmoid regime-shift model (mortality only, no climate input), 13 free parameters
Then predicts deaths under three climates the fits never saw (earlier window, longer window, harsher unfavourable periods).

    python run_seed_two_pathway.py 1        (set QUICK=1 in the environment for a fast smoke test)
"""
import os, sys, json, time
HERE_ = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from fit_lib import *

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
quick = os.environ.get('QUICK') == '1'
rng = np.random.default_rng(seed)
truth_dr = weekly_DR(dict(TRUTH2), climate)
inc = np.diff(np.r_[0.0, truth_dr])
obs = np.cumsum(rng.poisson(inc)).astype(float)
n = len(obs)
t0 = time.time()

CLIMATES = {
    'earlier_window': lambda t: climate(t, t_on=9.0, t_off=12.0),
    'longer_window': lambda t: climate(t, t_on=12.0, t_off=18.0),
    'harsher_baseline': lambda t: climate(t, base=-2.5),   # unfavourable periods more unfavourable (-2.5 instead of -1.5); favourable peak unchanged
}
out = dict(seed=seed, obs=obs.tolist(), truth_dr=truth_dr.tolist(), truth_params=TRUTH2)

# double-sigmoid regime-shift model (does not use climate)
print(f'seed {seed}: fitting the double-sigmoid model...', flush=True)
fA, xA = fit_double(obs, n_lhs=300 if quick else 2000, top_k=2 if quick else 4, seed=seed)
print(f'seed {seed}: double sigmoid done after {round(time.time()-t0)} s', flush=True)
predA = simulate_vetlyanka(xA)
out['double'] = dict(sse=fA, k=13, aic=aic(n, 13, fA), aicc=aicc(n, 13, fA), bic=bic(n, 13, fA), pred=predA.tolist(),
                     x0=float(xA[2]), x1=float(xA[3]),
                     cf={c: dict(pred=predA.tolist(), truth=weekly_DR(dict(TRUTH2), cl).tolist()) for c, cl in CLIMATES.items()})
for key, names, bH0 in (('one_pathway', ['lam', 'beta', 'tau_E', 'e0', 'tau_F', 'sens'], 0.0),
                        ('two_pathway', ['lam', 'beta', 'tau_E', 'e0', 'tau_F', 'sens', 'bH'], None)):
    base = dict(TRUTH2)
    if bH0 is not None: base['bH'] = bH0
    print(f'seed {seed}: fitting {key}...', flush=True)
    extra = None
    if key == 'two_pathway':   # the one-pathway model is the special case bH = 0, so start from its solution
        x1 = [out['one_pathway']['params'][a] for a in names[:-1]]
        extra = [x1 + [b] for b in (0.0, 0.3, 0.6)]
    f, x = fit_staged(obs, names, climate, n_starts=8 if quick else 40, seed=seed, base=base, n_polish=2 if quick else 5,
                      extra_starts=extra)
    print(f'seed {seed}: {key} done after {round(time.time()-t0)} s', flush=True)
    p = dict(base); p.update(dict(zip(names, map(float, x))))
    d = dict(sse=f, k=len(names), params={a: float(p[a]) for a in p}, aic=aic(n, len(names), f),
             aicc=aicc(n, len(names), f), bic=bic(n, len(names), f), pred=weekly_DR(p, climate).tolist(), cf={})
    for cname, cl in CLIMATES.items():
        d['cf'][cname] = dict(pred=weekly_DR(p, cl).tolist(), truth=weekly_DR(dict(TRUTH2), cl).tolist())
    out[key] = d
out['elapsed_s'] = time.time() - t0
os.makedirs(os.path.join(HERE_, 'results'), exist_ok=True)
json.dump(out, open(os.path.join(HERE_, 'results', f'two_pathway_seed{seed}.json'), 'w'))
print('done', seed, round(out['elapsed_s']))