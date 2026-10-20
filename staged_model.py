"""
Staged (framework) model: climate C -> environment E -> ecology F -> spillover S -> humans (H: perception P; T: S,I,R,corpses).
Human block follows the Vetlyanka model (constant beta, gamma, mu) so the only
thing that can end the epidemic is the climate-driven ecological stage, behaviour and depletion.
"""
import numpy as np
from scipy.integrate import solve_ivp

KE = 4.0
ALPHA = 0.000641025641025641
N_WEEKS = 22
T_OBS = np.arange(1, N_WEEKS + 1, dtype=float)
Y0 = [1699.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # S, I, R, DI, DDI, DR, P

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def climate(t, t_on=12.0, t_off=15.0, k=2.0, amp=1.5, base=None):
    """Artificial climate favourability (standardised): unfavourable level (-amp, or `base` if given), then a favourable
    window (t_on..t_off) that reaches +amp, then unfavourable again."""
    lo = -amp if base is None else base
    return lo + (amp - lo) * (sig(k * (t - t_on)) - sig(k * (t - t_off)))

HUMAN_FIXED = dict(gamma=0.5, mu=1.2, dispose=5.0, T_perceive=3.0)

def staged_rhs(t, y, p, clim):
    S, I, R, DI, DDI, DR, P, E, F = y
    lam, beta, tau_E, e0, tau_F, sens = p['lam'], p['beta'], p['tau_E'], p['e0'], p['tau_F'], p['sens']
    g, mu, disp, Tp = HUMAN_FIXED['gamma'], HUMAN_FIXED['mu'], HUMAN_FIXED['dispose'], HUMAN_FIXED['T_perceive']
    N = S + I + R
    C = clim(t) if clim is not None else p.get('C_const', 1.5)
    # environment and ecology
    dE = (C - E) / tau_E
    F_target = sig(KE * (E - e0))
    dF = (F_target - F) / tau_F
    # exposure / spillover (output) and human amplification
    spill = lam * F
    # direct route: climate also acts on person-to-person transmission (bH > 0: favourable climate lowers it; bH = 0: no direct route)
    beta_t = beta * np.exp(-p.get('bH', 0.0) * C)
    force = (spill + beta_t * (I + DI)) * (S / N) * np.exp(-sens * P)
    dS = -force - ALPHA * S
    dI = force - ALPHA * I - mu * I - g * I
    dR = g * I - ALPHA * R
    dDI = mu * I - disp * DI
    dDDI = mu * I
    dDR = mu * I + ALPHA * (S + I + R)
    dP = (mu * I - P) / Tp
    return [dS, dI, dR, dDI, dDDI, dDR, dP, dE, dF]

def simulate_staged(p, clim=climate, t_end=N_WEEKS, dense=False):
    E0 = clim(0.0) if clim is not None else p.get('C_const', 1.5)
    F0 = sig(KE * (E0 - p["e0"]))
    y0 = Y0 + [E0, F0]
    sol = solve_ivp(staged_rhs, [0, t_end], y0, args=(p, clim), t_eval=T_OBS if not dense else np.linspace(0, t_end, 441),
                    method='LSODA', rtol=1e-6, atol=1e-8)
    return sol

def weekly_DR(p, clim=climate):
    sol = simulate_staged(p, clim)
    if (not sol.success) or sol.y.shape[1] != N_WEEKS or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y[5]

TRUTH = dict(lam=200.0, beta=1.1, tau_E=1.0, e0=0.0, tau_F=0.5, sens=0.02)   # example 1: climate acts only through ecology
TRUTH2 = dict(lam=250.0, beta=0.45, tau_E=1.0, e0=0.0, tau_F=0.5, sens=0.02, bH=0.6)   # example 2: climate also acts on human transmission