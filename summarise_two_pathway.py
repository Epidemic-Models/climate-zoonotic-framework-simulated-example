"""
Read results/two_pathway_seed*.json; write results/two_pathway_summary.csv, results/two_pathway_counterfactual.csv
and figures/figure_two_pathway.png/.pdf.  All data are SIMULATED.
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from staged_model import climate, TRUTH2

HERE = os.path.dirname(os.path.abspath(__file__))
res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, 'results', 'two_pathway_seed*.json')))]
if not res: raise SystemExit('No two_pathway results found. Run run_all.py first.')

LAB = {'one_pathway': 'Climate on ecology only', 'two_pathway': 'Climate on ecology and on human transmission', 'double': 'Double sigmoid (mortality only)'}
rows = []
for r in res:
    for k in ('one_pathway', 'two_pathway', 'double'):
        d = r[k]
        rows.append(dict(seed=r['seed'], model=LAB[k], k=d['k'], SSE=d['sse'], AIC=d['aic'], AICc=d['aicc'], BIC=d['bic'],
                         bH_estimate=d['params']['bH'] if 'params' in d else float('nan'), bH_truth=TRUTH2['bH'],
                         shift_week_x1=d.get('x1', float('nan'))))
tab = pd.DataFrame(rows); tab.to_csv(os.path.join(HERE, 'results', 'two_pathway_summary.csv'), index=False)
print(tab.round(2).to_string(index=False))

cfr = []
for r in res:
    for k in ('one_pathway', 'two_pathway', 'double'):
        for c, v in r[k]['cf'].items():
            cfr.append(dict(seed=r['seed'], model=LAB[k], climate=c, truth_total=v['truth'][-1], predicted_total=v['pred'][-1],
                            error_pct=100 * (v['pred'][-1] - v['truth'][-1]) / v['truth'][-1],
                            # change relative to the same model's total under the original climate (direction of the response)
                            truth_change=v['truth'][-1] - r['truth_dr'][-1], predicted_change=v['pred'][-1] - r[k]['pred'][-1]))
cf = pd.DataFrame(cfr); cf.to_csv(os.path.join(HERE, 'results', 'two_pathway_counterfactual.csv'), index=False)
print(cf.round(1).to_string(index=False))

BLUE, ORANGE, GREEN, INK, MUTED = '#0072B2', '#D55E00', '#009E73', '#222222', '#777777'
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
inc = lambda c: np.diff(np.r_[0.0, np.array(c)])
weeks = np.arange(1, 23)
mid = sorted(range(len(res)), key=lambda i: res[i]['one_pathway']['aic'] - res[i]['two_pathway']['aic'])[len(res) // 2]
r = res[mid]
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
ax[0].plot(weeks, inc(r['obs']), 'o', color=INK, ms=4, label='Simulated deaths')
ax[0].plot(weeks, inc(r['one_pathway']['pred']), color=ORANGE, lw=4.5, alpha=0.55, label='Ecology only')
ax[0].plot(weeks, inc(r['two_pathway']['pred']), color=BLUE, lw=1.6, ls='--', label='Two pathways')
ax[0].plot(weeks, inc(r['double']['pred']), color=GREEN, lw=1.8, ls=':', label='Double sigmoid')
ax[0].set_xlabel('Week'); ax[0].set_ylabel('Deaths per week'); ax[0].set_title('a  Fits to simulated deaths', loc='left', fontsize=9)
ax[0].legend(frameon=False, fontsize=8, loc='upper left')
x = np.arange(len(res)); ax[1].axhline(TRUTH2['bH'], color=INK, lw=1.5)
ax[1].plot(x, [q['two_pathway']['params']['bH'] for q in res], 'o', color=BLUE, ms=6)
ax[1].text(len(res) - 0.6, TRUTH2['bH'] + 0.03, 'true value (0.6)', ha='right', color=INK, fontsize=8)
ax[1].text(0.0, 0.03, 'ecology-only model: fixed at 0, not estimated', ha='left', color=ORANGE, fontsize=7.5)
ax[1].set_ylim(-0.1, 1.65)
ax[1].set_xticks(x); ax[1].set_xticklabels([f'seed {q["seed"]}' for q in res], fontsize=8)
ax[1].set_ylabel(r'Climate effect on human transmission, $b_H$'); ax[1].set_title('b  Recovered effect on people', loc='left', fontsize=9)
cfd = r['two_pathway']['cf']['harsher_baseline']; cfo = r['one_pathway']['cf']['harsher_baseline']
ax[2].plot(weeks, inc(r['truth_dr']), color=MUTED, lw=1.2, label='Truth, original climate')
ax[2].plot(weeks, inc(cfd['truth']), color=INK, lw=2, label='Truth, harsher climate')
ax[2].plot(weeks, inc(cfd['pred']), color=BLUE, lw=2, ls='--', label='Two pathways')
ax[2].plot(weeks, inc(cfo['pred']), color=ORANGE, lw=2, ls='--', label='Ecology only')
ax[2].plot(weeks, inc(r['double']['cf']['harsher_baseline']['pred']), color=GREEN, lw=2, ls=':', label='Double sigmoid')
ax[2].set_xlabel('Week'); ax[2].set_ylabel('Deaths per week'); ax[2].set_title('c  New, harsher climate', loc='left', fontsize=9)
ax[2].legend(frameon=False, fontsize=8, loc='upper left')
for a in ax: a.grid(axis='y', color='#e5e5e5', lw=0.6); a.set_axisbelow(True)
fig.tight_layout(); os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
for ext in ('png', 'pdf'): fig.savefig(os.path.join(HERE, 'figures', f'figure_two_pathway.{ext}'), dpi=300)
print('figure from seed', r['seed'])