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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
        for cname, v in r[k]['cf'].items():
            cfr.append(dict(seed=r['seed'], model=LAB[k], climate=cname, truth_total=v['truth'][-1], predicted_total=v['pred'][-1],
                            error_pct=100 * (v['pred'][-1] - v['truth'][-1]) / v['truth'][-1],
                            # change relative to the same model's total under the original climate (direction of the response)
                            truth_change=v['truth'][-1] - r['truth_dr'][-1], predicted_change=v['pred'][-1] - r[k]['pred'][-1]))
cf = pd.DataFrame(cfr); cf.to_csv(os.path.join(HERE, 'results', 'two_pathway_counterfactual.csv'), index=False)
print(cf.round(1).to_string(index=False))

# =============================================================================
# Figure
# =============================================================================
BH = r'$b_H$'   # symbol shown on the figure; same as in the paper

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 11, "legend.fontsize": 9,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.linewidth": 0.9,
    "xtick.direction": "out", "ytick.direction": "out",
})
C_DATA = "#1f4e8c"      # simulated deaths and the true values
C_TWO = "#E08214"       # two-pathway model
C_ECO = "#2C7C2C"       # ecology-only model
C_DBL = "#7B3F98"       # double-sigmoid model
C_REF = "#7f7f7f"       # reference lines
C_WIN = "#5aaa5a"       # favourable climate window (pale green band)

inc = lambda cum: np.diff(np.r_[0.0, np.array(cum)])
weeks = np.arange(1, 23)
mid = sorted(range(len(res)), key=lambda i: res[i]['one_pathway']['aic'] - res[i]['two_pathway']['aic'])[len(res) // 2]
r = res[mid]
T_ON, T_OFF = 12.0, 15.0   # favourable climate window of the generating climate

# about the printed size (7.2 in), so the text stays readable when placed in the paper
fig = plt.figure(figsize=(7.2, 6.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.34)
a = fig.add_subplot(gs[0, :])
b = fig.add_subplot(gs[1, 0])
c = fig.add_subplot(gs[1, 1])

# (a) fits to the simulated deaths
a.axvspan(T_ON, T_OFF, color=C_WIN, alpha=0.16, linewidth=0, zorder=0)
a.plot(weeks, inc(r['obs']), marker='o', markersize=3.4, color=C_DATA, linewidth=1.0, zorder=4)
a.plot(weeks, inc(r['one_pathway']['pred']), color=C_ECO, lw=4.0, alpha=0.55, zorder=2)
a.plot(weeks, inc(r['double']['pred']), color=C_DBL, lw=1.8, ls=':', zorder=3)
a.plot(weeks, inc(r['two_pathway']['pred']), color=C_TWO, lw=1.8, ls='--', zorder=5)
a.set_ylabel('Weekly deaths'); a.set_xlabel('Week')
a.set_xlim(0.5, 22.5)
a.set_title(f'(a) Fits to the simulated deaths (seed {r["seed"]})', loc='left')

# (b) recovered climate effect on human transmission
x = np.arange(len(res))
b.axhline(TRUTH2['bH'], color=C_DATA, lw=1.5, zorder=1)
b.plot(x, [q['two_pathway']['params']['bH'] for q in res], 'o', color=C_TWO, ms=7, zorder=3)
b.text(len(res) - 0.55, TRUTH2['bH'] + 0.05, 'true value (0.6)', ha='right', color=C_DATA, fontsize=9)
b.text(-0.3, 0.04, 'ecology only: ' + BH + ' = 0', ha='left', color=C_ECO, fontsize=9)
b.set_ylim(-0.1, 1.65); b.set_xlim(-0.4, len(res) - 0.6)
b.set_xticks(x); b.set_xticklabels([str(q["seed"]) for q in res])
b.set_xlabel('Seed')
b.set_ylabel('Climate effect on people, ' + BH)
b.set_title('(b) Recovered climate effect', loc='left')

# (c) total deaths under a harsher climate, one point per seed and model
d_h = lambda q, k: q[k]['cf']['harsher_baseline']['pred'][-1]
truth_h = r['two_pathway']['cf']['harsher_baseline']['truth'][-1]
truth_o = r['truth_dr'][-1]
c.axhline(truth_h, color=C_DATA, lw=1.5, zorder=1)
c.axhline(truth_o, color=C_REF, lw=1.1, ls=':', zorder=1)
off = 0.2
c.plot(x - off, [d_h(q, 'one_pathway') for q in res], 's', color=C_ECO, ms=6.5, zorder=3)
c.plot(x, [d_h(q, 'two_pathway') for q in res], 'o', color=C_TWO, ms=7, zorder=4)
c.plot(x + off, [d_h(q, 'double') for q in res], '^', color=C_DBL, ms=6.5, zorder=3)
c.set_ylim(0, 720); c.set_xlim(-0.5, len(res) - 0.5)
c.set_xticks(x); c.set_xticklabels([str(q["seed"]) for q in res])
c.set_xlabel('Seed')
c.set_ylabel('Total deaths at week 22')
c.set_title('(c) Harsher climate', loc='left')

for p in (a, b, c):
    p.spines['top'].set_visible(False); p.spines['right'].set_visible(False); p.grid(False)

# one shared legend above the panels (the true-value lines are explained here, not on the axes)
handles = [
    Line2D([], [], color=C_DATA, marker='o', markersize=3.5, lw=1.0),
    Line2D([], [], color=C_TWO, lw=1.8, ls='--', marker='o', markersize=5),
    Line2D([], [], color=C_ECO, lw=3.5, alpha=0.6, marker='s', markersize=5),
    Line2D([], [], color=C_DBL, lw=1.8, ls=':', marker='^', markersize=5),
    Patch(facecolor=C_WIN, alpha=0.30, linewidth=0),
    Line2D([], [], color=C_DATA, lw=1.5),
    Line2D([], [], color=C_REF, lw=1.1, ls=':'),
]
labels = ['Simulated deaths', 'Two pathways', 'Ecology only', 'Double sigmoid',
          'Favourable climate window', f'True value (b); true total, harsher climate, {truth_h:.0f} (c)',
          f'True total, original climate, {truth_o:.0f} (c)']
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
           columnspacing=1.2, handlelength=2.0)
fig.subplots_adjust(top=0.84, bottom=0.08, left=0.09, right=0.98)

os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
for ext, kw in (('png', {'dpi': 600}), ('pdf', {})):
    fig.savefig(os.path.join(HERE, 'figures', f'figure_two_pathway.{ext}'), bbox_inches='tight', **kw)
print('figure from seed', r['seed'])