"""
Run the whole simulated example with one command.

    python run_all.py

This simulates five datasets (seeds 1-5) from the two-pathway truth, fits the three models
(double sigmoid, climate on ecology only, climate on ecology and human transmission) to each,
predicts the counterfactual climates, and then writes the summary tables and the figure
to results/ and figures/. Seeds run in parallel; each takes about 1-5 minutes.

Optional (not needed for the normal run):
    python run_all.py 1 2 3        # only these seeds
    python run_all.py --jobs 2     # run at most 2 seeds at once
    python run_all.py --first-version   # the earlier ecology-only example instead
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
two = '--first-version' not in args
if '--first-version' in args: args.remove('--first-version')
RUN = 'run_seed_two_pathway.py' if two else 'run_seed.py'
SUMMARY = 'summarise_two_pathway.py' if two else 'summarise_and_plot.py'
jobs = os.cpu_count() or 1
if '--jobs' in args:
    i = args.index('--jobs'); jobs = int(args[i + 1]); del args[i:i + 2]
seeds = [int(a) for a in args] or [1, 2, 3, 4, 5]

env = dict(os.environ, PYTHONWARNINGS='ignore')
pending, running, failed = list(seeds), {}, []
t0 = time.time()
print(f'Running seeds {seeds} with up to {jobs} at once...', flush=True)
while pending or running:
    while pending and len(running) < jobs:
        s = pending.pop(0)
        log = open(os.path.join(HERE, f'log_{"tp" if two else ""}seed{s}.txt'), 'w')
        running[s] = (subprocess.Popen([sys.executable, os.path.join(HERE, RUN), str(s)],
                                       cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env), log)
        print(f'  started seed {s}', flush=True)
    time.sleep(5)
    for s in list(running):
        p, log = running[s]
        if p.poll() is not None:
            log.close(); del running[s]
            ok = p.returncode == 0
            if not ok: failed.append(s)
            print(f'  seed {s} {"finished" if ok else "FAILED (see log_%sseed%d.txt)" % ("tp" if two else "", s)} after {round(time.time()-t0)} s', flush=True)

if failed:
    print('Some seeds failed:', failed)
print('Making the summary table and figure...', flush=True)
subprocess.run([sys.executable, os.path.join(HERE, SUMMARY)], cwd=HERE, check=False)