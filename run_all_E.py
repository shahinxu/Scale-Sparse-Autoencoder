#!/usr/bin/env python3
"""
Run test-combination.py for multiple E values and collect avg_similarity_scale values.
Produces a small CSV-like summary to stdout and saves per-run logs under logs/.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST_SCRIPT = ROOT / 'test-combination.py'
LOG_DIR = ROOT / 'logs' / 'run_all_E'
LOG_DIR.mkdir(parents=True, exist_ok=True)

E_VALUES = [2, 4, 8, 16]
results = []

for e in E_VALUES:
    env = os.environ.copy()
    env['E'] = str(e)
    logf = LOG_DIR / f'run_E{e}.log'
    print(f"Running E={e} -> log: {logf}")
    with open(logf, 'wb') as out:
        proc = subprocess.run(["python", str(TEST_SCRIPT)], env=env, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)
    # read logfile and extract AVG_SIMILARITY_SCALE
    txt = logf.read_text()
    avg = None
    for line in txt.splitlines():
        if line.startswith('AVG_SIMILARITY_SCALE='):
            try:
                avg = float(line.split('=', 1)[1].strip())
            except Exception:
                avg = None
    results.append((e, avg, str(logf)))

# Print summary
print('\nE,avg_similarity_scale,logfile')
for e, avg, path in results:
    print(f"{e},{'' if avg is None else avg},{path}")

# Save a small summary file
summary_p = ROOT / 'logs' / 'run_all_E' / 'summary.csv'
with summary_p.open('w') as f:
    f.write('E,avg_similarity_scale,logfile\n')
    for e, avg, path in results:
        f.write(f"{e},{'' if avg is None else avg},{path}\n")

print(f"Saved summary to {summary_p}")
