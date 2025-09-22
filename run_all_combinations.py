#!/usr/bin/env python3
"""
Run test-combination.py over multiple K and E values and collect results.
- Creates per-run logs under logs/run_all_KE/
- Parses averages from stdout and writes logs/run_all_KE/summary.csv

Usage (defaults K=[32,64,128], E=[2,4,8,16]):
    python run_all_combinations.py

Customize K/E via CLI:
    python run_all_combinations.py --k 32 64 --e 4 8 16
"""
import argparse
import os
import subprocess
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
TEST_SCRIPT = ROOT / 'test-combination.py'
LOG_DIR = ROOT / 'logs' / 'run_all_KE'
LOG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description='Run test-combination.py for grids of K and E')
    p.add_argument('--k', nargs='+', type=int, default=[2, 4, 8, 16, 32, 64, 128], help='List of K values')
    p.add_argument('--e', nargs='+', type=int, default=[2, 4, 8, 16], help='List of E values')
    p.add_argument('--python', type=str, default='python', help='Python executable to use')
    return p.parse_args()


def parse_metrics_from_text(txt: str):
    # Expected lines in test-combination.py output:
    #   avg_similarity (scale): <float>
    #   avg_similarity (plain): <float>
    m_scale = None
    m_plain = None
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith('avg_similarity (scale):'):
            try:
                m_scale = float(line.split(':', 1)[1].strip())
            except Exception:
                m_scale = None
        elif line.startswith('avg_similarity (plain):'):
            try:
                m_plain = float(line.split(':', 1)[1].strip())
            except Exception:
                m_plain = None
    return m_scale, m_plain


def main():
    args = parse_args()

    K_VALUES = list(args.k)
    E_VALUES = list(args.e)

    results = []  # (K, E, avg_scale, avg_plain, logfile)

    for k in K_VALUES:
        for e in E_VALUES:
            env = os.environ.copy()
            env['K'] = str(k)
            env['E'] = str(e)

            logf = LOG_DIR / f'run_K{k}_E{e}.log'
            print(f"Running K={k}, E={e} -> log: {logf}")

            with open(logf, 'wb') as out:
                subprocess.run(
                    [args.python, str(TEST_SCRIPT)],
                    env=env,
                    cwd=str(ROOT),
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            txt = logf.read_text(errors='ignore')
            avg_scale, avg_plain = parse_metrics_from_text(txt)
            results.append((k, e, avg_scale, avg_plain, str(logf)))

    # Print summary
    print('\nK,E,avg_similarity_scale,avg_similarity_plain,logfile')
    for k, e, s, p, path in results:
        s_str = '' if s is None else f'{s}'
        p_str = '' if p is None else f'{p}'
        print(f"{k},{e},{s_str},{p_str},{path}")

    # Save CSV
    summary_p = LOG_DIR / 'summary.csv'
    with summary_p.open('w') as f:
        f.write('K,E,avg_similarity_scale,avg_similarity_plain,logfile\n')
        for k, e, s, p, path in results:
            s_str = '' if s is None else f'{s}'
            p_str = '' if p is None else f'{p}'
            f.write(f"{k},{e},{s_str},{p_str},{path}\n")

    print(f"Saved summary to {summary_p}")


if __name__ == '__main__':
    main()
