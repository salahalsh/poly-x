#!/usr/bin/env python
"""One-command reproduction of every number, table and figure in the paper.

    python reproduce.py            # everything, in dependency order
    python reproduce.py --quick    # skip the two multi-hour steps
    python reproduce.py --list     # show the steps and their outputs

Each step is skipped if its outputs already exist, so an interrupted run can be
resumed by re-invoking the same command. Every step writes machine-readable
output to ``outputs/``; the figures are generated from those files only, and the
manuscript is assembled from them by the final step, which fails if any
reported value cannot be traced to one.

Reviewer 1 asked for "a reproducibility script for every figure and table";
Reviewer 2, Major 13 asked for "a one-command reproduction workflow".
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = (HERE.parent / 'analysis') if (HERE.parent / 'analysis').exists() \
    else (HERE / 'analysis')
OUT = ANALYSIS.parent / 'outputs'
FIGS = ANALYSIS.parent / 'figures'

# (script, args, human description, sentinel outputs, slow?)
STEPS = [
    ('s01_dataset_and_splits.py', [],
     'Dataset accounting, scaffold statistics, split characterisation',
     ['s01_dataset_accounting.json', 's01_scaffold_stats.json'], False),
    ('s03_polybert_embeddings.py', [],
     'polyBERT CLS and mean-pooled embeddings; mirror verification; '
     'SMILES-rewriting invariance',
     ['s03_embeddings_cls.npy', 's03_embeddings_mean.npy',
      's03_mirror_verification.json', 's03_invariance.json'], False),
    ('s03b_repeat_unit_invariance.py', [],
     'Invariance to repeat-unit multiplication',
     ['s03b_repeat_unit_invariance.json'], False),
    ('s06_polymetrix_descriptors.py', [],
     'PolyMetriX native hierarchical descriptors (baseline)',
     ['s06_polymetrix_descriptors.npy'], False),
    ('s12_capping_collapse.py', [],
     'How often the capping convention merges distinct polymers',
     ['s12_capping_collapse.json'], False),
    ('s04_gc_validation.py', [],
     'Tier 1 group contribution over the full collection, with coverage',
     ['s04_gc_summary.json', 's04_gc_predictions.csv'], False),
    ('s02_benchmark.py', ['8'],
     'Repeated-split representation benchmark (4 protocols x 12 '
     'representations)',
     ['s02_benchmark_summary.json'], True),
    ('s05_reliability_calibration.py', ['published', '42'],
     'Applicability domain and reliability index calibration; conformal '
     'intervals',
     ['s05_calibration_published_42.json',
      's05_perpolymer_published_42.csv'], True),
    ('s08_external_and_case_study.py', [],
     'External conjugated-polymer validation and the design case study',
     ['s08_external_conjugated.json', 's08_case_study.json'], False),
    ('s10_build_manuscript.py', [],
     'Significance testing, tier table, and assembly of the manuscript and '
     'response letter from the archived outputs',
     ['s10_tokens.json', 's10_significance.json', 's10_tier_table.json'],
     False),
    ('s09_figures.py', [],
     'All figures, each with a provenance sidecar (runs after step 10: '
     'Figure 4 reads the tier table that step 10 writes)',
     [], False),
    ('s11_supplementary.py', [],
     'Supporting Information tables, parsed from source',
     ['s11_gc_parameters.csv'], False),
]


def done(sentinels) -> bool:
    return bool(sentinels) and all((OUT / s).exists() for s in sentinels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='skip the steps marked slow')
    ap.add_argument('--force', action='store_true',
                    help='re-run steps whose outputs already exist')
    ap.add_argument('--list', action='store_true', dest='list_only')
    args = ap.parse_args()

    if args.list_only:
        for script, a, desc, sent, slow in STEPS:
            mark = ' [slow]' if slow else ''
            state = 'done' if done(sent) else 'pending'
            print(f'{script:36s} {state:8s}{mark}  {desc}')
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for script, extra, desc, sent, slow in STEPS:
        if slow and args.quick:
            print(f'--- SKIP (quick) {script}: {desc}')
            continue
        if done(sent) and not args.force:
            print(f'--- SKIP (done)  {script}: {desc}')
            continue
        print(f'\n=== RUN {script}: {desc}', flush=True)
        r = subprocess.run([sys.executable, '-u', script, *extra],
                           cwd=ANALYSIS)
        if r.returncode != 0:
            print(f'\nFAILED at {script} (exit {r.returncode}) after '
                  f'{(time.time()-t0)/60:.1f} min')
            return r.returncode
    print(f'\nAll steps complete in {(time.time()-t0)/60:.1f} min.')
    print(f'Outputs: {OUT}\nFigures: {FIGS}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
