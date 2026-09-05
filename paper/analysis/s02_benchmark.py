"""
Script 02 - repeated-split representation benchmark.

Answers:
  * Reviewer 1 and Reviewer 2, Major 3 - repeated splits, confidence intervals
    and significance testing for the ECFP4-versus-polyBERT comparison.
  * Reviewer 2, Major 2 - CLS versus mean-pooled polyBERT under identical
    splits and identical heads.
  * Reviewer 2, Major 4 - the ensemble blend weight is now genuinely optimised
    on the validation partition, and every component model's MAE and RMSE is
    recorded (Reviewer 2, Minor 3).
  * Reviewer 2, Major 6 - four splitting protocols are compared: the splitter
    as submitted, a genuinely randomised scaffold split, a polymer-aware
    Butina cluster split, and a random split (which quantifies how much the
    random-split figure inflates apparent accuracy).
  * Reviewer 2, Major 8 - a mean predictor, a regularised linear model,
    count fingerprints, alternative Morgan radii and lengths, and a physical
    2-D descriptor baseline are all evaluated under the same partitions.

Every run stores its validation and test predictions so that downstream
scripts (calibration, bootstrap tests, figures) never re-fit anything.

Runs are independent and are executed in a process pool; a run whose output
already exists is skipped, so the script is resumable.

Usage:  python s02_benchmark.py [n_workers]

Writes: outputs/s02_splits.npz
        outputs/runs/<protocol>_<seed>_<representation>.json
        outputs/runs/<protocol>_<seed>_<representation>.npz
        outputs/s02_benchmark_summary.json
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from polyx_rev import (OUT, SEEDS, butina_clusters, cluster_split, dump,
                       fit_outlier_filter, load_dataset, metrics,
                       morgan_matrix, optimize_weight, random_split,
                       rdkit_descriptor_matrix, scaffold_groups,
                       scaffold_split, scaffold_split_as_published)

RUNS = OUT / 'runs'
RUNS.mkdir(exist_ok=True)
SPLIT_CACHE = OUT / 's02_splits.npz'

# Hyperparameters exactly as submitted, so that representation - not tuning -
# is what is being compared. Script 05 relaxes this with an equal tuning
# budget per representation (Reviewer 2, Major 3 and Major 8).
RF_KW = dict(n_estimators=500, min_samples_leaf=2, random_state=42)
GB_KW = dict(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
             min_samples_leaf=5, random_state=42)

HEADLINE = ['ecfp4_2048', 'pb_cls', 'pb_mean']
BASELINES = ['mean', 'ridge_ecfp4', 'ecfp4_2048_counts', 'ecfp4_1024',
             'ecfp4_4096', 'ecfp6_2048', 'rdkit_desc', 'ecfp4_2048_methylcap',
             'ecfp4_2048_wildcard', 'polymetrix_desc']

_CACHE: dict = {}


def representation(rep, smiles):
    """Build one representation on demand (workers build only what they use)."""
    if rep in _CACHE:
        return _CACHE[rep]
    if rep == 'mean':
        X = np.zeros((len(smiles), 1), dtype=np.float32)
    elif rep in ('ecfp4_2048', 'ridge_ecfp4'):
        X = morgan_matrix(smiles, radius=2, n_bits=2048)
    elif rep == 'ecfp4_2048_counts':
        X = morgan_matrix(smiles, radius=2, n_bits=2048, counts=True)
    elif rep == 'ecfp4_2048_methylcap':
        X = morgan_matrix(smiles, radius=2, n_bits=2048, cap='methyl')
    elif rep == 'ecfp4_2048_wildcard':
        # Wildcards retained as dummy atoms rather than substituted: the
        # third treatment Reviewer 2 (Major 7) asks to be quantified.
        X = morgan_matrix(smiles, radius=2, n_bits=2048, cap='wildcard')
    elif rep == 'ecfp4_1024':
        X = morgan_matrix(smiles, radius=2, n_bits=1024)
    elif rep == 'ecfp4_4096':
        X = morgan_matrix(smiles, radius=2, n_bits=4096)
    elif rep == 'ecfp6_2048':
        X = morgan_matrix(smiles, radius=3, n_bits=2048)
    elif rep == 'rdkit_desc':
        X = rdkit_descriptor_matrix(smiles)
    elif rep == 'polymetrix_desc':
        # The native hierarchical descriptors of the PolyMetriX ecosystem
        # (Reviewer 2, Major 8); built by script 06.
        X = np.load(OUT / 's06_polymetrix_descriptors.npy')
    elif rep == 'pb_cls':
        X = np.load(OUT / 's03_embeddings_cls.npy')
    elif rep == 'pb_mean':
        X = np.load(OUT / 's03_embeddings_mean.npy')
    else:
        raise ValueError(rep)
    _CACHE[rep] = X
    return X


def fit_eval(X, y, tr, va, te, rep, n_jobs):
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if rep in ('mean', 'ridge_ecfp4'):
        if rep == 'mean':
            m = DummyRegressor(strategy='mean').fit(X[tr], y[tr])
        else:
            m = make_pipeline(StandardScaler(with_mean=False),
                              RidgeCV(alphas=np.logspace(-2, 4, 25)))
            m.fit(X[tr], y[tr])
        pv, pt = m.predict(X[va]), m.predict(X[te])
        res = {'components': {rep: {'val': metrics(y[va], pv),
                                    'test': metrics(y[te], pt)}},
               'ensemble': {'val': metrics(y[va], pv), 'test': metrics(y[te], pt),
                            'w_rf': None}}
        return res, {'rf_val': pv, 'gb_val': pv, 'ens_val': pv,
                     'rf_test': pt, 'gb_test': pt, 'ens_test': pt}

    rf = RandomForestRegressor(n_jobs=n_jobs, **RF_KW).fit(X[tr], y[tr])
    gb = GradientBoostingRegressor(**GB_KW).fit(X[tr], y[tr])
    rf_v, gb_v = rf.predict(X[va]), gb.predict(X[va])
    rf_t, gb_t = rf.predict(X[te]), gb.predict(X[te])

    w = optimize_weight(rf_v, gb_v, y[va])
    ens_v, ens_t = w * rf_v + (1 - w) * gb_v, w * rf_t + (1 - w) * gb_t
    eq_v, eq_t = 0.5 * (rf_v + gb_v), 0.5 * (rf_t + gb_t)

    res = {
        'components': {
            'rf': {'val': metrics(y[va], rf_v), 'test': metrics(y[te], rf_t)},
            'gb': {'val': metrics(y[va], gb_v), 'test': metrics(y[te], gb_t)},
        },
        'ensemble': {'val': metrics(y[va], ens_v), 'test': metrics(y[te], ens_t),
                     'w_rf': w},
        # The submitted model hard-coded an equal average; reported alongside
        # so the two are directly comparable (Reviewer 2, Major 4).
        'ensemble_equal_weight': {'val': metrics(y[va], eq_v),
                                  'test': metrics(y[te], eq_t), 'w_rf': 0.5},
    }
    return res, {'rf_val': rf_v, 'gb_val': gb_v, 'ens_val': ens_v,
                 'rf_test': rf_t, 'gb_test': gb_t, 'ens_test': ens_t}


def run_one(args):
    proto, seed, rep, n_jobs = args
    tag = f'{proto}_{seed}_{rep}'
    jpath, npath = RUNS / f'{tag}.json', RUNS / f'{tag}.npz'
    if jpath.exists() and npath.exists():
        return tag, None, 0.0

    df, _ = load_dataset()
    y = df['target'].values.astype(float)
    smiles = df['smiles'].tolist()

    z = np.load(SPLIT_CACHE)
    tr, va, te = z[f'{proto}_{seed}_train'], z[f'{proto}_{seed}_val'], z[f'{proto}_{seed}_test']

    # Target-dependent preprocessing fitted on TRAINING only (Reviewer 2,
    # Major 7). The submitted pipeline applied the 3-sigma filter globally,
    # before splitting, so test targets influenced preprocessing.
    lo, hi = fit_outlier_filter(y[tr])
    tr_f = tr[(y[tr] >= lo) & (y[tr] <= hi)]

    X = representation(rep, smiles)
    t0 = time.time()
    res, preds = fit_eval(X, y, tr_f, va, te, rep, n_jobs)
    res.update({'protocol': proto, 'seed': int(seed), 'representation': rep,
                'n_train': int(len(tr_f)), 'n_train_before_filter': int(len(tr)),
                'n_val': int(len(va)), 'n_test': int(len(te)),
                'n_features': int(X.shape[1]),
                'outlier_bounds_from_train': [float(lo), float(hi)],
                'seconds': round(time.time() - t0, 1)})
    with open(jpath, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(npath, train=tr_f, val=va, test=te,
                        y_val=y[va], y_test=y[te], **preds)
    return tag, res['ensemble']['test'], res['seconds']


def build_split_cache(df):
    if SPLIT_CACHE.exists():
        return
    groups = scaffold_groups(df, cap='hydrogen')
    store = {}

    def put(proto, seed, s):
        store[f'{proto}_{seed}_train'] = s[0]
        store[f'{proto}_{seed}_val'] = s[1]
        store[f'{proto}_{seed}_test'] = s[2]

    put('published', 42, scaffold_split_as_published(df, seed=42, groups=groups))
    for s in SEEDS:
        put('scaffold', s, scaffold_split(df, seed=s, groups=groups))
        put('random', s, random_split(df, seed=s))
    print('[split] butina clustering ...', flush=True)
    clusters = butina_clusters(df, cutoff=0.65)
    for s in SEEDS:
        put('cluster', s, cluster_split(df, clusters, seed=s))
    np.savez_compressed(SPLIT_CACHE, **store)
    print(f'[split] cached {len(store)//3} partitions', flush=True)


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_jobs = max(1, (os.cpu_count() or 8) // n_workers)

    df, _ = load_dataset()
    build_split_cache(df)

    # The randomised scaffold split is the headline protocol and gets all five
    # seeds. The cluster and random protocols are diagnostic (how much does the
    # protocol itself move the number?) and get three. The baseline sweep runs
    # on three scaffold seeds plus the split as submitted, which is enough to
    # rank representations with an across-seed spread.
    secondary = SEEDS[:3]
    protocols = ([('published', 42)]
                 + [('scaffold', s) for s in SEEDS]
                 + [('random', s) for s in secondary]
                 + [('cluster', s) for s in secondary])
    jobs = []
    for proto, seed in protocols:
        for rep in HEADLINE:
            jobs.append((proto, seed, rep, n_jobs))
        if proto == 'published' or (proto == 'scaffold' and seed in secondary):
            for rep in BASELINES:
                jobs.append((proto, seed, rep, n_jobs))
    todo = [j for j in jobs
            if not (RUNS / f'{j[0]}_{j[1]}_{j[2]}.json').exists()]
    print(f'[plan] {len(jobs)} runs, {len(todo)} outstanding, '
          f'{n_workers} workers x n_jobs={n_jobs}', flush=True)

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(run_one, j): j for j in todo}
        for fut in as_completed(futs):
            tag, m, secs = fut.result()
            done += 1
            if m:
                print(f'[{done}/{len(todo)}] {tag}: R2={m["r2"]:.4f} '
                      f'MAE={m["mae"]:.2f} RMSE={m["rmse"]:.2f} '
                      f'({secs}s, elapsed {(time.time()-t0)/60:.1f}m)', flush=True)
    summarise()
    print(f'[done] {(time.time()-t0)/60:.1f} min')


def summarise():
    rows = []
    for p in sorted(RUNS.glob('*.json')):
        with open(p, encoding='utf-8') as f:
            rows.append(json.load(f))
    summary = {}
    for proto in ('published', 'scaffold', 'cluster', 'random'):
        for rep in HEADLINE + BASELINES:
            sub = [r for r in rows
                   if r['protocol'] == proto and r['representation'] == rep]
            if not sub:
                continue
            def col(key):
                return [r['ensemble']['test'][key] for r in sub]
            summary[f'{proto}|{rep}'] = {
                'n_seeds': len(sub),
                'test_r2_mean': float(np.mean(col('r2'))),
                'test_r2_sd': float(np.std(col('r2'), ddof=1)) if len(sub) > 1 else 0.0,
                'test_mae_mean': float(np.mean(col('mae'))),
                'test_mae_sd': float(np.std(col('mae'), ddof=1)) if len(sub) > 1 else 0.0,
                'test_rmse_mean': float(np.mean(col('rmse'))),
                'test_rmse_sd': float(np.std(col('rmse'), ddof=1)) if len(sub) > 1 else 0.0,
                'w_rf_values': [r['ensemble']['w_rf'] for r in sub],
                'per_seed_test_r2': {str(r['seed']): r['ensemble']['test']['r2']
                                     for r in sub},
                'components_test_r2': {
                    k: float(np.mean([r['components'][k]['test']['r2'] for r in sub]))
                    for k in sub[0]['components']},
                'equal_weight_test_r2': (
                    float(np.mean([r['ensemble_equal_weight']['test']['r2']
                                   for r in sub]))
                    if 'ensemble_equal_weight' in sub[0] else None),
            }
    dump('s02_benchmark_summary.json', summary)


if __name__ == '__main__':
    main()
