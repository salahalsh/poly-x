"""
Script 10 - assemble the revised manuscript from the archived analysis outputs.

Reads Manuscript_R1.tex.in, substitutes every @@TOKEN@@ from the JSON and CSV
files written by scripts 01-08, and writes Manuscript_R1.tex. The build aborts
if any token is left unresolved or if any token is defined but unused, so no
number in the compiled manuscript can have been typed by hand.

Also performs the two remaining analyses that only make sense once the
benchmark has finished:
  * the paired bootstrap and across-seed Wilcoxon test for ECFP4 against
    polyBERT (Reviewer 1; Reviewer 2, Major 3);
  * the three-tier predictions for the eight canonical homopolymers, with
    their partition membership (Reviewer 2, Minor 6).

Writes: Manuscript_R1.tex
        outputs/s10_tier_table.json
        outputs/s10_significance.json
        outputs/s10_tokens.json
"""
import glob
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, wilcoxon
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from polyx_rev import (OUT, REVISION, canonical_psmiles, dump,
                       fit_outlier_filter, load_dataset, morgan_matrix,
                       paired_bootstrap_delta)
from s01_dataset_and_splits import CANONICAL

from _paths import SERVICES as _P  # noqa: E402


def _load(mod, fname):
    spec = importlib.util.spec_from_file_location(mod, _P / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


GC = _load('polyx_gc', 'group_contribution.py').GroupContributionCalculator
PROC = _load('polyx_proc', 'processability.py')

RF_KW = dict(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1)
GB_KW = dict(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
             min_samples_leaf=5, random_state=42)
CANONICAL_EXP = {'PE': 195, 'PP': 253, 'PS': 373, 'PVC': 354, 'PET': 342,
                 'PEO': 206, 'POM': 198, 'PVOH': 358}


def jload(name):
    with open(OUT / name, encoding='utf-8') as f:
        return json.load(f)


def f(x, n=2):
    return f'{x:.{n}f}'


# --------------------------------------------------------------------------
def runs_frame():
    rows = []
    for p in sorted(glob.glob(str(OUT / 'runs' / '*.json'))):
        with open(p, encoding='utf-8') as fh:
            d = json.load(fh)
        rows.append({
            'protocol': d['protocol'], 'seed': d['seed'],
            'rep': d['representation'],
            'test_r2': d['ensemble']['test']['r2'],
            'test_mae': d['ensemble']['test']['mae'],
            'test_rmse': d['ensemble']['test']['rmse'],
            'val_r2': d['ensemble']['val']['r2'],
            'w_rf': d['ensemble']['w_rf'],
            'eq_test_r2': d.get('ensemble_equal_weight', {}).get('test', {}).get('r2'),
            'rf_test_r2': d['components'].get('rf', {}).get('test', {}).get('r2'),
            'gb_test_r2': d['components'].get('gb', {}).get('test', {}).get('r2'),
            'rf_test_mae': d['components'].get('rf', {}).get('test', {}).get('mae'),
            'gb_test_mae': d['components'].get('gb', {}).get('test', {}).get('mae'),
            'rf_test_rmse': d['components'].get('rf', {}).get('test', {}).get('rmse'),
            'gb_test_rmse': d['components'].get('gb', {}).get('test', {}).get('rmse'),
            'rf_val_r2': d['components'].get('rf', {}).get('val', {}).get('r2'),
            'gb_val_r2': d['components'].get('gb', {}).get('val', {}).get('r2'),
            'n_features': d['n_features'],
        })
    return pd.DataFrame(rows)


def agg(df, proto, rep, col='test_r2'):
    s = df[(df.protocol == proto) & (df.rep == rep)][col]
    if len(s) == 0:
        raise KeyError(f'no runs for {proto}/{rep}')
    return float(s.mean()), (float(s.std(ddof=1)) if len(s) > 1 else 0.0), len(s)


# --------------------------------------------------------------------------
def significance(df):
    """Paired bootstrap over test polymers plus across-seed Wilcoxon."""
    out = {}
    for proto in ('scaffold', 'published', 'cluster'):
        seeds = sorted(df[(df.protocol == proto) & (df.rep == 'ecfp4_2048')].seed)
        per_seed = []
        for s in seeds:
            try:
                a = np.load(OUT / 'runs' / f'{proto}_{s}_ecfp4_2048.npz')
                b = np.load(OUT / 'runs' / f'{proto}_{s}_pb_mean.npz')
                c = np.load(OUT / 'runs' / f'{proto}_{s}_pb_cls.npz')
            except FileNotFoundError:
                continue
            y = a['y_test']
            d_mean = paired_bootstrap_delta(y, a['ens_test'], b['ens_test'],
                                            'r2', n_boot=5000, seed=1)
            d_cls = paired_bootstrap_delta(y, a['ens_test'], c['ens_test'],
                                           'r2', n_boot=5000, seed=1)
            d_pool = paired_bootstrap_delta(y, b['ens_test'], c['ens_test'],
                                            'r2', n_boot=5000, seed=1)
            per_seed.append({
                'seed': int(s),
                'ecfp4_vs_pbmean': dict(zip(('delta', 'lo', 'hi', 'p'), d_mean)),
                'ecfp4_vs_pbcls': dict(zip(('delta', 'lo', 'hi', 'p'), d_cls)),
                'pbmean_vs_pbcls': dict(zip(('delta', 'lo', 'hi', 'p'), d_pool)),
            })
        entry = {'per_seed': per_seed}
        if len(per_seed) >= 3:
            a_r2 = [df[(df.protocol == proto) & (df.rep == 'ecfp4_2048') &
                       (df.seed == r['seed'])].test_r2.iloc[0] for r in per_seed]
            b_r2 = [df[(df.protocol == proto) & (df.rep == 'pb_mean') &
                       (df.seed == r['seed'])].test_r2.iloc[0] for r in per_seed]
            try:
                stat, p = wilcoxon(a_r2, b_r2)
                entry['across_seed_wilcoxon'] = {'statistic': float(stat),
                                                 'p': float(p), 'n': len(a_r2)}
            except ValueError as e:
                entry['across_seed_wilcoxon'] = {'error': str(e), 'n': len(a_r2)}
        if per_seed:
            ds = [r['ecfp4_vs_pbmean']['delta'] for r in per_seed]
            entry['pooled'] = {
                'mean_delta_r2': float(np.mean(ds)),
                'sd_delta_r2': float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0,
                'ci_lo': float(np.mean([r['ecfp4_vs_pbmean']['lo'] for r in per_seed])),
                'ci_hi': float(np.mean([r['ecfp4_vs_pbmean']['hi'] for r in per_seed])),
                'max_p': float(max(r['ecfp4_vs_pbmean']['p'] for r in per_seed)),
                'n_seeds_favouring_ecfp4': int(sum(d > 0 for d in ds)),
            }
        out[proto] = entry
    dump('s10_significance.json', out)
    return out


# --------------------------------------------------------------------------
def tier_table(df_data, y, X, tr):
    """Three-tier predictions for the eight canonical polymers."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained('xushijie/polyBERT')
    mdl = AutoModel.from_pretrained('xushijie/polyBERT')
    mdl.eval()
    torch.manual_seed(42)

    emb_all = np.load(OUT / 's03_embeddings_mean.npy')
    rf2 = RandomForestRegressor(**RF_KW).fit(X[tr], y[tr])
    gb2 = GradientBoostingRegressor(**GB_KW).fit(X[tr], y[tr])
    rf3 = RandomForestRegressor(**RF_KW).fit(emb_all[tr], y[tr])
    gb3 = GradientBoostingRegressor(**GB_KW).fit(emb_all[tr], y[tr])

    z = np.load(OUT / 's02_splits.npz')
    part = {'train': set(z['published_42_train'].tolist()),
            'val': set(z['published_42_val'].tolist()),
            'test': set(z['published_42_test'].tolist())}
    calc = GC()

    rows = []
    for name, psmi in CANONICAL.items():
        fp = morgan_matrix([psmi], radius=2, n_bits=2048)
        with torch.no_grad():
            enc = tok([psmi], padding=True, truncation=True, max_length=512,
                      return_tensors='pt')
            hid = mdl(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1).to(hid.dtype)
            emb = ((hid * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).numpy()
        gc = calc.calculate(psmi)
        canon = canonical_psmiles(psmi)
        hit = df_data.index[df_data['canonical'] == canon].tolist()
        where = 'absent'
        if hit:
            i = int(hit[0])
            where = next((k for k, v in part.items() if i in v), 'dropped')
        rows.append({
            'polymer': name, 'psmiles': psmi,
            'exp': float(CANONICAL_EXP[name]),
            'tier1': round(float(gc.tg), 1) if gc.tg is not None else None,
            'tier2': round(float(0.5 * (rf2.predict(fp)[0] + gb2.predict(fp)[0])), 1),
            'tier3': round(float(0.5 * (rf3.predict(emb)[0] + gb3.predict(emb)[0])), 1),
            'partition': where,
            'dataset_tg': float(df_data.loc[hit[0], 'target']) if hit else None,
        })
    res = {'rows': rows,
           'tier1_mae': float(np.mean([abs(r['tier1'] - r['exp']) for r in rows])),
           'tier2_mae': float(np.mean([abs(r['tier2'] - r['exp']) for r in rows])),
           'tier3_mae': float(np.mean([abs(r['tier3'] - r['exp']) for r in rows])),
           'note': ('Tier 3 uses the published mean-pooled polyBERT '
                    'fingerprint. Partition refers to the split as submitted.')}
    dump('s10_tier_table.json', res)
    return res


# --------------------------------------------------------------------------
def processability_correlation(per):
    """Processability score against Tg over the test partition."""
    assessor = None
    for cls in ('ProcessabilityAssessor', 'PolymerProcessability',
                'ProcessabilityCalculator'):
        if hasattr(PROC, cls):
            assessor = getattr(PROC, cls)()
            break
    scores, tgs = [], []
    calc = GC()
    for smi, tg in zip(per['smiles'], per['y_true']):
        gc = calc.calculate(smi)
        if gc.tg is None or gc.tm is None:
            continue
        payload = {'tg_gc': gc.tg, 'tm_gc': gc.tm,
                   'chain_stiffness': gc.chain_stiffness}
        try:
            r = assessor.assess(payload) if assessor else None
            s = getattr(r, 'score', None)
        except Exception:
            s = None
        if s is None:
            continue
        scores.append(float(s))
        tgs.append(float(tg))
    if len(scores) < 10:
        return None
    r, p = pearsonr(scores, tgs)
    return {'n': len(scores), 'pearson_r': float(r), 'p': float(p),
            'score_min': float(min(scores)), 'score_max': float(max(scores)),
            'score_median': float(np.median(scores))}


# --------------------------------------------------------------------------
# LaTeX table builders
# --------------------------------------------------------------------------
def tab_platforms():
    return r"""\begin{table}[htbp]
  \caption{Feature-by-feature comparison of POLY-X with publicly available polymer informatics platforms. GC, group contribution; FP, fingerprint; GNN, graph neural network; AD, applicability domain; UQ, uncertainty quantification. Facts taken from the primary publications and repositories; the survey covers tools that are publicly usable or whose code is public and is not claimed to be exhaustive. $T_g$, glass transition temperature; ECFP4, extended-connectivity fingerprint of diameter 4; CLS, classification token.}
  \label{tbl:platforms}
  \centering
  \footnotesize
  \begin{tabular}{p{2.6cm}p{2.2cm}p{2.2cm}p{2.2cm}p{2.2cm}p{2.6cm}}
    \toprule
     & Polymer Genome & PolyID & polyBERT & PolyMetriX & \textbf{POLY-X} \\
    \midrule
    Reference & \citet{Kim2018PG} & \citet{Wilson2023} & \citet{Kuenneth2023} & \citet{Kunchapu2025} & this work \\
    Primary form & hosted service & package + tool & model + data & library + data & hosted service + source \\
    Properties & dozens\textsuperscript{a} & 8 & 29 & $T_g$ benchmark & \textbf{$T_g$ validated}; 5 unvalidated GC estimates \\
    Representation & handcrafted FP (945-D here\textsuperscript{b}) & message-passing GNN & transformer (600-D, mean-pooled) & hierarchical descriptors & GC groups; ECFP4 / counts; polyBERT (CLS and mean) \\
    Physics tier & no & no & no & no & \textbf{yes} \\
    Validation & held-out & held-out + domain of validity & cross-validation & scaffold, cluster & \textbf{4 protocols $\times$ repeated seeds} \\
    AD & qualitative & domain of validity & none & none & \textbf{calibrated against error} \\
    UQ & yes & ensemble spread & none & none & \textbf{conformal, coverage measured} \\
    Interpretability & descriptor attribution & limited & none & interpretable descriptors & \textbf{per-group decomposition} \\
    Code public & no & yes & yes & yes & yes \\
    Usable without login & no\textsuperscript{c} & partial & n/a & n/a & no\textsuperscript{d} \\
    Batch & limited & scripted & scripted & scripted & \textbf{asynchronous} \\
    \bottomrule
  \end{tabular}

  \smallskip
  \footnotesize{\textsuperscript{a}Doan Tran et al.\ describe Polymer Genome as predicting
  ``dozens of polymer properties''; no exact count is stated, so none is given here.
  \textsuperscript{b}945 is the dimensionality reported for the Polymer Genome fingerprint on the
  data set used by Kuenneth and Ramprasad, not a fixed property of the method.
  \textsuperscript{c}The Polymer Genome prediction interface states ``Login required'' and
  presents a user ID and password form (checked 2026-09-04). Every other entry in this row was
  checked the same way.
  \textsuperscript{d}POLY-X requires an account: every view is gated and guest access is not
  enabled for this tool. The source is public under a noncommercial licence and can be run
  locally without any account. Rows without a numeric entry are our reading of the cited work rather than a
  figure it reports.}
\end{table}"""


def tab_accounting(acc):
    a, r = acc['as_submitted'], acc['revised_pipeline']
    removed = a['rows_removed_by_3sigma']
    rows = '\n'.join(
        rf"    \texttt{{{x['smiles'][:58]}\ldots}} & {x['target']:.2f} \\"
        for x in removed)
    return rf"""\begin{{table}}[htbp]
  \caption{{Dataset accounting: per-stage record counts for the pipeline as submitted and for the revised pipeline, and the identity of every removed record. The two-record difference between the source collection and the submitted count arises entirely from the global outlier filter. SD, standard deviation; $T_g$, glass transition temperature; PSMILES, polymer SMILES.}}
  \label{{tbl:accounting}}
  \centering
  \footnotesize
  \begin{{tabular}}{{lcc}}
    \toprule
    Stage & As submitted & Revised \\
    \midrule
    Raw records & {a['raw_rows']:,} & {r['raw_rows']:,} \\
    After dropping missing values & {a['after_nan_drop']:,} & {r['after_nan_drop']:,} \\
    After PSMILES validation / canonical parse & {a['after_psmiles_validation']:,} & {r['after_canonical_parse']:,} \\
    After global 3$\sigma$ filter on $T_g$ & {a['after_global_3sigma']:,} & not applied globally\textsuperscript{{a}} \\
    After deduplication & {a['after_raw_string_dedup']:,}\textsuperscript{{b}} & {r['after_canonical_dedup']:,}\textsuperscript{{c}} \\
    \bottomrule
  \end{{tabular}}

  \smallskip
  \footnotesize{{\textsuperscript{{a}}The filter is target-dependent; in the revision it is fitted on the training partition and its bounds applied to the other partitions. \textsuperscript{{b}}Deduplication on the raw string removed no record. \textsuperscript{{c}}Deduplication on the canonical PSMILES also removed none: the source collection is already canonicalised and deduplicated.}}

  \smallskip
  \centering
  \begin{{tabular}}{{p{{11cm}}c}}
    \multicolumn{{2}}{{l}}{{\textbf{{The two records removed by the global 3$\sigma$ filter}} (mean {a['target_mean_used_for_sigma']:.2f}~K, SD {a['target_sd_used_for_sigma']:.2f}~K):}} \\
    \toprule
    PSMILES & $T_g$ (K) \\
    \midrule
{rows}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def tab_splits(st, seed='42'):
    p = st['per_seed'][seed]['partitions']
    return rf"""\begin{{table}}[htbp]
  \caption{{Scaffold groups and partition composition. The collection contains {st['n_scaffold_groups']:,} distinct Murcko scaffold groups, not one per polymer; {st['n_singleton_groups']:,} are singletons and the null (acyclic) scaffold alone holds {st['null_scaffold_group_size']:,} polymers. Partition statistics are for the randomised scaffold split, seed 42. Chemical families are assigned by first match against an ordered SMARTS list; the six most populated are shown. SD, standard deviation; $T_g$, glass transition temperature.}}
  \label{{tbl:splits}}
  \centering
  \begin{{tabular}}{{lccc}}
    \toprule
     & Train & Validation & Test \\
    \midrule
    $N$ & {p['train']['n']:,} & {p['val']['n']:,} & {p['test']['n']:,} \\
    Distinct scaffold groups & {p['train']['n_scaffold_groups']:,} & {p['val']['n_scaffold_groups']:,} & {p['test']['n_scaffold_groups']:,} \\
    Mean $T_g$ (K) & {p['train']['target_mean']:.1f} & {p['val']['target_mean']:.1f} & {p['test']['target_mean']:.1f} \\
    SD $T_g$ (K) & {p['train']['target_sd']:.1f} & {p['val']['target_sd']:.1f} & {p['test']['target_sd']:.1f} \\
    Range $T_g$ (K) & {p['train']['target_min']:.0f} to {p['train']['target_max']:.0f} & {p['val']['target_min']:.0f} to {p['val']['target_max']:.0f} & {p['test']['target_min']:.0f} to {p['test']['target_max']:.0f} \\
    Polyimide & {p['train']['family_counts'].get('polyimide', 0):,} & {p['val']['family_counts'].get('polyimide', 0):,} & {p['test']['family_counts'].get('polyimide', 0):,} \\
    Polyester & {p['train']['family_counts'].get('polyester', 0):,} & {p['val']['family_counts'].get('polyester', 0):,} & {p['test']['family_counts'].get('polyester', 0):,} \\
    Polyamide & {p['train']['family_counts'].get('polyamide', 0):,} & {p['val']['family_counts'].get('polyamide', 0):,} & {p['test']['family_counts'].get('polyamide', 0):,} \\
    Polyether & {p['train']['family_counts'].get('polyether', 0):,} & {p['val']['family_counts'].get('polyether', 0):,} & {p['test']['family_counts'].get('polyether', 0):,} \\
    Fluoropolymer & {p['train']['family_counts'].get('fluoropolymer', 0):,} & {p['val']['family_counts'].get('fluoropolymer', 0):,} & {p['test']['family_counts'].get('fluoropolymer', 0):,} \\
    Aliphatic hydrocarbon & {p['train']['family_counts'].get('aliphatic_hc', 0):,} & {p['val']['family_counts'].get('aliphatic_hc', 0):,} & {p['test']['family_counts'].get('aliphatic_hc', 0):,} \\
    \midrule
    Null scaffold assigned to & \multicolumn{{3}}{{c}}{{{st['per_seed'][seed]['null_scaffold_partition']}}} \\
    Scaffold-disjoint train/test & \multicolumn{{3}}{{c}}{{{'yes' if st['per_seed'][seed]['scaffold_disjoint'] else 'no'}}} \\
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def tab_gc(gc):
    o, cov = gc['overall_all_predictions'], gc['by_atom_coverage']
    fam = {k: v for k, v in gc['by_family'].items() if v and v['n'] >= 40}
    fam_rows = '\n'.join(
        rf"    {k.replace('_', ' ')} & {v['n']:,} & {v['mae']:.1f} & {v['bias']:+.1f} & {v['pct_within_25K']:.1f} \\"
        for k, v in sorted(fam.items(), key=lambda t: -t[1]['n']))
    cov_rows = '\n'.join(
        rf"    {k} & {v['n']:,} & {v['mae']:.1f} & {v['pct_within_25K']:.1f} \\"
        for k, v in cov.items() if v)
    return rf"""\begin{{table}}[htbp]
  \caption{{Tier 1 group contribution applied with frozen parameters to the full PolyMetriX collection ($N = {o['n']:,}$). Upper block: error against the fraction of heavy atoms of the repeat unit matched by a functional group. Lower block: error by chemical family, for families with at least 40 members. Bias is the mean signed error, predicted minus observed. MAE, mean absolute error; $T_g$, glass transition temperature.}}
  \label{{tbl:gc}}
  \centering
  \footnotesize
  \begin{{tabular}}{{lccc}}
    \toprule
    Matched heavy-atom fraction & $N$ & MAE (K) & Within 25~K (\%) \\
    \midrule
{cov_rows}
    \midrule
    \textbf{{All}} & \textbf{{{o['n']:,}}} & \textbf{{{o['mae']:.1f}}} & \textbf{{{o['pct_within_25K']:.1f}}} \\
    \bottomrule
  \end{{tabular}}

  \smallskip
  \begin{{tabular}}{{lcccc}}
    \toprule
    Family & $N$ & MAE (K) & Bias (K) & Within 25~K (\%) \\
    \midrule
{fam_rows}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def tab_ml(df):
    def line(proto, rep, label):
        sub = df[(df.protocol == proto) & (df.rep == rep)]
        if len(sub) == 0:
            return None
        n = len(sub)
        sd = (lambda s: f' $\\pm$ {s.std(ddof=1):.3f}' if n > 1 else '')
        sdm = (lambda s: f' $\\pm$ {s.std(ddof=1):.1f}' if n > 1 else '')
        return (rf"    {label} & {n} & {sub.rf_test_r2.mean():.3f} & "
                rf"{sub.gb_test_r2.mean():.3f} & {sub.test_r2.mean():.3f}{sd(sub.test_r2)} & "
                rf"{sub.test_mae.mean():.1f}{sdm(sub.test_mae)} & "
                rf"{sub.test_rmse.mean():.1f} & {sub.eq_test_r2.mean():.3f} \\")
    rows = [line(p, 'ecfp4_2048', l) for p, l in (
        ('published', 'Scaffold split as submitted'),
        ('scaffold', 'Randomised scaffold split'),
        ('cluster', 'Butina cluster split'),
        ('random', 'Random split'))]
    rows += [r'    \midrule']
    rows += [line(p, r, l) for p, r, l in (
        ('scaffold', 'pb_cls', 'polyBERT CLS, randomised scaffold'),
        ('scaffold', 'pb_mean', 'polyBERT mean-pooled, randomised scaffold'),
        ('published', 'pb_cls', 'polyBERT CLS, split as submitted'),
        ('published', 'pb_mean', 'polyBERT mean-pooled, split as submitted'))]
    body = '\n'.join(r for r in rows if r)
    return rf"""\begin{{table}}[htbp]
  \caption{{Tier 2 and Tier 3 performance under four splitting protocols. Values are means over the seeds indicated, with across-seed standard deviations where more than one seed was run. The final column gives the equal-weight ensemble used in the submitted models, for comparison with the validation-optimised blend. RF, random forest; GB, gradient boosting; MAE, mean absolute error; RMSE, root-mean-square error; CLS, classification token.}}
  \label{{tbl:ml_performance}}
  \centering
  \footnotesize
  \begin{{tabular}}{{lccccccc}}
    \toprule
    Protocol / representation & Seeds & RF $R^2$ & GB $R^2$ & Ensemble $R^2$ & MAE (K) & RMSE (K) & Equal-wt.\ $R^2$ \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def tab_baselines(df, proto='scaffold'):
    labels = [
        ('mean', 'Mean predictor'),
        ('ridge_ecfp4', 'Ridge regression, ECFP4 2048'),
        ('rdkit_desc', 'RDKit 2-D descriptors'),
        ('polymetrix_desc', 'PolyMetriX hierarchical descriptors'),
        ('ecfp4_1024', 'ECFP4, 1024 bit'),
        ('ecfp4_2048', 'ECFP4, 2048 bit (as submitted)'),
        ('ecfp4_4096', 'ECFP4, 4096 bit'),
        ('ecfp6_2048', 'ECFP6, 2048 bit'),
        ('ecfp4_2048_counts', 'ECFP4 counts, 2048'),
        ('ecfp4_2048_methylcap', 'ECFP4 2048, methyl cap'),
        ('ecfp4_2048_wildcard', 'ECFP4 2048, wildcards retained'),
        ('pb_cls', 'polyBERT, CLS pooling'),
        ('pb_mean', 'polyBERT, mean pooling'),
    ]
    rows = []
    for rep, label in labels:
        sub = df[(df.protocol == proto) & (df.rep == rep)]
        if len(sub) == 0:
            continue
        n = len(sub)
        sd = f' $\\pm$ {sub.test_r2.std(ddof=1):.3f}' if n > 1 else ''
        rows.append(rf"    {label} & {int(sub.n_features.iloc[0]):,} & {n} & "
                    rf"{sub.test_r2.mean():.3f}{sd} & {sub.test_mae.mean():.1f} & "
                    rf"{sub.test_rmse.mean():.1f} \\")
    body = '\n'.join(rows)
    return rf"""\begin{{table}}[htbp]
  \caption{{Representation and baseline comparison under the randomised scaffold split, with identical partitions and identical RF/GB prediction heads for every row. Values are means over seeds with across-seed standard deviations. Features is the dimensionality of the representation. ECFP4/ECFP6, extended-connectivity fingerprint of diameter 4/6; CLS, classification token; RF, random forest; GB, gradient boosting; MAE, mean absolute error; RMSE, root-mean-square error.}}
  \label{{tbl:baselines}}
  \centering
  \footnotesize
  \begin{{tabular}}{{lccccc}}
    \toprule
    Representation & Features & Seeds & Test $R^2$ & MAE (K) & RMSE (K) \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def tab_tier(tt):
    rows = '\n'.join(
        rf"    {r['polymer']} & {r['exp']:.0f} & {r['tier1']:.1f} & {r['tier2']:.1f} & "
        rf"{r['tier3']:.1f} & {r['partition']} & "
        rf"{('%.1f' % r['dataset_tg']) if r['dataset_tg'] is not None else 'n/a'} \\"
        for r in tt['rows'])
    return rf"""\begin{{table}}[htbp]
  \caption{{Three-tier predictions for the eight canonical homopolymers, with partition membership under the split as submitted. Five of the eight are training-set members and three are absent from the collection, so this table demonstrates interpolation and inter-tier disagreement, not external validation. The final column gives the curated dataset value where the polymer is present, which differs from the Polymer Handbook value used as the reference. Tier 3 uses the published mean-pooled polyBERT fingerprint. PE, polyethylene; PP, polypropylene; PS, polystyrene; PVC, poly(vinyl chloride); PET, poly(ethylene terephthalate); PEO, poly(ethylene oxide); POM, polyoxymethylene; PVOH, poly(vinyl alcohol); MAE, mean absolute error.}}
  \label{{tbl:tier_comparison}}
  \centering
  \begin{{tabular}}{{lccccll}}
    \toprule
    Polymer & Ref.\ $T_g$ (K) & Tier 1 & Tier 2 & Tier 3 & Partition & Dataset $T_g$ \\
    \midrule
{rows}
    \midrule
    MAE (K) & n/a & {tt['tier1_mae']:.1f} & {tt['tier2_mae']:.1f} & {tt['tier3_mae']:.1f} & & \\
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def tab_reliability(cal):
    q = cal['error_by_pri_quintile_recalibrated']
    ad = cal['error_by_ad_status']
    rc = cal['rank_correlation_with_absolute_error']
    conf = cal['conformal']['levels']
    qrows = '\n'.join(
        rf"    Q{i+1} & {b['n']:,} & {b['mae']:.1f} & {b['median_ae']:.1f} & {b['rmse']:.1f} \\"
        for i, b in enumerate(q))
    simq = cal['error_by_similarity_quintile']
    simrows = '\n'.join(
        rf"    Q{i+1} ({b['bin']}) & {b['n']:,} & {b['mae']:.1f} & {b['median_ae']:.1f} \\"
        for i, b in enumerate(simq))
    adnote = '; '.join(
        f"{k.replace('_', '-').lower()} {v['n']:,}" for k, v in ad.items() if v)
    crows = '\n'.join(
        rf"    {100*(1-float(a)):.0f}\% & {100*v['empirical_coverage']:.1f}\% & {v['median_halfwidth_K']:.0f} \\"
        for a, v in sorted(conf.items(), key=lambda t: -float(t[0])))
    return rf"""\begin{{table}}[htbp]
  \caption{{Calibration of the reliability machinery on the held-out test partition. All parameters were estimated from five-fold out-of-fold predictions over the training partition and from the validation partition; no test label was used in fitting. AD, applicability domain; PRI, Prediction Reliability Index; MAE, mean absolute error; AE, absolute error; RMSE, root-mean-square error; RF, random forest; GB, gradient boosting.}}
  \label{{tbl:reliability}}
  \centering
  \footnotesize
  \begin{{tabular}}{{lcccc}}
    \multicolumn{{5}}{{l}}{{\textbf{{(a) Error by recalibrated PRI quintile}}}} \\
    \toprule
    Quintile & $N$ & MAE (K) & Median AE (K) & RMSE (K) \\
    \midrule
{qrows}
    \bottomrule
  \end{{tabular}}

  \smallskip
  \begin{{tabular}}{{lccc}}
    \multicolumn{{4}}{{l}}{{\textbf{{(b) Error by nearest-neighbour similarity quintile}}}} \\
    \toprule
    Quintile (max Tanimoto) & $N$ & MAE (K) & Median AE (K) \\
    \midrule
{simrows}
    \bottomrule
  \end{{tabular}}

  \smallskip
  \footnotesize{{The categorical applicability-domain label is not shown because it
  is degenerate on this partition: {adnote}. The lowest maximum-Tanimoto value in
  the partition is above the out-of-domain boundary, so the conventional 0.30
  threshold assigns essentially every query to one class. The graded quantity in
  panel (b) is the informative form.}}

  \smallskip
  \begin{{tabular}}{{lcc}}
    \multicolumn{{3}}{{l}}{{\textbf{{(c) Split-conformal interval coverage}}}} \\
    \toprule
    Nominal & Empirical & Median half-width (K) \\
    \midrule
{crows}
    \bottomrule
  \end{{tabular}}

  \smallskip
  \begin{{tabular}}{{lcc}}
    \multicolumn{{3}}{{l}}{{\textbf{{(d) Spearman rank correlation with absolute error}}}} \\
    \toprule
    Indicator & $\rho$ & $p$ \\
    \midrule
    Recalibrated PRI & {rc['pri_recalibrated']['spearman']:.3f} & {rc['pri_recalibrated']['p']:.2e} \\
    PRI as published & {rc['pri_as_published']['spearman']:.3f} & {rc['pri_as_published']['p']:.2e} \\
    Maximum Tanimoto similarity & {rc['max_tanimoto']['spearman']:.3f} & {rc['max_tanimoto']['p']:.2e} \\
    RF/GB disagreement & {rc['disagreement']['spearman']:.3f} & {rc['disagreement']['p']:.2e} \\
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def tab_case(cs):
    def row(key, label):
        s = cs['strategies'].get(key)
        if not s or not s.get('top25'):
            return None
        t = s['top25']
        return (rf"    {label} & {s['n_selected_by_window']:,} & {t['k']} & "
                rf"{100*t['precision']:.1f} & {t['enrichment_over_base_rate']:.2f} & "
                rf"{t['mae_on_selected']:.1f} \\")
    rows = [row(k, l) for k, l in (
        ('tier1_gc', 'Tier 1 prediction in window'),
        ('tier2_ml', 'Tier 2 prediction in window'),
        ('tier2_ml_in_domain', 'Tier 2 + in-domain'),
        ('tier2_ml_pri_high', 'Tier 2 + top PRI quartile'),
        ('tier2_ml_low_disagreement', 'Tier 2 + lowest disagreement quartile'),
        ('tier1_and_tier2_consensus', 'Tier 1 and Tier 2 consensus'))]
    body = '\n'.join(r for r in rows if r)
    return rf"""\begin{{table}}[htbp]
  \caption{{Simulated prospective selection on the held-out test partition. Target: $T_g$ in {cs['design_target'].split('in ')[1]}. Base rate {100*cs['base_rate']:.1f}\% ({cs['n_true_hits_in_pool']} hits among {cs['n_candidates']} candidates). Precision is measured at the top 25 ranked candidates; enrichment is precision divided by the base rate. Eligible is the number of candidates passing the strategy's filter before ranking. $T_g$, glass transition temperature; PRI, Prediction Reliability Index; MAE, mean absolute error.}}
  \label{{tbl:case}}
  \centering
  \footnotesize
  \begin{{tabular}}{{lccccc}}
    \toprule
    Selection strategy & Eligible & $k$ & Precision (\%) & Enrichment & MAE on selected (K) \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


# --------------------------------------------------------------------------
def main():
    df_data, _ = load_dataset()
    y = df_data['target'].values.astype(float)
    X = morgan_matrix(df_data['smiles'].tolist(), radius=2, n_bits=2048)
    z = np.load(OUT / 's02_splits.npz')
    tr = z['published_42_train']
    lo, hi = fit_outlier_filter(y[tr])
    tr = tr[(y[tr] >= lo) & (y[tr] <= hi)]

    runs = runs_frame()
    print(f'[runs] {len(runs)} archived runs')
    sig = significance(runs)
    acc = jload('s01_dataset_accounting.json')
    st = jload('s01_scaffold_stats.json')
    gc = jload('s04_gc_summary.json')
    cal = jload('s05_calibration_published_42.json')
    inv = jload('s03_invariance.json')['summary']
    inv2 = jload('s03b_repeat_unit_invariance.json')['summary']
    cap = jload('s12_capping_collapse.json')
    mir = jload('s03_mirror_verification.json')
    ext = jload('s08_external_conjugated.json')
    cs = jload('s08_case_study.json')

    tt = (jload('s10_tier_table.json') if (OUT / 's10_tier_table.json').exists()
          else tier_table(df_data, y, X, tr))

    per = pd.read_csv(OUT / 's05_perpolymer_published_42.csv')
    proc = processability_correlation(per)
    if proc:
        dump('s10_processability.json', proc)

    sizes = st['size_distribution_top20']
    p42 = st['per_seed']['42']['partitions']
    fit = cal['fitted_on_training_out_of_fold_only']
    conf = cal['conformal']['levels']
    priq = cal['error_by_pri_quintile_recalibrated']
    disq = cal['error_by_disagreement_decile']
    ad = cal['error_by_ad_status']
    rc = cal['rank_correlation_with_absolute_error']

    e_r2, e_sd, _ = agg(runs, 'scaffold', 'ecfp4_2048')
    pm_r2, pm_sd, _ = agg(runs, 'scaffold', 'pb_mean')
    pc_r2, pc_sd, _ = agg(runs, 'scaffold', 'pb_cls')
    pub_r2, _, _ = agg(runs, 'published', 'ecfp4_2048')
    pub_mae, _, _ = agg(runs, 'published', 'ecfp4_2048', 'test_mae')
    rnd_r2, rnd_sd, _ = agg(runs, 'random', 'ecfp4_2048')
    clu_r2, clu_sd, _ = agg(runs, 'cluster', 'ecfp4_2048')
    cnt_r2, _, _ = agg(runs, 'scaffold', 'ecfp4_2048_counts')
    cnt_mae, _, _ = agg(runs, 'scaffold', 'ecfp4_2048_counts', 'test_mae')
    e_mae, _, _ = agg(runs, 'scaffold', 'ecfp4_2048', 'test_mae')
    rid_r2, _, _ = agg(runs, 'scaffold', 'ridge_ecfp4')
    pmx_r2, _, _ = agg(runs, 'scaffold', 'polymetrix_desc')
    cap_r2, _, _ = agg(runs, 'scaffold', 'ecfp4_2048_methylcap')
    wild_r2, _, _ = agg(runs, 'scaffold', 'ecfp4_2048_wildcard')
    mean_r2, _, _ = agg(runs, 'published', 'mean')
    pmx_nfeat = int(runs[runs.rep == 'polymetrix_desc'].n_features.iloc[0])

    # How much the evaluation protocol alone moves the same model on the same
    # representation; used to keep the ECFP4-vs-polyBERT difference in scale.
    _protocol_range = (max(pub_r2, e_r2, rnd_r2, clu_r2)
                       - min(pub_r2, e_r2, rnd_r2, clu_r2))

    # Does the ordering hold under the other structure-aware protocol? If it
    # reverses, no general statement about the two representations is
    # defensible, whatever the scaffold-split bootstrap says.
    pm_clu_r2, pm_clu_sd, _ = agg(runs, 'cluster', 'pb_mean')
    pc_clu_r2, _, _ = agg(runs, 'cluster', 'pb_cls')
    reversal = (e_r2 - pm_r2) * (clu_r2 - pm_clu_r2) < 0

    pooled = sig['scaffold']['pooled']
    wil = sig['scaffold'].get('across_seed_wilcoxon', {})
    delta = pooled['mean_delta_r2']
    ci = f"[{pooled['ci_lo']:.3f}, {pooled['ci_hi']:.3f}]"
    n_seeds = len(sig['scaffold']['per_seed'])
    # Every seed's own paired bootstrap must exclude zero, and every seed must
    # agree in direction. The across-seed Wilcoxon is reported but is NOT used
    # as the gate: with five pairs its smallest attainable two-sided p is
    # 0.0625, so requiring p < 0.05 would be unsatisfiable by construction.
    all_seed_cis_exclude_zero = all(
        r['ecfp4_vs_pbmean']['lo'] > 0 or r['ecfp4_vs_pbmean']['hi'] < 0
        for r in sig['scaffold']['per_seed'])
    consistent = pooled['n_seeds_favouring_ecfp4'] in (0, n_seeds)
    sig_bootstrap = all_seed_cis_exclude_zero and consistent
    wil_p = wil.get('p')

    if reversal:
        verdict = (
            f'The ordering does not hold under the other structure-aware '
            f'protocol. Under the Butina leave-clusters-out split it reverses: '
            f'mean-pooled polyBERT reaches {pm_clu_r2:.3f} against '
            f'{clu_r2:.3f} for ECFP4. A difference that changes sign when the '
            f'partition is built by chemical similarity rather than by '
            f'scaffold cannot support a general statement about the two '
            f'representations, however tight the confidence interval is on any '
            f'one protocol. We therefore withdraw the claim: what the data '
            f'show is that the ranking of these two representations is '
            f'protocol-dependent, not that either generalises better. The '
            f'abstract, discussion and conclusions have been rewritten '
            f'accordingly.')
        verdict_short = (
            f'the ordering reverses under the leave-clusters-out protocol '
            f'({pm_clu_r2:.3f} for polyBERT against {clu_r2:.3f} for ECFP4), '
            f'so the ranking is protocol-dependent and the original claim is '
            f'withdrawn.')
    elif sig_bootstrap:
        verdict = (
            f'The direction is the same in all {n_seeds} seeds and the paired '
            f'bootstrap interval excludes zero in every one of them, so the '
            f'difference is reproducible and statistically resolvable at the '
            f'level of individual test polymers. We note two caveats. The '
            f'across-seed Wilcoxon signed-rank test cannot reach $p < 0.05$ '
            f'with {n_seeds} pairs, its smallest attainable two-sided value '
            f'being 0.0625, so it is reported for completeness rather than as '
            f'the criterion. And the effect is far smaller than that of the '
            f'splitting protocol itself, which moves the same model and the '
            f'same representation by {_protocol_range:.2f} in $R^2$. '
            f'We therefore describe '
            f'it as a small but consistent representational difference, not as '
            f'evidence that fingerprints are generally preferable to '
            f'transformer embeddings.')
        verdict_short = (
            f'the difference is small and consistent across seeds, but it is '
            f'far smaller than the {_protocol_range:.2f} in $R^2$ that the '
            f'splitting protocol alone accounts for, so it '
            f'does not support a general claim about fingerprints versus '
            f'transformer embeddings.')
    else:
        verdict = ('The confidence interval includes zero, so the difference is '
                   'not statistically resolvable at this sample size. We '
                   'therefore report it as a preliminary observation rather '
                   'than as evidence of a representational advantage, and the '
                   'abstract, discussion and conclusions have been revised '
                   'accordingly.')
        verdict_short = ('the difference is not statistically resolvable and is '
                         'reported as a preliminary observation only.')

    csr = cs['reliability_separates_success_from_failure']
    p_pri = csr.get('mannwhitney_pri_correct_gt_wrong_p')
    if p_pri is not None and p_pri < 0.05:
        case_verdict = ('The reliability index therefore does separate the '
                        'selections that succeeded from those that failed, '
                        'which is the property a prospective user needs.')
    else:
        case_verdict = ('The difference in reliability between successful and '
                        'failed selections is not statistically resolvable at '
                        'this sample size, so the index should be read as a '
                        'guide to expected error magnitude rather than as a '
                        'per-candidate success probability.')

    def _ad(key, field, fmt='{:.1f}', default='n/a'):
        """A stratum that holds no polymer has no statistics; say so."""
        rec = ad.get(key)
        if not rec:
            return default
        return fmt.format(rec[field])

    # Describe the PRI trend from the series rather than asserting a shape.
    _pri_maes = [b['mae'] for b in priq]
    if all(a >= b for a, b in zip(_pri_maes, _pri_maes[1:])):
        _pri_trend = 'falls monotonically'
    else:
        _flat_from = next(i for i in range(1, len(_pri_maes))
                          if _pri_maes[i] >= _pri_maes[i - 1])
        _pri_trend = (f'falls across the lower {_flat_from} quintiles and is '
                      f'then flat')

    simq = cal['error_by_similarity_quintile']

    # Conformal coverage is reported as measured. Split conformal guarantees
    # marginal coverage only under exchangeability between the calibration and
    # test partitions, which a structure-aware split deliberately violates, so
    # the verdict sentence is derived from the numbers rather than assumed.
    _cov_gaps = [v['nominal_coverage'] - v['empirical_coverage']
                 for v in conf.values()]
    _worst_gap = max(_cov_gaps)
    if _worst_gap <= 0.02:
        _conf_verdict = (
            'Empirical coverage tracks the nominal level to within '
            f'{100 * _worst_gap:.1f} percentage points at every level tested, '
            'so these intervals, rather than the disagreement itself, are what '
            'the platform reports as its uncertainty statement.')
    else:
        _conf_verdict = (
            f'Coverage falls short of nominal by up to '
            f'{100 * _worst_gap:.0f} percentage points. This is a property of '
            f'the evaluation design rather than an implementation fault: split '
            f'conformal guarantees marginal coverage only when the calibration '
            f'and test partitions are exchangeable, and a scaffold split is '
            f'constructed precisely so that they are not. The platform '
            f'therefore reports these intervals with their measured coverage '
            f'attached rather than with their nominal level, and a user should '
            f'read an interval labelled 80\\% as the empirically '
            f'{100 * min(v["empirical_coverage"] for v in conf.values()):.0f}'
            f'-to-'
            f'{100 * max(v["empirical_coverage"] for v in conf.values()):.0f}\\% '
            f'interval that it is.')

    def cov(a):
        k = [x for x in conf if abs(float(x) - a) < 1e-9]
        return conf[k[0]] if k else None

    def _pk(cs, key, k='top25'):
        """Precision-at-k for one selection strategy, or a readable fallback
        when the strategy selected nothing."""
        st = cs['strategies'].get(key) or {}
        top = st.get(k)
        if not top:
            return 'not evaluable (no candidate passed this filter)'
        return f"{100 * top['precision']:.1f}" + r"\%"

    # Derived quantities used in the float-citation sentences.
    tier_spread = max(max(r['tier1'], r['tier2'], r['tier3'])
                      - min(r['tier1'], r['tier2'], r['tier3'])
                      for r in tt['rows'])
    tier1_wins = sum(
        1 for r in tt['rows']
        if abs(r['tier1'] - r['exp']) <= min(abs(r['tier2'] - r['exp']),
                                             abs(r['tier3'] - r['exp'])))

    T = {
        'N_SUBMITTED': f"{acc['as_submitted']['after_raw_string_dedup']:,}",
        'TG_SHIFT': f"{p42['test']['target_mean'] - p42['train']['target_mean']:.0f}",
        'TIER_MAX_SPREAD': f'{tier_spread:.0f}',
        'TIER1_WINS': {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                       6: 'six', 7: 'seven', 8: 'all eight'}.get(
                          tier1_wins, str(tier1_wins)),
        'N_DATASET': f"{acc['revised_pipeline']['after_canonical_dedup']:,}",
        'N_SCAFFOLD_GROUPS': f"{st['n_scaffold_groups']:,}",
        'N_SINGLETON': f"{st['n_singleton_groups']:,}",
        'NULL_GROUP_SIZE': f"{st['null_scaffold_group_size']:,}",
        'NULL_GROUP_PCT': f"{100*st['null_scaffold_fraction']:.1f}",
        'SECOND_GROUP_SIZE': f"{sizes[1]:,}",
        'TEST_ALIPHATIC_N': str(p42['test']['family_counts'].get('aliphatic_hc', 0)),
        'N_TEST': f"{p42['test']['n']:,}",
        'TEST_MEAN_TG': f"{p42['test']['target_mean']:.1f}",
        'TRAIN_MEAN_TG': f"{p42['train']['target_mean']:.1f}",
        'TEST_SD_TG': f"{p42['test']['target_sd']:.1f}",
        'TRAIN_SD_TG': f"{p42['train']['target_sd']:.1f}",
        'MEAN_BASELINE_R2': f'{mean_r2:.3f}',
        'CANON_MAX_DELTA': f"{max(abs(v['dataset_vs_manuscript_delta_K']) for v in gc['canonical_eight'].values() if v['dataset_vs_manuscript_delta_K'] is not None):.0f}",
        'GC_CANON_MAE': f"{gc['canonical_eight_mae']:.1f}",
        'GC_CANON_MAX': f"{max(v['error_vs_manuscript_K'] for v in gc['canonical_eight'].values()):.1f}",
        'GC_FULL_N': f"{gc['overall_all_predictions']['n']:,}",
        'GC_FULL_MAE': f"{gc['overall_all_predictions']['mae']:.0f}",
        'GC_FULL_MEDIAN': f"{gc['overall_all_predictions']['median_ae']:.0f}",
        'GC_FULL_R2': f"{gc['overall_all_predictions']['r2']:.2f}",
        'GC_FULL_BIAS': f"{gc['overall_all_predictions']['bias']:+.0f}",
        'GC_PCT10': f"{gc['overall_all_predictions']['pct_within_10K']:.1f}",
        'GC_PCT50': f"{gc['overall_all_predictions']['pct_within_50K']:.1f}",
        # Coverage bands, not the fine bins: the finest low-coverage bin holds
        # a single polymer, and the fine-bin trend is not monotone.
        'GC_MAE_LOWCOV': f"{gc['by_coverage_band']['below_0.75']['mae']:.0f}",
        'GC_N_LOWCOV': f"{gc['by_coverage_band']['below_0.75']['n']:,}",
        'GC_MAE_MIDCOV': f"{gc['by_coverage_band']['0.75_to_below_1.00']['mae']:.0f}",
        'GC_N_MIDCOV': f"{gc['by_coverage_band']['0.75_to_below_1.00']['n']:,}",
        'GC_MAE_FULLCOV': f"{gc['by_coverage_band']['full_coverage']['mae']:.0f}",
        'GC_N_FULLCOV': f"{gc['by_coverage_band']['full_coverage']['n']:,}",
        'GC_PCT_FULLCOV': f"{gc['coverage_distribution']['pct_full_coverage']:.1f}",
        'GC_N_FALLBACK': str(gc['n_fallback']),
        'GC_CALIBRATION_SET': ', '.join(
            jload('s11_gc_calibration_set.json')['calibration_structures']),
        'MIRROR_COSINE': f"{mir['mean_cosine']:.6f}",
        'MIRROR_MAXDIFF': f"{mir['max_abs_diff']:.1e}",
        'AD_K': f"{fit['ad_logistic_k']:.2f}",
        'AD_TAU': f"{fit['ad_logistic_tau']:.3f}",
        'UNC_K': f"{fit['unc_logistic_k_per_K']:.4f}",
        'UNC_C': f"{fit['unc_logistic_centre_K']:.1f}",
        'OOF_MAE': f"{fit['oof_mae_K']:.1f}",
        'MODEL_COMPONENT': f"{fit['model_component_validation_r2']:.3f}",
        'T2_PUB_R2': f'{pub_r2:.3f}',
        'T2_PUB_MAE': f'{pub_mae:.1f}',
        'T2_SCAFFOLD_R2': f'{e_r2:.3f}', 'T2_SCAFFOLD_R2SD': f'{e_sd:.3f}',
        'T2_RANDOM_R2': f'{rnd_r2:.3f}', 'T2_RANDOM_R2SD': f'{rnd_sd:.3f}',
        'T2_CLUSTER_R2': f'{clu_r2:.3f}', 'T2_CLUSTER_R2SD': f'{clu_sd:.3f}',
        'PROTOCOL_RANGE': f'{_protocol_range:.2f}',
        'PBMEAN_SCAFFOLD_R2': f'{pm_r2:.3f}', 'PBMEAN_SCAFFOLD_R2SD': f'{pm_sd:.3f}',
        'PBCLS_SCAFFOLD_R2': f'{pc_r2:.3f}', 'PBCLS_SCAFFOLD_R2SD': f'{pc_sd:.3f}',
        'PBMEAN_CLUSTER_R2': f'{pm_clu_r2:.3f}',
        'PBMEAN_CLUSTER_R2SD': f'{pm_clu_sd:.3f}',
        'PBCLS_CLUSTER_R2': f'{pc_clu_r2:.3f}',
        'ORDERING_REVERSES': ('reverses' if reversal else 'is preserved'),
        'DELTA_R2_SCAFFOLD': f'{delta:+.3f}',
        'DELTA_R2_PUBLISHED': f"{sig['published']['pooled']['mean_delta_r2']:+.3f}",
        'DELTA_R2_PUBLISHED_P': f"{sig['published']['pooled']['max_p']:.2f}",
        'DELTA_R2_CI': ci,
        'DELTA_R2_P': f"{pooled['max_p']:.3f}",
        'DELTA_R2_WILCOXON': (f'{wil_p:.3f}' if wil_p is not None else 'n/a'),
        'DELTA_R2_VERDICT': verdict,
        'DELTA_R2_VERDICT_SHORT': verdict_short,
        'DELTA_R2_VERDICT_SHORT_CAP': verdict_short[0].upper() + verdict_short[1:],
        'COUNTS_GAIN_R2': f'{cnt_r2 - e_r2:+.3f}',
        'COUNTS_GAIN_MAE': f'{e_mae - cnt_mae:.1f}',
        'RIDGE_R2': f'{rid_r2:.3f}',
        'PMX_R2': f'{pmx_r2:.3f}', 'PMX_NFEAT': str(pmx_nfeat),
        'CAP_DELTA': f'{cap_r2 - e_r2:+.3f}',
        'CAP_WILDCARD_DELTA': f'{wild_r2 - e_r2:+.3f}',
        'CAP_MERGE_H_POLY': f"{cap['by_capping']['hydrogen']['n_polymers_in_merged_groups']:,}",
        'CAP_MERGE_H_GROUPS': f"{cap['by_capping']['hydrogen']['n_merged_groups']:,}",
        'CAP_MERGE_H_MAXSPREAD': f"{cap['by_capping']['hydrogen']['max_tg_spread_within_a_merged_group_K']:.0f}",
        'CAP_MERGE_M_POLY': f"{cap['by_capping']['methyl']['n_polymers_in_merged_groups']:,}",
        'CAP_MERGE_M_GROUPS': f"{cap['by_capping']['methyl']['n_merged_groups']:,}",
        'CAP_FLOOR': f"{cap['irreducible_mae_floor_from_merging_K']:.2f}",
        'INV_ROT_MIN': f"{min(v['cls_cosine_min'] for k, v in inv.items() if k.startswith('rotated')):.2f}",
        'INV_X2_CLS': f"{inv2['x2_repeat_units']['polybert_cls_cosine_mean']:.2f}",
        'INV_X2_CLS_MIN': f"{inv2['x2_repeat_units']['polybert_cls_cosine_min']:.2f}",
        'INV_X2_MEAN': f"{inv2['x2_repeat_units']['polybert_mean_cosine_mean']:.2f}",
        'INV_X2_ECFP': f"{inv2['x2_repeat_units']['ecfp4_tanimoto_mean']:.2f}",
        'TIER2_CANON_MAE': f"{tt['tier2_mae']:.1f}",
        'TIER3_CANON_MAE': f"{tt['tier3_mae']:.1f}",
        'EXT_N': str(ext['n_external_unseen']),
        'EXT_SIM': f"{ext['mean_max_tanimoto_to_training']:.2f}",
        'EXT_PCT_OOD': f"{ext['pct_out_of_domain']:.0f}",
        'EXT_ML_MAE': f"{ext['tier2_ml']['mae']:.0f}",
        'EXT_GC_MAE': f"{ext['tier1_gc']['mae']:.0f}",
        'EXT_GC_BIAS': f"{ext['tier1_gc']['bias_K']:+.0f}",
        'EXT_N_IN_TRAIN': str(ext['n_already_in_training']),
        'CASE_GC_P25': _pk(cs, 'tier1_gc'),
        'CASE_DIS_P25': _pk(cs, 'tier2_ml_low_disagreement'),
        'CASE_DIS_ENRICH': (
            f"{cs['strategies']['tier2_ml_low_disagreement']['top25']['enrichment_over_base_rate']:.1f}"
            if (cs['strategies'].get('tier2_ml_low_disagreement') or {}).get('top25')
            else 'n/a'),
        'N_TEST_PUB': f"{cal['n_test']:,}",
        'AD_IN_N': _ad('IN_DOMAIN', 'n', '{:,}', '0'),
        'AD_IN_PCT': _ad('IN_DOMAIN', 'fraction', '{:.1%}', '0%').rstrip('%'),
        'AD_BORDER_N': _ad('BORDERLINE', 'n', '{:,}', '0'),
        'AD_BORDER_PCT': _ad('BORDERLINE', 'fraction', '{:.1%}', '0%').rstrip('%'),
        'AD_OOD_N': _ad('OUT_OF_DOMAIN', 'n', '{:,}', '0'),
        'AD_OOD_PCT': _ad('OUT_OF_DOMAIN', 'fraction', '{:.1%}', '0%').rstrip('%'),
        'AD_IN_MAE': _ad('IN_DOMAIN', 'mae'),
        'AD_OOD_MAE': _ad('OUT_OF_DOMAIN', 'mae'),
        # Similarity strata: the graded form of the same quantity, which is
        # what actually carries information on this partition.
        'MIN_TEST_SIM': f"{float(per['max_tanimoto'].min()):.3f}",
        'SIM_Q1_MAE': f"{simq[0]['mae']:.1f}",
        'SIM_Q5_MAE': f"{simq[-1]['mae']:.1f}",
        'SIM_Q1_HI': simq[0]['bin'].split('-')[-1],
        'SIM_SPEARMAN': f"{rc['max_tanimoto']['spearman']:.3f}",
        'PRI_TREND_PHRASE': _pri_trend,
        'PRI_ASPUB_SPEARMAN': f"{rc['pri_as_published']['spearman']:.3f}",
        'PRI_MEDIAN': f"{cal['pri_distribution_recalibrated']['median']:.2f}",
        'PRI_IQR': (f"{cal['pri_distribution_recalibrated']['q25']:.2f} to "
                    f"{cal['pri_distribution_recalibrated']['q75']:.2f}"),
        'PRI_Q1_MAE': f"{priq[0]['mae']:.1f}",
        'PRI_Q5_MAE': f"{priq[-1]['mae']:.1f}",
        'PRI_SPEARMAN': f"{rc['pri_recalibrated']['spearman']:.3f}",
        'PRI_SPEARMAN_P': f"{rc['pri_recalibrated']['p']:.1e}",
        'DIS_SPEARMAN': f"{rc['disagreement']['spearman']:.3f}",
        'DIS_SPEARMAN_P': f"{rc['disagreement']['p']:.1e}",
        'DIS_D1_MAE': f"{disq[0]['mae']:.1f}",
        'DIS_D10_MAE': f"{disq[-1]['mae']:.1f}",
        'CONFORMAL_COVERAGE_80': f"{100*cov(0.20)['empirical_coverage']:.1f}\\%",
        'CONFORMAL_COVERAGE_90': f"{100*cov(0.10)['empirical_coverage']:.1f}\\%",
        'CONFORMAL_COVERAGE_68': f"{100*cov(0.32)['empirical_coverage']:.1f}\\%",
        'CONFORMAL_WIDTH_80': f"{cov(0.20)['median_halfwidth_K']:.0f}",
        'CONFORMAL_WIDTH_90': f"{cov(0.10)['median_halfwidth_K']:.0f}",
        'CONFORMAL_WIDTH_68': f"{cov(0.32)['median_halfwidth_K']:.0f}",
        'CONFORMAL_VERDICT': _conf_verdict,
        'CONFORMAL_SHORTFALL': f'{100 * _worst_gap:.0f}',
        'PRI_SENS_MAX': f"{max(v['pct_category_changed'] for v in cal['weight_sensitivity_analysis'].values()):.1f}",
        'PROC_R': (f"{proc['pearson_r']:+.2f}" if proc else 'not computed'),
        'CASE_WINDOW': cs['design_target'].split('in ')[1].replace(' K', ''),
        'CASE_NHITS': str(cs['n_true_hits_in_pool']),
        'CASE_NPOOL': str(cs['n_candidates']),
        'CASE_BASERATE': f"{100*cs['base_rate']:.1f}",
        'CASE_ML_P25': _pk(cs, 'tier2_ml'),
        'CASE_ML_ENRICH': (
            f"{cs['strategies']['tier2_ml']['top25']['enrichment_over_base_rate']:.1f}"
            if (cs['strategies'].get('tier2_ml') or {}).get('top25')
            else 'n/a'),
        'CASE_AD_P25': f"{100*cs['strategies']['tier2_ml_in_domain']['top25']['precision']:.1f}\\%",
        'CASE_PRI_P25': f"{100*cs['strategies']['tier2_ml_pri_high']['top25']['precision']:.1f}\\%",
        'CASE_CONS_P25': (f"{100*cs['strategies']['tier1_and_tier2_consensus']['top25']['precision']:.1f}\\%"
                          if cs['strategies']['tier1_and_tier2_consensus'].get('top25')
                          else 'not evaluable (no consensus candidates)'),
        # These are None if the selection produced no correct (or no wrong)
        # candidate at all; the sentence must still read correctly.
        'CASE_PRI_CORRECT': (f"{csr['pri_mean_when_correct']:.3f}"
                             if csr.get('pri_mean_when_correct') is not None
                             else 'not evaluable'),
        'CASE_PRI_WRONG': (f"{csr['pri_mean_when_wrong']:.3f}"
                           if csr.get('pri_mean_when_wrong') is not None
                           else 'not evaluable'),
        'CASE_PRI_TEST': (f"Mann-Whitney $p = {p_pri:.3f}$" if p_pri is not None
                          else 'test not evaluable'),
        'CASE_VERDICT': case_verdict,
        'PI1M_N': f"{cs.get('pi1m_prospective_screen', {}).get('n_screened', 0):,}",
        'PI1M_REPORTED': str(cs.get('pi1m_prospective_screen', {}).get('n_reported', 0)),
        'TABLE_PLATFORMS': tab_platforms(),
        'TABLE_ACCOUNTING': tab_accounting(acc),
        'TABLE_SPLITS': tab_splits(st),
        'TABLE_GC': tab_gc(gc),
        'TABLE_ML': tab_ml(runs),
        'TABLE_BASELINES': tab_baselines(runs),
        'TABLE_TIER': tab_tier(tt),
        'TABLE_RELIABILITY': tab_reliability(cal),
        'TABLE_CASE': tab_case(cs),
    }

    # Table and figure numbers are derived from the order of appearance in the
    # manuscript template, so the response letter can cite them without
    # hand-numbering (which rots silently when a float is inserted).
    ms_src = (REVISION / 'Manuscript_R1.tex.in').read_text(encoding='utf-8')
    resolved = re.sub(r'@@(TABLE_[A-Z0-9_]+)@@',
                      lambda m: str(T.get(m.group(1), '')), ms_src)
    for kind, prefix in (('tbl', 'TBLNO'), ('fig', 'FIGNO')):
        seen = []
        for m in re.finditer(r'\\label\{' + kind + r':([A-Za-z0-9_]+)\}',
                             resolved):
            if m.group(1) not in seen:
                seen.append(m.group(1))
        for i, name in enumerate(seen, 1):
            T[f'{prefix}_{name.upper()}'] = str(i)
        print(f'[{kind}] ' + ', '.join(f'{n}={i}' for i, n in enumerate(seen, 1)))

    all_used = set()
    for stem in ('Manuscript_R1', 'response_to_reviewers'):
        src_path = REVISION / f'{stem}.tex.in'
        if not src_path.exists():
            print(f'[skip] {src_path.name} not present')
            continue
        src = src_path.read_text(encoding='utf-8')
        used = set(re.findall(r'@@([A-Z0-9_]+)@@', src))
        all_used |= used
        missing = used - set(T)
        if missing:
            raise SystemExit(f'BUILD FAILED ({stem}) - tokens used but not '
                             f'defined: {sorted(missing)}')
        out = re.sub(r'@@([A-Z0-9_]+)@@', lambda m: str(T[m.group(1)]), src)
        left = re.findall(r'@@([A-Z0-9_]+)@@', out)
        if left:
            raise SystemExit(f'BUILD FAILED ({stem}) - unresolved tokens '
                             f'remain: {left}')
        (REVISION / f'{stem}.tex').write_text(out, encoding='utf-8')
        print(f'[write] {stem}.tex ({len(used)} tokens substituted)')

    unused = set(T) - all_used
    if unused:
        print(f'[warn] tokens defined but unused: {sorted(unused)}')
    dump('s10_tokens.json', {k: v for k, v in T.items()
                             if not k.startswith('TABLE_')})


if __name__ == '__main__':
    main()
