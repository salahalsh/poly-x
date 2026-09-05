"""
Script 09 - all manuscript figures, regenerated from archived prediction files.

Reviewer 2, Minor 11: "all plotted data should be generated directly from
archived prediction files. Figure captions should state the data source,
partition, sample number, and whether values are observed, calculated, or
illustrative."

Every panel below reads a file written by scripts 01-08. Nothing is generated,
assumed or hard-coded. Each figure also writes a sidecar ``.provenance.txt``
naming the exact input files and the number of points drawn, so a reader can
trace any plotted value back to its source.

Figures produced (revised numbering):
  fig2_gc_validation.pdf   Tier 1: the eight canonical polymers AND the full
                           7,367-polymer evaluation, side by side.
  fig3_ml_parity.pdf       Tier 2 parity with residual and error-distribution
                           panels (Reviewer 2, Minor 11).
  fig5_enhanced_features.pdf  The reliability panel, computed on the test
                           partition: error versus similarity, PRI
                           distribution, error by PRI quintile, and conformal
                           coverage.

Not in the default set:
  fig4_tier_comparison.pdf Cross-tier and cross-protocol performance. The same
                           numbers appear in the tier table, and the referees
                           asked that results not be duplicated across text,
                           table and figure. Build it explicitly with
                           ``python s09_figures.py 4`` if wanted for the SI.

Note that the file names carry historical digits and no longer match the figure
numbers LaTeX assigns; ``build_documents.py`` writes correctly numbered copies
into ``submission_figures/`` from the order of appearance in the manuscript.
  fig6_representation_benchmark.pdf  Representation and baseline comparison.
  fig7_invariance.pdf      Representation invariance to equivalent PSMILES.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polyx_rev import FIGS, OUT

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 300, 'savefig.bbox': 'tight',
})
GREEN, ORANGE, RED, BLUE, PURPLE = ('#27AE60', '#F39C12', '#E74C3C',
                                    '#2E86C1', '#7D3C98')
PROV: dict[str, list[str]] = {}


def note(fig, *lines):
    PROV.setdefault(fig, []).extend(lines)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'{name}.{ext}')
    plt.close(fig)
    with open(FIGS / f'{name}.provenance.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(PROV.get(name, ['(no provenance recorded)'])) + '\n')
    print(f'[fig] {name}')


def load(name):
    with open(OUT / name, encoding='utf-8') as f:
        return json.load(f)


# --------------------------------------------------------------------------
def fig2_gc():
    gc = pd.read_csv(OUT / 's04_gc_predictions.csv')
    s = load('s04_gc_summary.json')
    canon = s['canonical_eight']

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) the eight canonical polymers
    names = list(canon)
    exp = [canon[n]['manuscript_experimental_tg_K'] for n in names]
    pred = [canon[n]['gc_predicted_tg_K'] for n in names]
    ax[0].scatter(exp, pred, s=70, c=BLUE, edgecolor='k', zorder=3)
    for n, x, y in zip(names, exp, pred):
        ax[0].annotate(n, (x, y), textcoords='offset points', xytext=(5, 5),
                       fontsize=9)
    lim = [150, 420]
    ax[0].plot(lim, lim, 'k-', lw=1)
    ax[0].fill_between(lim, [v - 10 for v in lim], [v + 10 for v in lim],
                       color='r', alpha=0.10, label='$\\pm$10 K')
    ax[0].set(xlim=lim, ylim=lim, xlabel='Experimental $T_g$ (K)',
              ylabel='Tier 1 predicted $T_g$ (K)',
              title=f'(a) Eight canonical homopolymers\n'
                    f'MAE = {s["canonical_eight_mae"]:.1f} K (N = 8)')
    ax[0].legend(loc='upper left')
    ax[0].grid(alpha=0.3, ls='--')

    # (b) the full dataset
    ok = gc[gc['pred_tg'].notna()]
    ax[1].scatter(ok['exp_tg'], ok['pred_tg'], s=4, alpha=0.18, c=BLUE,
                  edgecolor='none')
    lim2 = [100, 800]
    ax[1].plot(lim2, lim2, 'k-', lw=1)
    o = s['overall_all_predictions']
    ax[1].set(xlim=lim2, ylim=lim2, xlabel='Experimental $T_g$ (K)',
              ylabel='Tier 1 predicted $T_g$ (K)',
              title=f'(b) Full PolyMetriX collection\n'
                    f'MAE = {o["mae"]:.0f} K, $R^2$ = {o["r2"]:.2f} '
                    f'(N = {o["n"]:,})')
    ax[1].grid(alpha=0.3, ls='--')

    # (c) error by coverage band. Bands rather than the fine bins: the finest
    # low-coverage bin contains a single polymer, so plotting it would imply a
    # precision the data do not support.
    bands = s['by_coverage_band']
    labels = {'below_0.75': '$\\leq$ 0.75', '0.75_to_below_1.00': '0.75-1.00',
              'full_coverage': '1.00 (complete)'}
    ks = [k for k in labels if isinstance(bands.get(k), dict)]
    mae = [bands[k]['mae'] for k in ks]
    ns = [bands[k]['n'] for k in ks]
    b = ax[2].bar(range(len(ks)), mae, color=ORANGE, edgecolor='k', lw=0.5,
                  width=0.62)
    for bar, n in zip(b, ns):
        ax[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                   f'n = {n:,}', ha='center', fontsize=9)
    ax[2].set_xticks(range(len(ks)))
    ax[2].set_xticklabels([labels[k] for k in ks])
    ax[2].set_ylim(0, max(mae) * 1.18)
    ax[2].set(xlabel='Fraction of heavy atoms matched by a group',
              ylabel='MAE (K)', title='(c) Error versus group coverage')
    ax[2].grid(axis='y', alpha=0.3, ls='--')

    note('fig2_gc_validation',
         'Panel (a): outputs/s04_gc_summary.json -> canonical_eight; '
         'observed values are the Table 1 literature Tg, calculated values are '
         'Tier 1 output. N = 8.',
         f'Panel (b): outputs/s04_gc_predictions.csv, all parseable polymers, '
         f'N = {len(ok)}. Observed = PolyMetriX experimental Tg; '
         f'calculated = Tier 1 output with frozen parameters.',
         'Panel (c): outputs/s04_gc_summary.json -> by_atom_coverage.')
    fig.tight_layout()
    save(fig, 'fig2_gc_validation')


# --------------------------------------------------------------------------
def fig3_parity(tag='published_42_ecfp4_2048'):
    z = np.load(OUT / 'runs' / f'{tag}.npz')
    y, p = z['y_test'], z['ens_test']
    res = p - y
    d = load('s02_benchmark_summary.json')

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    hb = ax[0].hexbin(y, p, gridsize=40, cmap='viridis', mincnt=1)
    lim = [min(y.min(), p.min()) - 20, max(y.max(), p.max()) + 20]
    ax[0].plot(lim, lim, 'k-', lw=1.2)
    from sklearn.metrics import r2_score, mean_absolute_error
    ax[0].set(xlim=lim, ylim=lim, xlabel='Experimental $T_g$ (K)',
              ylabel='Predicted $T_g$ (K)',
              title=f'(a) Tier 2 parity (N = {len(y)})\n'
                    f'$R^2$ = {r2_score(y, p):.3f}, '
                    f'MAE = {mean_absolute_error(y, p):.1f} K')
    fig.colorbar(hb, ax=ax[0], label='count')

    ax[1].scatter(p, res, s=8, alpha=0.35, c=BLUE, edgecolor='none')
    ax[1].axhline(0, color='k', lw=1)
    ax[1].axhline(res.mean(), color=RED, ls='--', lw=1.2,
                  label=f'bias = {res.mean():+.1f} K')
    ax[1].set(xlabel='Predicted $T_g$ (K)', ylabel='Residual (K)',
              title='(b) Residuals versus prediction')
    ax[1].legend()
    ax[1].grid(alpha=0.3, ls='--')

    ax[2].hist(res, bins=45, color=BLUE, edgecolor='k', lw=0.4, alpha=0.85)
    ax[2].axvline(0, color='k', lw=1)
    ax[2].set(xlabel='Residual (K)', ylabel='Count',
              title=f'(c) Residual distribution\n'
                    f'SD = {res.std(ddof=1):.1f} K, '
                    f'skew = {pd.Series(res).skew():.2f}')
    ax[2].grid(axis='y', alpha=0.3, ls='--')

    note('fig3_ml_parity',
         f'All panels: outputs/runs/{tag}.npz (archived test predictions of '
         f'the Tier 2 ensemble). N = {len(y)}. Observed = PolyMetriX '
         f'experimental Tg; calculated = model output.')
    fig.tight_layout()
    save(fig, 'fig3_ml_parity')


# --------------------------------------------------------------------------
def fig4_tiers():
    s = load('s02_benchmark_summary.json')
    protos = ['published', 'scaffold', 'cluster', 'random']
    reps = [('ecfp4_2048', 'ECFP4 (binary)'), ('pb_cls', 'polyBERT CLS'),
            ('pb_mean', 'polyBERT mean-pooled')]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    w = 0.25
    xs = np.arange(len(protos))
    for i, (rep, label) in enumerate(reps):
        m, e = [], []
        for pr in protos:
            k = f'{pr}|{rep}'
            m.append(s[k]['test_r2_mean'] if k in s else np.nan)
            e.append(s[k]['test_r2_sd'] if k in s else 0.0)
        ax[0].bar(xs + (i - 1) * w, m, w, yerr=e, capsize=3, label=label,
                  edgecolor='k', lw=0.5,
                  color=[BLUE, ORANGE, GREEN][i])
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(['as submitted\n(1 split)', 'scaffold\n(5 seeds)',
                           'Butina cluster\n(5 seeds)', 'random\n(5 seeds)'])
    ax[0].set(ylabel='Test $R^2$', title='(a) Representation by split protocol')
    ax[0].legend()
    ax[0].grid(axis='y', alpha=0.3, ls='--')
    ax[0].axhline(0, color='k', lw=0.8)

    # (b) canonical polymer multi-tier comparison
    tc = load('s10_tier_table.json') if (OUT / 's10_tier_table.json').exists() else None
    if tc:
        names = [r['polymer'] for r in tc['rows']]
        xs2 = np.arange(len(names))
        for i, (key, label, c) in enumerate(
                [('exp', 'Experimental', 'k'), ('tier1', 'Tier 1 GC', BLUE),
                 ('tier2', 'Tier 2 ML', ORANGE), ('tier3', 'Tier 3 polyBERT', GREEN)]):
            vals = [r[key] for r in tc['rows']]
            ax[1].bar(xs2 + (i - 1.5) * 0.2, vals, 0.2, label=label,
                      color=c, edgecolor='k', lw=0.4)
        ax[1].set_xticks(xs2)
        ax[1].set_xticklabels(names, rotation=45, ha='right')
        ax[1].set(ylabel='$T_g$ (K)', title='(b) Canonical homopolymers')
        ax[1].legend(fontsize=8)
        ax[1].grid(axis='y', alpha=0.3, ls='--')
    note('fig4_tier_comparison',
         'Panel (a): outputs/s02_benchmark_summary.json; bars are the mean '
         'test R2 across seeds, error bars the across-seed standard deviation.',
         'Panel (b): outputs/s10_tier_table.json, recomputed predictions for '
         'the eight canonical polymers. Experimental values are literature.')
    fig.tight_layout()
    save(fig, 'fig4_tier_comparison')


# --------------------------------------------------------------------------
def fig5_reliability(proto='published', seed=42):
    per = pd.read_csv(OUT / f's05_perpolymer_{proto}_{seed}.csv')
    s = load(f's05_calibration_{proto}_{seed}.json')
    n = len(per)

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))

    # (a) Error against similarity. The categorical domain label is degenerate
    # on this partition (nearly every query is "in-domain"), so plotting the
    # categories would spend a quarter of the figure on a classification the
    # text dismisses. The graded quantity, with the conventional threshold
    # drawn on it, shows in one glance why that threshold does no work here.
    sim = per['max_tanimoto'].values
    err = per['abs_error'].values
    ax[0].scatter(sim, err, s=9, alpha=0.30, c=BLUE, edgecolor='none')
    qs = np.quantile(sim, np.linspace(0, 1, 6))
    centres, means = [], []
    for i in range(5):
        m = (sim >= qs[i]) & (sim <= qs[i + 1])
        if m.sum():
            centres.append(0.5 * (qs[i] + qs[i + 1]))
            means.append(err[m].mean())
    ax[0].plot(centres, means, '-o', color=RED, lw=2, ms=6,
               label='quintile mean absolute error')
    ax[0].axvline(0.30, color='k', ls='--', lw=1.2)
    ax[0].text(0.305, ax[0].get_ylim()[1] * 0.94,
               'conventional\nin-domain\nthreshold (0.30)', fontsize=8,
               va='top', color='k')
    ax[0].set(xlabel='Maximum Tanimoto similarity to training set',
              ylabel='Absolute error (K)',
              title=f'(a) Error versus similarity\n(test partition, N = {n})')
    ax[0].legend(loc='upper right', fontsize=8)
    ax[0].grid(alpha=0.3, ls='--')

    # (b) PRI distribution
    pri = per['pri_recalibrated'].values
    ax[1].hist(pri, bins=np.arange(0, 1.02, 0.04), color=BLUE,
               edgecolor='k', lw=0.4, alpha=0.85)
    for t, c in ((0.75, GREEN), (0.50, ORANGE), (0.30, RED)):
        ax[1].axvline(t, color=c, ls='--', lw=1.2)
    d = s['pri_distribution_recalibrated']
    ax[1].set(xlabel='Prediction Reliability Index', ylabel='Count',
              title=f'(b) PRI distribution\nmedian {d["median"]:.2f} '
                    f'(IQR {d["q25"]:.2f}-{d["q75"]:.2f})')
    ax[1].grid(axis='y', alpha=0.3, ls='--')

    # (c) THE calibration panel: does a higher PRI mean a lower error?
    q = s['error_by_pri_quintile_recalibrated']
    xs = np.arange(len(q))
    ax[2].bar(xs, [b_['mae'] for b_ in q], color=PURPLE, edgecolor='k', lw=0.5)
    for i, b_ in enumerate(q):
        ax[2].text(i, b_['mae'] + 1, f'n={b_["n"]}', ha='center', fontsize=8)
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels([f'Q{i+1}' for i in xs])
    rho = s['rank_correlation_with_absolute_error']['pri_recalibrated']
    ax[2].set(xlabel='PRI quintile (low to high)', ylabel='MAE (K)',
              title=f'(c) Error versus reliability\n'
                    f'Spearman $\\rho$ = {rho["spearman"]:.3f} '
                    f'(p = {rho["p"]:.1e})')
    ax[2].grid(axis='y', alpha=0.3, ls='--')

    # (d) conformal coverage
    lv = s['conformal']['levels']
    nom = [lv[k]['nominal_coverage'] for k in lv]
    emp = [lv[k]['empirical_coverage'] for k in lv]
    ax[3].plot([0.5, 1.0], [0.5, 1.0], 'k--', lw=1, label='ideal')
    ax[3].scatter(nom, emp, s=90, c=RED, edgecolor='k', zorder=3)
    for x, y_, k in zip(nom, emp, lv):
        ax[3].annotate(f'{lv[k]["median_halfwidth_K"]:.0f} K',
                       (x, y_), textcoords='offset points', xytext=(6, -12),
                       fontsize=9)
    ax[3].set(xlabel='Nominal coverage', ylabel='Empirical coverage',
              xlim=(0.6, 1.0), ylim=(0.6, 1.0),
              title='(d) Conformal interval coverage\n(labels: median '
                    'half-width)')
    ax[3].legend()
    ax[3].grid(alpha=0.3, ls='--')

    note('fig5_enhanced_features',
         f'All panels computed on the {proto} split (seed {seed}) test '
         f'partition, N = {n}, from outputs/s05_perpolymer_{proto}_{seed}.csv '
         f'and outputs/s05_calibration_{proto}_{seed}.json.',
         'All values are calculated from archived model predictions; none is '
         'illustrative. AD, PRI and conformal parameters were fitted on '
         'training out-of-fold data and the validation partition only.')
    fig.tight_layout()
    save(fig, 'fig5_enhanced_features')


# --------------------------------------------------------------------------
def fig6_representations(proto='scaffold'):
    s = load('s02_benchmark_summary.json')
    labels = {
        'mean': 'Mean predictor', 'ridge_ecfp4': 'Ridge / ECFP4',
        'rdkit_desc': 'RDKit 2-D descriptors',
        'polymetrix_desc': 'PolyMetriX hierarchical',
        'ecfp4_1024': 'ECFP4 1024 bit', 'ecfp4_2048': 'ECFP4 2048 bit',
        'ecfp4_4096': 'ECFP4 4096 bit', 'ecfp6_2048': 'ECFP6 2048 bit',
        'ecfp4_2048_counts': 'ECFP4 counts',
        'ecfp4_2048_methylcap': 'ECFP4, methyl cap',
        'pb_cls': 'polyBERT CLS', 'pb_mean': 'polyBERT mean-pooled',
    }
    rows = [(labels[r], s[f'{proto}|{r}']['test_r2_mean'],
             s[f'{proto}|{r}']['test_r2_sd'], s[f'{proto}|{r}']['test_mae_mean'])
            for r in labels if f'{proto}|{r}' in s]
    rows.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ys = np.arange(len(rows))
    cols = [GREEN if 'polyBERT' not in r[0] else ORANGE for r in rows]
    ax[0].barh(ys, [r[1] for r in rows], xerr=[r[2] for r in rows], capsize=3,
               color=cols, edgecolor='k', lw=0.5)
    ax[0].set_yticks(ys)
    ax[0].set_yticklabels([r[0] for r in rows])
    ax[0].set(xlabel='Test $R^2$ (mean $\\pm$ SD over 5 seeds)',
              title=f'(a) Representations, {proto} split')
    ax[0].grid(axis='x', alpha=0.3, ls='--')
    ax[0].axvline(0, color='k', lw=0.8)

    ax[1].barh(ys, [r[3] for r in rows], color=cols, edgecolor='k', lw=0.5)
    ax[1].set_yticks(ys)
    ax[1].set_yticklabels([])
    ax[1].set(xlabel='Test MAE (K)', title='(b) Same runs, MAE')
    ax[1].grid(axis='x', alpha=0.3, ls='--')
    note('fig6_representation_benchmark',
         f'outputs/s02_benchmark_summary.json, protocol "{proto}". Bars are '
         f'means over 5 seeds; error bars are across-seed standard deviations. '
         f'All representations use identical partitions and identical RF/GB '
         f'heads.')
    fig.tight_layout()
    save(fig, 'fig6_representation_benchmark')


# --------------------------------------------------------------------------
def fig7_invariance():
    a = load('s03_invariance.json')['summary']
    b = load('s03b_repeat_unit_invariance.json')['summary']
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

    kinds = [k for k in ('canonical', 'rotated_0', 'rotated_1') if k in a]
    xs = np.arange(len(kinds))
    ax[0].bar(xs - 0.2, [a[k]['cls_cosine_mean'] for k in kinds], 0.4,
              yerr=[a[k]['cls_cosine_mean'] - a[k]['cls_cosine_min'] for k in kinds],
              capsize=3, label='polyBERT CLS', color=ORANGE, edgecolor='k', lw=0.5)
    ax[0].bar(xs + 0.2, [a[k]['mean_cosine_mean'] for k in kinds], 0.4,
              yerr=[a[k]['mean_cosine_mean'] - a[k]['mean_cosine_min'] for k in kinds],
              capsize=3, label='polyBERT mean-pooled', color=GREEN,
              edgecolor='k', lw=0.5)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(['canonicalised', 'rewriting 1', 'rewriting 2'])
    ax[0].axhline(1.0, color='k', ls='--', lw=1)
    ax[0].set(ylabel='Cosine to the original embedding', ylim=(0.6, 1.03),
              title='(a) Same molecule, different SMILES writing')
    ax[0].legend(loc='lower left')
    ax[0].grid(axis='y', alpha=0.3, ls='--')

    ks = list(b)
    xs2 = np.arange(len(ks))
    for i, (key, label, c) in enumerate([
            ('ecfp4_tanimoto_mean', 'ECFP4 (Tanimoto)', BLUE),
            ('polybert_cls_cosine_mean', 'polyBERT CLS', ORANGE),
            ('polybert_mean_cosine_mean', 'polyBERT mean-pooled', GREEN)]):
        ax[1].bar(xs2 + (i - 1) * 0.27, [b[k][key] for k in ks], 0.27,
                  label=label, color=c, edgecolor='k', lw=0.5)
    ax[1].set_xticks(xs2)
    ax[1].set_xticklabels(['2 repeat units', '3 repeat units'])
    ax[1].axhline(1.0, color='k', ls='--', lw=1)
    ax[1].set(ylabel='Similarity to the single repeat unit', ylim=(0, 1.05),
              title='(b) Same polymer, multiplied repeat unit')
    ax[1].legend(loc='lower left')
    ax[1].grid(axis='y', alpha=0.3, ls='--')
    note('fig7_invariance',
         'Panel (a): outputs/s03_invariance.json, 200 probe polymers. '
         'Bars are means, whiskers reach the observed minimum.',
         'Panel (b): outputs/s03b_repeat_unit_invariance.json, 300 probe '
         'polymers, repeat units multiplied with RDKit.')
    fig.tight_layout()
    save(fig, 'fig7_invariance')


if __name__ == '__main__':
    import sys
    # Figure 4 (tier comparison) is deliberately not in the default set:
    # the same numbers are in the tier table, and the reviewers asked that
    # results not be duplicated across text, table and figure.
    only = sys.argv[1:] or ['2', '3', '5', '6', '7']
    fns = {'2': fig2_gc, '3': fig3_parity, '4': fig4_tiers,
           '5': fig5_reliability, '6': fig6_representations, '7': fig7_invariance}
    for k in only:
        try:
            fns[k]()
        except FileNotFoundError as e:
            print(f'[skip] figure {k}: missing {e.filename}')
